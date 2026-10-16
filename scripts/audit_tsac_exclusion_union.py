"""Calculate TSAC train-only exclusion candidates without preparing a dataset.

Run from the project root:
    python3 scripts/audit_tsac_exclusion_union.py
    python3 scripts/audit_tsac_exclusion_union.py --private-output data/interim/tsac_exclusion_candidates_2026-10-16_v2.json

Printed output contains aggregate counts only. Optional row references are written
only inside Git-ignored data/interim/, never with corpus text. Neither source
files nor the official test are changed.
"""

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from audit_structure import ROOT, TSAC_DIR, decoded_lines


EXACT_INDEX = ROOT / "data/interim/exact_review_2026-10-04.tsv"
NEAR_INDEX = ROOT / "data/interim/tsac_near_review_2026-10-04.tsv"
SHORT_LEDGER = ROOT / "data/interim/short_tsac_review_decisions_2026-10-16.json"
SHORT_WORKBOOK = ROOT / "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4/tuniscope_short_tsac_review_decisions_2026-10-16.xlsx"
HUMAN_REVIEW = ROOT / "data/interim/human_review_2026-10-06.md"
EXPECTED_NEAR_INDEX_SHA256 = "5927ec7e2de4e5e00c687d068e092d75c561aa14d71264883db230e1e03f25e7"
EXPECTED_EXACT_INDEX_SHA256 = "74dcb39c7f5baeafdf0b4bc0ef3dd87d57fff8ee1f274a8b03dec6fab26cc9f0"
REFERENCE_PATTERN = re.compile(r"^((?:train|test)_(?:pos|neg)\.txt):(\d+) \((\d+) ligne\(s\)\)$")


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def form(text):
    """Candidate-search key only, not the future training normalization."""
    return " ".join(unicodedata.normalize("NFKC", text).split())


def load_rows():
    rows = {}
    by_split_raw = {"train": defaultdict(list), "test": defaultdict(list)}
    by_split_form = {"train": defaultdict(list), "test": defaultdict(list)}
    physical = Counter()
    for split in ("train", "test"):
        for label in ("POS", "NEG"):
            filename = f"{split}_{label.lower()}.txt"
            for line_number, text in decoded_lines(TSAC_DIR / filename):
                if text is None:
                    raise ValueError(f"Invalid UTF-8 in {filename}:{line_number}")
                ref = (filename, line_number)
                if ref in rows:
                    raise ValueError(f"Repeated source reference: {ref}")
                rows[ref] = {"text": text, "split": split, "label": label}
                physical[split] += 1
                if text.strip():
                    by_split_raw[split][text].append(ref)
                    by_split_form[split][form(text)].append(ref)
    return rows, by_split_raw, by_split_form, physical


def read_tsv(path, expected_sha256):
    if sha256(path) != expected_sha256:
        raise ValueError(f"Review index changed: {path.name}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


def parse_reference(value, split, rows, by_form):
    match = REFERENCE_PATTERN.fullmatch(value)
    if not match:
        raise ValueError(f"Malformed private row reference: {value!r}")
    filename, line_number, count = match.group(1), int(match.group(2)), int(match.group(3))
    ref = (filename, line_number)
    if ref not in rows or rows[ref]["split"] != split:
        raise ValueError(f"Reference outside {split} source files: {ref}")
    matching = by_form[split][form(rows[ref]["text"])]
    if len(matching) != count:
        raise ValueError(f"Occurrence count changed for {ref}")
    return ref, matching


def punctuation_only(text):
    visible = text.strip()
    return bool(visible) and all(unicodedata.category(char).startswith("P") for char in visible)


def load_exact_conflict_choices(exact_index, rows):
    """Bind the recorded single-reviewer labels to exact text groups in memory."""
    content = HUMAN_REVIEW.read_text(encoding="utf-8")
    case_to_label = {}
    for label, numbers in re.findall(r"^\| (POS|NEG) \| ([\d, ]+) \| \d+ \|$", content, re.MULTILINE):
        for number in numbers.split(","):
            case_id = int(number.strip())
            if case_id in case_to_label:
                raise ValueError("Repeated human conflict case ID")
            case_to_label[case_id] = label
    if len(case_to_label) != 12 or 325 in case_to_label:
        raise ValueError("Expected 12 single-reviewer POS/NEG conflict choices")
    text_to_label = {}
    for item in exact_index:
        if item["category"] != "exact_pos_neg_conflict":
            continue
        case_id = int(item["case_id"])
        if case_id == 325:
            continue
        if case_id not in case_to_label:
            raise ValueError(f"Missing human decision for conflict case {case_id}")
        ref = (item["file_a"], int(item["line_a"]))
        other = (item["file_b"], int(item["line_b"]))
        if ref not in rows or other not in rows or rows[ref]["text"] != rows[other]["text"]:
            raise ValueError(f"Conflict index changed in case {case_id}")
        text_to_label[rows[ref]["text"]] = case_to_label[case_id]
    if len(text_to_label) != 12:
        raise ValueError("Expected 12 distinct reviewed exact conflict texts")
    return text_to_label


def calculate():
    rows, by_raw, by_form, physical = load_rows()
    if sum(physical.values()) < 17065:
        raise ValueError("Unexpectedly small TSAC source files")
    reasons = defaultdict(set)

    # A single train line may match several distinct controls. Record all motifs.
    for text in by_raw["test"]:
        for ref in by_raw["train"].get(text, ()):
            reasons[ref].add("exact_train_test")
    for key in by_form["test"]:
        if key not in by_form["train"]:
            continue
        for ref in by_form["train"][key]:
            if rows[ref]["text"] not in by_raw["test"]:
                reasons[ref].add("normalized_exact_train_test")

    near_index = read_tsv(NEAR_INDEX, EXPECTED_NEAR_INDEX_SHA256)
    if len(near_index) != 124 or [int(item["case_id"]) for item in near_index] != list(range(1, 125)):
        raise ValueError("Expected 124 fixed near-pair review cases")
    for item in near_index:
        ref = (item["train_file"], int(item["train_line"]))
        test_ref = (item["test_file"], int(item["test_line"]))
        if ref not in rows or test_ref not in rows or rows[ref]["split"] != "train" or rows[test_ref]["split"] != "test":
            raise ValueError(f"Invalid near-pair source references in case {item['case_id']}")
        for train_ref in by_form["train"][form(rows[ref]["text"])]:
            reasons[train_ref].add("reviewed_near_train_test")

    exact_index = read_tsv(EXACT_INDEX, EXPECTED_EXACT_INDEX_SHA256)
    invalid_exact = [item for item in exact_index if item["case_id"] == "325"]
    if len(invalid_exact) != 1 or invalid_exact[0]["category"] != "exact_pos_neg_conflict":
        raise ValueError("Missing reviewed invalid TSAC case 325")
    invalid_ref = (invalid_exact[0]["file_a"], int(invalid_exact[0]["line_a"]))
    if invalid_ref not in rows or not punctuation_only(rows[invalid_ref]["text"]):
        raise ValueError("Reviewed case 325 is not punctuation-only")
    for ref in by_raw["train"][rows[invalid_ref]["text"]]:
        reasons[ref].add("reviewed_non_linguistic")

    with SHORT_LEDGER.open("r", encoding="utf-8") as stream:
        ledger = json.load(stream)
    if sha256(SHORT_WORKBOOK) != ledger["final_workbook_sha256"]:
        raise ValueError("Finalized short-review workbook changed after the decision ledger")
    decisions = ledger["decisions"]
    if len(decisions) != 160 or [item["case_id"] for item in decisions] != list(range(1, 161)):
        raise ValueError("Expected 160 fixed short-pair review cases")
    if Counter(item["decision"] for item in decisions) != {
        "Même sens": 152, "Sens différent": 7, "Non linguistique": 1,
    }:
        raise ValueError("Short-pair decision counts changed")
    for item in decisions:
        train_ref, matching_train = parse_reference(item["train_reference"], "train", rows, by_form)
        test_ref, _ = parse_reference(item["test_reference"], "test", rows, by_form)
        if form(rows[train_ref]["text"]) == form(rows[test_ref]["text"]):
            raise ValueError(f"Short candidate unexpectedly identical in case {item['case_id']}")
        if item["decision"] == "Même sens":
            for ref in matching_train:
                reasons[ref].add("reviewed_short_near_train_test")
        elif item["decision"] == "Non linguistique":
            if item["case_id"] != 158 or not punctuation_only(rows[train_ref]["text"]):
                raise ValueError("Non-linguistic short decision requires side-specific review")
            for ref in matching_train:
                reasons[ref].add("reviewed_non_linguistic")

    blank_refs = [ref for ref, row in rows.items() if row["split"] == "train" and not row["text"].strip()]
    for ref in blank_refs:
        reasons[ref].add("blank_train_line")

    train_refs = {ref for ref, row in rows.items() if row["split"] == "train"}
    if not set(reasons).issubset(train_refs):
        raise AssertionError("An official test row was marked for exclusion")
    first_stage_count = len(reasons)
    kept_train = train_refs - set(reasons)
    kept_by_raw = defaultdict(list)
    for ref in kept_train:
        kept_by_raw[rows[ref]["text"]].append(ref)
    duplicate_groups = [group for group in kept_by_raw.values() if len(group) > 1]
    conflicts = [group for group in duplicate_groups if len({rows[ref]["label"] for ref in group}) > 1]

    # Deduplicate only after train/test exclusions. A reviewed POS/NEG conflict
    # keeps an occurrence bearing the chosen original label; no source relabeling.
    choices = load_exact_conflict_choices(exact_index, rows)
    for group in duplicate_groups:
        labels = {rows[ref]["label"] for ref in group}
        if len(labels) > 1:
            text = rows[group[0]]["text"]
            chosen = choices.get(text)
            if chosen is None or chosen not in labels:
                raise ValueError("Remaining exact label conflict lacks applicable human choice")
            representatives = [ref for ref in group if rows[ref]["label"] == chosen]
        else:
            representatives = group
        representative = min(representatives)
        for ref in group:
            if ref != representative:
                reasons[ref].add("within_train_exact_duplicate")
    after_raw_dedup = train_refs - set(reasons)
    if len(after_raw_dedup) != len(kept_by_raw):
        raise AssertionError("Expected one retained row per distinct exact train text")
    # A few raw-distinct strings still collapse to the same light screening key.
    # They have the same label in this source snapshot; keep one deterministically.
    kept_by_form = defaultdict(list)
    for ref in after_raw_dedup:
        kept_by_form[form(rows[ref]["text"])].append(ref)
    normalized_internal_groups = [group for group in kept_by_form.values() if len(group) > 1]
    if any(len({rows[ref]["label"] for ref in group}) > 1 for group in normalized_internal_groups):
        raise ValueError("Normalized within-train duplicate has conflicting labels; needs review")
    for group in normalized_internal_groups:
        representative = min(group)
        for ref in group:
            if ref != representative:
                reasons[ref].add("within_train_normalized_duplicate")
    final_kept_train = train_refs - set(reasons)
    if len(final_kept_train) != len(kept_by_form):
        raise AssertionError("Expected one retained row per light-normalized train form")
    residual_normalized_overlap = sum(
        form(rows[ref]["text"]) in by_form["test"] for ref in final_kept_train
    )
    if residual_normalized_overlap:
        raise AssertionError("Normalized-exact train/test overlap remains in the candidate train")

    by_reason = Counter(reason for row_reasons in reasons.values() for reason in row_reasons)
    by_label = Counter(rows[ref]["label"] for ref in reasons)
    reason_multiplicity = Counter(len(row_reasons) for row_reasons in reasons.values())
    summary = {
        "scope": "TSAC official train only; read-only exclusion candidates, not a prepared dataset",
        "source_physical_rows": dict(sorted(physical.items())),
        "train_rows_candidate_after_leakage_and_invalid_controls": len(kept_train),
        "train_rows_candidate_after_raw_exact_dedup": len(after_raw_dedup),
        "unique_train_rows_candidate_excluded_before_within_train_dedup": first_stage_count,
        "excluded_unique_train_rows_candidate": len(reasons),
        "candidate_rows_by_reason_non_additive": dict(sorted(by_reason.items())),
        "candidate_rows_by_label": dict(sorted(by_label.items())),
        "rows_with_multiple_reasons": sum(count for multiplicity, count in reason_multiplicity.items() if multiplicity > 1),
        "candidate_reason_multiplicity": dict(sorted(reason_multiplicity.items())),
        "train_rows_remaining_after_candidate_union": len(final_kept_train),
        "exact_duplicate_groups_within_train_after_first_stage": len(duplicate_groups),
        "rows_in_exact_duplicate_groups_within_train_after_first_stage": sum(len(group) for group in duplicate_groups),
        "exact_label_conflict_groups_within_train_after_first_stage": len(conflicts),
        "exact_label_conflict_groups_with_reviewed_choice_applied": len(conflicts),
        "normalized_duplicate_groups_within_train_after_raw_dedup": len(normalized_internal_groups),
        "residual_normalized_exact_train_test_rows_after_candidate_union": residual_normalized_overlap,
        "residual_exact_duplicate_groups_within_candidate_train": 0,
        "residual_normalized_duplicate_groups_within_candidate_train": 0,
        "official_test_rows_marked_for_exclusion": 0,
    }
    provenance = {
        "tsac_files_sha256": {
            filename: sha256(TSAC_DIR / filename)
            for filename in ("train_pos.txt", "train_neg.txt", "test_pos.txt", "test_neg.txt")
        },
        "near_index_sha256": sha256(NEAR_INDEX),
        "exact_index_sha256": sha256(EXACT_INDEX),
        "short_decision_ledger_sha256": sha256(SHORT_LEDGER),
        "human_review_sha256": sha256(HUMAN_REVIEW),
    }
    private_rows = [
        {"file": ref[0], "line": ref[1], "label": rows[ref]["label"], "reasons": sorted(row_reasons)}
        for ref, row_reasons in sorted(reasons.items())
    ]
    return summary, provenance, private_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-output", type=Path, help="Optional JSON under Git-ignored data/interim/")
    args = parser.parse_args()
    summary, provenance, private_rows = calculate()
    if args.private_output is not None:
        target = args.private_output.resolve()
        allowed = (ROOT / "data/interim").resolve()
        if allowed not in target.parents:
            raise ValueError("Private output must be a file inside data/interim/")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite existing private ledger: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            json.dump({"schema_version": "1.0", "summary": summary, "provenance": provenance,
                       "candidate_train_rows": private_rows}, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        summary["private_output"] = str(target.relative_to(ROOT))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
