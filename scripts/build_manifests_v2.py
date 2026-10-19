"""Version TSAC groups after human repetition review; preserve immutable v1.

All repetition-normalized families constrain splitting conservatively, even if
they originally occupied one partition. This does NOT assert semantic identity
for unreviewed pairs, change texts/labels, or enable repetition normalization.
Only original train/validation texts are loaded. Test manifest rows are copied.
"""

from collections import Counter, defaultdict
import csv
import json

from audit_elongation_policy import collapse_repeated_letters
from audit_tsac_candidate_groups import load_groups
from audit_tsac_validation_split import SEED, VALIDATION_FRACTION, split_groups
from build_data_manifests import ROOT, ref_group_id, sha256, write_csv
from load_working_data import load_records
from normalize_text import normalize


ORIGINAL = ROOT / "data/manifests/v1"
OUTPUT = ROOT / "data/manifests/v2"
LEDGER = ROOT / "data/interim/repetition_review_decisions_2026-10-17.json"
INDEX = ROOT / "data/interim/repetition_review_index_2026-10-17.json"
RULE = "tsac-candidate-components-repetition-v2"


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def plan():
    ledger = json.loads(LEDGER.read_text(encoding="utf-8"))
    if (ledger["provenance"]["manifest_sha256"] != sha256(ORIGINAL / "tsac.csv") or
            ledger["provenance"]["index_sha256"] != sha256(INDEX)):
        raise ValueError("Review ledger provenance mismatch")
    index = json.loads(INDEX.read_text(encoding="utf-8"))
    if len(ledger["pairs"]) != 12 or len(index["pairs"]) != 12:
        raise ValueError("Expected twelve explicit human decisions")
    # This revision is authorized only for the reviewed all-same outcome.
    if any(p["decision"] != "Même sens" or p["decision_origin"] != "explicit_human_cell"
           for p in ledger["pairs"]):
        raise ValueError("This revision requires a different policy for non-same decisions")
    for pair, candidate in zip(ledger["pairs"], index["pairs"]):
        if any(pair[key] != candidate[key] for key in ("case_id", "family", "a", "b")):
            raise ValueError("Review pair reference changed")
    rows = load_records("tsac")  # v1; never reads official test text.
    by_ref = {(row["source_file"], row["source_line"]): row for row in rows}
    refs, old_groups, edge_count = load_groups()
    if set(refs) != set(by_ref):
        raise ValueError("Original grouping and manifest retained references differ")
    parent = {ref: ref for ref in refs}

    def find(ref):
        while parent[ref] != ref:
            parent[ref] = parent[parent[ref]]
            ref = parent[ref]
        return ref

    def join(group):
        for ref in group[1:]:
            parent[find(ref)] = find(group[0])

    for group in old_groups:
        join(group)
    repetition = defaultdict(list)
    for ref, row in by_ref.items():
        repetition[collapse_repeated_letters(normalize(row["text_raw"]))].append(ref)
    families = [tuple(sorted(group)) for group in repetition.values() if len(group) > 1]
    old_crossing = sum(len({by_ref[ref]["split"] for ref in group}) > 1 for group in families)
    if old_crossing != 9:
        raise ValueError("Original repetition collision count changed")
    for pair in ledger["pairs"]:
        a, b = ((pair[side]["file"], pair[side]["line"]) for side in ("a", "b"))
        if (a not in refs or b not in refs or
                collapse_repeated_letters(normalize(by_ref[a]["text_raw"])) !=
                collapse_repeated_letters(normalize(by_ref[b]["text_raw"]))):
            raise ValueError("Human pair does not match repetition candidate rule")
    # Include within-partition families too, so repartitioning cannot create
    # fresh collisions. They remain heuristic constraints, not human labels.
    for family in families:
        join(family)
    components = defaultdict(list)
    for ref in refs:
        components[find(ref)].append(ref)
    groups = sorted(tuple(sorted(group)) for group in components.values())
    validation, _, _ = split_groups(refs, groups)
    group_ids = {ref: ref_group_id("tsac", group) for group in groups for ref in group}
    if any(len({ref in validation for ref in group}) != 1 for group in families):
        raise ValueError("Repetition family crosses revised partitions")
    revised = read_csv(ORIGINAL / "tsac.csv")
    fields = list(revised[0])
    moved = Counter()
    for row in revised:
        if row["split"] == "test":
            continue
        ref = (row["source_file"], int(row["source_line"]))
        split = "validation" if ref in validation else "train"
        if split != row["split"]:
            moved[f"{row['split']}_to_{split}"] += 1
        row.update(split=split, group_id=group_ids[ref])
    multi = [group for group in groups if len(group) > 1]
    summary = {
        "version": "v2", "group_rule": RULE, "split_seed": SEED,
        "tsac_validation_fraction_target": VALIDATION_FRACTION,
        "original_manifest_sha256": sha256(ORIGINAL / "tsac.csv"),
        "human_decision_ledger_sha256": sha256(LEDGER),
        "original_candidate_edges": edge_count, "human_same_meaning_pairs": 12,
        "repetition_candidate_families_total": len(families),
        "original_crossing_repetition_families": old_crossing,
        "revised_crossing_repetition_families": 0,
        "groups_total": len(groups), "multi_row_groups": len(multi),
        "rows_in_multi_row_groups": sum(map(len, multi)), "largest_group": max(map(len, groups)),
        "mixed_label_groups": sum(len({refs[ref] for ref in group}) > 1 for group in multi),
        "rows_moved": dict(sorted(moved.items())),
        "tsac_partition_labels": {f"{split}/{label}": count for (split, label), count in
                                  sorted(Counter((r["split"], r["label"]) for r in revised).items())},
        "tsac_rows": len(revised), "tsac_excluded_train_rows": 1842,
        "madar_rows": 7988, "madar_excluded_train_rows": 12,
        "official_test_texts_loaded": 0, "source_test_modified": False,
        "texts_or_labels_modified": False,
    }
    return revised, fields, summary


def main():
    if OUTPUT.exists():
        raise FileExistsError("Refusing to overwrite existing v2 manifests")
    rows, fields, summary = plan()
    OUTPUT.mkdir()
    write_csv(OUTPUT / "tsac.csv", rows, fields)
    for name in ("madar.csv", "madar_exclusions.csv", "tsac_exclusions.csv"):
        (OUTPUT / name).write_bytes((ORIGINAL / name).read_bytes())
    write_csv(OUTPUT / "file_hashes.csv",
              [{"file": p.name, "sha256": sha256(p)} for p in sorted(OUTPUT.glob("*.csv"))],
              ["file", "sha256"])
    (OUTPUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2,
                                                   sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
