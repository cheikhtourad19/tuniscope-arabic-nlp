"""Read-only screen for related TSAC texts inside the projected working train.

Run from the project root:
    python3 scripts/screen_tsac_within_train.py
    python3 scripts/screen_tsac_within_train.py --private-index data/interim/tsac_within_train_near_2026-10-16.json

The output contains aggregate counts only. An optional Git-ignored private index
contains source references and similarity scores, never corpus text. A candidate
pair is not a confirmed same-meaning pair or a decided validation group.
"""

import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path

from audit_structure import ROOT, TSAC_DIR, decoded_lines
from audit_tsac_exclusion_union import form
from screen_near_duplicates import (
    LONG_MINIMUM_LENGTH,
    LONG_SEQUENCE_RATIO,
    LONG_TRIGRAM_JACCARD,
    MAX_GRAM_DOCUMENT_FREQUENCY,
    SHORT_MINIMUM_LENGTH,
    SHORT_SEQUENCE_RATIO,
    deletion_keys,
    trigrams,
)


UNION_LEDGER = ROOT / "data/interim/tsac_exclusion_candidates_2026-10-16_v2.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retained_forms():
    with UNION_LEDGER.open("r", encoding="utf-8") as stream:
        ledger = json.load(stream)
    if ledger["summary"]["excluded_unique_train_rows_candidate"] != 1842:
        raise ValueError("Exclusion ledger does not match the verified v2 audit")
    excluded = {(item["file"], item["line"]) for item in ledger["candidate_train_rows"]}
    if len(excluded) != 1842 or any(not file.startswith("train_") for file, _ in excluded):
        raise ValueError("Invalid train-only exclusion references")
    for filename, expected in ledger["provenance"]["tsac_files_sha256"].items():
        if sha256(TSAC_DIR / filename) != expected:
            raise ValueError(f"TSAC source file changed: {filename}")
    kept = {}
    for label in ("POS", "NEG"):
        filename = f"train_{label.lower()}.txt"
        for line_number, text in decoded_lines(TSAC_DIR / filename):
            if text is None:
                raise ValueError(f"Invalid UTF-8 in {filename}:{line_number}")
            ref = (filename, line_number)
            if ref in excluded:
                continue
            key = form(text)
            if key in kept:
                raise ValueError("Projected train still contains a repeated screening form")
            kept[key] = {"file": filename, "line": line_number, "label": label}
    if len(kept) != 11827:
        raise ValueError("Projected retained train count changed")
    return kept


def screen(forms):
    texts = sorted(forms)
    grams = [trigrams(text) for text in texts]
    deletion_index = defaultdict(set)
    postings = defaultdict(list)
    for index, text in enumerate(texts):
        if len(text) < LONG_MINIMUM_LENGTH + 4:
            for key in deletion_keys(text):
                deletion_index[key].add(index)
        if len(text) >= LONG_MINIMUM_LENGTH - 4:
            for gram in grams[index]:
                postings[gram].append(index)

    screened = set()
    primary = {}
    short = {}
    candidate_comparisons = Counter()
    long_without_informative = 0
    for index, text in enumerate(texts):
        if len(text) < SHORT_MINIMUM_LENGTH:
            candidates = set()
            for key in deletion_keys(text):
                candidates.update(deletion_index.get(key, ()))
            for other in candidates:
                if other == index or len(texts[other]) >= 9 or abs(len(texts[other]) - len(text)) > 1:
                    continue
                pair = tuple(sorted((index, other)))
                if pair in screened:
                    continue
                screened.add(pair)
                candidate_comparisons["short_under_8"] += 1
                ratio = SequenceMatcher(None, texts[pair[0]], texts[pair[1]], autojunk=False).ratio()
                short[pair] = ratio
            continue

        if len(text) < LONG_MINIMUM_LENGTH:
            candidates = set()
            for key in deletion_keys(text):
                candidates.update(deletion_index.get(key, ()))
            for other in candidates:
                if other == index or len(texts[other]) < SHORT_MINIMUM_LENGTH or abs(len(texts[other]) - len(text)) > 2:
                    continue
                pair = tuple(sorted((index, other)))
                if pair in screened:
                    continue
                screened.add(pair)
                candidate_comparisons["length_8_to_19"] += 1
                ratio = SequenceMatcher(None, texts[pair[0]], texts[pair[1]], autojunk=False).ratio()
                if ratio >= SHORT_SEQUENCE_RATIO:
                    primary[pair] = ratio
            continue

        shared_informative = Counter()
        for gram in grams[index]:
            matches = postings.get(gram, ())
            if len(matches) <= MAX_GRAM_DOCUMENT_FREQUENCY:
                shared_informative.update(matches)
        if not shared_informative:
            long_without_informative += 1
        minimum_shared = 2 if len(text) < 30 else 3
        for other, shared_count in shared_informative.items():
            if other == index or shared_count < minimum_shared:
                continue
            pair = tuple(sorted((index, other)))
            if pair in screened:
                continue
            other_text = texts[other]
            if min(len(text), len(other_text)) / max(len(text), len(other_text)) < 0.8:
                continue
            jaccard = len(grams[index] & grams[other]) / len(grams[index] | grams[other])
            if jaccard < LONG_TRIGRAM_JACCARD:
                continue
            screened.add(pair)
            candidate_comparisons["length_20_plus"] += 1
            ratio = SequenceMatcher(None, texts[pair[0]], texts[pair[1]], autojunk=False).ratio()
            if ratio >= LONG_SEQUENCE_RATIO:
                primary[pair] = ratio

    # A short form can also be found while processing a length-8 neighbor.
    # Keep such pairs in the cautious short category, not the primary screen.
    for pair in list(primary):
        if len(texts[pair[0]]) < SHORT_MINIMUM_LENGTH or len(texts[pair[1]]) < SHORT_MINIMUM_LENGTH:
            short[pair] = primary.pop(pair)

    return texts, primary, short, candidate_comparisons, long_without_informative


def components(pairs, texts, forms):
    parent = {}

    def find(index):
        parent.setdefault(index, index)
        if parent[index] != index:
            parent[index] = find(parent[index])
        return parent[index]

    for first, second in pairs:
        parent[find(first)] = find(second)
    grouped = defaultdict(list)
    for index in parent:
        grouped[find(index)].append(index)
    groups = list(grouped.values())
    return {
        "components": len(groups),
        "forms_in_components": sum(len(group) for group in groups),
        "largest_component_forms": max((len(group) for group in groups), default=0),
        "components_with_both_source_labels": sum(
            len({forms[texts[index]]["label"] for index in group}) > 1 for group in groups
        ),
    }


def pair_rows(pairs, texts, forms, category):
    records = []
    for first, second in sorted(pairs, key=lambda pair: (-pairs[pair], pair)):
        a, b = forms[texts[first]], forms[texts[second]]
        records.append({
            "category": category,
            "similarity": round(pairs[(first, second)], 4),
            "file_a": a["file"], "line_a": a["line"], "label_a": a["label"],
            "file_b": b["file"], "line_b": b["line"], "label_b": b["label"],
        })
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-index", type=Path, help="Optional JSON inside Git-ignored data/interim/")
    args = parser.parse_args()
    forms = retained_forms()
    texts, primary, short, comparisons, no_informative = screen(forms)
    involved_primary = {index for pair in primary for index in pair}
    involved_short = {index for pair in short for index in pair}
    summary = {
        "scope": "Projected TSAC train after v2 exclusions; similarity candidates only, no split or exclusion",
        "retained_train_rows_and_distinct_forms": len(forms),
        "primary_candidate_pairs_length_8_plus": len(primary),
        "primary_candidate_forms": len(involved_primary),
        "primary_pairs_with_opposite_source_labels": sum(
            forms[texts[a]]["label"] != forms[texts[b]]["label"] for a, b in primary
        ),
        "primary_candidate_components": components(primary, texts, forms),
        "short_under_8_candidate_pairs": len(short),
        "short_candidate_forms": len(involved_short),
        "short_candidate_components": components(short, texts, forms),
        "all_candidate_components_if_grouped_conservatively": components(
            {**primary, **short}, texts, forms
        ),
        "short_pairs_with_opposite_source_labels": sum(
            forms[texts[a]]["label"] != forms[texts[b]]["label"] for a, b in short
        ),
        "candidate_comparisons_after_retrieval": dict(sorted(comparisons.items())),
        "long_forms_without_informative_trigrams": no_informative,
        "source_test_rows_examined": 0,
    }
    if args.private_index is not None:
        target = args.private_index.resolve()
        allowed = (ROOT / "data/interim").resolve()
        if allowed not in target.parents:
            raise ValueError("Private index must be a file inside data/interim/")
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite private index: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as stream:
            json.dump({
                "schema_version": "1.0", "summary": summary,
                "exclusion_ledger_sha256": sha256(UNION_LEDGER),
                "primary_pairs": pair_rows(primary, texts, forms, "length_8_plus"),
                "short_pairs": pair_rows(short, texts, forms, "under_8_one_edit"),
            }, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        summary["private_index"] = str(target.relative_to(ROOT))
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
