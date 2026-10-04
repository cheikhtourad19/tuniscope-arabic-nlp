"""Screen TSAC train/test for possible near-duplicates without changing data.

Run ``python3 scripts/screen_near_duplicates.py`` from the project root.
This heuristic prints aggregate counts only; a match is NOT an exclusion rule.
It intentionally does not emit protected comments, individual text hashes, or IDs.
"""

import argparse
from collections import Counter, defaultdict
import csv
from difflib import SequenceMatcher
import json
from pathlib import Path
import unicodedata

from audit_structure import ROOT, TSAC_DIR, decoded_lines


MAX_GRAM_DOCUMENT_FREQUENCY = 800
LONG_MINIMUM_LENGTH = 20
SHORT_MINIMUM_LENGTH = 8
LONG_SEQUENCE_RATIO = 0.92
LONG_TRIGRAM_JACCARD = 0.65
SHORT_SEQUENCE_RATIO = 0.90


def screening_form(text):
    # Candidate search only. This is not the project's training normalization.
    return " ".join(unicodedata.normalize("NFKC", text).split())


def trigrams(text):
    return {text[index:index + 3] for index in range(len(text) - 2)}


def deletion_keys(text):
    return {text} | {text[:index] + text[index + 1:] for index in range(len(text))}


def load_by_split():
    result = {"train": defaultdict(list), "test": defaultdict(list)}
    for split in ("train", "test"):
        for label in ("POS", "NEG"):
            filename = f"{split}_{label.lower()}.txt"
            for line_number, line in decoded_lines(TSAC_DIR / filename):
                if line is None:
                    raise ValueError(f"Invalid UTF-8 in {filename}, line {line_number}")
                if line.strip():
                    result[split][screening_form(line)].append((filename, line_number, label))
    return result


def write_private_review_index(path, near_pairs, train, test):
    allowed_dir = (ROOT / "data/interim").resolve()
    target = path.resolve()
    if target == allowed_dir or allowed_dir not in target.parents:
        raise ValueError("Review index must be inside Git-ignored data/interim/")
    if target.exists():
        raise FileExistsError(f"Review index already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow((
            "case_id", "similarity", "train_file", "train_line", "train_labels",
            "train_occurrences", "test_file", "test_line", "test_labels", "test_occurrences",
        ))
        for case_id, (test_text, train_text, ratio) in enumerate(
            sorted(near_pairs, key=lambda item: (-item[2], item[0], item[1])), start=1
        ):
            train_rows = train[train_text]
            test_rows = test[test_text]
            writer.writerow((
                case_id, f"{ratio:.4f}", train_rows[0][0], train_rows[0][1],
                ",".join(sorted({row[2] for row in train_rows})), len(train_rows),
                test_rows[0][0], test_rows[0][1],
                ",".join(sorted({row[2] for row in test_rows})), len(test_rows),
            ))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--review-index", type=Path,
        help="Write a TSV of file/line references under Git-ignored data/interim/",
    )
    args = parser.parse_args()
    by_split = load_by_split()
    train = by_split["train"]
    test = by_split["test"]
    train_texts = list(train)
    test_texts = list(test)
    train_grams = [trigrams(text) for text in train_texts]
    postings = defaultdict(list)
    short_deletion_index = defaultdict(set)
    for index, text in enumerate(train_texts):
        if len(text) < LONG_MINIMUM_LENGTH + 4:
            for key in deletion_keys(text):
                short_deletion_index[key].add(index)
        if len(text) >= LONG_MINIMUM_LENGTH - 4:
            for gram in train_grams[index]:
                postings[gram].append(index)

    normalized_exact = []
    near_pairs = []
    candidate_pairs_scored = 0
    short_skipped = 0
    long_without_informative_grams = 0
    for test_text in test_texts:
        if test_text in train:
            normalized_exact.append(test_text)
            continue
        if len(test_text) < SHORT_MINIMUM_LENGTH:
            short_skipped += 1
            continue
        if len(test_text) < LONG_MINIMUM_LENGTH:
            candidates = set()
            for key in deletion_keys(test_text):
                candidates.update(short_deletion_index.get(key, ()))
            for index in sorted(candidates):
                train_text = train_texts[index]
                if abs(len(train_text) - len(test_text)) > 2:
                    continue
                candidate_pairs_scored += 1
                ratio = SequenceMatcher(None, test_text, train_text, autojunk=False).ratio()
                if ratio >= SHORT_SEQUENCE_RATIO:
                    near_pairs.append((test_text, train_text, ratio))
            continue

        test_grams = trigrams(test_text)
        shared_informative_grams = Counter()
        for gram in test_grams:
            gram_postings = postings.get(gram, ())
            if len(gram_postings) <= MAX_GRAM_DOCUMENT_FREQUENCY:
                shared_informative_grams.update(gram_postings)
        if not shared_informative_grams:
            long_without_informative_grams += 1
        minimum_shared = 2 if len(test_text) < 30 else 3
        for index, shared_count in shared_informative_grams.items():
            if shared_count < minimum_shared:
                continue
            train_text = train_texts[index]
            if min(len(test_text), len(train_text)) / max(len(test_text), len(train_text)) < 0.8:
                continue
            grams = train_grams[index]
            jaccard = len(test_grams & grams) / len(test_grams | grams)
            if jaccard < LONG_TRIGRAM_JACCARD:
                continue
            candidate_pairs_scored += 1
            ratio = SequenceMatcher(None, test_text, train_text, autojunk=False).ratio()
            if ratio >= LONG_SEQUENCE_RATIO:
                near_pairs.append((test_text, train_text, ratio))

    test_near_texts = {test_text for test_text, _, _ in near_pairs}
    train_near_texts = {train_text for _, train_text, _ in near_pairs}
    opposite_label_pairs = sum(
        any(test_row[2] != train_row[2]
            for test_row in test[test_text] for train_row in train[train_text])
        for test_text, train_text, _ in near_pairs
    )
    result = {
        "scope": "TSAC train versus official test only; screening candidates, not confirmed duplicates",
        "parameters": {
            "screening_form": "Unicode NFKC plus collapsed whitespace, only for candidate search",
            "short_length_range": "8–19 code points; shared one-deletion key and SequenceMatcher >= 0.90",
            "long_length_range": "20+ code points; informative character trigrams, Jaccard >= 0.65, SequenceMatcher >= 0.92",
            "max_trigram_document_frequency": MAX_GRAM_DOCUMENT_FREQUENCY,
            "minimum_length_ratio_for_long": 0.8,
        },
        "unique_screening_forms_train": len(train),
        "unique_screening_forms_test": len(test),
        "normalized_exact_overlap_forms": len(normalized_exact),
        "normalized_exact_overlap_test_rows": sum(len(test[text]) for text in normalized_exact),
        "near_candidate_pairs": len(near_pairs),
        "near_candidate_unique_test_forms": len(test_near_texts),
        "near_candidate_test_rows": sum(len(test[text]) for text in test_near_texts),
        "near_candidate_unique_train_forms": len(train_near_texts),
        "near_candidate_pairs_with_possible_opposite_labels": opposite_label_pairs,
        "candidate_pairs_scored_after_filters": candidate_pairs_scored,
        "test_forms_shorter_than_eight_not_screened": short_skipped,
        "long_test_forms_without_informative_trigrams": long_without_informative_grams,
    }
    if args.review_index is not None:
        write_private_review_index(args.review_index, near_pairs, train, test)
        result["private_review_index"] = str(args.review_index.resolve().relative_to(ROOT))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
