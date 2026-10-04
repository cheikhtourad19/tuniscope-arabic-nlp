"""Read-only MADAR group and TSAC exact-duplicate audit.

Run ``python3 scripts/audit_duplicates.py`` from the project root.
Prints aggregate counts only: no protected corpus text, text hashes, or row IDs.
No cleaning, partitioning, or near-duplicate decisions are made here.
"""

from collections import Counter, defaultdict
import json

from audit_structure import MADAR_DIR, TSAC_DIR, decoded_lines


MADAR_CITIES = ("MSA", "Tunis", "Sfax", "Cairo")
EXPECTED_MADAR_LANG = {"MSA": "MSA", "Tunis": "TUN", "Sfax": "SFX", "Cairo": "CAI"}
CORPUS26_SPLITS = {
    "corpus-6-test-corpus-26-train": "train",
    "corpus-6-test-corpus-26-dev": "validation",
    "corpus-6-test-corpus-26-test": "test",
}


def madar_audit():
    # Keys and sentences remain only in memory while this script runs.
    all_id_splits = defaultdict(set)
    scoped = defaultdict(lambda: defaultdict(list))
    scoped_texts = defaultdict(list)
    city_rows = Counter()
    unexpected_language_rows = Counter()
    for city in MADAR_CITIES:
        path = MADAR_DIR / f"MADAR.corpus.{city}.tsv"
        for line_number, line in decoded_lines(path):
            if line_number == 1:
                continue
            if line is None:
                raise ValueError(f"Invalid UTF-8 in {path.name}, line {line_number}")
            fields = line.split("\t")
            if len(fields) != 4:
                raise ValueError(f"Malformed TSV row in {path.name}, line {line_number}")
            sentence_id, source_split, language, sentence = fields
            if language != EXPECTED_MADAR_LANG[city]:
                unexpected_language_rows[city] += 1
            all_id_splits[sentence_id].add(source_split)
            if source_split in CORPUS26_SPLITS:
                scoped[sentence_id][city].append(source_split)
                scoped_texts[sentence].append((sentence_id, CORPUS26_SPLITS[source_split]))
                city_rows[city] += 1

    split_counts = Counter()
    missing_city_groups = 0
    repeated_id_city_rows = 0
    split_conflict_groups = 0
    complete_four_city_groups = 0
    for city_map in scoped.values():
        seen_splits = set()
        if set(city_map) != set(MADAR_CITIES):
            missing_city_groups += 1
        else:
            complete_four_city_groups += 1
        for splits in city_map.values():
            repeated_id_city_rows += len(splits) - 1
            seen_splits.update(splits)
        if len(seen_splits) > 1:
            split_conflict_groups += 1
        elif len(seen_splits) == 1:
            split_counts[CORPUS26_SPLITS[next(iter(seen_splits))]] += 1

    # An identical string across different source IDs and official splits is a
    # potential lexical leakage issue, even if the ID-based grouping is sound.
    cross_split_text_groups = [
        rows for rows in scoped_texts.values()
        if len({split for _, split in rows}) > 1
    ]
    return {
        "corpus26_rows_by_city": dict(sorted(city_rows.items())),
        "unexpected_language_rows_by_city": {
            city: unexpected_language_rows[city] for city in MADAR_CITIES
        },
        "corpus26_unique_source_ids": len(scoped),
        "corpus26_complete_four_city_groups": complete_four_city_groups,
        "corpus26_groups_missing_a_city": missing_city_groups,
        "corpus26_repeated_id_city_rows": repeated_id_city_rows,
        "corpus26_groups_in_multiple_official_splits": split_conflict_groups,
        "corpus26_groups_by_project_split": dict(sorted(split_counts.items())),
        "all_selected_city_ids_in_multiple_source_splits": sum(
            len(splits) > 1 for splits in all_id_splits.values()
        ),
        "corpus26_exact_text_groups_crossing_splits": len(cross_split_text_groups),
        "corpus26_rows_in_cross_split_exact_text_groups": sum(
            len(rows) for rows in cross_split_text_groups
        ),
    }


def group_summary(groups):
    return {
        "groups": len(groups),
        "rows_in_groups": sum(len(rows) for rows in groups),
    }


def tsac_audit():
    by_text = defaultdict(list)
    by_file = defaultdict(lambda: defaultdict(list))
    by_trimmed_text = defaultdict(list)
    rows_by_file = Counter()
    for split in ("train", "test"):
        for label in ("POS", "NEG"):
            filename = f"{split}_{label.lower()}.txt"
            for line_number, line in decoded_lines(TSAC_DIR / filename):
                if line is None:
                    raise ValueError(f"Invalid UTF-8 in {filename}, line {line_number}")
                if not line.strip():
                    continue
                occurrence = (filename, line_number, split, label)
                by_text[line].append(occurrence)
                by_file[filename][line].append(occurrence)
                by_trimmed_text[line.strip()].append((line, occurrence))
                rows_by_file[filename] += 1

    duplicates = [rows for rows in by_text.values() if len(rows) > 1]
    cross_split = [
        rows for rows in by_text.values()
        if {row[2] for row in rows} == {"train", "test"}
    ]
    label_conflicts = [
        rows for rows in by_text.values()
        if len({row[3] for row in rows}) > 1
    ]
    train_conflicts = [
        rows for rows in label_conflicts
        if any(row[2] == "train" for row in rows)
    ]
    conflicts_within_train = sum(
        len({row[3] for row in rows if row[2] == "train"}) > 1
        for rows in label_conflicts
    )
    conflicts_within_test = sum(
        len({row[3] for row in rows if row[2] == "test"}) > 1
        for rows in label_conflicts
    )
    trimmed_variants = [
        rows for rows in by_trimmed_text.values()
        if len({raw_text for raw_text, _ in rows}) > 1
    ]
    trimmed_cross_split = [
        rows for rows in by_trimmed_text.values()
        if {occurrence[2] for _, occurrence in rows} == {"train", "test"}
    ]
    duplicate_split_presence = Counter(
        "+".join(sorted({row[2] for row in rows})) for rows in duplicates
    )
    conflict_split_presence = Counter(
        "+".join(sorted({row[2] for row in rows})) for rows in label_conflicts
    )
    return {
        "nonempty_rows_by_file": dict(sorted(rows_by_file.items())),
        "unique_exact_texts": len(by_text),
        "exact_duplicate_groups": group_summary(duplicates),
        "exact_duplicate_groups_by_split_presence": dict(sorted(duplicate_split_presence.items())),
        "exact_duplicate_excess_rows": sum(len(rows) - 1 for rows in duplicates),
        "exact_duplicate_excess_rows_within_file": {
            filename: sum(len(rows) - 1 for rows in text_map.values())
            for filename, text_map in sorted(by_file.items())
        },
        "exact_train_test_overlap": {
            **group_summary(cross_split),
            "train_rows": sum(row[2] == "train" for rows in cross_split for row in rows),
            "test_rows": sum(row[2] == "test" for rows in cross_split for row in rows),
        },
        "exact_pos_neg_label_conflicts": group_summary(label_conflicts),
        "exact_pos_neg_conflict_groups_by_split_presence": dict(sorted(conflict_split_presence.items())),
        "exact_pos_neg_conflict_groups_touching_train": len(train_conflicts),
        "exact_pos_neg_conflict_groups_within_train": conflicts_within_train,
        "exact_pos_neg_conflict_groups_within_test": conflicts_within_test,
        "whitespace_trimmed_groups_with_distinct_raw_texts": len(trimmed_variants),
        "whitespace_trimmed_train_test_overlap_groups": len(trimmed_cross_split),
    }


def main():
    print(json.dumps({"madar": madar_audit(), "tsac": tsac_audit()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
