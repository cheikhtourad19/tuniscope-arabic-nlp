"""Independently verify the text-free v1 source-reference manifests.

Run: python3 scripts/verify_data_manifests.py
"""

from collections import Counter, defaultdict
import csv
import json

from audit_structure import ROOT, MADAR_DIR, TSAC_DIR, decoded_lines
from audit_tsac_candidate_groups import load_groups, sha256
from audit_tsac_exclusion_union import form
from build_data_manifests import MADAR_HASHES


OUTPUT = ROOT / "data/manifests/v1"


def read_csv(filename, expected_fields):
    with (OUTPUT / filename).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected_fields:
            raise ValueError(f"Unexpected columns in {filename}")
        return list(reader)


def check_partition(rows, expected, name):
    counts = Counter((row["split"], row["label"]) for row in rows)
    if counts != expected:
        raise ValueError(f"{name} split counts mismatch: {counts}")
    group_splits = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
    if any(len(splits) > 1 for splits in group_splits.values()):
        raise ValueError(f"{name} group crosses partitions")
    return len(group_splits)


def main():
    with (OUTPUT / "file_hashes.csv").open(encoding="utf-8", newline="") as stream:
        hashes = list(csv.DictReader(stream))
    if {item["file"] for item in hashes} != {
        "tsac.csv", "tsac_exclusions.csv", "madar.csv", "madar_exclusions.csv"
    } or any(sha256(OUTPUT / item["file"]) != item["sha256"] for item in hashes):
        raise ValueError("Manifest hashes do not match")
    summary = json.loads((OUTPUT / "summary.json").read_text(encoding="utf-8"))
    if summary["version"] != "v1" or summary["split_seed"] != 42:
        raise ValueError("Unexpected manifest version or seed")

    tsac = read_csv("tsac.csv", ["id", "source_file", "source_line", "source_id",
                                  "group_id", "official_split", "split", "label"])
    tsac_ex = read_csv("tsac_exclusions.csv", ["source_file", "source_line",
                                                "original_label", "reasons"])
    tsac_counts = {("train", "POS"): 5255, ("train", "NEG"): 4798,
                   ("validation", "POS"): 927, ("validation", "NEG"): 847,
                   ("test", "POS"): 1700, ("test", "NEG"): 1700}
    tsac_groups = check_partition(tsac, tsac_counts, "TSAC")
    tsac_refs = {(row["source_file"], int(row["source_line"])) for row in tsac}
    ex_refs = {(row["source_file"], int(row["source_line"])) for row in tsac_ex}
    if len(tsac_refs) != len(tsac) or len(ex_refs) != len(tsac_ex) or tsac_refs & ex_refs:
        raise ValueError("TSAC references repeated or retained and excluded")
    all_tsac_refs = set()
    for split in ("train", "test"):
        for label in ("POS", "NEG"):
            filename = f"{split}_{label.lower()}.txt"
            all_tsac_refs.update((filename, line) for line, _ in decoded_lines(TSAC_DIR / filename))
    if tsac_refs | ex_refs != all_tsac_refs or len(ex_refs) != 1842:
        raise ValueError("TSAC manifest does not account for every source row")
    if any(not file.startswith("train_") for file, _ in ex_refs):
        raise ValueError("Official test row excluded")
    if any(row["official_split"] != ("test" if row["split"] == "test" else "train")
           for row in tsac):
        raise ValueError("TSAC official split provenance mismatch")
    retained, candidate_groups, _ = load_groups()
    by_ref = {(row["source_file"], int(row["source_line"])): row for row in tsac}
    for group in candidate_groups:
        if len({by_ref[ref]["split"] for ref in group}) != 1:
            raise ValueError("TSAC candidate family crosses train/validation")
        if len({by_ref[ref]["group_id"] for ref in group}) != 1:
            raise ValueError("TSAC candidate family has multiple group IDs")
    if set(retained) != {ref for ref, row in by_ref.items() if row["split"] != "test"}:
        raise ValueError("TSAC projected train differs from manifest")
    test_forms = {form(text) for label in ("POS", "NEG")
                  for _, text in decoded_lines(TSAC_DIR / f"test_{label.lower()}.txt")}
    if any(form(text) in test_forms for label in ("POS", "NEG")
           for line, text in decoded_lines(TSAC_DIR / f"train_{label.lower()}.txt")
           if (f"train_{label.lower()}.txt", line) in retained):
        raise ValueError("Exact normalized TSAC train/test overlap remains")

    madar = read_csv("madar.csv", ["id", "source_file", "source_line", "source_id",
                                     "group_id", "city", "official_split", "split", "label"])
    madar_ex = read_csv("madar_exclusions.csv", ["id", "source_file", "source_line",
                                                   "source_id", "group_id", "city",
                                                   "official_split", "split", "label", "reason"])
    madar_counts = {("train", "MSA"): 1597, ("train", "TN"): 3194,
                    ("train", "EG"): 1597, ("validation", "MSA"): 200,
                    ("validation", "TN"): 400, ("validation", "EG"): 200,
                    ("test", "MSA"): 200, ("test", "TN"): 400, ("test", "EG"): 200}
    madar_groups = check_partition(madar, madar_counts, "MADAR")
    madar_refs = {(row["source_file"], int(row["source_line"])) for row in madar}
    madar_by_ref = {(row["source_file"], int(row["source_line"])): row for row in madar}
    madar_ex_refs = {(row["source_file"], int(row["source_line"])) for row in madar_ex}
    if len(madar_refs) != len(madar) or len(madar_ex_refs) != len(madar_ex) or madar_refs & madar_ex_refs:
        raise ValueError("MADAR references repeated or retained and excluded")
    selected_refs = set()
    text_splits = defaultdict(set)
    for city in MADAR_HASHES:
        filename = f"MADAR.corpus.{city}.tsv"
        path = MADAR_DIR / filename
        if sha256(path) != MADAR_HASHES[city]:
            raise ValueError(f"MADAR source changed: {filename}")
        for line, record in decoded_lines(path):
            if line == 1:
                continue
            sentence_id, source_split, language, sentence = record.split("\t")
            if source_split.startswith("corpus-6-test-corpus-26-"):
                ref = (filename, line)
                selected_refs.add(ref)
                if ref in madar_refs:
                    row = madar_by_ref[ref]
                    if row["source_id"] != sentence_id:
                        raise ValueError("MADAR source ID mismatch")
                    text_splits[sentence].add(row["split"])
    if madar_refs | madar_ex_refs != selected_refs or len(madar_ex_refs) != 12:
        raise ValueError("MADAR manifest does not account for selected source rows")
    if len({row["source_id"] for row in madar_ex}) != 3 or any(row["split"] != "train" for row in madar_ex):
        raise ValueError("MADAR whole-train-group exclusion mismatch")
    if any(len(splits) > 1 for splits in text_splits.values()):
        raise ValueError("MADAR exact text crosses splits")
    print(json.dumps({"tsac_rows": len(tsac), "tsac_excluded_train": len(tsac_ex),
                      "tsac_manifest_groups": tsac_groups, "madar_rows": len(madar),
                      "madar_excluded_train": len(madar_ex), "madar_source_groups": madar_groups,
                      "source_test_modified": False, "all_checks": "passed"},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
