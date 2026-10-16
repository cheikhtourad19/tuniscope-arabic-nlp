"""Build text-free MADAR and TSAC source-reference manifests.

Run: python3 scripts/build_data_manifests.py
The output directory must not exist. Raw corpus files are read but never changed.
"""

from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path

from audit_structure import ROOT, MADAR_DIR, TSAC_DIR, decoded_lines
from audit_duplicates import MADAR_CITIES, EXPECTED_MADAR_LANG, CORPUS26_SPLITS
from audit_tsac_candidate_groups import UNION_LEDGER, load_groups, sha256
from audit_tsac_validation_split import SEED, VALIDATION_FRACTION, split_groups
from audit_tsac_exclusion_union import form


OUTPUT = ROOT / "data/manifests/v1"
MADAR_HASHES = {
    "MSA": "2b09710ece69651a0bee40f84c26993d017527386d3049a1061ed1158353d283",
    "Tunis": "4567b4622a49d1a8c0a661ac890d2fefa7c6546bf351ded6410b2505f3e0f8a6",
    "Sfax": "d6ea0d9ad1715fcac42f9ef59b20142715764f294ae0056dc285c2ee9f36c21b",
    "Cairo": "288bd22455ff0d5421348bf3a5c9e713b95c2f860bb8ed2eb1908a138c660429",
}


def ref_group_id(prefix, refs):
    payload = "\n".join(f"{file}:{line}" for file, line in sorted(refs))
    return f"{prefix}:{hashlib.sha256(payload.encode()).hexdigest()[:20]}"


def build_tsac():
    refs, groups, _ = load_groups()
    validation, _, _ = split_groups(refs, groups)
    group_ids = {ref: ref_group_id("tsac", group) for group in groups for ref in group}
    with UNION_LEDGER.open(encoding="utf-8") as stream:
        ledger = json.load(stream)
    excluded = {(item["file"], item["line"]): item for item in ledger["candidate_train_rows"]}
    if len(excluded) != 1842:
        raise ValueError("TSAC exclusion ledger count changed")
    for filename, expected in ledger["provenance"]["tsac_files_sha256"].items():
        if sha256(TSAC_DIR / filename) != expected:
            raise ValueError(f"TSAC source changed: {filename}")

    rows = []
    excluded_rows = []
    test_forms = defaultdict(list)
    for source_split in ("train", "test"):
        for label in ("POS", "NEG"):
            filename = f"{source_split}_{label.lower()}.txt"
            for line, text in decoded_lines(TSAC_DIR / filename):
                if text is None:
                    raise ValueError(f"Invalid UTF-8: {filename}:{line}")
                ref = (filename, line)
                if ref in excluded:
                    if source_split != "train":
                        raise ValueError("Official test row marked excluded")
                    excluded_rows.append({"source_file": filename, "source_line": line,
                                          "original_label": label,
                                          "reasons": "|".join(excluded[ref]["reasons"])})
                    continue
                if source_split == "train":
                    if ref not in refs or refs[ref] != label:
                        raise ValueError("Retained TSAC reference mismatch")
                    split = "validation" if ref in validation else "train"
                    group_id = group_ids[ref]
                else:
                    split = "test"
                    test_forms[form(text)].append(ref)
                    group_id = None
                rows.append({"id": f"tsac:{filename}:{line}", "source_file": filename,
                             "source_line": line, "source_id": f"{filename}:{line}",
                             "group_id": group_id, "official_split": source_split,
                             "split": split, "label": label})

    test_group_ids = {ref: ref_group_id("tsac-test", members)
                      for members in test_forms.values() for ref in members}
    for row in rows:
        if row["split"] == "test":
            row["group_id"] = test_group_ids[(row["source_file"], row["source_line"])]
    counts = Counter((row["split"], row["label"]) for row in rows)
    if counts != {("train", "POS"): 5255, ("train", "NEG"): 4798,
                  ("validation", "POS"): 927, ("validation", "NEG"): 847,
                  ("test", "POS"): 1700, ("test", "NEG"): 1700}:
        raise ValueError(f"Unexpected TSAC split counts: {counts}")
    if len(excluded_rows) != 1842:
        raise ValueError("TSAC exclusion count mismatch")
    return rows, excluded_rows, ledger


def build_madar():
    rows = []
    text_occurrences = defaultdict(list)
    source_groups = defaultdict(list)
    for city in MADAR_CITIES:
        filename = f"MADAR.corpus.{city}.tsv"
        path = MADAR_DIR / filename
        if sha256(path) != MADAR_HASHES[city]:
            raise ValueError(f"MADAR source changed: {filename}")
        for line, text in decoded_lines(path):
            if line == 1:
                if text != "sentID.BTEC\tsplit\tlang\tsent":
                    raise ValueError(f"Unexpected header: {filename}")
                continue
            if text is None:
                raise ValueError(f"Invalid UTF-8: {filename}:{line}")
            fields = text.split("\t")
            if len(fields) != 4:
                raise ValueError(f"Malformed MADAR row: {filename}:{line}")
            sentence_id, official, language, sentence = fields
            if official not in CORPUS26_SPLITS:
                continue
            if language != EXPECTED_MADAR_LANG[city]:
                raise ValueError(f"Unexpected MADAR language: {filename}:{line}")
            split = CORPUS26_SPLITS[official]
            row = {"id": f"madar:{city}:{sentence_id}", "source_file": filename,
                   "source_line": line, "source_id": sentence_id,
                   "group_id": f"madar:{sentence_id}", "city": city,
                   "official_split": official, "split": split,
                   "label": {"MSA": "MSA", "Tunis": "TN", "Sfax": "TN", "Cairo": "EG"}[city]}
            rows.append(row)
            source_groups[sentence_id].append(row)
            text_occurrences[sentence].append(row)
    if len(rows) != 8000 or len(source_groups) != 2000:
        raise ValueError("MADAR Corpus-26 scope changed")
    for members in source_groups.values():
        if len(members) != 4 or {row["city"] for row in members} != set(MADAR_CITIES):
            raise ValueError("Incomplete MADAR source group")
        if len({row["split"] for row in members}) != 1:
            raise ValueError("MADAR source group crosses official splits")
    excluded_ids = {
        row["source_id"] for occurrences in text_occurrences.values()
        if len({item["split"] for item in occurrences}) > 1
        for row in occurrences if row["split"] == "train"
    }
    if len(excluded_ids) != 3:
        raise ValueError("MADAR exact cross-split decision changed")
    kept = [row for row in rows if row["source_id"] not in excluded_ids]
    excluded = [{**row, "reason": "exact_text_cross_split_whole_source_group"}
                for row in rows if row["source_id"] in excluded_ids]
    if len(kept) != 7988 or len(excluded) != 12 or any(row["split"] != "train" for row in excluded):
        raise ValueError("MADAR whole-group exclusion failed")
    remaining_texts = defaultdict(set)
    for sentence, occurrences in text_occurrences.items():
        for row in occurrences:
            if row["source_id"] not in excluded_ids:
                remaining_texts[sentence].add(row["split"])
    if any(len(splits) > 1 for splits in remaining_texts.values()):
        raise ValueError("MADAR exact text still crosses splits")
    counts = Counter((row["split"], row["label"]) for row in kept)
    if counts != {("train", "MSA"): 1597, ("train", "TN"): 3194,
                  ("train", "EG"): 1597, ("validation", "MSA"): 200,
                  ("validation", "TN"): 400, ("validation", "EG"): 200,
                  ("test", "MSA"): 200, ("test", "TN"): 400, ("test", "EG"): 200}:
        raise ValueError(f"Unexpected MADAR split counts: {counts}")
    return kept, excluded


def write_csv(path, rows, fields):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    if OUTPUT.exists():
        raise FileExistsError(f"Refusing to overwrite existing manifests: {OUTPUT}")
    tsac, tsac_excluded, ledger = build_tsac()
    madar, madar_excluded = build_madar()
    OUTPUT.mkdir(parents=True)
    write_csv(OUTPUT / "tsac.csv", tsac,
              ["id", "source_file", "source_line", "source_id", "group_id", "official_split", "split", "label"])
    write_csv(OUTPUT / "tsac_exclusions.csv", tsac_excluded,
              ["source_file", "source_line", "original_label", "reasons"])
    write_csv(OUTPUT / "madar.csv", madar,
              ["id", "source_file", "source_line", "source_id", "group_id", "city", "official_split", "split", "label"])
    write_csv(OUTPUT / "madar_exclusions.csv", madar_excluded,
              ["id", "source_file", "source_line", "source_id", "group_id", "city", "official_split", "split", "label", "reason"])
    summary = {
        "version": "v1", "tsac_rows": len(tsac), "tsac_excluded_train_rows": len(tsac_excluded),
        "madar_rows": len(madar), "madar_excluded_train_rows": len(madar_excluded),
        "tsac_exclusion_ledger_sha256": sha256(UNION_LEDGER),
        "split_seed": SEED, "tsac_validation_fraction_target": VALIDATION_FRACTION,
        "source_test_modified": False,
    }
    write_csv(OUTPUT / "file_hashes.csv",
              [{"file": path.name, "sha256": sha256(path)} for path in sorted(OUTPUT.glob("*.csv"))],
              ["file", "sha256"])
    with (OUTPUT / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
