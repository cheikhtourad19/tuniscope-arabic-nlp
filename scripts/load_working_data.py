"""Load manifest-selected corpus rows into memory without writing text files.

The default reads train and validation only. Access to the frozen official test
requires an explicit allow_test=True. Pandas is optional and imported only by
as_pandas(); the core loader uses only the Python standard library.
"""

from collections import Counter
import csv
from pathlib import Path

from audit_structure import ROOT, MADAR_DIR, TSAC_DIR, decoded_lines
from audit_duplicates import CORPUS26_SPLITS, EXPECTED_MADAR_LANG
from build_data_manifests import MADAR_HASHES
from audit_tsac_candidate_groups import sha256


MANIFESTS = ROOT / "data/manifests/v1"
TSAC_HASHES = {
    "train_pos.txt": "1ddedc6fce42e9d3e09a5892088700a1c9065dcd2f1739983babdf964fc21556",
    "train_neg.txt": "4babaf50959d556ea879a471acc3f5c0b794c6e606f31da8002633ab8b199668",
    "test_pos.txt": "36e64d31103fb770988337b0a3a71340ac2187c568d52e9eec139df4571b7eb1",
    "test_neg.txt": "41e8e829d1b74196ba82c03171e793ac641dd3c936de6a662d9fa399d438a3c6",
}
ALLOWED_SPLITS = {"train", "validation", "test"}
MANIFEST_FIELDS = {
    "tsac": ["id", "source_file", "source_line", "source_id", "group_id",
             "official_split", "split", "label"],
    "madar": ["id", "source_file", "source_line", "source_id", "group_id",
              "city", "official_split", "split", "label"],
}


def manifest_directory(version="v1"):
    """Select an explicit version without silently changing historical runs."""
    if version not in {"v1", "v2"}:
        raise ValueError("Manifest version must be v1 or v2")
    return ROOT / "data/manifests" / version


def _manifest_rows(corpus, manifest_version="v1"):
    directory = manifest_directory(manifest_version)
    expected = MANIFEST_FIELDS[corpus]
    filename = f"{corpus}.csv"
    with (directory / "file_hashes.csv").open(encoding="utf-8", newline="") as stream:
        file_hashes = {row["file"]: row["sha256"] for row in csv.DictReader(stream)}
    if sha256(directory / filename) != file_hashes.get(filename):
        raise ValueError(f"Manifest checksum mismatch: {filename}")
    with (directory / filename).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != expected:
            raise ValueError(f"Unexpected manifest columns: {filename}")
        return list(reader)


def _source_texts(corpus, selected):
    by_file = {}
    source_dir = TSAC_DIR if corpus == "tsac" else MADAR_DIR
    expected_hashes = TSAC_HASHES if corpus == "tsac" else {
        f"MADAR.corpus.{city}.tsv": digest for city, digest in MADAR_HASHES.items()
    }
    for filename in selected:
        if filename not in expected_hashes or Path(filename).name != filename:
            raise ValueError(f"Unexpected source file in manifest: {filename}")
        path = source_dir / filename
        if sha256(path) != expected_hashes[filename]:
            raise ValueError(f"Source checksum mismatch: {filename}")
        wanted = selected[filename]
        lines = {}
        for number, text in decoded_lines(path):
            if number in wanted:
                if text is None:
                    raise ValueError(f"Invalid UTF-8: {filename}:{number}")
                lines[number] = text
        if set(lines) != wanted:
            raise ValueError(f"Missing source lines: {filename}")
        by_file[filename] = lines
    return by_file


def load_records(corpus, splits=("train", "validation"), *, allow_test=False, manifest_version="v1"):
    """Return canonical raw-v0 records as a list of dictionaries in RAM.

    No normalization is applied: text_norm equals text_raw. Script and language
    detection have not yet been implemented, so those fields remain unknown.
    """
    if corpus not in MANIFEST_FIELDS:
        raise ValueError("Corpus must be 'tsac' or 'madar'")
    splits = tuple(splits)
    if not splits or len(set(splits)) != len(splits) or set(splits) - ALLOWED_SPLITS:
        raise ValueError("Choose distinct valid splits")
    if "test" in splits and not allow_test:
        raise PermissionError("Frozen test requires explicit allow_test=True")
    manifest = [row for row in _manifest_rows(corpus, manifest_version) if row["split"] in splits]
    selected = {}
    for row in manifest:
        if not row["id"] or not row["group_id"] or row["label"] not in (
            {"POS", "NEG"} if corpus == "tsac" else {"MSA", "TN", "EG"}
        ):
            raise ValueError("Missing ID, group, or valid label")
        filename = row["source_file"]
        number = int(row["source_line"])
        selected.setdefault(filename, set()).add(number)
    texts = _source_texts(corpus, selected)
    if sum(map(len, selected.values())) != len(manifest):
        raise ValueError("Repeated source reference in manifest selection")

    records = []
    for row in manifest:
        filename, number = row["source_file"], int(row["source_line"])
        source_line = texts[filename][number]
        if corpus == "tsac":
            original_split, original_label = filename[:-4].split("_")
            if row["official_split"] != original_split or row["label"] != original_label.upper():
                raise ValueError(f"TSAC source provenance mismatch: {filename}:{number}")
            raw = source_line
            source = "TSAC"
            license_ref = "TSAC repository licence; see SOURCE_REGISTER.md"
        else:
            sentence_id, official, language, raw = source_line.split("\t")
            city = row["city"]
            if (filename != f"MADAR.corpus.{city}.tsv" or
                    language != EXPECTED_MADAR_LANG[city] or
                    sentence_id != row["source_id"] or official != row["official_split"] or
                    CORPUS26_SPLITS.get(official) != row["split"] or
                    row["label"] != {"MSA": "MSA", "Tunis": "TN", "Sfax": "TN", "Cairo": "EG"}[city]):
                raise ValueError(f"MADAR source provenance mismatch: {filename}:{number}")
            source = "MADAR Parallel Corpora v1.1, Corpus-26"
            license_ref = "MADAR registered research access; see SOURCE_REGISTER.md"
        if not raw.strip() and row["split"] != "test":
            raise ValueError(f"Empty retained working text: {filename}:{number}")
        records.append({
            "id": row["id"], "text_raw": raw, "text_norm": raw,
            "task": "sentiment" if corpus == "tsac" else "dialect",
            "label": row["label"], "script": "UNKNOWN", "languages": ["und"],
            "source": source, "source_id": row["source_id"],
            "source_file": filename, "source_line": number,
            "group_id": row["group_id"], "split": row["split"],
            "official_split": row["official_split"],
            "annotation_status": "original", "license": license_ref,
            "normalization_version": "raw-v0",
        })
    if len({record["id"] for record in records}) != len(records):
        raise ValueError("Repeated record ID")
    return records


def as_pandas(records):
    """Convert records to a DataFrame only when pandas is already installed."""
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is optional and is not installed in this environment") from exc
    return pd.DataFrame.from_records(records)


def main():
    # Do not print, serialize, or sample protected corpus text.
    for corpus in ("tsac", "madar"):
        records = load_records(corpus)
        print(corpus, len(records), dict(sorted(Counter((item["split"], item["label"])
                                                      for item in records).items())))


if __name__ == "__main__":
    main()
