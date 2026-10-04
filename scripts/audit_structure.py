"""Audit source integrity and file structure without exporting corpus text.

Run from the project root with ``python3 scripts/audit_structure.py``.
Only aggregate counts, metadata values, and file hashes are printed.
This is the first audit pass; it does not check duplicates or create splits.
"""

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "madar": {
        "archive": ROOT / "data/raw/madar/MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021.zip",
        "original": ROOT / "data/raw/madar/original",
        "expected_sha256": "fc9c34638a9c8f8d5266d6431966b6ed933ad07b1e987514fa75f0b5eaf50a6a",
    },
    "tsac": {
        "archive": ROOT / "data/raw/tsac/TSAC-master.zip",
        "original": ROOT / "data/raw/tsac/original",
        "expected_sha256": "757f2873b516574accb0e5c21d7be7ef86db414f110d5fc109b59988a1e69268",
    },
}
MADAR_DIR = SOURCES["madar"]["original"] / "MADAR.Parallel-Corpora-Public-Version1.1-25MAR2021/MADAR_Corpus"
TSAC_DIR = SOURCES["tsac"]["original"] / "TSAC-master"
MADAR_HEADER = ["sentID.BTEC", "split", "lang", "sent"]


def sha256_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(block)
    return digest.hexdigest()


def verify_archive(source):
    archive = source["archive"]
    original = source["original"]
    with archive.open("rb") as stream:
        actual_sha = sha256_stream(stream)
    if actual_sha != source["expected_sha256"]:
        raise ValueError(f"Archive SHA-256 mismatch: {archive.name}")

    with zipfile.ZipFile(archive) as zipped:
        bad_member = zipped.testzip()
        if bad_member is not None:
            raise ValueError(f"ZIP integrity failure in {archive.name}: {bad_member}")
        members = [info for info in zipped.infolist() if not info.is_dir()]
        member_names = {info.filename for info in members}
        if len(member_names) != len(members):
            raise ValueError(f"Duplicate ZIP member names in {archive.name}")
        matched_extracted_names = set()
        renamed_empty_icons = 0
        for info in members:
            relative = Path(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe ZIP member path in {archive.name}")
            extracted = original / relative
            if not extracted.is_file() and info.filename.endswith("Icon\r") and info.file_size == 0:
                # macOS unzip strips the trailing CR from these empty Finder icons.
                relative = Path(info.filename[:-1])
                extracted = original / relative
                if extracted.is_file() and extracted.stat().st_size == 0:
                    renamed_empty_icons += 1
            if not extracted.is_file():
                raise ValueError(f"Missing extracted file: {relative}")
            matched_extracted_names.add(relative.as_posix())
            with zipped.open(info) as archived_stream, extracted.open("rb") as extracted_stream:
                if sha256_stream(archived_stream) != sha256_stream(extracted_stream):
                    raise ValueError(f"Extracted file differs from ZIP: {relative}")

    if len(matched_extracted_names) != len(members):
        raise ValueError(f"Multiple ZIP members map to one extracted file in {archive.name}")
    extracted_names = {p.relative_to(original).as_posix() for p in original.rglob("*") if p.is_file()}
    extra_names = sorted(extracted_names - matched_extracted_names)
    if extra_names:
        raise ValueError(f"Unexpected extracted files in {archive.name}: {extra_names}")
    return {
        "archive_sha256": actual_sha,
        "zip_integrity": "ok",
        "extracted_files_identical": len(members),
        "renamed_empty_macos_icons": renamed_empty_icons,
        "missing_or_extra_extracted_files": 0,
    }


def decoded_lines(path):
    with path.open("rb") as stream:
        for line_number, raw in enumerate(stream, start=1):
            try:
                yield line_number, raw.decode("utf-8-sig" if line_number == 1 else "utf-8").rstrip("\r\n")
            except UnicodeDecodeError:
                yield line_number, None


def audit_madar(path):
    stats = Counter()
    splits = Counter()
    languages = Counter()
    header = None
    for line_number, line in decoded_lines(path):
        stats["physical_lines"] += 1
        if line_number == 1:
            if line is None:
                raise ValueError(f"Invalid UTF-8 header in {path.name}")
            header = line.split("\t")
            continue
        stats["data_lines"] += 1
        if line is None:
            stats["invalid_utf8_lines"] += 1
            continue
        if not line.strip():
            stats["empty_lines"] += 1
            continue
        fields = line.split("\t")
        if len(fields) != len(MADAR_HEADER):
            stats["malformed_column_count"] += 1
            continue
        stats["valid_four_column_rows"] += 1
        sentence_id, split, language, sentence = fields
        if not sentence_id.strip():
            stats["missing_ids"] += 1
        if not split.strip():
            stats["missing_splits"] += 1
        if not language.strip():
            stats["missing_languages"] += 1
        if not sentence.strip():
            stats["empty_sentences"] += 1
        splits[split] += 1
        languages[language] += 1
    if header != MADAR_HEADER:
        raise ValueError(f"Unexpected MADAR header in {path.name}: {header}")
    return {
        "file": path.name,
        "sha256": sha256_path(path),
        "header": header,
        "counts": {key: stats[key] for key in (
            "physical_lines", "data_lines", "valid_four_column_rows", "invalid_utf8_lines",
            "empty_lines", "malformed_column_count", "missing_ids", "missing_splits",
            "missing_languages", "empty_sentences",
        )},
        "source_splits": dict(sorted(splits.items())),
        "source_languages": dict(sorted(languages.items())),
    }


def sha256_path(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def audit_tsac(path):
    stats = Counter()
    for _, line in decoded_lines(path):
        stats["physical_lines"] += 1
        if line is None:
            stats["invalid_utf8_lines"] += 1
        elif not line.strip():
            stats["empty_lines"] += 1
        else:
            stats["nonempty_lines"] += 1
    return {
        "file": path.name,
        "sha256": sha256_path(path),
        "source_split": path.stem.split("_")[0],
        "label_from_filename": path.stem.split("_")[1].upper(),
        "counts": {key: stats[key] for key in (
            "physical_lines", "nonempty_lines", "empty_lines", "invalid_utf8_lines",
        )},
    }


def main():
    report = {name: {"integrity": verify_archive(source)} for name, source in SOURCES.items()}
    report["madar"]["task_files"] = [
        audit_madar(MADAR_DIR / f"MADAR.corpus.{city}.tsv")
        for city in ("MSA", "Tunis", "Sfax", "Cairo")
    ]
    report["tsac"]["task_files"] = [
        audit_tsac(TSAC_DIR / f"{split}_{label}.txt")
        for split in ("train", "test") for label in ("pos", "neg")
    ]
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(f"Audit failed: {exc}", file=sys.stderr)
        sys.exit(1)
