"""Show one flagged audit case locally without copying corpus text to a report.

Run, for example:
  python3 scripts/review_audit_case.py --index exact --case 321

The original files and review indexes are read-only. Do not save or paste
the displayed protected text into a tracked file or a public issue.
"""

import argparse
import csv
from pathlib import Path

from audit_structure import MADAR_DIR, ROOT, TSAC_DIR, decoded_lines


INDEXES = {
    "exact": ROOT / "data/interim/exact_review_2026-10-04.tsv",
    "near": ROOT / "data/interim/tsac_near_review_2026-10-04.tsv",
}
MADAR_FILES = {f"MADAR.corpus.{city}.tsv" for city in ("MSA", "Tunis", "Sfax", "Cairo")}
TSAC_FILES = {f"{split}_{label}.txt" for split in ("train", "test") for label in ("pos", "neg")}


def find_case(index_path, case_id):
    with index_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row.get("case_id") == str(case_id):
                return row
    raise ValueError(f"Case {case_id} was not found in the {index_path.name} index")


def source_path(source, filename):
    if Path(filename).name != filename:
        raise ValueError("Invalid filename in review index")
    if source == "MADAR" and filename in MADAR_FILES:
        return MADAR_DIR / filename
    if source == "TSAC" and filename in TSAC_FILES:
        return TSAC_DIR / filename
    raise ValueError("Unexpected source or filename in review index")


def source_text(source, filename, line_number):
    if line_number < 1:
        raise ValueError("Invalid line number in review index")
    path = source_path(source, filename)
    for current, line in decoded_lines(path):
        if current == line_number:
            if line is None:
                raise ValueError("Invalid UTF-8 at indexed line")
            if source == "MADAR":
                fields = line.split("\t")
                if len(fields) != 4 or current == 1:
                    raise ValueError("Indexed MADAR line is not a data row")
                return fields[3]
            return line
    raise ValueError("Indexed line exceeds source file length")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", choices=INDEXES, required=True)
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument(
        "--metadata-only", action="store_true",
        help="Show file/line references without printing protected corpus text",
    )
    parser.add_argument(
        "--private-file", type=Path,
        help="Write the case to a new Git-ignored text file under data/interim",
    )
    args = parser.parse_args()
    if args.case < 1:
        parser.error("--case must be a positive integer")
    if args.metadata_only and args.private_file:
        parser.error("--metadata-only cannot be combined with --private-file")

    output_path = None
    if args.private_file:
        output_path = args.private_file.resolve()
        interim_dir = (ROOT / "data/interim").resolve()
        if output_path.parent != interim_dir or output_path.suffix != ".txt":
            parser.error("--private-file must be a .txt file directly under data/interim")
        if output_path.exists():
            parser.error("--private-file must not overwrite an existing file")

    row = find_case(INDEXES[args.index], args.case)
    source = row.get("source", "TSAC")
    category = row.get("category", "near_duplicate_candidate")
    lines = [f"Case {args.case}: {source}, {category}"]
    for side in ("a", "b"):
        if args.index == "near":
            prefix = "train" if side == "a" else "test"
            filename = row[f"{prefix}_file"]
            line_number = int(row[f"{prefix}_line"])
            labels = row[f"{prefix}_labels"]
            split = prefix
        else:
            filename = row[f"file_{side}"]
            line_number = int(row[f"line_{side}"])
            labels = row[f"label_{side}"]
            split = row[f"split_{side}"]
        lines.append(f"{side.upper()}: {filename}, line {line_number}, split {split}, label {labels}")
        if not args.metadata_only:
            lines.append(source_text(source, filename, line_number))
    if not args.metadata_only:
        lines.append("Review privately: Is the sentiment clearly POS, clearly NEG, or ambiguous?")
        lines.append("Do not edit the originals or treat this display as an annotation decision.")
    if output_path:
        output_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8", errors="strict")
        print(f"Created private review file: {output_path}")
    else:
        print("\n\n".join(lines))


if __name__ == "__main__":
    main()
