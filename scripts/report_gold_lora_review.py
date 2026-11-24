"""Bind declared AI hypotheses to the fixed local LoRA diagnostic selection."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from load_working_data import load_records
from run_gold_validation import ROOT, digest, read_pinned


def report(diagnostic, notes_file, output, review_file):
    if (ROOT / "outputs").resolve() not in review_file.resolve().parents:
        raise ValueError("The case-level review belongs in private outputs")
    if any(path.exists() for path in (output, review_file)):
        raise FileExistsError("Preserve earlier evidence")
    notes = json.loads(notes_file.read_text())
    source = read_pinned(diagnostic, notes["diagnostic_sha256"])
    if source["split"] != "validation" or source["test_loaded"] is not False or source["training"]:
        raise ValueError("Use only the new diagnostic validation predictions")
    result = {"scope": "new-local-lora-diagnostic-validation", "test_loaded": False,
              "diagnostic_sha256": notes["diagnostic_sha256"],
              "notes_sha256": digest(notes_file), "origin": notes["origin"], "analysis": notes["analysis"],
              "independent_human_review": False, "tasks": {}}
    checks = {}
    for corpus, task in source["tasks"].items():
        predictions = read_pinned(notes_file.parent / f"{corpus}.json", task["predictions_sha256"])
        if len(predictions) != task["n"] or len({r["group_id"] for r in predictions}) != task["n"]:
            raise ValueError("Expected the complete distinct-group diagnostic sample")
        errors = [row for row in predictions if row["truth"] != row["prediction"]]
        chosen = sorted(errors, key=lambda row: (sha256(f'42|{row["id"]}'.encode()).digest(), row["id"]))[:30]
        records = {row["id"]: row for row in load_records(corpus, splits=("validation",), manifest_version="v2")}
        checked = []
        for (category, hypothesis), row in zip(notes["notes"][corpus], chosen, strict=True):
            record = records[row["id"]]
            if (not category.strip() or not hypothesis.strip() or
                    (record["label"], record["group_id"]) != (row["truth"], row["group_id"])):
                raise ValueError("Empty review or source alignment mismatch")
            checked.append({"id": row["id"], "group_id": row["group_id"], "text": record["text_raw"],
                            "reference": row["truth"], "prediction": row["prediction"] or "INVALID",
                            "writing": row["writing"], "category": category, "hypothesis": hypothesis,
                            "hypothesis_author": "Codex"})
        checks[corpus] = checked
        result["tasks"][corpus] = {
            "sample_rows": task["n"], "sample_errors": len(errors), "invalid_rows": task["invalid_rows"],
            "reviewed": len(checked), "categories": dict(Counter(row["category"] for row in checked)),
            "reference_to_prediction": dict(Counter(f'{row["reference"]} → {row["prediction"]}' for row in checked))}
    for path, payload in ((review_file, checks), (output, result)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic", required=True, type=Path)
    parser.add_argument("--notes", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--private-review", required=True, type=Path)
    args = parser.parse_args()
    report(args.diagnostic, args.notes, args.output, args.private_review)
