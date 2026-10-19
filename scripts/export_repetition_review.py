"""Prepare private TSAC repetition-collision review inputs, never decisions.

Uses v1 train/validation only, independently of any model predictions or scores.
Original texts go only to ignored outputs/. A text-free reference index is kept
under ignored data/interim/. Existing human-review artifacts are not overwritten.
"""

from collections import defaultdict
import hashlib
import itertools
import json
from pathlib import Path

from audit_elongation_policy import collapse_repeated_letters
from audit_structure import ROOT
from load_working_data import load_records
from normalize_text import normalize


OUTPUT = ROOT / "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4/repetition_review_2026-10-17.json"
INDEX = ROOT / "data/interim/repetition_review_index_2026-10-17.json"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_key(row):
    return (row["source_file"], row["source_line"])


def reference(row):
    return {"id": row["id"], "file": row["source_file"], "line": row["source_line"],
            "split": row["split"], "group_id": row["group_id"]}


def main():
    if OUTPUT.exists() or INDEX.exists():
        raise FileExistsError("Refusing to overwrite existing review inputs")
    groups = defaultdict(list)
    for row in load_records("tsac"):
        groups[collapse_repeated_letters(normalize(row["text_raw"]))].append(row)
    families = [rows for rows in groups.values() if {row["split"] for row in rows} == {"train", "validation"}]
    families.sort(key=lambda rows: min(source_key(row) for row in rows))
    if len(families) != 9:
        raise ValueError("Expected the nine candidate families found in the previous audit")
    pairs = []
    family_refs = []
    for family_number, rows in enumerate(families, 1):
        rows.sort(key=source_key)
        family_refs.append({"family": family_number, "members": [reference(row) for row in rows]})
        train = [row for row in rows if row["split"] == "train"]
        validation = [row for row in rows if row["split"] == "validation"]
        for a, b in itertools.product(train, validation):
            if normalize(a["text_raw"]) == normalize(b["text_raw"]):
                raise ValueError("Expected repetition-related variants, not an existing v1 exact overlap")
            pairs.append({"case_id": len(pairs) + 1, "family": family_number,
                          "a": reference(a), "b": reference(b),
                          "text_a": a["text_raw"], "text_b": b["text_raw"],
                          "decision": "", "note": ""})
    edge_refs = [(item["a"]["id"], item["b"]["id"]) for item in pairs]
    if len(set(edge_refs)) != len(pairs):
        raise ValueError("Repeated review pair")
    summary = {
        "families": len(families), "cross_split_pairs": len(pairs),
        "train_rows": len({item["a"]["id"] for item in pairs}),
        "validation_rows": len({item["b"]["id"] for item in pairs}),
        "decisions_recorded": 0, "official_test_rows_loaded": 0,
        "selection_uses_model_predictions_or_scores": False,
    }
    provenance = {"manifest_sha256": sha256(ROOT / "data/manifests/v1/tsac.csv"),
                  "rule": "NFC/spacing v1 then collapse identical Arabic/Latin letter runs >=3 to 2",
                  "script_sha256": sha256(__file__)}
    # Paths are explicit and confined to the two Git-ignored artifact locations.
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps({"summary": summary, "provenance": provenance, "pairs": pairs},
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    INDEX.write_text(json.dumps({"summary": summary, "provenance": provenance,
                                "payload_sha256": sha256(OUTPUT), "families": family_refs,
                                "pairs": [{key: value for key, value in item.items()
                                           if key not in {"text_a", "text_b"}} for item in pairs]},
                               ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
