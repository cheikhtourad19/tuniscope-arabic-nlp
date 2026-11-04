"""Shared fixed experiment contract; no data loaded on import."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from load_working_data import ROOT, load_records


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def validate_config(config):
    if (config.get("version") != "char-lm-raw-v2-v1" or config.get("manifest_version") != "v2"
            or config.get("variant") != "raw-v0" or config.get("corpora") != ["tsac", "madar"]
            or config.get("orders") != [1, 3] or config.get("alpha") != 0.1
            or config.get("unit") != "unicode-codepoint"
            or config.get("boundaries") != "reset-per-document-bos-eos"
            or config.get("vocabulary") != "all-train-characters-plus-unk-eos"
            or config.get("smoke_rows_per_partition") != 100
            or config.get("smoke_selection") != "sha256-id-sort-v1"):
        raise ValueError("Configuration outside the fixed initial character-LM protocol")


def partitions(corpus, mode, config):
    if mode not in {"smoke", "full"}:
        raise ValueError("Expected smoke or full")
    rows = load_records(corpus, splits=("train", "validation"), manifest_version="v2")
    result = {}
    for split in ("train", "validation"):
        selected = [r for r in rows if r["split"] == split]
        if mode == "smoke":
            selected = sorted(selected, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[
                :config["smoke_rows_per_partition"]]
        if not selected or any(r["text_raw"] != r["text_norm"] or r["normalization_version"] != "raw-v0" for r in selected):
            raise ValueError("Expected nonempty raw-v0 rows")
        result[split] = selected
    if {r["group_id"] for r in result["train"]} & {r["group_id"] for r in result["validation"]}:
        raise ValueError("Train/validation group overlap")
    return result["train"], result["validation"]


def partition_summary(rows):
    return {"rows": len(rows), "groups": len({r["group_id"] for r in rows}),
            "labels": dict(sorted(Counter(r["label"] for r in rows).items())),
            "ids_sha256": object_digest([r["id"] for r in rows]),
            "characters": sum(len(r["text_raw"]) for r in rows)}


def read_artifact(directory, filename):
    path = directory / filename
    if Path(filename).name != filename or path.resolve().parent != directory.resolve():
        raise ValueError("Artifact path escapes run directory")
    return path
