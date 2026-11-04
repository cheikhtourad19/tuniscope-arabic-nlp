"""Aggregate train-only vocabulary and sequence-length audit; no test selection."""
import argparse
from collections import Counter
import json
from pathlib import Path

from bigru_data import audit, build_vocabulary, partitions
from character_lm_experiment import ROOT, digest


def audit_inputs(config):
    results = {}
    for corpus in config["corpora"]:
        train, validation = partitions(corpus, "full", config)
        vocabulary = build_vocabulary([r["text_raw"] for r in train], config["max_vocabulary"], config["min_frequency"])
        results[corpus] = {"vocabulary_size": len(vocabulary) + 2,
                           "train": audit([r["text_raw"] for r in train], vocabulary, config["max_length"]),
                           "validation": audit([r["text_raw"] for r in validation], vocabulary, config["max_length"]),
                           "train_labels": dict(Counter(r["label"] for r in train)),
                           "validation_labels": dict(Counter(r["label"] for r in validation)),
                           "shared_train_validation_groups": 0}
    return {"config": config, "manifest_version": "v2", "test_loaded": False,
            "vocabulary_from": "train_only", "corpora": results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bigru_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (ROOT / "outputs").resolve() not in args.output.resolve().parents or args.output.exists():
        raise ValueError("Choose a fresh file under outputs/")
    result = audit_inputs(json.loads(args.config.read_text()))
    result["config_sha256"] = digest(args.config)
    result["manifests_sha256"] = {c: digest(ROOT / f"data/manifests/v2/{c}.csv") for c in result["corpora"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
