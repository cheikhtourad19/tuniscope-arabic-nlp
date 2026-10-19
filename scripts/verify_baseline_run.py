"""Verify saved predictions, model reloads and hand-recomputed validation metrics."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path

import joblib
import numpy as np

from load_working_data import ROOT, load_records, manifest_directory
from run_baselines import digest, sample_rows, text_for
from baseline_models import tfidf_system, zero_feature_rows, verify_model_config, vocabulary_sizes


def verify(run_dir):
    metadata = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    learned_system = tfidf_system(config)
    manifest_version = config.get("manifest_version", "v1")
    manifests = manifest_directory(manifest_version)
    if metadata["test_loaded"] or len(metadata["results"]) != 6:
        raise ValueError("Expected six complete train/validation-only results")
    if digest(ROOT / "uv.lock") != metadata["uv_lock_sha256"]:
        raise ValueError("Dependency lock changed since run")
    alignment = {}
    for corpus in config["corpora"]:
        if digest(manifests / f"{corpus}.csv") != metadata["manifests_sha256"][corpus]:
            raise ValueError("Manifest changed since run")
        rows = load_records(corpus, manifest_version=manifest_version)
        train = [row for row in rows if row["split"] == "train"]
        validation = [row for row in rows if row["split"] == "validation"]
        if metadata["mode"] == "smoke":
            train = sample_rows(train, config["smoke_rows_per_partition"], config["seed"])
            validation = sample_rows(validation, config["smoke_rows_per_partition"], config["seed"])
        results = [r for r in metadata["results"] if r["corpus"] == corpus]
        if {(r["system"], r["variant"]) for r in results} != {
            ("majority", "raw-v0"), (learned_system, "raw-v0"),
            (learned_system, "norm-nfc-space-v1")
        }:
            raise ValueError("Missing or duplicate baseline system")
        for result in results:
            checkpoint = run_dir / result["checkpoint"]
            path = run_dir / result["predictions"]
            if digest(checkpoint) != result["checkpoint_sha256"] or digest(path) != result["predictions_sha256"]:
                raise ValueError("Run artifacts changed")
            predictions = json.loads(path.read_text(encoding="utf-8"))
            if [p["id"] for p in predictions] != [row["id"] for row in validation]:
                raise ValueError("Validation IDs do not match saved predictions")
            if (result["validation_rows"] != len(validation) or
                    [p["group_id"] for p in predictions] != [row["group_id"] for row in validation] or
                    [p["truth"] for p in predictions] != [row["label"] for row in validation]):
                raise ValueError("Validation truth does not match manifest")
            if result["train_rows"] != len(train) or result["train_labels"] != dict(Counter(row["label"] for row in train)):
                raise ValueError("Training counts do not match source rows")
            labels = config["labels"][corpus]
            matrix = [[0] * len(labels) for _ in labels]
            for p in predictions:
                matrix[labels.index(p["truth"])][labels.index(p["predicted"])]+=1
            f1s = []
            for i in range(len(labels)):
                true_positive = matrix[i][i]
                denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
                f1s.append(2 * true_positive / denominator if denominator else 0)
            metrics = result["metrics"]
            if matrix != metrics["confusion_matrix"] or not math.isclose(
                sum(f1s) / len(labels), metrics["macro_f1"], abs_tol=1e-12
            ) or not math.isclose(sum(matrix[i][i] for i in range(len(labels))) / len(predictions),
                                 metrics["accuracy"], abs_tol=1e-12):
                raise ValueError("Metrics do not match hand-recomputed confusion matrix")
            model = joblib.load(checkpoint)
            if result["system"] == "majority":
                inputs = np.zeros((len(validation), 1))
                expected_majority = max(Counter(row["label"] for row in train),
                                        key=Counter(row["label"] for row in train).get)
                if {p["predicted"] for p in predictions} != {expected_majority}:
                    raise ValueError("Majority is not derived from the train")
            else:
                inputs = [text_for(row, result["variant"]) for row in validation]
                verify_model_config(model, config)
                if sum(vocabulary_sizes(model).values()) != result["vocabulary_size"]:
                    raise ValueError("Checkpoint feature dimensions mismatch")
                if "vocabulary_sizes" in result and vocabulary_sizes(model) != result["vocabulary_sizes"]:
                    raise ValueError("Checkpoint branch dimensions mismatch")
                if "zero_feature_validation_rows" in result:
                    if (zero_feature_rows(model, inputs) != result["zero_feature_validation_rows"] or
                            zero_feature_rows(model, [text_for(row, result["variant"]) for row in train]) !=
                            result["zero_feature_train_rows"]):
                        raise ValueError("Zero-feature counts do not match checkpoint")
            if model.predict(inputs).tolist() != [p["predicted"] for p in predictions]:
                raise ValueError("Reloaded checkpoint predictions differ")
            alignment.setdefault(corpus, [p["id"] for p in predictions])
            if alignment[corpus] != [p["id"] for p in predictions]:
                raise ValueError("Systems evaluated on different examples")
    print(f"Verified {metadata['mode']} run: six results, aligned validation IDs, model reloads, metrics recalculated")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    verify(parser.parse_args().run_dir)
