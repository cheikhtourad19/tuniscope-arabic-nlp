"""Train initial CPU baselines on train and evaluate on validation only.

All row-level outputs and model vocabularies are kept under ignored outputs/.
No test flag exists in this runner. Use --mode smoke before --mode full.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time
import warnings

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedShuffleSplit
from threadpoolctl import threadpool_limits

from audit_structure import ROOT
from audit_text_composition import writing_system
from baseline_metrics import classification_metrics
from baseline_models import build_tfidf_model, tfidf_system, zero_feature_rows, vocabulary_sizes
from load_working_data import load_records, manifest_directory
from normalize_text import VERSION, normalize


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sample_rows(rows, n, seed):
    if len(rows) <= n:
        return rows
    indices, _ = next(StratifiedShuffleSplit(n_splits=1, train_size=n, random_state=seed).split(
        np.zeros(len(rows)), [row["label"] for row in rows]
    ))
    return [rows[index] for index in sorted(indices)]


def text_for(row, variant):
    if variant == "raw-v0":
        return row["text_raw"]
    if variant == VERSION:
        return normalize(row["text_raw"])
    raise ValueError("Unsupported normalization version")


def evaluate(rows, predicted, labels):
    truth = [row["label"] for row in rows]
    result = classification_metrics(truth, predicted, labels)
    result["by_writing_system"] = {}
    scripts = [writing_system(row["text_raw"]) for row in rows]
    for script in sorted(set(scripts)):
        indices = [i for i, value in enumerate(scripts) if value == script]
        sub_truth = [truth[i] for i in indices]
        sub_pred = [predicted[i] for i in indices]
        metrics = classification_metrics(sub_truth, sub_pred, labels)
        metrics["small_or_missing_class_slice"] = len(indices) < 30 or set(sub_truth) != set(labels)
        result["by_writing_system"][script] = metrics
    return result


def peak_rss_bytes():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(value if sys.platform == "darwin" else value * 1024)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/baselines_initial.json")
    parser.add_argument("--mode", required=True, choices=("smoke", "full"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    manifest_version = config.get("manifest_version", "v1")
    manifests = manifest_directory(manifest_version)
    learned_system = tfidf_system(config)
    target = args.output.resolve()
    allowed = (ROOT / "outputs/baselines").resolve()
    if allowed not in target.parents or target.exists():
        raise ValueError("Choose a new run directory under outputs/baselines/")
    if config["variants"] != ["raw-v0", VERSION] or config["threads"] != 1:
        raise ValueError("Initial protocol requires raw/v1 and one CPU thread")
    target.mkdir(parents=True)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": args.mode, "config": config, "config_sha256": digest(args.config),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
        "python": platform.python_version(), "platform": platform.system(),
        "architecture": platform.machine(), "device": "cpu", "test_loaded": False,
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "uv_lock_sha256": digest(ROOT / "uv.lock"),
        "code_sha256": {path.name: digest(path) for path in sorted((ROOT / "scripts").glob("*.py"))},
        "manifests_sha256": {corpus: digest(manifests / f"{corpus}.csv")
                             for corpus in config["corpora"]},
        "results": [],
    }
    with threadpool_limits(limits=config["threads"]):
        for corpus in config["corpora"]:
            rows = load_records(corpus, manifest_version=manifest_version)
            train = [row for row in rows if row["split"] == "train"]
            validation = [row for row in rows if row["split"] == "validation"]
            labels = config["labels"][corpus]
            if {row["group_id"] for row in train} & {row["group_id"] for row in validation}:
                raise ValueError("Train/validation group overlap")
            if args.mode == "smoke":
                train = sample_rows(train, config["smoke_rows_per_partition"], config["seed"])
                validation = sample_rows(validation, config["smoke_rows_per_partition"], config["seed"])
            truth = [row["label"] for row in train]
            for system, variant in [("majority", "raw-v0")] + [
                (learned_system, variant) for variant in config["variants"]
            ]:
                name = f"{corpus}__{system}__{variant}"
                print(f"Running {name} ({args.mode}; {len(train)} train, {len(validation)} validation)", flush=True)
                if system == "majority":
                    model = DummyClassifier(strategy="most_frequent")
                    x_train = np.zeros((len(train), 1))
                    x_val = np.zeros((len(validation), 1))
                else:
                    model = build_tfidf_model(config)
                    x_train = [text_for(row, variant) for row in train]
                    x_val = [text_for(row, variant) for row in validation]
                started = time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("error", ConvergenceWarning)
                    model.fit(x_train, truth)
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                predicted = model.predict(x_val).tolist()
                predict_seconds = time.perf_counter() - started
                metrics = evaluate(validation, predicted, labels)
                checkpoint = target / f"{name}.joblib"
                joblib.dump(model, checkpoint)
                reloaded = joblib.load(checkpoint)
                if reloaded.predict(x_val[:10]).tolist() != predicted[:10]:
                    raise ValueError("Reload check failed")
                predictions_file = target / f"{name}.predictions.json"
                predictions_file.write_text(json.dumps([
                    {"id": row["id"], "group_id": row["group_id"], "truth": row["label"],
                     "predicted": prediction, "writing_system": writing_system(row["text_raw"])}
                    for row, prediction in zip(validation, predicted)
                ], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                result = {
                    "corpus": corpus, "system": system, "variant": variant,
                    "train_rows": len(train), "train_labels": dict(Counter(truth)),
                    "validation_rows": len(validation), "metrics": metrics,
                    "fit_seconds": fit_seconds, "predict_seconds": predict_seconds,
                    "process_peak_rss_bytes_cumulative": peak_rss_bytes(),
                    "checkpoint": checkpoint.name, "checkpoint_sha256": digest(checkpoint),
                    "checkpoint_bytes": checkpoint.stat().st_size, "reload_first_ten": "passed",
                    "predictions": predictions_file.name, "predictions_sha256": digest(predictions_file),
                }
                if system != "majority":
                    result["zero_feature_train_rows"] = zero_feature_rows(model, x_train)
                    result["zero_feature_validation_rows"] = zero_feature_rows(model, x_val)
                    result["vocabulary_sizes"] = vocabulary_sizes(model)
                    result["vocabulary_size"] = sum(result["vocabulary_sizes"].values())
                    result["solver_iterations"] = model["classifier"].n_iter_.tolist()
                    result["classifier_parameters"] = int(model["classifier"].coef_.size +
                                                           model["classifier"].intercept_.size)
                metadata["results"].append(result)
                (target / "results.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2,
                                                              sort_keys=True) + "\n", encoding="utf-8")
                print(f"Validation macro-F1={metrics['macro_f1']:.6f}, fit={fit_seconds:.2f}s", flush=True)


if __name__ == "__main__":
    main()
