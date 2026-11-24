"""Cross-fit one temperature per saved CAMeLBERT checkpoint on validation."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.model_selection import GroupKFold

from audit_text_composition import writing_system
from compare_encoder_errors import checked_predictions, compare_predictions
from load_working_data import load_records
from run_gold_validation import ROOT, PROTOCOL, alignment_rows, abstention_metrics, confidence_rows, digest, read_pinned


BINS = 10
FOLDS = 5


def fit_temperature(log_scores, truth):
    indices = np.arange(len(truth))

    def loss(log_temperature):
        logits = log_scores / np.exp(log_temperature)
        return float(np.mean(logsumexp(logits, axis=1) - logits[indices, truth]))

    fitted = minimize_scalar(loss, bounds=(-4, 4), method="bounded")
    if not fitted.success or not np.isfinite(fitted.fun):
        raise ValueError("Temperature fitting failed")
    return float(np.exp(fitted.x))


def cross_fit(probabilities, truth, partitions):
    log_scores = np.log(np.clip(probabilities, np.finfo(float).tiny, 1))
    held_out = np.empty_like(probabilities)
    temperatures = []
    for fit, measure in partitions:
        temp = fit_temperature(log_scores[fit], truth[fit])
        held_out[measure] = softmax(log_scores[measure] / temp, axis=1)
        temperatures.append(temp)
    return held_out, temperatures


def metrics(probabilities, truth):
    confidence = probabilities.max(axis=1)
    correct = probabilities.argmax(axis=1) == truth
    indices = np.arange(len(truth))
    bucket = np.minimum((confidence * BINS).astype(int), BINS - 1)
    bins = []
    for i in range(BINS):
        chosen = bucket == i
        if chosen.any():
            bins.append({"bin": i, "n": int(chosen.sum()), "confidence": float(confidence[chosen].mean()),
                         "accuracy": float(correct[chosen].mean())})
    return {"n": len(truth), "accuracy": float(correct.mean()),
            "nll": float(-np.log(np.clip(probabilities[indices, truth], np.finfo(float).tiny, 1)).mean()),
            "brier": float(np.square(probabilities - np.eye(probabilities.shape[1])[truth]).sum(axis=1).mean()),
            "ece_10": float(sum(row["n"] * abs(row["confidence"] - row["accuracy"]) for row in bins) / len(truth)),
            "bins": bins}


def audit(output):
    if output.exists():
        raise FileExistsError("Preserve earlier results")
    plan = json.loads(PROTOCOL.read_text())
    validation_receipt = ROOT / "reports/gold_validation_2026-11-22/summary.json"
    earlier = json.loads(validation_receipt.read_text())
    if earlier["test_loaded"] is not False or earlier["protocol_sha256"] != digest(PROTOCOL):
        raise ValueError("Expected original validation abstention evidence")
    root = ROOT / plan["encoder_run"]
    campaign = read_pinned(root / "confirmation.json", plan["confirmation_sha256"])
    data = {"split": "validation", "test_loaded": False, "folds": FOLDS, "bins": BINS,
            "fit_unit": "manifest group_id", "exploratory": True,
            "confirmation_sha256": plan["confirmation_sha256"],
            "source_abstention_sha256": digest(validation_receipt), "script_sha256": digest(Path(__file__)),
            "tasks": {}}
    for entry in campaign["entries"]:
        corpus = entry["corpus"]
        folder = root / entry["full"]["directory"]
        stage = read_pinned(folder / "results.json", entry["full"]["results_sha256"])
        if stage["test_loaded"] is not False or stage["config"]["manifest_version"] != "v2":
            raise ValueError("Unexpected classification source")
        records = load_records(corpus, splits=("validation",), manifest_version="v2")
        aligned = alignment_rows(records)
        labels = stage["config"]["labels"][corpus]
        truth = np.array([labels.index(row["label"]) for row in records])
        groups = np.array([row["group_id"] for row in records])
        partitions = list(GroupKFold(n_splits=FOLDS).split(truth, groups=groups))
        if (any(set(groups[train]) & set(groups[measure]) for train, measure in partitions)
                or sorted(np.concatenate([measure for _, measure in partitions]).tolist()) != list(range(len(records)))):
            raise ValueError("Group leakage or incomplete out-of-fold selection")
        threshold = earlier["encoder"][corpus]["threshold_selected_on_validation"]
        groups_by_writing = defaultdict(list)
        for i, record in enumerate(records):
            groups_by_writing[writing_system(record["text_raw"])].append(i)
        seed_reports, raw, calibrated = [], [], []
        raw_scores_by_writing = defaultdict(list)
        for row in sorted(stage["results"], key=lambda r: r["seed"]):
            path = f'{row["checkpoint"]}/predictions.json'
            predictions = checked_predictions(folder, path, stage["artifacts"][path])
            compare_predictions(aligned, predictions)
            scores = confidence_rows(predictions, labels)
            probabilities = np.array([r["probabilities"] for r in predictions])
            adjusted, temperatures = cross_fit(probabilities, truth, partitions)
            if not np.array_equal(probabilities.argmax(axis=1), adjusted.argmax(axis=1)):
                raise ValueError("Positive temperature should preserve class ordering")
            seed_reports.append({"seed": row["seed"], "temperature_folds": temperatures,
                                 "prediction_sha256": stage["artifacts"][path],
                                 "raw": metrics(probabilities, truth), "cross_fit": metrics(adjusted, truth)})
            raw.append(probabilities)
            calibrated.append(adjusted)
            for writing, indices in groups_by_writing.items():
                raw_scores_by_writing[writing].extend(scores[i] for i in indices)
        if [row["seed"] for row in seed_reports] != plan["seeds"]:
            raise ValueError("Incomplete seed set")
        all_truth = np.tile(truth, len(raw))
        data["tasks"][corpus] = {"validation_rows": len(records),
                                 "fold_measurement_rows": [len(measure) for _, measure in partitions],
                                 "raw": metrics(np.concatenate(raw), all_truth),
                                 "cross_fit": metrics(np.concatenate(calibrated), all_truth), "seeds": seed_reports,
                                 "raw_abstention_threshold": threshold, "raw_abstention_by_writing": {
                                     writing: {"distinct_examples": len(groups_by_writing[writing]),
                                               "before": abstention_metrics(scores, 0.0),
                                               "after": abstention_metrics(scores, threshold)}
                                     for writing, scores in sorted(raw_scores_by_writing.items())}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    audit(parser.parse_args().output)
