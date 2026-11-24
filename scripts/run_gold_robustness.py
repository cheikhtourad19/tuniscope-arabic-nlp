"""Replay two saved TSAC classifiers on paired validation text variants."""

import argparse
from collections import Counter, defaultdict
from hashlib import file_digest
import json
import re
from pathlib import Path
from time import perf_counter

import joblib
import torch
from threadpoolctl import threadpool_limits

from audit_text_composition import writing_system
from baseline_metrics import classification_metrics
from bigru_model import resolve_device
from compare_encoder_errors import checked_predictions, compare_predictions
from encoder_finetuning import evaluate, tokens
from encoder_finetuning_inference import load_checkpoint
from load_working_data import load_records
from normalize_text import normalize
from run_gold_validation import ROOT, PROTOCOL, alignment_rows, read_pinned, validation_reference


SEED = 42
VARIANTS = ("original", "normalisation_nfc_espaces", "espaces_triplement", "lettre_dupliquee")


def duplicate_letter(text):
    # Modify at most one word. Exclude short tokens and combining marks at the insertion.
    for match in re.finditer(r"[^\W\d_]{4,}", text):
        position = match.start() + len(match.group()) // 2
        letter = text[position]
        if letter.isalpha() and letter not in (text[position - 1], text[position + 1]):
            return text[:position] + letter + text[position:]
    return text


def variants(text):
    return dict(zip(VARIANTS, (text, normalize(text), text.replace(" ", "   "),
                               duplicate_letter(text)), strict=True))


def counts(truth, original, perturbed, labels):
    before = classification_metrics(truth, original, labels)
    after = classification_metrics(truth, perturbed, labels)
    return {"n": len(truth), "macro_f1_original": before["macro_f1"],
            "macro_f1_variant": after["macro_f1"], "macro_f1_difference": after["macro_f1"] - before["macro_f1"],
            "prediction_flips": sum(a != b for a, b in zip(original, perturbed, strict=True)),
            "new_errors": sum(a == y != b for y, a, b in zip(truth, original, perturbed, strict=True)),
            "corrected_errors": sum(a != y == b for y, a, b in zip(truth, original, perturbed, strict=True)),
            "label_counts": dict(Counter(truth))}


def analyze(rows, forms, predictions, labels):
    truth = [r["label"] for r in rows]
    original = predictions["original"]
    groups = {"writing_system": defaultdict(list), "length_characters": defaultdict(list)}
    for i, row in enumerate(rows):
        text = row["text_raw"]
        groups["writing_system"][writing_system(text)].append(i)
        length = len(text)
        groups["length_characters"][next(label for bound, label in
                                        ((20, "0–20"), (50, "21–50"), (100, "51–100"), (float("inf"), ">100"))
                                        if length <= bound)].append(i)

    def subset(indices, predicted):
        result = counts([truth[i] for i in indices], [original[i] for i in indices],
                        [predicted[i] for i in indices], labels)
        result["small_or_missing_class_slice"] = len(indices) < 100 or set(result["label_counts"]) != set(labels)
        return result

    summary = {}
    for name, predicted in predictions.items():
        changed = [i for i, form in enumerate(forms) if form[name] != form["original"]]
        if any(predicted[i] != original[i] for i, form in enumerate(forms) if form[name] == form["original"]):
            raise ValueError("Replayed identical text changed label")
        summary[name] = {"all": counts(truth, original, predicted, labels),
                         "changed": subset(changed, predicted) if changed else None,
                         **{axis: {key: subset(indices, predicted) for key, indices in sorted(slices.items())}
                            for axis, slices in groups.items()}}
    return summary


def check_hash(path, expected):
    with path.open("rb") as stream:
        if file_digest(stream, "sha256").hexdigest() != expected:
            raise ValueError(f"Saved model file changed: {path.name}")


def run(target, device_name):
    if target.exists():
        raise FileExistsError("Choose a new robustness result directory")
    protocol = json.loads(PROTOCOL.read_text())
    classical = ROOT / protocol["baseline_run"]
    baseline = read_pinned(classical / "results.json", protocol["baseline_results_sha256"])
    ref, tfidf_original = validation_reference(baseline, classical, "tsac")
    checkpoint = classical / ref["checkpoint"]
    check_hash(checkpoint, ref["checkpoint_sha256"])
    encoder_source = ROOT / protocol["encoder_run"]
    campaign = read_pinned(encoder_source / "confirmation.json", protocol["confirmation_sha256"])
    stage_info, = [r["full"] for r in campaign["entries"] if r["corpus"] == "tsac"]
    stage = encoder_source / stage_info["directory"]
    stage_data = read_pinned(stage / "results.json", stage_info["results_sha256"])
    selected, = [r for r in stage_data["results"] if r["seed"] == SEED]
    name = selected["checkpoint"]
    prediction_file = f"{name}/predictions.json"
    encoder_original = checked_predictions(stage, prediction_file, stage_data["artifacts"][prediction_file])
    for filename in ("model.safetensors", "tuniscope.json", "config.json", "tokenizer.json",
                     "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        key = f"{name}/{filename}"
        check_hash(stage / key, stage_data["artifacts"][key])
    rows = load_records("tsac", splits=("validation",), manifest_version="v2")
    compare_predictions(alignment_rows(rows), tfidf_original)
    compare_predictions(tfidf_original, encoder_original)
    labels = stage_data["config"]["labels"]["tsac"]
    forms = [variants(row["text_raw"]) for row in rows]
    device = resolve_device(device_name)
    if device.type == "cpu":
        torch.set_num_threads(4)
    started = perf_counter()
    with threadpool_limits(limits=1):
        tfidf = joblib.load(checkpoint)
        scores_tfidf = {variant: tfidf.predict([row[variant] for row in forms]).tolist() for variant in VARIANTS}
    model, tokenizer, meta = load_checkpoint(stage / name, device)
    if meta["labels"] != labels or meta["seed"] != SEED:
        raise ValueError("Unexpected checkpoint")
    scores_encoder = {}
    for variant in VARIANTS:
        records = [{"text_raw": row[variant]} for row in forms]
        scores_encoder[variant] = evaluate(
            model, tokens(records, tokenizer, meta["config"]),
            [labels.index(row["label"]) for row in rows], tokenizer, labels, meta["config"], device
        )["predicted"]
        print(f"CAMeLBERT {variant} terminé", flush=True)
    for scores, expected in ((scores_tfidf, tfidf_original), (scores_encoder, encoder_original)):
        if scores["original"] != [r["predicted"] for r in expected]:
            raise ValueError("Raw predictions differ from saved reference")
    data = {"corpus": "tsac", "split": "validation", "manifest_version": "v2",
            "test_loaded": False, "training_started": False, "encoder_seed": SEED, "device": str(device),
            "encoder_original_prediction_sha256": stage_data["artifacts"][prediction_file],
            "tfidf_original_prediction_sha256": ref["predictions_sha256"],
            "wall_seconds_inference_and_summary": 0,
            "models": {key: analyze(rows, forms, scores, labels) for key, scores in
                       (("tfidf_char", scores_tfidf), ("camelbert", scores_encoder))}}
    data["wall_seconds_inference_and_summary"] = perf_counter() - started
    target.mkdir(parents=True)
    (target / "robustness.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    args = parser.parse_args()
    run(args.output, args.device)
