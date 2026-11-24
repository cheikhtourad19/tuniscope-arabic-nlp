"""Rehearse the grouped bootstrap and inventory errors on retained validation predictions."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

from threadpoolctl import threadpool_limits

from compare_encoder_errors import checked_predictions, compare_predictions
from grouped_bootstrap import group_bootstrap
from run_gold_validation import ROOT, PROTOCOL, read_pinned, validation_reference


BIGRU = ROOT / "outputs/neural_baselines/2026-11-04_bigru_v1_cpu_full"
BIGRU_SHA = "58c615f9b652ea79599747b54db154c91d7945faf8cf82e882f630119ada2178"
QWEN = ROOT / "outputs/qwen/2026-11-07_validation_v2_mps_b16"
QWEN_SHA = "8d74e9fb979c02cc06dabc4fe67553994f2628195234082114a899b9a6d12250"
SEEDS = (13, 42, 2026)


def error_queue(rows, limit=30):
    wrong = sorted((row for row in rows if row["predicted"] != row["truth"]),
                   key=lambda r: sha256(f'42|{r["id"]}'.encode()).digest())
    chosen = {}
    for row in wrong:
        chosen.setdefault(row["group_id"], row)
    selected = [{key: row[key] for key in ("id", "group_id", "truth", "predicted")}
                for row in list(chosen.values())[:limit]]
    return {"predictions": len(rows), "errors": len(wrong), "error_groups": len(chosen),
            "invalid_predictions": sum(row["predicted"] == "INVALID" for row in rows),
            "selected_for_review": len(selected)}, selected


def run(public_file, private_file):
    if public_file.exists() or private_file.exists():
        raise FileExistsError("Preserve earlier evidence")
    protocol = json.loads(PROTOCOL.read_text())
    base_path = ROOT / protocol["baseline_run"]
    baseline = read_pinned(base_path / "results.json", protocol["baseline_results_sha256"])
    encoder = ROOT / protocol["encoder_run"]
    campaign = read_pinned(encoder / "confirmation.json", protocol["confirmation_sha256"])
    bigru = read_pinned(BIGRU / "results.json", BIGRU_SHA)
    qwen = read_pinned(QWEN / "completion.json", QWEN_SHA)
    qwen_plan = read_pinned(QWEN / "plan.json", qwen["plan_sha256"])
    if (any(run["test_loaded"] is not False for run in (baseline, campaign, bigru))
            or qwen_plan["split"] != "validation" or qwen_plan["training"] is not False):
        raise ValueError("Use validation artifacts only")
    data = {"split": "validation", "test_loaded": False, "manifest_version": "v2",
            "bootstrap_contrast": "camelbert-confirmed-mean-three-seeds minus tfidf-char-raw",
            "bootstrap": {}, "error_inventory": {},
            "missing_error_inventory": {"qwen_lora_cuda": "original row-level predictions unavailable"},
            "source_sha256": {"baseline": protocol["baseline_results_sha256"],
                              "encoder": protocol["confirmation_sha256"], "bigru": BIGRU_SHA,
                              "qwen": QWEN_SHA}}
    private = {}
    for entry in campaign["entries"]:
        corpus = entry["corpus"]
        folder = encoder / entry["full"]["directory"]
        stage = read_pinned(folder / "results.json", entry["full"]["results_sha256"])
        ref, reference = validation_reference(baseline, base_path, corpus)
        by_seed = {}
        for row in stage["results"]:
            path = f'{row["checkpoint"]}/predictions.json'
            by_seed[row["seed"]] = checked_predictions(folder, path, stage["artifacts"][path])
        labels = stage["config"]["labels"][corpus]
        alternatives = [by_seed[seed] for seed in SEEDS]
        data["bootstrap"][corpus] = group_bootstrap(reference, alternatives, labels)
        chosen_bigru, = [r for r in bigru["results"] if r["corpus"] == corpus and r["seed"] == 42]
        systems = {
            "tfidf_char": reference, "camelbert_seed_42": by_seed[42],
            "bigru_seed_42": checked_predictions(BIGRU, chosen_bigru["predictions"],
                                                chosen_bigru["predictions_sha256"])}
        for mode in ("zero_shot", "few_shot"):
            filename = f"{corpus}_{mode}.json"
            stage_qwen = read_pinned(QWEN / filename, qwen["stages"][f"{corpus}_{mode}"])
            if (stage_qwen["plan_sha256"], stage_qwen["corpus"], stage_qwen["mode"]) != (
                qwen["plan_sha256"], corpus, mode
            ):
                raise ValueError("Unexpected Qwen condition")
            systems[f"qwen_base_{mode}"] = [{**r, "predicted": r["prediction"] or "INVALID"}
                                            for r in stage_qwen["predictions"]]
        data["error_inventory"][corpus], private[corpus] = {}, {}
        for system, rows in systems.items():
            compare_predictions(reference, rows)
            data["error_inventory"][corpus][system], private[corpus][system] = error_queue(rows)
    for output, payload in ((public_file, data), (private_file, private)):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-output", type=Path, required=True)
    args = parser.parse_args()
    if ROOT / "outputs" not in args.private_output.resolve().parents:
        parser.error("Keep the row-level review queue under outputs/")
    with threadpool_limits(limits=1):
        run(args.output, args.private_output)
