"""Train separate small BiGRU classifiers from scratch; validation only."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import torch

from audit_text_composition import writing_system
from bigru_data import audit, build_vocabulary, partitions
from bigru_model import BiGRUClassifier, resolve_device, set_seed
from bigru_training import add_writing_slices, evaluate, make_loader, save_json, train_epoch, validate_config
from character_lm_experiment import ROOT, digest, object_digest, partition_summary


def train_one(train, validation, corpus, seed, mode, config, target, device):
    set_seed(seed, config["cpu_threads"])
    labels = config["labels"][corpus]
    vocabulary = build_vocabulary([r["text_raw"] for r in train], config["max_vocabulary"], config["min_frequency"])
    model = BiGRUClassifier(len(vocabulary) + 2, len(labels), config).to(device)
    initial = model.classifier.weight.detach().cpu().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    train_loader = make_loader(train, vocabulary, labels, config, True, seed)
    validation_loader = make_loader(validation, vocabulary, labels, config, False, seed)
    best_score, best_epoch, best_state, stale = -1.0, None, None, 0
    history = []
    epochs = config["smoke_epochs"] if mode == "smoke" else config["max_epochs"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        training = train_epoch(model, train_loader, optimizer, device, config["gradient_clip"])
        if training["rows"] != len(train) or not training["nonzero_gradient_steps"]:
            raise ValueError("Training coverage or gradient check failed")
        evaluated = evaluate(model, validation_loader, labels, device)
        score = evaluated["metrics"]["macro_f1"]
        history.append({"epoch": epoch, "train": training, "validation_loss": evaluated["loss"],
                        "validation_macro_f1": score, "validation_accuracy": evaluated["metrics"]["accuracy"],
                        "validation_seconds": evaluated["seconds"]})
        print(f"{corpus.upper()} seed={seed} epoch={epoch}/{epochs} "
              f"train_loss={training['loss']:.4f} val_loss={evaluated['loss']:.4f} "
              f"val_macro_f1={score:.6f} ({training['seconds']:.2f}s train)", flush=True)
        if score > best_score:
            best_score, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if mode == "full" and stale >= config["patience"]:
            break
    train_eval_seconds = time.perf_counter() - started
    if best_state is None or torch.equal(initial, best_state["classifier.weight"]):
        raise ValueError("No selected checkpoint weight update")
    model.load_state_dict(best_state)
    selected = evaluate(model, validation_loader, labels, device)
    if selected["metrics"]["macro_f1"] != best_score:
        raise ValueError("Restored best state differs from epoch selection")
    name = f"{corpus}__bigru__seed_{seed}__raw-v0"
    checkpoint = target / f"{name}.pt"
    # Primitive metadata + tensors only, reloadable with weights_only=True.
    torch.save({"state_dict": best_state, "vocabulary": vocabulary, "labels": labels,
                "config": config, "corpus": corpus, "seed": seed, "best_epoch": best_epoch}, checkpoint)
    prediction_path = target / f"{name}.predictions.json"
    predictions = [{"id": row["id"], "group_id": row["group_id"], "truth": row["label"],
                    "predicted": predicted, "scores": dict(zip(labels, probabilities)),
                    "writing_system": writing_system(row["text_raw"])}
                   for row, predicted, probabilities in zip(validation, selected["predicted"], selected["probabilities"])]
    save_json(prediction_path, predictions)
    history_path = target / f"{name}.history.json"
    save_json(history_path, history)
    metrics = add_writing_slices(selected["metrics"], validation, selected["predicted"], labels)
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"corpus": corpus, "seed": seed, "variant": "raw-v0", "system": "bigru_word",
            "train": partition_summary(train), "validation": partition_summary(validation),
            "vocabulary_size": len(vocabulary) + 2, "vocabulary_sha256": object_digest(vocabulary),
            "train_token_audit": audit([r["text_raw"] for r in train], vocabulary, config["max_length"]),
            "validation_token_audit": audit([r["text_raw"] for r in validation], vocabulary, config["max_length"]),
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "best_epoch": best_epoch, "epochs_completed": len(history), "early_stopped": len(history) < epochs,
            "selected_validation_loss": selected["loss"], "metrics": metrics,
            "train_eval_seconds": train_eval_seconds,
            "process_peak_rss_bytes_cumulative": int(memory if sys.platform == "darwin" else memory * 1024),
            "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
            "checkpoint": checkpoint.name, "checkpoint_sha256": digest(checkpoint), "checkpoint_bytes": checkpoint.stat().st_size,
            "predictions": prediction_path.name, "predictions_sha256": digest(prediction_path),
            "history": history_path.name, "history_sha256": digest(history_path), "weights_changed": True,
            "scores_calibrated": False}


def run(config_path, mode, output, smoke_run=None, requested_device=None):
    config = json.loads(config_path.read_text())
    validate_config(config)
    target = output.resolve()
    if (ROOT / "outputs/neural_baselines").resolve() not in target.parents or target.exists():
        raise ValueError("Choose a new directory under outputs/neural_baselines/")
    device = resolve_device(requested_device or config["device"])
    if mode == "full":
        if smoke_run is None:
            raise ValueError("Full requires a verified smoke via --smoke-run")
        from verify_bigru import verify
        smoke = verify(smoke_run)
        if smoke["mode"] != "smoke" or smoke["config"] != config or smoke["device"] != str(device):
            raise ValueError("Smoke mode/config/device does not match")
    elif mode != "smoke":
        raise ValueError("Expected smoke or full")
    target.mkdir(parents=True)
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": mode, "config": config,
                "config_sha256": digest(config_path), "device": str(device), "python": platform.python_version(),
                "platform": platform.system(), "architecture": platform.machine(), "test_loaded": False,
                "packages": {n: importlib.metadata.version(n) for n in ("torch", "numpy", "scikit-learn", "matplotlib")},
                "cuda_version": torch.version.cuda, "mps_available": torch.backends.mps.is_available(),
                "torch_deterministic_algorithms": True,
                "environment_lock_sha256": digest(ROOT / "environments/bigru/uv.lock"),
                "classical_lock_sha256": digest(ROOT / "uv.lock"),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                "protocol_sha256": digest(ROOT / "docs/38_protocole_bigru.md"),
                "manifests_sha256": {c: digest(ROOT / f"data/manifests/v2/{c}.csv") for c in config["corpora"]},
                "code_sha256": {n: digest(ROOT / "scripts" / n) for n in ("bigru_data.py", "bigru_model.py", "bigru_training.py", "run_bigru.py", "verify_bigru.py")},
                "smoke_results_sha256": digest(smoke_run / "results.json") if smoke_run else None, "results": []}
    seeds = [config["smoke_seed"]] if mode == "smoke" else config["seeds"]
    print(f"BiGRU — {mode} — device={device} — seeds={seeds}", flush=True)
    for corpus in config["corpora"]:
        train, validation = partitions(corpus, mode, config)
        for seed in seeds:
            result = train_one(train, validation, corpus, seed, mode, config, target, device)
            metadata["results"].append(result)
            save_json(target / "results.json", metadata)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/bigru_v1.json")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-run", type=Path)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda", "auto"])
    args = parser.parse_args()
    run(args.config, args.mode, args.output, args.smoke_run, args.device)
