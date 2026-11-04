"""Fit unigram/trigram CPU LMs, evaluate validation, never open official test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

from character_lm import CharacterLM, aggregate
from character_lm_experiment import ROOT, digest, object_digest, partitions, partition_summary, validate_config


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def run(config_path, mode, output, smoke_run=None):
    config = json.loads(config_path.read_text())
    validate_config(config)
    target = output.resolve()
    if (ROOT / "outputs/language_models").resolve() not in target.parents or target.exists():
        raise ValueError("Choose a fresh directory under outputs/language_models/")
    if mode == "full":
        if smoke_run is None:
            raise ValueError("Full requires --smoke-run; smoke must be verified first")
        from verify_character_lm import verify
        smoke = verify(smoke_run)
        if smoke["mode"] != "smoke" or smoke["config"] != config:
            raise ValueError("Smoke configuration/mode does not match full")
    elif mode != "smoke":
        raise ValueError("Expected smoke or full")
    target.mkdir(parents=True)
    metadata = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "mode": mode,
                "config": config, "config_sha256": digest(config_path), "variant": "raw-v0",
                "python": platform.python_version(), "platform": platform.system(),
                "architecture": platform.machine(), "device": "cpu", "seed": None,
                "deterministic": True, "test_loaded": False,
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                "uv_lock_sha256": digest(ROOT / "uv.lock"),
                "manifests_sha256": {c: digest(ROOT / f"data/manifests/v2/{c}.csv") for c in config["corpora"]},
                "code_sha256": {name: digest(ROOT / "scripts" / name) for name in (
                    "character_lm.py", "character_lm_experiment.py", "run_character_lm.py", "verify_character_lm.py", "load_working_data.py")},
                "protocol_sha256": digest(ROOT / "docs/37_modele_langage_caracteres.md"),
                "smoke_results_sha256": digest(smoke_run / "results.json") if smoke_run else None,
                "results": []}
    for corpus in config["corpora"]:
        train, validation = partitions(corpus, mode, config)
        texts = [r["text_raw"] for r in train]
        vocabulary = sorted(set("".join(texts)))
        for order in config["orders"]:
            name = f"{corpus}__char_{order}gram__raw-v0"
            started = time.perf_counter()
            model = CharacterLM(order, config["alpha"], vocabulary).fit(texts)
            fit_seconds = time.perf_counter() - started
            checkpoint = target / f"{name}.json"
            write_json(checkpoint, model.to_dict())
            reloaded = CharacterLM.from_dict(json.loads(checkpoint.read_text()))
            if reloaded.score("exemple fictif اختبار") != model.score("exemple fictif اختبار"):
                raise ValueError("Checkpoint reload mismatch")
            started = time.perf_counter()
            scores = [{"id": row["id"], "group_id": row["group_id"], **model.score(row["text_raw"])}
                      for row in validation]
            evaluate_seconds = time.perf_counter() - started
            score_file = target / f"{name}.scores.json"
            write_json(score_file, scores)
            memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result = {"corpus": corpus, "order": order, "variant": "raw-v0",
                      "train": partition_summary(train), "validation": partition_summary(validation),
                      "vocabulary_size": model.vocabulary_size, "vocabulary_sha256": object_digest(vocabulary),
                      "contexts": len(model.counts), "observed_ngrams": sum(len(c) for c in model.counts.values()),
                      "train_targets": sum(model.totals.values()), "metrics": aggregate(scores),
                      "fit_seconds": fit_seconds, "evaluation_seconds": evaluate_seconds,
                      "process_peak_rss_bytes_cumulative": int(memory if sys.platform == "darwin" else memory * 1024),
                      "checkpoint": checkpoint.name, "checkpoint_sha256": digest(checkpoint),
                      "checkpoint_bytes": checkpoint.stat().st_size,
                      "scores": score_file.name, "scores_sha256": digest(score_file)}
            metadata["results"].append(result)
            write_json(target / "results.json", metadata)
            print(f"{corpus.upper()} | {order}-gram | {len(train)} train / {len(validation)} validation | "
                  f"perplexité={result['metrics']['perplexity']:.6f} | "
                  f"inconnus={result['metrics']['unknown_character_rate']:.4%}", flush=True)
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/character_lm_v1.json")
    parser.add_argument("--mode", choices=["smoke", "full"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-run", type=Path)
    args = parser.parse_args()
    run(args.config, args.mode, args.output, args.smoke_run)
