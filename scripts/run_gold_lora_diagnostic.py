"""Replay a fixed validation subset with the archived Qwen LoRA seed 42."""

import argparse
from collections import Counter
from hashlib import sha256 as stable_hash
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import GenerationConfig

from fetch_qwen3 import MODEL_DIR, ROOT, sha256, verify_model
from qwen_prompting import PROMPT_CONFIG, load_tokenizer
from run_qwen_lora_campaign import reload_adapter, score_stage
from run_qwen_validation import AUDIT, generate_batch, prepare, save_json


ADAPTER = ROOT / "outputs/qwen/2026-11-07_cuda_lora_c10d02666e154607ac9d7faddbd47838_full/seed_42"
ADAPTER_HASHES = {
    "adapter_config.json": "d6ee95bad07c71771aa81256e8ffcecdd092241432dd2a585188aa6637cb5433",
    "adapter_model.safetensors": "e4a24bc6949d9f11645e230da42751b2c2308686bd0a6d5ea6d9403618b30e9b",
}
ROWS_PER_TASK = 144
BATCH_SIZE = 8


def sample_distinct_groups(rows, labels):
    ordered = sorted(rows, key=lambda row: (stable_hash(f'42|{row["id"]}'.encode()).digest(), row["id"]))
    quota = ROWS_PER_TASK // len(labels)
    used_groups, selected = set(), set()
    counts = Counter()
    for row in ordered:
        label, group = row["label"], row["group_id"]
        if counts[label] < quota and group not in used_groups:
            selected.add(row["id"])
            used_groups.add(group)
            counts[label] += 1
    if counts != {label: quota for label in labels}:
        raise ValueError("Insufficient distinct groups for balanced validation sample")
    return selected


def run(private, public):
    if (ROOT / "outputs").resolve() not in private.resolve().parents:
        raise ValueError("Store example-level audit material privately")
    if public.exists() or private.exists():
        raise FileExistsError("Choose new diagnostic outputs")
    for name, expected in ADAPTER_HASHES.items():
        if sha256(ADAPTER / "adapter" / name) != expected:
            raise ValueError(f"Adapter hash mismatch: {name}")
    base = verify_model()
    prompting = json.loads(PROMPT_CONFIG.read_text())
    audit = json.loads(AUDIT.read_text())
    if audit["config_sha256"] != sha256(PROMPT_CONFIG):
        raise ValueError("Prompt audit hash mismatch")
    for corpus, expected in audit["manifests_sha256"].items():
        if sha256(ROOT / f"data/manifests/v2/{corpus}.csv") != expected:
            raise ValueError("Manifest hash mismatch")

    started = perf_counter()
    device = torch.device("cpu")
    torch.set_num_threads(4)
    tokenizer, _ = load_tokenizer()
    tokenizer.padding_side = "left"
    generation = GenerationConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    options = {**prompting["initial_generation"], "max_new_tokens": prompting["max_new_tokens"], "use_cache": True}
    model = reload_adapter(ADAPTER, device).requires_grad_(False).eval()
    private.mkdir(parents=True)
    summary = {"split": "validation", "manifest_version": "v2", "test_loaded": False, "training": False,
               "scope": "new-balanced-distinct-group-diagnostic-not-original-cuda-run",
               "settings": {"device": "cpu", "dtype": "float32", "seed": 42, "prompt_mode": "zero_shot",
                            "batch_size": BATCH_SIZE, "selection": "fixed hash order with label quotas"},
               "base_sha256": base["files"]["model.safetensors"]["sha256"], "adapter_hashes": ADAPTER_HASHES,
               "prompt_config_sha256": sha256(PROMPT_CONFIG), "manifests_sha256": audit["manifests_sha256"],
               "code_sha256": sha256(Path(__file__)), "tasks": {}}
    with torch.inference_mode():
        for corpus, mode, validation, pairs in prepare(tokenizer, prompting, audit):
            if mode != "zero_shot":
                continue
            labels = prompting["tasks"][corpus]["labels"]
            ids = sample_distinct_groups(validation, labels)
            chosen = [(row, tokens) for row, tokens in pairs if row["id"] in ids]
            predictions = []
            for start in range(0, len(chosen), BATCH_SIZE):
                predictions.extend(generate_batch(model, tokenizer, generation, options, device,
                                                  chosen[start:start + BATCH_SIZE], labels))
            if len(predictions) != ROWS_PER_TASK or {r["id"] for r in predictions} != ids:
                raise ValueError("Diagnostic size or alignment mismatch")
            output = private / f"{corpus}.json"
            save_json(output, predictions)
            summary["tasks"][corpus] = {**score_stage(predictions, labels),
                                        "predictions_sha256": sha256(output)}
            print(f"{corpus.upper()} · {len(predictions)} avis · {perf_counter() - started:.1f} s", flush=True)
    summary["seconds"] = perf_counter() - started
    public.parent.mkdir(parents=True, exist_ok=True)
    save_json(public, summary)


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--private", required=True, type=Path)
    cli.add_argument("--output", required=True, type=Path)
    args = cli.parse_args()
    run(args.private, args.output)
