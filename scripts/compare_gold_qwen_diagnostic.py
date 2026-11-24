"""Run the frozen Qwen base on the exact local LoRA diagnostic samples."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import torch
from transformers import GenerationConfig

from fetch_qwen3 import MODEL_DIR, ROOT, sha256, verify_model
from grouped_bootstrap import group_bootstrap
from qwen_prompting import PROMPT_CONFIG, load_tokenizer
from run_qwen_lora_campaign import load_base, metrics
from run_qwen_validation import AUDIT, generate_batch, prepare, save_json


def run(diagnostic, private, output):
    if not private.resolve().is_relative_to((ROOT / "outputs").resolve()):
        raise ValueError("Keep individual generations in private outputs")
    if output.exists():
        raise FileExistsError("Preserve earlier diagnostic comparisons")
    source = json.loads(diagnostic.read_text())
    settings = source["settings"]
    if (source["split"] != "validation" or source["test_loaded"] or source["training"]
            or (settings["device"], settings["dtype"], settings["prompt_mode"]) != ("cpu", "float32", "zero_shot")
            or sha256(PROMPT_CONFIG) != source["prompt_config_sha256"]):
        raise ValueError("Expected a compatible local validation-only diagnostic")
    base = verify_model()
    if base["files"]["model.safetensors"]["sha256"] != source["base_sha256"]:
        raise ValueError("Base model differs from the LoRA diagnostic")
    prompting = json.loads(PROMPT_CONFIG.read_text())
    audit = json.loads(AUDIT.read_text())
    for corpus, expected in source["manifests_sha256"].items():
        if sha256(ROOT / f"data/manifests/v2/{corpus}.csv") != expected:
            raise ValueError("Validation manifest changed")
    started = perf_counter()
    torch.set_num_threads(4)
    device = torch.device("cpu")
    tokenizer, _ = load_tokenizer()
    tokenizer.padding_side = "left"
    generation = GenerationConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    options = {**prompting["initial_generation"], "max_new_tokens": prompting["max_new_tokens"], "use_cache": True}
    model = load_base().requires_grad_(False).eval().to(device)
    result = {"split": "validation", "test_loaded": False, "scope": "paired-local-balanced-diagnostic",
              "lora_diagnostic_sha256": sha256(diagnostic), "base_sha256": source["base_sha256"],
              "settings": settings, "script_sha256": sha256(Path(__file__)), "tasks": {}}
    batch_size = settings["batch_size"]
    with torch.inference_mode():
        for corpus, mode, _, pairs in prepare(tokenizer, prompting, audit):
            if mode != "zero_shot":
                continue
            labels = prompting["tasks"][corpus]["labels"]
            path = private / f"{corpus}.json"
            if sha256(path) != source["tasks"][corpus]["predictions_sha256"]:
                raise ValueError("Original local LoRA predictions changed")
            lora = json.loads(path.read_text())
            selected = {r["id"] for r in lora}
            batch_rows = [pair for pair in pairs if pair[0]["id"] in selected]
            base_rows = []
            for start in range(0, len(batch_rows), batch_size):
                base_rows.extend(generate_batch(model, tokenizer, generation, options, device,
                                                batch_rows[start:start + batch_size], labels))
            if [r["id"] for r in base_rows] != [r["id"] for r in lora]:
                raise ValueError("Different validation sample or decoding batch order")
            base_path = private / f"{corpus}_base.json"
            if base_path.exists():
                raise FileExistsError("Preserve previous base predictions")
            save_json(base_path, base_rows)

            def normalized(rows):
                return [{**r, "predicted": r["prediction"] or "INVALID"} for r in rows]

            result["tasks"][corpus] = {"base": metrics(base_rows, labels), "lora": metrics(lora, labels),
                                       "paired_bootstrap": group_bootstrap(normalized(base_rows),
                                                                           [normalized(lora)], labels),
                                       "base_predictions_sha256": sha256(base_path)}
            print(f"{corpus.upper()} · comparatif terminé · {perf_counter() - started:.1f} s", flush=True)
    result["seconds"] = perf_counter() - started
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--diagnostic", required=True, type=Path)
    p.add_argument("--private", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    run(args.diagnostic, args.private, args.output)
