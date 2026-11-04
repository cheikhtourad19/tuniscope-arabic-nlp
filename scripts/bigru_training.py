"""Training/evaluation helpers for the fixed small BiGRU experiment."""
import json
import math
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from audit_text_composition import writing_system
from baseline_metrics import classification_metrics
from bigru_data import encode
from bigru_model import collate


def validate_config(config):
    if (config.get("version") != "bigru-word-raw-v2-v1" or config.get("manifest_version") != "v2"
            or config.get("variant") != "raw-v0" or config.get("tokenizer") != "unicode-word-symbol-v1"
            or config.get("corpora") != ["tsac", "madar"] or config.get("layers") != 1
            or config.get("pooling") != "masked-mean" or config.get("seeds") != [13, 42, 2026]
            or config.get("labels") != {"tsac": ["NEG", "POS"], "madar": ["MSA", "TN", "EG"]}
            or config.get("selection_metric") != "validation_macro_f1" or config.get("tie_break") != "earliest_epoch"):
        raise ValueError("Unexpected initial BiGRU protocol")
    for key in ("max_length", "embedding_dim", "hidden_dim", "batch_size", "max_epochs", "patience", "cpu_threads"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 <= config["dropout"] < 1 or not 0 < config["learning_rate"] < 1 or config["gradient_clip"] <= 0:
        raise ValueError("Invalid optimization settings")


def make_loader(rows, vocabulary, labels, config, shuffle, seed):
    mapping = {label: i for i, label in enumerate(labels)}
    data = [(encode(row["text_raw"], vocabulary, config["max_length"]), mapping[row["label"]]) for row in rows]
    return DataLoader(data, batch_size=config["batch_size"], shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(seed), collate_fn=collate)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def train_epoch(model, loader, optimizer, device, clip):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = total = 0
    nonzero_gradient_steps = 0
    sync(device)
    started = time.perf_counter()
    for tokens, lengths, targets in loader:
        tokens, targets = tokens.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens, lengths)
        loss = loss_fn(logits, targets)
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        norm = nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        nonzero_gradient_steps += float(norm) > 0
        optimizer.step()
        total_loss += float(loss.detach()) * len(targets)
        total += len(targets)
    sync(device)
    return {"loss": total_loss / total, "rows": total, "gradient_steps": len(loader),
            "nonzero_gradient_steps": nonzero_gradient_steps, "seconds": time.perf_counter() - started}


def evaluate(model, loader, labels, device):
    model.eval()
    probabilities, truth_ids, losses = [], [], []
    sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for tokens, lengths, targets in loader:
            logits = model(tokens.to(device), lengths)
            loss = nn.functional.cross_entropy(logits, targets.to(device), reduction="sum")
            if not torch.isfinite(loss):
                raise ValueError("Non-finite validation loss")
            losses.append(float(loss))
            probabilities.extend(logits.softmax(-1).cpu().tolist())
            truth_ids.extend(targets.tolist())
    sync(device)
    predicted = [labels[max(range(len(labels)), key=lambda i: scores[i])] for scores in probabilities]
    truth = [labels[i] for i in truth_ids]
    return {"loss": math.fsum(losses) / len(truth), "probabilities": probabilities,
            "predicted": predicted, "metrics": classification_metrics(truth, predicted, labels),
            "seconds": time.perf_counter() - started}


def add_writing_slices(metrics, rows, predicted, labels):
    metrics["by_writing_system"] = {}
    scripts = [writing_system(r["text_raw"]) for r in rows]
    for script in sorted(set(scripts)):
        indices = [i for i, s in enumerate(scripts) if s == script]
        truth = [rows[i]["label"] for i in indices]
        item = classification_metrics(truth, [predicted[i] for i in indices], labels)
        item["small_or_missing_class_slice"] = len(indices) < 30 or set(truth) != set(labels)
        metrics["by_writing_system"][script] = item
    return metrics


def save_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
