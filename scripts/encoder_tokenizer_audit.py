"""Local-only WordPiece audit, no model weights or neural runtime required."""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path

from audit_text_composition import writing_system
from fetch_camelbert_tokenizer import ASSET_DIR, MODEL_ID, REVISION, verify_assets
from load_working_data import ROOT, load_records

PROBES = [
    ("arabizi", "el service behi barcha"),
    ("negation_latine", "el service moch behi"),
    ("negation_tunisienne", "الخدمة موش باهية"),
    ("negation_attachee", "ما عجبنيش"),
    ("francais", "Le service est très agréable."),
    ("melange", "service behi ama livraison très lente"),
    ("emojis", "🙂😍"),
    ("chiffres_arabizi", "7لو 3lech"),
    ("msa", "أين المحطة؟"),
    ("voyelles", "بَاهي"),
    ("repetition", "باااهي برشا"),
    ("controle_invisible", "موش\u200b باهي"),
]


def object_digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def load_tokenizer(directory=ASSET_DIR):
    source = verify_assets(directory)
    configuration = json.loads((directory / "config.json").read_text())
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(directory), use_fast=True,
                                              model_max_length=configuration["max_position_embeddings"],
                                              local_files_only=True, trust_remote_code=False)
    if type(tokenizer).__name__ != "BertTokenizerFast" or not tokenizer.is_fast:
        raise ValueError("Expected the pinned fast BERT tokenizer")
    if len(tokenizer) != configuration["vocab_size"] or tokenizer.num_special_tokens_to_add(pair=False) != 2:
        raise ValueError("Unexpected vocabulary or special-token format")
    return tokenizer, configuration, source


def parameter_counts(config):
    """Closed-form count for standard absolute-position BERT; no checkpoint read."""
    if (config["model_type"] != "bert" or config.get("position_embedding_type", "absolute") != "absolute"
            or config["hidden_size"] % config["num_attention_heads"]):
        raise ValueError("Unsupported BERT architecture")
    hidden, intermediate = config["hidden_size"], config["intermediate_size"]
    embeddings = (config["vocab_size"] + config["max_position_embeddings"] + config["type_vocab_size"]) * hidden + 2 * hidden
    attention = 4 * (hidden * hidden + hidden) + 2 * hidden
    feedforward = hidden * intermediate + intermediate + intermediate * hidden + hidden + 2 * hidden
    encoder = embeddings + config["num_hidden_layers"] * (attention + feedforward)
    pooler = hidden * hidden + hidden
    return {"method": "theoretical-standard-bert-formula-not-checkpoint-numel",
            "embeddings": embeddings, "encoder_without_pooler": encoder,
            "encoder_with_pooler": encoder + pooler,
            "sequence_classification_2_labels": encoder + pooler + (hidden + 1) * 2,
            "sequence_classification_3_labels": encoder + pooler + (hidden + 1) * 3,
            "masked_lm_tied_no_pooler": encoder + hidden * hidden + hidden + 2 * hidden + config["vocab_size"],
            "checkpoint_loaded": False}


def features(rows, tokenizer):
    encoded = tokenizer([row["text_raw"] for row in rows], add_special_tokens=False, truncation=False,
                        padding=False, return_attention_mask=False, return_token_type_ids=False, verbose=False)
    special = tokenizer.num_special_tokens_to_add(pair=False)
    return [{"id": row["id"], "group_id": row["group_id"], "script": writing_system(row["text_raw"]),
             "content_tokens": len(ids), "length_with_special": len(ids) + special,
             "unknown_tokens": ids.count(tokenizer.unk_token_id), "whitespace_fields": len(row["text_raw"].split())}
            for row, ids in zip(rows, encoded["input_ids"])]


def stats(records, candidates, minimum):
    if not records:
        raise ValueError("Cannot summarize an empty group")
    lengths = sorted(r["length_with_special"] for r in records)
    tokens = sum(r["content_tokens"] for r in records)
    fields = sum(r["whitespace_fields"] for r in records)
    unknown = sum(r["unknown_tokens"] for r in records)
    return {"rows": len(records), "groups": len({r["group_id"] for r in records}),
            "ids_sha256": object_digest([r["id"] for r in records]),
            "minimum_100_reached": len(records) >= minimum,
            "content_tokens": tokens, "whitespace_fields": fields,
            "subwords_per_whitespace_field": tokens / fields if fields else None,
            "unknown_tokens": unknown, "unknown_token_rate": unknown / tokens if tokens else 0,
            "rows_with_unknown": sum(r["unknown_tokens"] > 0 for r in records),
            "rows_with_unknown_rate": sum(r["unknown_tokens"] > 0 for r in records) / len(records),
            "empty_content_rows": sum(r["content_tokens"] == 0 for r in records),
            "length_with_special": {"p50": lengths[math.ceil(.5 * len(lengths)) - 1],
                                    "p95": lengths[math.ceil(.95 * len(lengths)) - 1],
                                    "p99": lengths[math.ceil(.99 * len(lengths)) - 1], "max": max(lengths)},
            "candidate_truncation": {str(limit): {
                "rows": sum(n > limit for n in lengths),
                "row_rate": sum(n > limit for n in lengths) / len(records),
                "removed_tokens": sum(max(n - limit, 0) for n in lengths)} for limit in candidates}}


def choose_length(corpora, config):
    eligible = [(corpus, script, item) for corpus, partitions in corpora.items()
                for script, item in partitions["train"]["by_writing_system"].items()
                if item["rows"] >= config["minimum_rows_per_writing_group"]]
    if not eligible:
        raise ValueError("No sufficiently populated training group")
    candidates = config["candidate_max_lengths"]
    selected = next((limit for limit in candidates if all(
        item["candidate_truncation"][str(limit)]["row_rate"] <= config["max_train_truncated_row_rate"] + 1e-12
        for _, _, item in eligible)), None)
    return {"max_length": selected if selected is not None else max(candidates),
            "coverage_target_met": selected is not None, "chosen_from": "train_only",
            "criterion": config["length_selection"],
            "eligible_training_groups": [f"{c}:{s}" for c, s, _ in eligible],
            "small_training_groups_not_used_for_selection": [f"{c}:{s}" for c, p in corpora.items()
                for s, item in p["train"]["by_writing_system"].items() if item["rows"] < config["minimum_rows_per_writing_group"]]}


def probe_results(tokenizer):
    result = []
    normalizer = tokenizer.backend_tokenizer.normalizer
    for name, text in PROBES:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        result.append({"name": name, "text": text, "invented": True,
                       "normalizer_output": normalizer.normalize_str(text),
                       "tokens": tokenizer.convert_ids_to_tokens(ids),
                       "unknown_tokens": ids.count(tokenizer.unk_token_id)})
    return result


def audit(config, directory=ASSET_DIR, loader=load_records):
    if (config["model_id"] != MODEL_ID or config["revision"] != REVISION or config["manifest_version"] != "v2"
            or config["variant"] != "raw-v0" or config["splits"] != ["train", "validation"]):
        raise ValueError("Expected pinned raw-v0/v2 train/validation audit")
    tokenizer, architecture, source = load_tokenizer(directory)
    if max(config["candidate_max_lengths"]) > architecture["max_position_embeddings"]:
        raise ValueError("Candidate exceeds architecture limit")
    corpora = {}
    for corpus in config["corpora"]:
        rows = loader(corpus, splits=("train", "validation"), manifest_version="v2")
        groups = {split: {r["group_id"] for r in rows if r["split"] == split} for split in config["splits"]}
        if groups["train"] & groups["validation"]:
            raise ValueError("Train/validation groups overlap")
        if any(r["split"] not in {"train", "validation"} for r in rows):
            raise ValueError("Unexpected held-out rows")
        corpora[corpus] = {}
        for split in config["splits"]:
            selected = [r for r in rows if r["split"] == split]
            encoded = features(selected, tokenizer)
            by_script = {script: stats([r for r in encoded if r["script"] == script], config["candidate_max_lengths"],
                                      config["minimum_rows_per_writing_group"])
                         for script in sorted({r["script"] for r in encoded})}
            corpora[corpus][split] = {"all": stats(encoded, config["candidate_max_lengths"], config["minimum_rows_per_writing_group"]),
                                      "by_writing_system": by_script,
                                      "labels": dict(sorted(Counter(r["label"] for r in selected).items()))}
    return {"model_id": MODEL_ID, "revision": REVISION, "model_license_declared": source["model_license_declared"],
            "architecture": architecture, "parameter_counts": parameter_counts(architecture),
            "manifest_version": "v2", "variant": "raw-v0", "test_loaded": False, "weights_downloaded": False,
            "tokenizer_class": type(tokenizer).__name__, "tokenizer_length": len(tokenizer),
            "normalizer": json.loads(tokenizer.backend_tokenizer.to_str())["normalizer"],
            "source_files": source["files"], "corpora": corpora, "length_decision": choose_length(corpora, config),
            "invented_probes": probe_results(tokenizer)}


def readable_summary(result):
    lines = ["CAMeLBERT Mix — audit du tokenizer, sans poids ni entraînement",
             f"Révision : {result['revision']}", "Texte brut ; train/validation v2, test officiel non sélectionné.",
             f"{'Corpus':<8}{'Split':<12}{'Écriture':<10}{'N':>7}{'Sous-mots/champ':>18}{'UNK tokens':>13}{'P99':>7}{'Max':>7}"]
    for corpus, partitions in result["corpora"].items():
        for split, data in partitions.items():
            for script, s in data["by_writing_system"].items():
                ratio = f"{s['subwords_per_whitespace_field']:.2f}" if s["subwords_per_whitespace_field"] is not None else "—"
                lines.append(f"{corpus.upper():<8}{split:<12}{script:<10}{s['rows']:>7}{ratio:>18}"
                             f"{s['unknown_token_rate']:>12.2%}{s['length_with_special']['p99']:>7}{s['length_with_special']['max']:>7}")
    decision = result["length_decision"]
    lines += ["", f"Longueur initiale proposée : {decision['max_length']} tokens, CLS/SEP compris.",
              f"Critère 99 % satisfait sur les groupes train retenus : {decision['coverage_target_met']}",
              "UNK, découpage et longueur ne mesurent pas la qualité de classification."]
    return "\n".join(lines)
