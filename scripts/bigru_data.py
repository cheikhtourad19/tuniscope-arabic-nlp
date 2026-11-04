"""Train-only word/symbol vocabulary and tokenization; no torch dependency."""
from collections import Counter
import hashlib
import math
import re

from load_working_data import load_records

TOKENIZER_VERSION = "unicode-word-symbol-v1"
PAD, UNK = 0, 1


def tokenize(text):
    # Unicode alphanumerics including Arabizi digits; keep emoji/punctuation.
    # No casefold, transliteration, normalization or regex-based label inference.
    return re.findall(r"[^\W_]+|[^\s]", text, flags=re.UNICODE)


def build_vocabulary(texts, max_vocabulary=20000, min_frequency=2):
    if max_vocabulary < 2 or min_frequency < 1:
        raise ValueError("Invalid vocabulary limits")
    counts = Counter(token for text in texts for token in tokenize(text))
    retained = sorted(((token, n) for token, n in counts.items() if n >= min_frequency),
                      key=lambda item: (-item[1], item[0]))[:max_vocabulary - 2]
    return {token: i + 2 for i, (token, _) in enumerate(retained)}


def encode(text, vocabulary, max_length):
    if max_length < 1:
        raise ValueError("Positive maximum length required")
    tokens = tokenize(text)
    # One UNK for an empty token sequence: no zero-length sequence reaches GRU.
    return [vocabulary.get(token, UNK) for token in tokens[:max_length]] or [UNK]


def audit(texts, vocabulary, max_length):
    lengths = []
    unknown = total = truncated = removed = empty = 0
    for text in texts:
        tokens = tokenize(text)
        lengths.append(len(tokens))
        total += len(tokens)
        unknown += sum(token not in vocabulary for token in tokens)
        truncated += len(tokens) > max_length
        removed += max(0, len(tokens) - max_length)
        empty += not tokens
    ordered = sorted(lengths)
    return {"rows": len(texts), "tokens_before_truncation": total,
            "unknown_tokens_before_truncation": unknown,
            "unknown_token_rate_before_truncation": unknown / total if total else 0,
            "truncated_rows": truncated, "truncated_tokens": removed, "empty_token_rows": empty,
            "p50_tokens": ordered[max(0, math.ceil(.5 * len(ordered)) - 1)],
            "p95_tokens": ordered[max(0, math.ceil(.95 * len(ordered)) - 1)],
            "p99_tokens": ordered[max(0, math.ceil(.99 * len(ordered)) - 1)],
            "max_tokens": max(ordered)}


def partitions(corpus, mode, config):
    if mode not in {"smoke", "full"}:
        raise ValueError("Expected smoke or full")
    records = load_records(corpus, splits=("train", "validation"), manifest_version="v2")
    result = []
    for split in ("train", "validation"):
        selected = [r for r in records if r["split"] == split]
        if mode == "smoke":
            selected = sorted(selected, key=lambda r: hashlib.sha256(r["id"].encode()).hexdigest())[
                :config["smoke_rows_per_partition"]]
        if not selected:
            raise ValueError("Empty partition")
        result.append(selected)
    if {r["group_id"] for r in result[0]} & {r["group_id"] for r in result[1]}:
        raise ValueError("Train/validation group overlap")
    return tuple(result)
