"""Train-derived lexical voting and a bounded, explicitly heuristic negation rule."""
from collections import Counter
import hashlib
import json
import math
import re
import unicodedata

TOKEN_RE = re.compile(r"_x[0-9a-f]{4}_|[^\W_]+|[.!?؟,،;؛:\n]", re.IGNORECASE)
PUNCTUATION = set(".!?؟,،;؛:\n")


def tokenize(text):
    """Keep word/Arabizi units and clause boundaries, not emoji/punctuation cues."""
    text = unicodedata.normalize("NFC", text).casefold()
    return [match.group() for match in TOKEN_RE.finditer(text)]


def boundary(token):
    return token in PUNCTUATION or bool(re.fullmatch(r"_x[0-9a-f]{4}_", token))


def controls(config):
    n = config["negation"]
    return set(n["markers"]) | set(n["clause_breaks"])


def validate_config(config):
    lex = config["lexicon"]
    if config["manifest_version"] != "v2" or config["tokenization"] != "rule-nfc-casefold-v1" or config["fallback"] != "train-majority":
        raise ValueError("Unsupported rule protocol")
    if any(type(lex[k]) is not int or lex[k] < 1 for k in
           ("min_class_document_frequency", "max_terms_per_class", "min_token_length")):
        raise ValueError("Positive integer lexicon limits required")
    if lex["smoothing"] <= 0 or lex["min_log_rate_ratio"] <= 0 or not all(math.isfinite(lex[k]) for k in ("smoothing", "min_log_rate_ratio")):
        raise ValueError("Positive finite smoothing and threshold required")
    if type(config["negation"]["window_words"]) is not int or config["negation"]["window_words"] < 1:
        raise ValueError("Positive negation window required")
    items = config["negation"]["markers"] + config["negation"]["clause_breaks"]
    if any(not item or tokenize(item) != [item] for item in items):
        raise ValueError("Controls must be distinct atomic lowercased tokens")
    for phrase in config["negation"]["exceptions"]:
        if not phrase or any(tokenize(t) != [t] for t in phrase):
            raise ValueError("Exceptions must be explicit token sequences")


def fit_rules(rows, labels, config):
    validate_config(config)
    if not rows or any(r["split"] != "train" or r["label"] not in labels for r in rows):
        raise ValueError("Lexicon construction requires valid train rows only")
    if len(set(labels)) != len(labels) or len(labels) < 2 or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Distinct labels and train IDs required")
    counts = Counter(r["label"] for r in rows)
    if any(not counts[label] for label in labels):
        raise ValueError("Every configured class requires training examples")
    token_lists = [tokenize(r["text_raw"]) for r in rows]
    token_sets = [set(tokens) for tokens in token_lists]
    attested = set().union(*token_sets)
    frequencies = {label: Counter() for label in labels}
    lex = config["lexicon"]
    excluded = controls(config)
    for row, tokens in zip(rows, token_sets):
        frequencies[row["label"]].update(t for t in tokens if len(t) >= lex["min_token_length"]
                                          and not boundary(t) and t not in excluded)
    vocabulary = sorted(set().union(*(set(frequencies[label]) for label in labels)))
    candidates = {label: [] for label in labels}
    alpha = lex["smoothing"]
    for token in vocabulary:
        associations = []
        for label in labels:
            inside = frequencies[label][token]
            outside = sum(frequencies[other][token] for other in labels if other != label)
            log_ratio = math.log((inside + alpha) / (counts[label] + 2 * alpha)) - math.log(
                (outside + alpha) / (len(rows) - counts[label] + 2 * alpha))
            associations.append((log_ratio, label, inside))
        # Lexical tie order is fixed by the configured class order, not the
        # validation label, corpus order, or any later outcome.
        best = max(range(len(labels)), key=lambda i: associations[i][0])
        ratio, label, frequency = associations[best]
        if frequency >= lex["min_class_document_frequency"] and ratio >= lex["min_log_rate_ratio"]:
            candidates[label].append({"token": token, "label": label,
                                      "log_rate_ratio": ratio, "class_document_frequency": frequency})
    entries = []
    for label in labels:
        entries.extend(sorted(candidates[label], key=lambda e: (-e["log_rate_ratio"], e["token"]))[:lex["max_terms_per_class"]])
    n = config["negation"]
    return {"schema_version": "1.0", "config": config, "labels": labels,
            "majority": max(labels, key=lambda label: counts[label]),
            "train_rows": len(rows), "train_labels": {label: counts[label] for label in labels},
            "train_id_sha256": hashlib.sha256(json.dumps(sorted(r["id"] for r in rows)).encode()).hexdigest(),
            "entries": entries,
            "active_negation_markers": [t for t in n["markers"] if t in attested],
            "active_clause_breaks": [t for t in n["clause_breaks"] if t in attested],
            "active_exceptions": [p for p in n["exceptions"] if any(
                tokens[i:i+len(p)] == p for tokens in token_lists for i in range(len(tokens)))]}


def predict_rule(text, model, *, negation=False):
    labels = model["labels"]
    if negation and set(labels) != {"NEG", "POS"}:
        raise ValueError("Negation polarity flip only applies to binary sentiment")
    lexicon = {e["token"]: e["label"] for e in model["entries"]}
    tokens = tokenize(text)
    remaining, negative = 0, False
    votes = set()
    hits, flipped = set(), set()
    triggers = 0
    for i, token in enumerate(tokens):
        if boundary(token) or token in model["active_clause_breaks"]:
            remaining, negative = 0, False
            continue
        exception = any(tokens[i:i+len(p)] == p for p in model["active_exceptions"])
        if token in model["active_negation_markers"]:
            if negation and not exception:
                negative = not negative if remaining else True
                remaining = model["config"]["negation"]["window_words"]
                triggers += 1
            continue
        if token in lexicon:
            original = lexicon[token]
            label = ("NEG" if original == "POS" else "POS") if negation and remaining and negative else original
            votes.add((token, label))
            hits.add(token)
            if label != original:
                flipped.add(token)
        if remaining:
            remaining -= 1
    scores = {label: sum(vote == label for _,vote in votes) for label in labels}
    maximum = max(scores.values())
    winners = [label for label in labels if scores[label] == maximum]
    decisive = maximum > 0 and len(winners) == 1
    return {"predicted": winners[0] if decisive else model["majority"],
            "decisive": decisive, "fallback": not decisive, "lexical_hit": bool(hits),
            "lexical_hits": len(hits), "flipped_cues": len(flipped), "negation_triggers": triggers,
            "scores": scores}
