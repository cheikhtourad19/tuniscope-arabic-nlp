"""Sparse character/lexical concatenation with unchanged character block."""
from collections import Counter
import hashlib
import json

import numpy as np
from scipy import sparse
from sklearn.linear_model import LogisticRegression

from baseline_models import vectorizer
from rule_baselines import fit_rules, predict_rule


def systems(corpus):
    if corpus == "tsac":
        return ["char_control", "char_lexicon", "char_lexicon_negation"]
    if corpus == "madar":
        return ["char_control", "char_lexicon"]
    raise ValueError("Unknown corpus")


def validate_config(config, reference):
    for key in ("manifest_version", "corpora", "labels", "seed", "tfidf", "classifier", "threads", "smoke_rows_per_partition"):
        if config.get(key) != reference.get(key):
            raise ValueError(f"Character reference condition changed: {key}")
    if (config["variant"] != "raw-v0" or config["manifest_version"] != "v2" or
        config["rule_feature_schema"] != "class-vote-fractions-plus-hit-v1" or
        config["rule_feature_weight"] != 1.0 or config["global_normalization"] is not None):
        raise ValueError("Unsupported augmented-feature protocol")


def rule_features(rows, rules, *, negation=False):
    """Scores are bounded fractions plus a hit flag, not class probabilities."""
    values = []
    for row in rows:
        result = predict_rule(row["text_raw"], rules, negation=negation)
        total = sum(result["scores"].values())
        values.append([result["scores"][label] / total if total else 0.0 for label in rules["labels"]]
                      + [float(result["lexical_hit"])])
    return sparse.csr_matrix(np.asarray(values, dtype=np.float64).reshape(len(rows), len(rules["labels"])+1))


def feature_matrix(model, rows):
    char = model["vectorizer"].transform([r["text_raw"] for r in rows])
    if model["system"] == "char_control":
        return char
    lexical = rule_features(rows, model["rules"], negation=model["system"] == "char_lexicon_negation")
    return sparse.hstack([char, lexical], format="csr")


def fit_model(rows, corpus, system, config, rule_config):
    if system not in systems(corpus) or not rows or any(r["split"] != "train" for r in rows):
        raise ValueError("Fit requires a valid system and train rows only")
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate training IDs")
    labels = config["labels"][corpus]
    if set(r["label"] for r in rows) != set(labels) or rule_config["labels"][corpus] != labels:
        raise ValueError("Unexpected or missing training classes")
    v = vectorizer(config["tfidf"])
    v.fit([r["text_raw"] for r in rows])
    # The control also records the same train lexicon for provenance, but does
    # not use it in its feature matrix or classifier.
    rules = fit_rules(rows, labels, rule_config)
    model = {"schema_version":"1.0", "corpus":corpus, "system":system, "config":config,
             "vectorizer":v, "rules":rules, "labels":labels,
             "train_rows":len(rows), "train_labels":dict(Counter(r["label"] for r in rows)),
             "train_ids_sha256":hashlib.sha256(json.dumps([r["id"] for r in rows]).encode()).hexdigest()}
    matrix = feature_matrix(model, rows)
    classifier = LogisticRegression(**config["classifier"])
    classifier.fit(matrix, [r["label"] for r in rows])
    model["classifier"] = classifier
    model["feature_count"] = matrix.shape[1]
    return model


def predict_rows(model, rows):
    matrix = feature_matrix(model, rows)
    if matrix.shape[1] != model["feature_count"]:
        raise ValueError("Feature schema changed")
    return model["classifier"].predict(matrix).tolist()


def coverage(model, rows):
    lexical = rule_features(rows, model["rules"], negation=model["system"] == "char_lexicon_negation")
    char = model["vectorizer"].transform([r["text_raw"] for r in rows])
    return {"character_features":char.shape[1],
            "additional_features":0 if model["system"] == "char_control" else lexical.shape[1],
            "rows":len(rows), "zero_character_rows":int(np.count_nonzero(char.getnnz(axis=1)==0)),
            "zero_rule_block_rows":int(np.count_nonzero(lexical.getnnz(axis=1)==0)),
            "zero_combined_rows":int(np.count_nonzero(feature_matrix(model,rows).getnnz(axis=1)==0))}
