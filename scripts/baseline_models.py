"""Shared classical model construction; no corpus access or fitting at import."""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import Normalizer


def tfidf_system(config):
    analyzer = config["tfidf"]["analyzer"]
    if analyzer not in {"char", "word", "word_char"}:
        raise ValueError("Supported TF-IDF analyzers are char, word and word_char")
    return f"tfidf_{analyzer}"


def vectorizer(params):
    params = {**params, "ngram_range": tuple(params["ngram_range"]),
              "dtype": np.float64}
    return TfidfVectorizer(**params)


def build_tfidf_model(config):
    system = tfidf_system(config)
    if system == "tfidf_word_char":
        params = config["tfidf"]
        if params["global_norm"] != "l2" or params["block_weights"] != {"word": 1.0, "char": 1.0}:
            raise ValueError("Initial combination requires equal blocks and global L2")
        if any(params[name]["analyzer"] != name for name in ("word", "char")):
            raise ValueError("Combined branch analyzer mismatch")
        features = FeatureUnion([(name, vectorizer(params[name])) for name in ("word", "char")],
                                transformer_weights=params["block_weights"], n_jobs=1)
        steps = [("tfidf", features), ("normalize", Normalizer(norm=params["global_norm"]))]
    else:
        steps = [("tfidf", vectorizer(config["tfidf"]))]
    return Pipeline(steps + [("classifier", LogisticRegression(**config["classifier"]))])


def vectorizer_branches(model):
    features = model["tfidf"]
    if isinstance(features, FeatureUnion):
        return dict(features.transformer_list)
    return {features.analyzer: features}


def vocabulary_sizes(model):
    return {name: len(branch.vocabulary_) for name, branch in vectorizer_branches(model).items()}


def verify_model_config(model, config):
    """Check saved estimators against their declared branch/classifier settings."""
    combined = tfidf_system(config) == "tfidf_word_char"
    expected = {name: config["tfidf"][name] for name in ("word", "char")} if combined else {
        config["tfidf"]["analyzer"]: config["tfidf"]}
    branches = vectorizer_branches(model)
    if set(branches) != set(expected):
        raise ValueError("Checkpoint branch mismatch")
    for name, params in expected.items():
        actual = branches[name].get_params()
        for key, value in params.items():
            if key == "ngram_range":
                value = tuple(value)
            if key == "dtype":
                if np.dtype(actual[key]) != np.dtype(value):
                    raise ValueError("Checkpoint dtype mismatch")
            elif actual[key] != value:
                raise ValueError(f"Checkpoint TF-IDF parameter mismatch: {name}/{key}")
    if combined:
        if (model["tfidf"].transformer_weights != config["tfidf"]["block_weights"] or
                model["normalize"].norm != config["tfidf"]["global_norm"]):
            raise ValueError("Checkpoint block scaling mismatch")
    if any(model["classifier"].get_params()[key] != value for key, value in config["classifier"].items()):
        raise ValueError("Checkpoint classifier parameter mismatch")


def zero_feature_rows(model, texts):
    """Count rows with no retained features, without exposing their texts."""
    matrix = model[:-1].transform(texts)
    return int(np.count_nonzero(matrix.getnnz(axis=1) == 0))
