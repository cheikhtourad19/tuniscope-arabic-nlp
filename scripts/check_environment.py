"""Check the baseline environment with fictitious data only.

Run: sh scripts/project_uv.sh run --locked python scripts/check_environment.py
No authorized corpus is needed and no checkpoint is saved.
"""

import importlib.metadata
import json
import platform
import sys


def main():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Use the Python 3.12 environment described in docs/20_environnement_python.md")

    import numpy as np
    import pandas as pd
    from scipy import sparse
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    # These are invented examples for installation verification, not corpus data.
    frame = pd.DataFrame({
        "text_raw": ["behi service", "service behi", "service moch behi", "moch behi service"],
        "label": ["POS", "POS", "NEG", "NEG"],
    })
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), lowercase=False,
                                 min_df=2, sublinear_tf=True)
    analyzer = vectorizer.build_analyzer()
    if analyzer("abc  def") != analyzer("abc def"):
        raise RuntimeError("Recheck the assumed built-in whitespace processing before the ablation")
    features = vectorizer.fit_transform(frame["text_raw"])
    if not sparse.issparse(features) or features.shape[0] != len(frame):
        raise RuntimeError("TF-IDF sparse matrix check failed")
    classifier = LogisticRegression(C=1.0, class_weight=None, max_iter=1000, solver="lbfgs")
    classifier.fit(features, frame["label"])
    prediction = classifier.predict(features)
    if prediction.shape != (4,) or not set(prediction).issubset({"POS", "NEG"}):
        raise RuntimeError("Synthetic classifier check failed")
    if not np.isfinite(features.data).all():
        raise RuntimeError("Non-finite synthetic features")
    if pd.DataFrame.from_records([{"text": "مرحبا café 😄"}]).iloc[0]["text"] != "مرحبا café 😄":
        raise RuntimeError("DataFrame Unicode round-trip failed")
    print(json.dumps({
        "python": platform.python_version(),
        "platform": platform.system(),
        "architecture": platform.machine(),
        "versions": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")},
        "synthetic_sparse_tfidf_classifier": "passed",
        "unicode_dataframe": "passed",
        "char_tfidf_collapses_internal_whitespace": True,
        "corpus_read": False,
        "device": "cpu",
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
