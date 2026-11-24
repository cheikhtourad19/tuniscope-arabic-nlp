"""Paired macro-F1 intervals from fixed labels and manifest groups."""

import numpy as np


def macro_f1(matrices):
    diagonal = matrices.diagonal(axis1=-2, axis2=-1)
    denominators = matrices.sum(axis=-1) + matrices.sum(axis=-2)[..., :diagonal.shape[-1]]
    return np.divide(2 * diagonal, denominators, out=np.zeros_like(diagonal, dtype=float),
                     where=denominators > 0).mean(axis=-1)


def group_bootstrap(reference, alternatives, labels, *, samples=1000, seed=42):
    def alignment(rows):
        return [(row["id"], row["group_id"], row["truth"]) for row in rows]

    aligned = alignment(reference)
    if not aligned or len({row[0] for row in aligned}) != len(aligned) or not alternatives:
        raise ValueError("Expected unique, nonempty paired examples")
    if any(alignment(rows) != aligned for rows in alternatives):
        raise ValueError("Validation ID/group/truth mismatch")
    groups = {key: i for i, key in enumerate(sorted({r["group_id"] for r in reference}))}
    target = {label: i for i, label in enumerate(labels)}
    n, k, m = len(groups), len(labels), 1 + len(alternatives)
    columns = {**target, "INVALID": k}
    matrix = np.zeros((n, m, k, k + 1), dtype=np.int64)
    i = [groups[r["group_id"]] for r in reference]
    j = [target[r["truth"]] for r in reference]
    for system, predictions in enumerate((reference, *alternatives)):
        np.add.at(matrix, (i, system, j, [columns[r["predicted"]] for r in predictions]), 1)
    point_scores = macro_f1(matrix.sum(axis=0))
    random = np.random.default_rng(seed)
    gaps = np.empty(samples)
    for iteration in range(samples):
        weights = np.bincount(random.integers(n, size=n), minlength=n)
        scores = macro_f1(np.einsum("i,imjk->mjk", weights, matrix))
        gaps[iteration] = scores[1:].mean() - scores[0]
    low, high = np.quantile(gaps, [.025, .975])
    return {"groups": n, "rows": len(reference), "samples": samples, "seed": seed,
            "reference_macro_f1": float(point_scores[0]), "mean_alt_macro_f1": float(point_scores[1:].mean()),
            "difference": float(point_scores[1:].mean() - point_scores[0]),
            "percentile_interval_95": [float(low), float(high)]}
