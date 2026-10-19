"""Classification metrics with a fixed class list and explicit zero divisions."""

from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score,
)


def classification_metrics(truth, predicted, labels):
    if len(truth) != len(predicted) or not truth:
        raise ValueError("Expected equal, nonempty truth and prediction arrays")
    if set(truth) - set(labels) or set(predicted) - set(labels):
        raise ValueError("Unexpected classification label")
    matrix = confusion_matrix(truth, predicted, labels=labels).tolist()
    recalls_present = [row[i] / sum(row) for i, row in enumerate(matrix) if sum(row)]
    return {
        "n": len(truth), "labels": labels,
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": sum(recalls_present) / len(recalls_present),
        "macro_f1": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
        "per_class": {label: item for label, item in classification_report(
            truth, predicted, labels=labels, output_dict=True, zero_division=0
        ).items() if label in labels},
        "confusion_matrix": matrix,
        "zero_division": 0,
    }
