"""Render presentation figures from aggregate Gold validation results."""

import argparse
import json
import os
from pathlib import Path

from audit_structure import ROOT


def render(report, target):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    data = json.loads(report.read_text(encoding="utf-8"))
    if data["test_loaded"] is not False or data["protocol"]["version"] != "gold-validation-v1":
        raise ValueError("Expected the Gold validation report")
    if target.exists():
        raise FileExistsError("Preserve previous figures")
    target.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#1e293b", "text.color": "#172438"})

    budget = data["budget"]["summary"]
    fig, ax = plt.subplots(figsize=(9, 5))
    fig.subplots_adjust(left=.13, right=.91, top=.78, bottom=.21)
    fig.suptitle("Sentiment tunisien · Effet du volume de données",
                 x=.13, y=.94, ha="left", fontweight="bold", fontsize=19)
    fig.text(.13, .845, "TF-IDF caractères + régression logistique · Validation TSAC", fontsize=13)
    x = [row["percent"] for row in budget]
    means = [row["mean_macro_f1"] * 100 for row in budget]
    ax.errorbar(x, means, yerr=[row["sd_macro_f1"] * 100 for row in budget],
                marker="o", capsize=5, linewidth=2.5, color="#006e73")
    for pct, score in zip(x, means, strict=True):
        label = f"{score:.2f} %".replace(".", ",")
        ax.annotate(label, (pct, score), xytext=(0, 20), textcoords="offset points",
                    ha="center", fontweight="bold")
    ax.set(xlabel="Part des données d’entraînement", ylabel="Macro-F1 moyen",
           xticks=x, xlim=(2, 109), ylim=(82, 96))
    ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.grid(axis="y", alpha=.23)
    fig.text(.13, .055, "Mêmes 1 774 textes de validation · Barres = écart-type sur trois graines",
             fontsize=11, color="#475569")
    fig.savefig(target / "budget_tfidf_tsac.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharey=True)
    fig.subplots_adjust(left=.095, right=.965, top=.74, bottom=.21, wspace=.21)
    fig.suptitle("CAMeLBERT · Couverture et erreurs avec abstention", x=.095, y=.94,
                 ha="left", fontweight="bold", fontsize=19)
    fig.text(.095, .845, "Chaque point représente un seuil évalué sur les trois modèles de chaque tâche",
             fontsize=12)
    for ax, (corpus, result) in zip(axes, data["encoder"].items(), strict=True):
        rows = [r for r in result["summary"] if r["accepted_error_rate"] is not None]
        ax.plot([100 * r["coverage"] for r in rows], [100 * r["accepted_error_rate"] for r in rows],
                "o-", color="#006e73", linewidth=2, markersize=5)
        chosen = next(r for r in rows if r["threshold"] == result["threshold_selected_on_validation"])
        x, y = 100 * chosen["coverage"], 100 * chosen["accepted_error_rate"]
        ax.scatter(x, y, color="#b54820", s=95, zorder=3)
        ax.annotate(f"Seuil {chosen['threshold']:.2f}".replace(".", ","), (x, y),
                    xytext=(-8, 11), textcoords="offset points", ha="right", color="#9e3713")
        ax.set(title=corpus.upper(), xlabel="Textes acceptés", xlim=(77, 102), ylim=(0, 7.5))
        ax.xaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
        ax.grid(axis="y", alpha=.23)
    axes[0].set_ylabel("Erreurs parmi les textes acceptés")
    fig.text(.095, .055, "Point orange : seuil le plus élevé qui accepte au moins 90 % des textes pour chaque graine.",
             fontsize=11, color="#475569")
    fig.savefig(target / "abstention_camelbert.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    render(arguments.report, arguments.output)
