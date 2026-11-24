"""Plot paired validation audit results for TuniScope's Gold review."""

import argparse
import json
import os
from pathlib import Path

from audit_structure import ROOT


def plot(report, target):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    audit = json.loads((report / "validation_audit.json").read_text())
    robustness = json.loads((report / "robustness.json").read_text())
    if any(data["split"] != "validation" or data["test_loaded"] for data in (audit, robustness)):
        raise ValueError("Expected validation-only evidence")
    target.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "text.color": "#172438", "axes.labelcolor": "#172438"})
    colors = "#006e73", "#b54820"

    fig, ax = plt.subplots(figsize=(9, 4.7))
    fig.subplots_adjust(left=.17, right=.94, top=.76, bottom=.22)
    fig.suptitle("CAMeLBERT et TF-IDF · Écart de macro-F1",
                 x=.17, y=.94, ha="left", fontweight="bold", fontsize=19)
    fig.text(.17, .83, "Validation · 1 000 rééchantillonnages appariés par groupe", fontsize=12)
    ax.axvline(0, color="#94a3b8", linestyle="--", linewidth=1.5, zorder=0)
    for i, (corpus, estimates) in enumerate(reversed(list(audit["bootstrap"].items()))):
        point = estimates["difference"] * 100
        low, high = (value * 100 for value in estimates["percentile_interval_95"])
        ax.errorbar(point, i, xerr=[[point - low], [high - point]], fmt="o", markersize=8,
                    linewidth=2.5, capsize=6, color=colors[i])
        ax.text(.49, i + .16, f"{point:+.2f} pt [{low:+.2f} ; {high:+.2f}]".replace(".", ","),
                ha="center", fontsize=11, color=colors[i])
    ax.set(yticks=[0, 1], yticklabels=["MADAR", "TSAC"], ylim=(-.42, 1.43), xlim=(-1.6, 2.5),
           xlabel="CAMeLBERT (moyenne de trois graines) − TF-IDF (points)")
    ax.grid(axis="x", alpha=.20)
    fig.text(.17, .055, "La ligne à zéro représente un score égal entre les deux systèmes.", fontsize=11)
    fig.savefig(target / "bootstrap_validation.png", dpi=180)
    plt.close(fig)

    keys = ("normalisation_nfc_espaces", "espaces_triplement", "lettre_dupliquee")
    names = ("Normalisation NFC\net espaces", "Espaces ordinaires\ntriplés", "Une lettre\ndupliquée")
    fig, ax = plt.subplots(figsize=(9, 5.1))
    fig.subplots_adjust(left=.14, right=.95, top=.72, bottom=.22)
    fig.suptitle("TSAC · Effet de trois modifications de texte",
                 x=.14, y=.94, ha="left", fontweight="bold", fontsize=19)
    fig.text(.14, .845, "Comparaison avec le texte original sur les mêmes 1 774 avis",
             fontsize=12)
    ax.axhline(0, color="#94a3b8", linewidth=1.2, zorder=0)
    minimum = -.15
    for i, (model, title) in enumerate((("tfidf_char", "TF-IDF caractères"), ("camelbert", "CAMeLBERT · graine 42"))):
        values = [robustness["models"][model][key]["all"]["macro_f1_difference"] * 100 for key in keys]
        minimum = min(minimum, *values)
        positions = [x + (i - .5) * .32 for x in range(len(keys))]
        ax.bar(positions, values, .30, label=title, color=colors[i])
        for x, value in zip(positions, values, strict=True):
            label = "0" if value == 0 else f"{value:.2f}".replace("-", "−").replace(".", ",")
            ax.text(x, .035 if value == 0 else value + .09, label, ha="center",
                    color=colors[i] if value == 0 else "white", fontweight="bold", fontsize=11)
    ax.set(xticks=list(range(len(keys))), xticklabels=names, ylabel="Écart de macro-F1 (points)",
           ylim=(minimum - .10, .17))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}".replace(".", ",")))
    ax.grid(axis="y", alpha=.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.015), ncols=2, frameon=False)
    fig.text(.14, .045, "Variations artificielles sur la validation. Les labels originaux sont conservés.",
             fontsize=11)
    fig.savefig(target / "robustesse_tsac.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    plot(args.report, args.output)
