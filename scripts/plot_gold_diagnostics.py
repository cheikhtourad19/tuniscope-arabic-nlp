"""Render calibration and local LoRA diagnostic figures for the Gold review."""

import argparse
import json
import os
from pathlib import Path

from audit_structure import ROOT


def plot(source, target):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    import numpy as np

    calibration = json.loads((source / "calibration.json").read_text())
    lora = json.loads((source / "diagnostic.json").read_text())
    if any(r["test_loaded"] or r["split"] != "validation" for r in (calibration, lora)):
        raise ValueError("Expected validation evidence")
    target.mkdir(parents=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.labelcolor": "#1e293b", "text.color": "#172438"})

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.10, right=.96, top=.74, bottom=.19, wspace=.16)
    fig.suptitle("CAMeLBERT · La confiance correspond-elle aux réussites ?",
                 x=.10, y=.95, ha="left", fontweight="bold", fontsize=18)
    fig.text(.10, .85, "Validation groupée en cinq plis · Mêmes classes prédites après calibration")
    for ax, (corpus, result) in zip(axes, calibration["tasks"].items(), strict=True):
        ax.plot([.35, 1], [.35, 1], "--", color="#94a3b8", linewidth=1.5,
                label="Confiance bien alignée")
        for key, name, color in (("raw", "Scores bruts", "#b54820"),
                                 ("cross_fit", "Après calibration", "#006e73")):
            bins = result[key]["bins"]
            ax.scatter([r["confidence"] for r in bins], [r["accuracy"] for r in bins],
                       s=[18 + 60 * r["n"] / result[key]["n"] for r in bins],
                       label=name, color=color)
        ax.set(title=corpus.upper(), xlabel="Confiance moyenne", xlim=(.35, 1.025), ylim=(.25, 1.025))
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(alpha=.18)
    axes[0].set_ylabel("Réponses correctes observées")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.54, .025),
               ncols=3, frameon=False)
    fig.savefig(target / "calibration_camelbert.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5.3))
    fig.subplots_adjust(left=.105, right=.98, top=.75, bottom=.20, wspace=.24)
    fig.suptitle("Qwen + LoRA · Où le modèle se trompe",
                 x=.10, y=.95, ha="left", fontweight="bold", fontsize=19)
    fig.text(.10, .86, "Essai local sans exemples dans la consigne · 144 textes équilibrés par tâche")
    for ax, (corpus, result) in zip(axes, lora["tasks"].items(), strict=True):
        labels, columns = result["labels"], result["prediction_columns"]
        matrix = np.array(result["confusion_matrix"])
        ax.imshow(matrix, cmap="GnBu", vmin=0, vmax=int(matrix.max()), aspect="auto")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                ax.text(j, i, str(value), ha="center", va="center", fontweight="bold",
                        color="white" if value > matrix.max() * .72 else "#172438")
        ax.set(title=corpus.upper(), xticks=range(len(columns)), xticklabels=columns,
               yticks=range(len(labels)), yticklabels=labels, xlabel="Prédiction")
    axes[0].set_ylabel("Label de référence")
    fig.text(.10, .06, "Effectifs du nouveau diagnostic CPU avec l’adaptateur de la graine 42.",
             fontsize=11)
    fig.savefig(target / "erreurs_qwen_lora.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    plot(args.report, args.output)
