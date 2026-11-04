"""Export actual epoch histories and selected-checkpoint confusion matrices."""
import argparse
import json
import os
from pathlib import Path

from character_lm_experiment import ROOT, digest, read_artifact


def export_figures(directory):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib-cache"))
    # A Jupyter parent may export an inline backend unavailable in this profile.
    os.environ["MPLBACKEND"] = "Agg"
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from verify_bigru import verify
    metadata = verify(directory)
    target = directory / "figures"
    if target.exists():
        raise ValueError("Figures already exist; preserve previous exports")
    target.mkdir()
    provenance = {"source_results_sha256": digest(directory / "results.json"), "split": "validation", "files": {}}
    for result in metadata["results"]:
        history = json.loads(read_artifact(directory, result["history"]).read_text())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout="constrained")
        epochs = [h["epoch"] for h in history]
        axes[0].plot(epochs, [h["train"]["loss"] for h in history], marker=".", label="Train")
        axes[0].plot(epochs, [h["validation_loss"] for h in history], marker=".", label="Validation")
        axes[0].set(title="Perte (entropie croisée)", xlabel="Époque", ylabel="Perte moyenne")
        axes[0].legend()
        axes[1].plot(epochs, [h["validation_macro_f1"] for h in history], marker="o", color="#167a8a")
        axes[1].axvline(result["best_epoch"], linestyle="--", color="gray", label="Époque retenue")
        axes[1].set(title="Macro-F1 de validation", xlabel="Époque", ylabel="Macro-F1", ylim=(0, 1))
        axes[1].legend()
        matrix, labels = result["metrics"]["confusion_matrix"], result["metrics"]["labels"]
        axes[2].imshow(matrix, cmap="Blues", vmin=0)
        axes[2].set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
                    title="Checkpoint retenu", xlabel="Prédiction", ylabel="Label réel")
        maximum = max(map(max, matrix))
        for i in range(len(labels)):
            for j in range(len(labels)):
                axes[2].text(j, i, str(matrix[i][j]), ha="center", va="center", color="white" if matrix[i][j] > maximum / 2 else "black")
        fig.suptitle(f"BiGRU — {result['corpus'].upper()} — graine {result['seed']} — {metadata['mode']}\n"
                     "Validation v2 ; test officiel non évalué", fontsize=12)
        filename = f"{result['corpus']}_seed_{result['seed']}.png"
        fig.savefig(target / filename, dpi=160)
        plt.close(fig)
        provenance["files"][filename] = digest(target / filename)
    (target / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"Figures enregistrées : {target}")
    return target


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    export_figures(parser.parse_args().run.resolve())
