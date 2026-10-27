"""Plot saved v2 TF-IDF validation runs, without fitting or opening raw data.

Only the optional --verify subprocess reloads trusted checkpoints through the
unchanged project environment. Matplotlib belongs in the notebook environment.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
NAMES = {"majority": "Majorité", "tfidf_char": "Caractères",
         "tfidf_word": "Mots", "tfidf_word_char": "Mots + caractères"}
COLORS = ("#8795a5", "#087f8c", "#d98a20")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def local_path(root, name):
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError("Artifact must stay inside its source run")
    return path


def metrics_from_predictions(predictions, labels):
    """Independent stdlib calculation, so plotting does not need sklearn."""
    if not predictions or len(set(labels)) != len(labels) or not labels:
        raise ValueError("Nonempty predictions and distinct labels required")
    matrix = [[0] * len(labels) for _ in labels]
    for row in predictions:
        if row["truth"] not in labels or row["predicted"] not in labels:
            raise ValueError("Unknown prediction label")
        actual_index = labels.index(row["truth"])
        predicted_index = labels.index(row["predicted"])
        matrix[actual_index][predicted_index] += 1
    per_class, recalls = {}, []
    for i, label in enumerate(labels):
        support, announced, correct = sum(matrix[i]), sum(r[i] for r in matrix), matrix[i][i]
        precision = correct / announced if announced else 0
        recall = correct / support if support else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1-score": f1, "support": support}
        if support:
            recalls.append(recall)
    return {"n": len(predictions), "labels": labels, "confusion_matrix": matrix,
            "accuracy": sum(matrix[i][i] for i in range(len(labels))) / len(predictions),
            "macro_f1": sum(r["f1-score"] for r in per_class.values()) / len(labels),
            "balanced_accuracy": sum(recalls) / len(recalls), "per_class": per_class}


def check_metrics(actual, saved):
    for field in ("n", "labels", "confusion_matrix"):
        if actual[field] != saved[field]:
            raise ValueError(f"Saved {field} differs from predictions")
    for field in ("accuracy", "macro_f1", "balanced_accuracy"):
        if not math.isfinite(saved[field]) or not math.isclose(actual[field], saved[field], abs_tol=1e-12, rel_tol=0):
            raise ValueError(f"Saved {field} differs from predictions")
    for label, row in actual["per_class"].items():
        for field, value in row.items():
            if not math.isclose(value, saved["per_class"][label][field], abs_tol=1e-12, rel_tol=0):
                raise ValueError(f"Saved per-class {label}/{field} differs")


def collect(run):
    """Check saved predictions and return aggregates, never source texts/IDs."""
    run = Path(run).resolve()
    if (ROOT / "outputs/baselines").resolve() not in run.parents:
        raise ValueError("Select a local run below outputs/baselines/")
    data = json.loads((run / "results.json").read_text(encoding="utf-8"))
    if data["test_loaded"] is not False or data["config"].get("manifest_version") != "v2":
        raise ValueError("Only v2 train/validation runs without test access are allowed")
    if data["mode"] not in {"smoke", "full"} or data["config"]["corpora"] != ["tsac", "madar"]:
        raise ValueError("Expected TSAC/MADAR full or smoke results")
    results = data["results"]
    if len(results) != 6:
        raise ValueError("Expected six complete TF-IDF baseline results")
    system = "tfidf_" + data["config"]["tfidf"]["analyzer"]
    if system not in NAMES:
        raise ValueError("Unsupported TF-IDF system")
    aggregate = []
    for corpus in ("tsac", "madar"):
        selected = [r for r in results if r["corpus"] == corpus]
        expected = {("majority", "raw-v0"), (system, "raw-v0"), (system, "norm-nfc-space-v1")}
        if len(selected) != 3 or {(r["system"], r["variant"]) for r in selected} != expected:
            raise ValueError("Missing or duplicate corpus/variant results")
        reference = None
        for row in selected:
            path = local_path(run, row["predictions"])
            if digest(path) != row["predictions_sha256"]:
                raise ValueError("Prediction checksum differs")
            predictions = json.loads(path.read_text(encoding="utf-8"))
            identity = [(p["id"], p["group_id"], p["truth"]) for p in predictions]
            if len({p["id"] for p in predictions}) != len(predictions):
                raise ValueError("Repeated validation ID")
            if reference is not None and identity != reference:
                raise ValueError("Systems use different validation examples")
            reference = identity
            labels = data["config"]["labels"][corpus]
            metrics = metrics_from_predictions(predictions, labels)
            check_metrics(metrics, row["metrics"])
            if metrics["n"] != row["validation_rows"]:
                raise ValueError("Validation count differs")
            slices = {}
            for writing in sorted({p["writing_system"] for p in predictions}):
                if writing not in {"AR", "LATIN", "MIXED", "UNKNOWN"}:
                    raise ValueError("Unknown writing-system category")
                subset = [p for p in predictions if p["writing_system"] == writing]
                metrics_slice = metrics_from_predictions(subset, labels)
                if writing in row["metrics"].get("by_writing_system", {}):
                    check_metrics(metrics_slice, row["metrics"]["by_writing_system"][writing])
                metrics_slice["small_or_missing_class_slice"] = (
                    len(subset) < 30 or {p["truth"] for p in subset} != set(labels))
                slices[writing] = metrics_slice
            times = {key: row[key] for key in ("fit_seconds", "predict_seconds")}
            if any(not math.isfinite(v) or v < 0 for v in times.values()):
                raise ValueError("Invalid saved duration")
            aggregate.append({"corpus": corpus, "system": row["system"], "variant": row["variant"],
                              "train_rows": row["train_rows"], "metrics": metrics,
                              "by_writing_system": slices, **times})
    return {"source_run": str(run.relative_to(ROOT.resolve())), "results_sha256": digest(run / "results.json"),
            "mode": data["mode"], "manifest_version": "v2", "learned_system": system,
            "source_created_at_utc": data["created_at_utc"], "results": aggregate}


def pick(data, corpus, system=None, variant="raw-v0"):
    system = system or data["learned_system"]
    return next(r for r in data["results"]
                if (r["corpus"], r["system"], r["variant"]) == (corpus, system, variant))


def short_name(row):
    return "Majorité" if row["system"] == "majority" else (
        NAMES[row["system"]] + (" brut" if row["variant"] == "raw-v0" else " NFC / espaces"))


def render(data, output, checkpoint_verified=False):
    """Six PNGs and an aggregate provenance JSON; never opens a GUI window."""
    output = Path(output).resolve()
    if (ROOT / "outputs").resolve() not in output.parents or output.exists():
        raise ValueError("Choose a NEW figure directory below outputs/")
    # Set caches before importing Matplotlib; no global user configuration writes.
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".tools/matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titleweight": "bold", "savefig.facecolor": "white"})
    output.mkdir(parents=True)
    images = []
    status = "FULL : validation complète" if data["mode"] == "full" else "SMOKE : petit test technique, pas le résultat complet"
    source = Path(data["source_run"]).name
    note = "Scores de validation de l'expérience source ; test officiel non évalué."

    def save(fig, filename, title, explanation):
        fig.suptitle(title, fontsize=20, x=0.045, ha="left", y=.975)
        fig.text(.045, .915, status, fontsize=11, color="#36536a")
        fig.text(.045, .065, explanation, fontsize=10, color="#36536a")
        fig.text(.045, .035, note, fontsize=9, color="#52677a")
        fig.text(.045, .013, f"Source : {source}", fontsize=8, color="#52677a")
        fig.subplots_adjust(left=.09, right=.96, top=.82, bottom=.24, wspace=.35)
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)
        images.append(filename)

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, corpus in zip(axes, ("tsac", "madar")):
        rows = [pick(data, corpus, "majority"), pick(data, corpus),
                pick(data, corpus, variant="norm-nfc-space-v1")]
        x = list(range(3))
        for offset, key, label, color in [(-.18, "macro_f1", "Macro-F1", COLORS[1]),
                                          (.18, "accuracy", "Accuracy", COLORS[2])]:
            bars = ax.bar([v + offset for v in x], [r["metrics"][key] for r in rows], .34,
                          label=label, color=color)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xticks(x, [short_name(r).replace(" NFC / espaces", "\nNFC / espaces") for r in rows], fontsize=9)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.set_ylabel("Score (0 à 1)")
        ax.set_title(f"{corpus.upper()} · n = {rows[0]['metrics']['n']}")
        ax.legend(loc="upper left", fontsize=9)
    save(fig, "01_scores.png", "Scores globaux de validation",
         "Macro-F1 : chaque classe compte autant. Accuracy : proportion de bonnes réponses. Valeurs arrondies à 3 décimales.")

    cmap = LinearSegmentedColormap.from_list("tuniscope", ["#f3f6fa", COLORS[1]])
    for number, corpus in enumerate(("tsac", "madar"), 2):
        fig, axes = plt.subplots(1, 2, figsize=(12, 7))
        rows = [pick(data, corpus), pick(data, corpus, variant="norm-nfc-space-v1")]
        maximum = max(max(line) for r in rows for line in r["metrics"]["confusion_matrix"])
        for ax, row in zip(axes, rows):
            m = row["metrics"]
            ax.imshow(m["confusion_matrix"], cmap=cmap, vmin=0, vmax=maximum)
            for i, line in enumerate(m["confusion_matrix"]):
                for j, value in enumerate(line):
                    ax.text(j, i, str(value), ha="center", va="center", fontsize=18,
                            color="white" if value > maximum * .55 else "#17324d")
            ax.set_xticks(range(len(m["labels"])), m["labels"])
            ax.set_yticks(range(len(m["labels"])), m["labels"])
            ax.set_xlabel("Catégorie prédite")
            ax.set_ylabel("Catégorie réelle")
            errors = m["n"] - sum(m["confusion_matrix"][i][i] for i in range(len(m["labels"])))
            ax.set_title(f"{short_name(row)}\n{errors} erreurs / {m['n']} · Macro-F1 {m['macro_f1']:.6f}", fontsize=12)
        save(fig, f"0{number}_matrice_{corpus}.png", f"{corpus.upper()} : où le classifieur se trompe-t-il ?",
             "Diagonale : réponses correctes. Autres cases : confusions. Comptages bruts ; échelle de couleur commune aux deux variantes.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, corpus in zip(axes, ("tsac", "madar")):
        m = pick(data, corpus)["metrics"]
        for offset, key, label, color in zip((-.24, 0, .24), ("precision", "recall", "f1-score"),
                                            ("Précision", "Rappel", "F1"), COLORS):
            bars = ax.bar([i + offset for i in range(len(m["labels"]))],
                          [m["per_class"][c][key] for c in m["labels"]], .23, label=label, color=color)
            ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        ax.set_xticks(range(len(m["labels"])),
                      [f"{c}\nn = {m['per_class'][c]['support']}" for c in m["labels"]])
        ax.set_ylim(0, 1.17)
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.set_title(corpus.upper())
        ax.legend(loc="upper center", ncol=3, fontsize=9)
    save(fig, "04_classes.png", f"Résultats par classe : {NAMES[data['learned_system']].lower()}, texte brut",
         "Précision : fiabilité des réponses annoncées. Rappel : part des vrais cas retrouvés. F1 : compromis entre les deux.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, corpus in zip(axes, ("tsac", "madar")):
        row = pick(data, corpus)
        slices = row["by_writing_system"]
        labels = [s for s in ("AR", "LATIN", "MIXED", "UNKNOWN") if s in slices]
        bars = ax.bar(range(len(labels)), [slices[s]["macro_f1"] for s in labels],
                      color=[COLORS[2] if slices[s]["small_or_missing_class_slice"] else COLORS[1] for s in labels])
        ax.bar_label(bars, fmt="%.3f", padding=4)
        ax.set_xticks(range(len(labels)), [f"{s}\nn = {slices[s]['n']}" for s in labels])
        ax.set_ylim(0, 1.1)
        ax.set_yticks([0, .2, .4, .6, .8, 1])
        ax.set_ylabel("Macro-F1")
        ax.set_title(corpus.upper())
    save(fig, "05_ecritures.png", f"Résultats par écriture : {NAMES[data['learned_system']].lower()}, texte brut",
         "Écriture ≠ langue. Orange : moins de 30 exemples ou classe absente ; interprétation prudente, surtout UNKNOWN.")

    fig, axes = plt.subplots(1, 2, figsize=(13, 7))
    for ax, key, title in zip(axes, ("fit_seconds", "predict_seconds"),
                              ("Apprentissage enregistré (s)", "Prédiction du dev enregistrée (s)")):
        rows = [pick(data, corpus, variant=variant) for corpus in ("tsac", "madar")
                for variant in ("raw-v0", "norm-nfc-space-v1")]
        bars = ax.bar(range(4), [r[key] for r in rows], color=[COLORS[1], COLORS[2]] * 2)
        ax.bar_label(bars, fmt="%.3f", padding=4, fontsize=10)
        ax.set_xticks(range(4), [r["corpus"].upper() + ("\nbrut" if r["variant"] == "raw-v0" else "\nnormalisé") for r in rows])
        ax.set_ylim(0, max(r[key] for r in rows) * 1.3 + .001)
        ax.set_title(title)
    save(fig, "06_durees.png", "Durées de l'expérience source",
         "Valeurs historiques : pas une mesure de cette relecture, ni une latence applicative. Apprentissage = TF-IDF + classifieur.")

    summary = {**data, "generated_at_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint_verified_this_export": checkpoint_verified,
               "saved_predictions_recomputed": True, "new_training": False,
               "test_evaluated": False, "matplotlib": matplotlib.__version__,
               "renderer_sha256": digest(Path(__file__)),
               "images": [{"file": name, "sha256": digest(output / name)} for name in images]}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"directory": str(output), "images": images, "mode": data["mode"],
            "source_run": data["source_run"], "checkpoint_verified": checkpoint_verified}


def export_figures(run, output, verify=False):
    data = collect(run)
    if verify:
        subprocess.run(["sh", str(ROOT / "scripts/project_uv.sh"), "run", "--locked", "--offline",
                        "python", str(ROOT / "scripts/verify_baseline_run.py"), str(Path(run).resolve())],
                       cwd=ROOT, check=True)
    return render(data, output, checkpoint_verified=verify)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    print(json.dumps(export_figures(args.run, args.output, args.verify), ensure_ascii=False, indent=2))
