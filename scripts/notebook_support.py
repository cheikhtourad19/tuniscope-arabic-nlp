"""Small notebook bridge to existing, locked project scripts.

No data acquisition or neural model download. Data/model imports run in project
uv; the optional figure export uses Matplotlib in the notebook kernel only.
"""

from collections import Counter
import json
import os
from pathlib import Path
import subprocess
import uuid


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = {
    "characters": "baselines_grouped_v2.json",
    "words": "baselines_words_v2.json",
    "combined": "baselines_combined_v2.json",
}


def run_project(script, *arguments, capture=False):
    """Run a repository script in the existing offline Python 3.12 environment."""
    path = ROOT / "scripts" / script
    if path.parent != ROOT / "scripts" or not path.is_file():
        raise ValueError("Expected an existing script directly inside scripts/")
    environment = os.environ.copy()
    # Always target .venv, not the active Jupyter kernel environment.
    environment.pop("VIRTUAL_ENV", None)
    try:
        completed = subprocess.run(
            ["sh", str(ROOT / "scripts/project_uv.sh"), "run", "--locked", "--offline",
             "python", str(path), *(str(arg) for arg in arguments)],
            cwd=ROOT, check=True, text=True, capture_output=capture, env=environment,
        )
    except subprocess.CalledProcessError as error:
        if capture:
            print(error.stdout or "", end="")
            print(error.stderr or "", end="")
        raise
    return completed.stdout if capture else None


def synthetic_records():
    """Four invented comments, not a sample or evaluation of either corpus."""
    result = []
    for index, (text, label, split) in enumerate([
        ("  behi\t👍 ", "POS", "train"), ("moch behi", "NEG", "train"),
        ("service ممتاز", "POS", "validation"), ("service ضعيف", "NEG", "validation"),
    ]):
        result.append({
            "id": f"fiction-{index}", "text_raw": text, "text_norm": text,
            "task": "sentiment", "label": label, "script": "UNKNOWN",
            "languages": ["und"], "source": "invented", "source_id": str(index),
            "source_file": "not-a-corpus", "source_line": index + 1,
            "group_id": f"fiction-group-{index}", "split": split,
            "official_split": split, "annotation_status": "synthetic",
            "license": "invented demonstration", "normalization_version": "raw-v0",
        })
    return result


def summarize_records(records):
    """Return aggregates only; reject held-out rows before any transformation."""
    from audit_text_composition import writing_system
    from load_working_data import as_pandas
    from normalize_text import normalize_records

    if not records or any(row["split"] not in {"train", "validation"} for row in records):
        raise ValueError("Only nonempty train/validation selections are allowed")
    before = [(row["text_raw"], row["text_norm"], row["normalization_version"])
              for row in records]
    normalized = normalize_records(records)
    frame = as_pandas(normalized)
    groups = {split: {row["group_id"] for row in records if row["split"] == split}
              for split in ("train", "validation")}
    shared = groups["train"] & groups["validation"]
    if shared:
        raise ValueError("Train/validation groups must be disjoint")
    preserved = before == [(row["text_raw"], row["text_norm"], row["normalization_version"])
                           for row in records]
    if not preserved or any(a["text_raw"] != b["text_raw"]
                            for a, b in zip(records, normalized)):
        raise ValueError("Raw inputs changed")
    partitions = {}
    for split in ("train", "validation"):
        rows = [row for row in normalized if row["split"] == split]
        partitions[split] = {
            "rows": len(rows), "groups": len(groups[split]),
            "labels": dict(sorted(Counter(row["label"] for row in rows).items())),
            "writing_system": dict(sorted(Counter(writing_system(row["text_raw"])
                                                   for row in rows).items())),
            "normalization_changed_rows": sum(row["text_raw"] != row["text_norm"] for row in rows),
        }
    return {"dataframe_shape": list(frame.shape), "columns": frame.columns.tolist(),
            "partitions": partitions, "shared_train_validation_groups": len(shared),
            "raw_input_preserved": preserved, "selected_test_rows": 0}


def data_overview(mode):
    if mode not in {"synthetic", "real"}:
        raise ValueError("Mode must be synthetic or real")
    if mode == "synthetic":
        corpora = {"invented": summarize_records(synthetic_records())}
    else:
        from load_working_data import load_records
        corpora = {corpus: summarize_records(load_records(
            corpus, splits=("train", "validation"), manifest_version="v2",
        )) for corpus in ("tsac", "madar")}
    return {"mode": mode, "manifest_version": "v2" if mode == "real" else None,
            "corpus_read": mode == "real", "writing_system_is_language_id": False,
            "processed_text_written": False, "corpora": corpora}


def train_baselines(family, mode, *, concise=False):
    """Explicit opt-in; smoke before full, new directories and verification."""
    if family not in CONFIGS or mode not in {"smoke", "full"}:
        raise ValueError("Choose an allowed TF-IDF family and smoke/full mode")
    config = ROOT / "configs" / CONFIGS[family]
    if json.loads(config.read_text(encoding="utf-8")).get("manifest_version") != "v2":
        raise ValueError("Notebook comparisons require explicit v2 manifests")
    parent = ROOT / "outputs/baselines/notebooks"
    parent.mkdir(parents=True, exist_ok=True)
    identifier = uuid.uuid4().hex
    runs = []
    for stage in (["smoke", "full"] if mode == "full" else ["smoke"]):
        target = parent / f"{identifier}_{family}_{stage}"
        if concise:
            print("Contrôle préalable (smoke, 100 exemples par partition)…" if stage == "smoke"
                  else "Entraînement complet (full, tout le train ; évaluation sur tout le dev)…", flush=True)
        run_project("run_baselines.py", "--config", config, "--mode", stage,
                    "--output", target, capture=concise)
        run_project("verify_baseline_run.py", target, capture=concise)
        result = json.loads((target / "results.json").read_text(encoding="utf-8"))
        if result["mode"] != stage or result["test_loaded"] is not False:
            raise ValueError("Unexpected result mode or test access")
        runs.append({"directory": str(target.relative_to(ROOT)), "mode": stage,
                     "validation": [
                         {key: row[key] for key in ("corpus", "system", "variant",
                                                   "train_rows", "validation_rows")}
                         | {key: row["metrics"][key] for key in ("macro_f1", "accuracy")}
                         for row in result["results"]]})
        if concise:
            print("Contrôle préalable réussi." if stage == "smoke" else "Entraînement complet et rechargement vérifiés.", flush=True)
    return runs


def select_training_run(runs, requested_mode):
    """Select the requested final stage, never silently fall back to smoke."""
    expected = ["smoke", "full"] if requested_mode == "full" else ["smoke"]
    if requested_mode not in {"smoke", "full"} or [r["mode"] for r in runs] != expected:
        raise ValueError("Requested training stage is missing or unexpected; no fallback is allowed")
    selected = runs[-1]
    if len(selected["validation"]) != 6:
        raise ValueError("Expected six verified results")
    return selected


def training_summary(run):
    """Readable table instead of the smoke/full JSON array."""
    lines = ["ENTRAÎNEMENT COMPLET — FULL" if run["mode"] == "full" else "SMOKE — PETIT CONTRÔLE TECHNIQUE",
             "Scores sur validation (dev), pas sur le test officiel.",
             f"{'Corpus':<8}{'Système':<19}{'Texte':<14}{'Train':>8}{'Dev':>8}{'Accuracy':>12}{'Macro-F1':>12}"]
    for row in run["validation"]:
        variant = "brut" if row["variant"] == "raw-v0" else "normalisé"
        lines.append(f"{row['corpus'].upper():<8}{row['system']:<19}{variant:<14}"
                     f"{row['train_rows']:>8}{row['validation_rows']:>8}"
                     f"{row['accuracy']:>12.6f}{row['macro_f1']:>12.6f}")
    lines += ["", f"Modèles et résultats : {ROOT / run['directory']}"]
    return "\n".join(lines)


def export_training_figures(run):
    """Export this selected training run, not a historical/latest directory."""
    from plot_validation_results import export_figures
    source = ROOT / run["directory"]
    saved = json.loads((source / "results.json").read_text(encoding="utf-8"))
    if saved["mode"] != run["mode"]:
        raise ValueError("Selected mode differs from saved training result")
    output = source / "figures"
    if output.exists():
        output = source / f"figures_{uuid.uuid4().hex[:12]}"
    return export_figures(source, output, verify=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("synthetic", "real"), default="synthetic")
    args = parser.parse_args()
    print(json.dumps(data_overview(args.mode), ensure_ascii=False, indent=2, sort_keys=True))
