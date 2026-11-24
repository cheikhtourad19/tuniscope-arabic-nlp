"""Measure the TSAC data budget and CAMeLBERT abstention on validation v2."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
from hashlib import sha256
from itertools import accumulate
import json
from math import isclose, isfinite
from pathlib import Path
from statistics import mean, stdev
from time import perf_counter
import warnings

from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from audit_structure import ROOT
from baseline_metrics import classification_metrics
from baseline_models import build_tfidf_model
from compare_encoder_errors import checked_predictions, compare_predictions
from load_working_data import load_records, manifest_directory
from run_baselines import digest


PROTOCOL = ROOT / "configs/gold_validation_v1.json"


def read_pinned(path, expected_hash):
    if digest(path) != expected_hash:
        raise ValueError(f"Unexpected file hash: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def nested_train_subsets(train, seed, percentages):
    """Keep whole groups and approximate the row target in each label stratum."""
    groups = defaultdict(list)
    for row in train:
        groups[row["group_id"]].append(row)
    strata = defaultdict(list)
    for group, rows in groups.items():
        strata[tuple(sorted({row["label"] for row in rows}))].append(group)
    selected = {pct: set() for pct in percentages}
    for members in strata.values():
        ordered = sorted(members, key=lambda group: (sha256(f"{seed}:{group}".encode()).digest(), group))
        cumulative = [0, *accumulate(len(groups[group]) for group in ordered)]
        for pct in percentages:
            target = cumulative[-1] * pct / 100
            right = bisect_left(cumulative, target)
            left = max(0, right - 1)
            stop = min((left, right), key=lambda i: (abs(cumulative[i] - target), i))
            selected[pct].update(ordered[:stop])
    return {pct: [row for row in train if row["group_id"] in selected[pct]] for pct in percentages}


def validation_reference(baseline, run, corpus):
    result, = [r for r in baseline["results"] if
               (r["corpus"], r["system"], r["variant"]) == (corpus, "tfidf_char", "raw-v0")]
    rows = checked_predictions(run, result["predictions"], result["predictions_sha256"])
    return result, rows


def alignment_rows(rows):
    return [{"id": row["id"], "group_id": row["group_id"],
             "truth": row["label"], "predicted": row["label"]} for row in rows]


def run_budget(protocol, config, reference, validation_predictions):
    start = perf_counter()
    records = load_records("tsac", splits=("train", "validation"), manifest_version="v2")
    train = [row for row in records if row["split"] == "train"]
    validation = [row for row in records if row["split"] == "validation"]
    if len(train) != reference["train_rows"] or len(validation) != reference["validation_rows"]:
        raise ValueError("Training or validation size differs from the reference")
    compare_predictions(alignment_rows(validation), validation_predictions)
    v_groups = {r["group_id"] for r in validation}
    if v_groups & {r["group_id"] for r in train}:
        raise ValueError("Train and validation groups overlap")
    labels = config["labels"]["tsac"]
    truth = [row["label"] for row in validation]
    texts = [row["text_raw"] for row in validation]
    baseline_predictions = [row["predicted"] for row in validation_predictions]
    results = []
    with threadpool_limits(limits=config["threads"]), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for seed in protocol["seeds"]:
            previous_ids = set()
            for pct, rows in nested_train_subsets(train, seed, protocol["budget_percentages"]).items():
                ids = [row["id"] for row in rows]
                if not previous_ids < set(ids):
                    raise ValueError("Expected strictly nested training subsets")
                previous_ids = set(ids)
                targets = [row["label"] for row in rows]
                if set(targets) != set(labels):
                    raise ValueError("Training subset is missing a label")
                model = build_tfidf_model(config)
                fit_start = perf_counter()
                model.fit([row["text_raw"] for row in rows], targets)
                fit_seconds = perf_counter() - fit_start
                predicted = model.predict(texts).tolist()
                if pct == 100 and (rows != train or predicted != baseline_predictions):
                    raise ValueError("Full budget failed to reproduce the saved validation predictions")
                score = classification_metrics(truth, predicted, labels)
                results.append({"seed": seed, "target_percent": pct, "train_rows": len(rows),
                                "train_groups": len({row["group_id"] for row in rows}),
                                "train_labels": dict(Counter(targets)),
                                "ids_sha256": sha256("\n".join(ids).encode()).hexdigest(),
                                "macro_f1": score["macro_f1"], "accuracy": score["accuracy"],
                                "fit_seconds": fit_seconds})
    summary = []
    for pct in protocol["budget_percentages"]:
        runs = [row for row in results if row["target_percent"] == pct]
        scores = [row["macro_f1"] for row in runs]
        counts = [row["train_rows"] for row in runs]
        summary.append({"percent": pct, "mean_macro_f1": mean(scores), "sd_macro_f1": stdev(scores),
                        "min_train_rows": min(counts), "max_train_rows": max(counts)})
    return {"validation_rows": len(validation), "runs": results, "summary": summary,
            "validation_prediction_source_sha256": reference["predictions_sha256"],
            "full_budget_matches_reference": True, "seconds": perf_counter() - start}


def confidence_rows(rows, labels):
    result = []
    for row in rows:
        scores = dict(zip(labels, row["probabilities"], strict=True))
        if not all(isfinite(v) and 0 <= v <= 1 for v in scores.values()) or not isclose(
            sum(scores.values()), 1, abs_tol=1e-5
        ):
            raise ValueError("Invalid saved probabilities")
        if scores[row["predicted"]] != max(scores.values()):
            raise ValueError("Predicted label disagrees with probabilities")
        result.append((max(scores.values()), row["predicted"] != row["truth"]))
    return result


def abstention_metrics(rows, threshold):
    accepted = [wrong for score, wrong in rows if score >= threshold]
    return {"total": len(rows), "accepted": len(accepted), "errors": sum(accepted),
            "coverage": len(accepted) / len(rows),
            "accepted_error_rate": sum(accepted) / len(accepted) if accepted else None}


def evaluate_encoder(protocol, baseline, manifests):
    source = ROOT / protocol["encoder_run"]
    campaign = read_pinned(source / "confirmation.json", protocol["confirmation_sha256"])
    if campaign["mode"] != "full" or campaign["test_loaded"] is not False:
        raise ValueError("Expected validation-only FULL confirmation")
    output = {}
    for entry in campaign["entries"]:
        corpus = entry["corpus"]
        stage_info = entry["full"]
        folder = source / stage_info["directory"]
        stage = read_pinned(folder / "results.json", stage_info["results_sha256"])
        if (stage["mode"], stage["test_loaded"], stage["config"]["variant"],
            stage["config"]["manifest_version"], stage["manifests_sha256"][corpus]) != (
            "full", False, "raw-v0", "v2", manifests[corpus]
        ):
            raise ValueError(f"Unexpected validation run for {corpus}")
        if sorted(row["seed"] for row in stage["results"]) != protocol["seeds"]:
            raise ValueError("Unexpected confirmation seeds")
        labels = stage["config"]["labels"][corpus]
        ref, baseline_rows = validation_reference(baseline, ROOT / protocol["baseline_run"], corpus)
        if len(baseline_rows) != ref["validation_rows"]:
            raise ValueError("Wrong number of validation rows")
        by_seed = []
        for row in sorted(stage["results"], key=lambda r: r["seed"]):
            name = f'{row["checkpoint"]}/predictions.json'
            predictions = checked_predictions(folder, name, stage["artifacts"][name])
            comparison = compare_predictions(baseline_rows, predictions)
            scores = confidence_rows(predictions, labels)
            if sum(error for _, error in scores) != comparison["encoder_errors"]:
                raise ValueError("Inconsistent validation error count")
            by_seed.append({"seed": row["seed"], "prediction_sha256": stage["artifacts"][name],
                            "comparison": comparison, "thresholds": [
                                {"threshold": t, **abstention_metrics(scores, t)}
                                for t in protocol["abstention_thresholds"]]})
        summary = []
        for i, threshold in enumerate(protocol["abstention_thresholds"]):
            metrics = [seed["thresholds"][i] for seed in by_seed]
            total, accepted, errors = (sum(r[key] for r in metrics)
                                       for key in ("total", "accepted", "errors"))
            summary.append({"threshold": threshold, "coverage": accepted / total,
                            "min_seed_coverage": min(r["coverage"] for r in metrics),
                            "accepted_error_rate": errors / accepted if accepted else None,
                            "accepted": accepted, "errors": errors, "total": total})
        chosen = max(row["threshold"] for row in summary
                     if row["min_seed_coverage"] >= protocol["minimum_coverage"])
        output[corpus] = {"validation_rows": len(baseline_rows),
                          "baseline_predictions_sha256": ref["predictions_sha256"],
                          "threshold_selected_on_validation": chosen, "summary": summary,
                          "by_seed": by_seed}
    if set(output) != {"tsac", "madar"}:
        raise ValueError("Both tasks are required")
    return output


def format_report(data):
    def pc(value):
        return "—" if value is None else f"{100 * value:.2f} %".replace(".", ",")

    lines = ["# TuniScope — Expériences Gold sur la validation", "",
             "Validation v2. Les résultats de classification sur le test officiel restent à produire.", "",
             "## Quantité de données et performance de TF-IDF caractères", "",
             "Même validation TSAC (1 774 avis) et mêmes hyperparamètres à chaque essai. "
             "Le vocabulaire TF-IDF et la régression logistique sont ajustés à nouveau sur chaque sous-ensemble. "
             "Les graines fixent l’ordre des groupes. Les tailles sont imbriquées dans chaque graine. "
             "Les groupes restent entiers, ce qui explique les petits écarts d’effectifs.", "",
             "| Budget cible | Avis d’entraînement min–max | Macro-F1 moyen | Écart-type |",
             "| ---: | ---: | ---: | ---: |"]
    for row in data["budget"]["summary"]:
        lines.append(f'| {row["percent"]} % | {row["min_train_rows"]}–{row["max_train_rows"]} '
                     f'| {pc(row["mean_macro_f1"])} | {pc(row["sd_macro_f1"])} |')
    lines += ["", "L’écart-type est calculé entre trois graines. À 100 %, les trois essais reprennent les mêmes "
              "données dans le même ordre et reproduisent toutes les prédictions de la référence. "
              "Le score à 100 % est donc un point de référence déterministe.", "",
              "## Abstention exploratoire de CAMeLBERT", "",
              "Le modèle accepte un texte si son score le plus élevé atteint le seuil. La couverture est la "
              "part des textes acceptés. Le taux d’erreur est calculé uniquement parmi ces textes. Nous retenons "
              "pour chaque tâche le seuil de la grille le plus haut qui accepte au moins 90 % des textes pour "
              "chacune des trois graines de validation.", "",
              "| Corpus | Seuil | Couverture | Erreurs parmi les textes acceptés |",
              "| --- | ---: | ---: | ---: |"]
    for corpus, result in data["encoder"].items():
        for row in result["summary"]:
            suffix = " (retenu)" if row["threshold"] == result["threshold_selected_on_validation"] else ""
            lines.append(f'| {corpus.upper()} | {row["threshold"]:.2f}{suffix} | {pc(row["coverage"])} '
                         f'| {pc(row["accepted_error_rate"])} ({row["errors"]}/{row["accepted"]}) |')
    lines += ["", "Les comptes regroupent trois séries de prédictions sur les mêmes textes. Le seuil et les "
              "performances sont calculés sur la même validation. Les scores sont les probabilités brutes du "
              "modèle et leur calibration reste à étudier.", "",
              "## Comparaison appariée des erreurs", "",
              "| Corpus | Graine | Erreurs TF-IDF | Erreurs CAMeLBERT | Corrigées par CAMeLBERT "
              "| Introduites par CAMeLBERT | Communes |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for corpus, result in data["encoder"].items():
        for entry in result["by_seed"]:
            comp = entry["comparison"]
            lines.append(f'| {corpus.upper()} | {entry["seed"]} | {comp["baseline_errors"]} '
                         f'| {comp["encoder_errors"]} | {comp["corrected"]} | {comp["introduced"]} '
                         f'| {comp["both_wrong"]} |')
    lines += ["", "Une erreur corrigée est un texte que TF-IDF classe mal et que CAMeLBERT classe correctement. "
              "Une erreur introduite correspond à la situation inverse. Ces comptes utilisent les labels de "
              "référence et les prédictions sur les mêmes textes.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    target = args.output.resolve()
    if target.exists() or not any(parent in target.parents for parent in (ROOT / "reports", ROOT / "outputs")):
        raise ValueError("Choose a new directory in reports/ or outputs/")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    config = read_pinned(ROOT / protocol["baseline_config"], protocol["baseline_config_sha256"])
    baseline_run = ROOT / protocol["baseline_run"]
    baseline = read_pinned(baseline_run / "results.json", protocol["baseline_results_sha256"])
    manifests = {corpus: digest(manifest_directory("v2") / f"{corpus}.csv")
                 for corpus in ("tsac", "madar")}
    if (baseline["mode"], baseline["test_loaded"], baseline["manifests_sha256"], baseline["config_sha256"]) != (
        "full", False, manifests, protocol["baseline_config_sha256"]
    ):
        raise ValueError("Baseline reference does not match the protocol")
    reference, rows = validation_reference(baseline, baseline_run, "tsac")
    started = perf_counter()
    data = {"protocol": protocol, "protocol_sha256": digest(PROTOCOL), "test_loaded": False,
            "manifests_sha256": manifests, "script_sha256": digest(Path(__file__)),
            "uv_lock_sha256": digest(ROOT / "uv.lock"),
            "budget": run_budget(protocol, config, reference, rows),
            "encoder": evaluate_encoder(protocol, baseline, manifests)}
    data["seconds"] = perf_counter() - started
    target.mkdir(parents=True)
    (target / "summary.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    (target / "README.md").write_text(format_report(data), encoding="utf-8")
    print(f'{len(data["budget"]["runs"])} entraînements et 6 jeux de prédictions contrôlés '
          f'en {data["seconds"]:.2f} s. Résultats : {target}')


if __name__ == "__main__":
    main()
