"""Reload all selected BiGRU checkpoints and recalculate validation metrics."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import statistics

import torch

from baseline_metrics import classification_metrics
from bigru_data import audit, build_vocabulary, partitions
from bigru_model import BiGRUClassifier, resolve_device, set_seed
from bigru_training import add_writing_slices, evaluate, make_loader, validate_config
from character_lm_experiment import ROOT, digest, object_digest, partition_summary, read_artifact


def compare_nested(actual, expected):
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("Metric keys differ")
        for key in expected:
            compare_nested(actual[key], expected[key])
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError("Metric lengths differ")
        for a, e in zip(actual, expected):
            compare_nested(a, e)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7):
            raise ValueError("Recomputed metric differs")
    elif actual != expected:
        raise ValueError("Metric value differs")


def verify(directory, report=None, reference_run=None):
    directory = directory.resolve()
    metadata = json.loads((directory / "results.json").read_text())
    config = metadata["config"]
    validate_config(config)
    if metadata["mode"] not in {"smoke", "full"} or metadata["test_loaded"] is not False:
        raise ValueError("Expected complete validation-only run")
    if digest(ROOT / "environments/bigru/uv.lock") != metadata["environment_lock_sha256"]:
        raise ValueError("Neural environment lock changed")
    if digest(ROOT / "uv.lock") != metadata["classical_lock_sha256"]:
        raise ValueError("Classical environment lock changed")
    if metadata["packages"]["torch"] != importlib.metadata.version("torch"):
        raise ValueError("PyTorch version changed")
    if object_digest(config) != object_digest(json.loads((ROOT / "configs/bigru_v1.json").read_text())):
        raise ValueError("Configuration changed")
    seeds = [config["smoke_seed"]] if metadata["mode"] == "smoke" else config["seeds"]
    results = metadata["results"]
    if len(results) != len(seeds) * 2 or {(r["corpus"], r["seed"]) for r in results} != {(c, s) for c in config["corpora"] for s in seeds}:
        raise ValueError("Incomplete or duplicate task/seed results")
    device = resolve_device(metadata["device"])
    references = {}
    if reference_run:
        if metadata["mode"] != "full":
            raise ValueError("Compare full validation only")
        reference = json.loads((reference_run / "results.json").read_text())
        if reference["mode"] != "full" or reference["config"]["manifest_version"] != "v2" or reference["test_loaded"] is not False:
            raise ValueError("Expected full v2 TF-IDF reference")
        for c in config["corpora"]:
            candidates = [r for r in reference["results"] if r["corpus"] == c and r["system"] == "tfidf_char" and r["variant"] == "raw-v0"]
            if len(candidates) != 1:
                raise ValueError("Missing unique raw character reference")
            item = candidates[0]
            path = read_artifact(reference_run, item["predictions"])
            if digest(path) != item["predictions_sha256"]:
                raise ValueError("TF-IDF prediction checksum mismatch")
            references[c] = (item, json.loads(path.read_text()))
    for corpus in config["corpora"]:
        if digest(ROOT / f"data/manifests/v2/{corpus}.csv") != metadata["manifests_sha256"][corpus]:
            raise ValueError("Manifest changed")
        train, validation = partitions(corpus, metadata["mode"], config)
        labels = config["labels"][corpus]
        vocabulary = build_vocabulary([r["text_raw"] for r in train], config["max_vocabulary"], config["min_frequency"])
        if references:
            ref_result, ref_predictions = references[corpus]
            if [(p["id"], p["group_id"], p["truth"]) for p in ref_predictions] != [(r["id"], r["group_id"], r["label"]) for r in validation]:
                raise ValueError("TF-IDF and BiGRU evaluated on different examples")
            compare_nested({k: v for k, v in ref_result["metrics"].items() if k != "by_writing_system"},
                           classification_metrics([r["label"] for r in validation], [p["predicted"] for p in ref_predictions], labels))
        for result in [r for r in results if r["corpus"] == corpus]:
            if result["train"] != partition_summary(train) or result["validation"] != partition_summary(validation):
                raise ValueError("Partition coverage mismatch")
            if (result["vocabulary_size"] != len(vocabulary) + 2 or result["vocabulary_sha256"] != object_digest(vocabulary)
                    or result["variant"] != "raw-v0" or result["scores_calibrated"] is not False):
                raise ValueError("Vocabulary or inference protocol mismatch")
            for key in ("checkpoint", "predictions", "history"):
                if digest(read_artifact(directory, result[key])) != result[f"{key}_sha256"]:
                    raise ValueError("Saved artifact checksum mismatch")
            checkpoint = torch.load(read_artifact(directory, result["checkpoint"]), map_location="cpu", weights_only=True)
            if (checkpoint["vocabulary"] != vocabulary or checkpoint["labels"] != labels
                    or checkpoint["config"] != config or checkpoint["corpus"] != corpus
                    or checkpoint["seed"] != result["seed"] or checkpoint["best_epoch"] != result["best_epoch"]):
                raise ValueError("Checkpoint metadata mismatch")
            set_seed(result["seed"], config["cpu_threads"])
            model = BiGRUClassifier(len(vocabulary) + 2, len(labels), config).to(device)
            initial = model.classifier.weight.detach().cpu().clone()
            model.load_state_dict(checkpoint["state_dict"])
            if torch.equal(initial, model.classifier.weight.detach().cpu()) or result["weights_changed"] is not True:
                raise ValueError("No checkpoint classifier update")
            count = sum(p.numel() for p in model.parameters())
            if result["parameters"] != count or result["trainable_parameters"] != count or count > 1_000_000_000:
                raise ValueError("Parameter count mismatch")
            if read_artifact(directory, result["checkpoint"]).stat().st_size != result["checkpoint_bytes"]:
                raise ValueError("Checkpoint size mismatch")
            compare_nested(result["train_token_audit"], audit([r["text_raw"] for r in train], vocabulary, config["max_length"]))
            compare_nested(result["validation_token_audit"], audit([r["text_raw"] for r in validation], vocabulary, config["max_length"]))
            history = json.loads(read_artifact(directory, result["history"]).read_text())
            if not history or len(history) != result["epochs_completed"] or [h["epoch"] for h in history] != list(range(1, len(history) + 1)):
                raise ValueError("Incomplete history")
            best = max(history, key=lambda h: h["validation_macro_f1"])
            if best["epoch"] != result["best_epoch"]:
                raise ValueError("Wrong checkpoint selection/tie-break")
            for h in history:
                if h["train"]["rows"] != len(train) or h["train"]["nonzero_gradient_steps"] <= 0:
                    raise ValueError("Training coverage or gradients missing")
                if not all(math.isfinite(h[k]) for k in ("validation_loss", "validation_macro_f1", "validation_accuracy")):
                    raise ValueError("Non-finite validation history")
            replay = evaluate(model, make_loader(validation, vocabulary, labels, config, False, result["seed"]), labels, device)
            predictions = json.loads(read_artifact(directory, result["predictions"]).read_text())
            if len(predictions) != len(validation):
                raise ValueError("Missing validation predictions")
            for row, saved, predicted, scores in zip(validation, predictions, replay["predicted"], replay["probabilities"]):
                if (saved["id"], saved["group_id"], saved["truth"], saved["predicted"]) != (row["id"], row["group_id"], row["label"], predicted):
                    raise ValueError("Reloaded predictions/IDs/truth differ")
                compare_nested(saved["scores"], dict(zip(labels, scores)))
            # Hand confusion/F1 recalculation, independent of the sklearn contract.
            matrix = [[0 for _ in labels] for _ in labels]
            for p in predictions:
                matrix[labels.index(p["truth"])][labels.index(p["predicted"])]+=1
            f1s = []
            for i in range(len(labels)):
                denominator = sum(matrix[i]) + sum(row[i] for row in matrix)
                f1s.append(2 * matrix[i][i] / denominator if denominator else 0)
            compare_nested(result["metrics"]["confusion_matrix"], matrix)
            compare_nested(result["metrics"]["macro_f1"], math.fsum(f1s) / len(labels))
            compare_nested(result["metrics"], add_writing_slices(replay["metrics"], validation, replay["predicted"], labels))
            compare_nested(result["selected_validation_loss"], replay["loss"])
            compare_nested(best["validation_macro_f1"], replay["metrics"]["macro_f1"])
    if report:
        write_report(directory, metadata, report, references)
    print(f"Verified {metadata['mode']}: {len(results)} BiGRU checkpoints, train-only vocabularies, full validation replay and metrics", flush=True)
    return metadata


def write_report(directory, metadata, report, references):
    if report.exists() or (ROOT / "reports").resolve() not in report.resolve().parents:
        raise ValueError("Choose a fresh report directory under reports/")
    if metadata["mode"] != "full":
        raise ValueError("Summary with three seeds requires full")
    report.mkdir(parents=True)
    table = ["| Corpus | Graine | Époque retenue | Macro-F1 | Accuracy | Paramètres |",
             "| --- | ---: | ---: | ---: | ---: | ---: |"]
    aggregate = {}
    for r in metadata["results"]:
        table.append(f"| {r['corpus'].upper()} | {r['seed']} | {r['best_epoch']} | {r['metrics']['macro_f1']:.6f} | "
                     f"{r['metrics']['accuracy']:.6f} | {r['parameters']} |")
    for c in metadata["config"]["corpora"]:
        values = [r["metrics"]["macro_f1"] for r in metadata["results"] if r["corpus"] == c]
        aggregate[c] = {"seeds": metadata["config"]["seeds"], "macro_f1_values": values,
                        "mean_macro_f1": statistics.mean(values), "sample_std_macro_f1": statistics.stdev(values)}
        if references:
            aggregate[c]["tfidf_character_raw_macro_f1"] = references[c][0]["metrics"]["macro_f1"]
    summary = {"run": directory.name, "source_results_sha256": digest(directory / "results.json"),
               "mode": "full", "split": "validation", "test_loaded": False,
               "device": metadata["device"], "packages": metadata["packages"], "config": metadata["config"],
               "manifests_sha256": metadata["manifests_sha256"], "aggregate": aggregate,
               "results": [{k: v for k, v in r.items() if k not in {"checkpoint", "predictions", "history"}} for r in metadata["results"]]}
    paragraphs = ["# BiGRU — résultats de validation v2", "",
                  f"Run `{directory.name}`, texte raw-v0, appareil `{metadata['device']}`. Six modèles, trois graines par tâche.", "",
                  *table, "", "## Moyenne et variabilité", ""]
    for c, a in aggregate.items():
        line = f"- {c.upper()} : macro-F1 moyen **{a['mean_macro_f1']:.6f} ± {a['sample_std_macro_f1']:.6f}**, écart-type d'échantillon sur trois graines."
        if references:
            line += f" Référence TF-IDF caractères brut sur les mêmes exemples : {a['tfidf_character_raw_macro_f1']:.6f}."
        paragraphs.append(line)
    paragraphs.extend(["", "Sélection d'époque sur validation, égalités au premier checkpoint. Aucun choix de meilleure graine. "
                       "Les scores ne sont pas ceux du test officiel ; ils ne prouvent pas une généralisation hors domaine. "
                       "La comparaison des systèmes change à la fois représentation et classifieur.", "",
                       "Vocabulaire et audit reconstruits depuis le train ; six checkpoints rechargés, prédictions rejouées et métriques recalculées. "
                       "Les courbes et artefacts individuels restent dans le run privé. L04/L07/L12 restent partiels."])
    (report / "README.md").write_text("\n".join(paragraphs) + "\n", encoding="utf-8")
    (report / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--reference-run", type=Path)
    args = parser.parse_args()
    verify(args.run, args.report, args.reference_run)
