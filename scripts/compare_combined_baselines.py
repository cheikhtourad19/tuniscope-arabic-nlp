"""Generate an aligned, aggregate-only three-representation validation report."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from baseline_models import vectorizer_branches
from compare_baseline_runs import compatible, paired_summary
from run_baselines import ROOT, digest
from verify_baseline_run import verify


def compatible_combination(character, word, combined):
    compatible(character, word)
    if combined["mode"] != "full" or combined["test_loaded"]:
        raise ValueError("Full, validation-only combined run required")
    for key in ("manifests_sha256", "uv_lock_sha256", "packages", "device", "python"):
        if character[key] != combined[key]:
            raise ValueError(f"Different combined run condition: {key}")
    for key in ("manifest_version", "corpora", "labels", "variants", "classifier", "threads", "seed"):
        if character["config"].get(key) != combined["config"].get(key):
            raise ValueError(f"Different combined configuration: {key}")
    features = combined["config"]["tfidf"]
    if (features["analyzer"] != "word_char" or features["global_norm"] != "l2" or
            features["block_weights"] != {"word": 1.0, "char": 1.0}):
        raise ValueError("Unexpected combined feature rule")
    if features["char"] != character["config"]["tfidf"] or features["word"] != word["config"]["tfidf"]:
        raise ValueError("Combined branches differ from standalone configurations")


def paired_combination(reference, combined, labels):
    # Preserve the existing word/character comparison interface, but use neutral
    # names in this report, where the challenger really is the combined model.
    return {key.replace("character", "reference").replace("word", "combined"): value
            for key, value in paired_summary(reference, combined, labels).items()}


def check_branch_identity(standalone, combined, branch):
    a = vectorizer_branches(standalone)[branch]
    b = vectorizer_branches(combined)[branch]
    if a.vocabulary_ != b.vocabulary_ or not np.array_equal(a.idf_, b.idf_):
        raise ValueError("Learned branch vocabulary or IDF differs from standalone reference")


def compare(char_dir, word_dir, combined_dir, report_dir):
    directories = {"char": char_dir, "word": word_dir, "combined": combined_dir}
    runs = {}
    for name, directory in directories.items():
        verify(directory)
        runs[name] = json.loads((directory / "results.json").read_text())
    compatible_combination(runs["char"], runs["word"], runs["combined"])
    report_dir = report_dir.resolve()
    if (ROOT / "reports").resolve() not in report_dir.parents or report_dir.exists():
        raise ValueError("Choose a new directory under reports/")
    comparisons, scores = [], []
    for corpus in runs["combined"]["config"]["corpora"]:
        for variant in runs["combined"]["config"]["variants"]:
            models, predictions, results = {}, {}, {}
            for name in directories:
                system = "tfidf_word_char" if name == "combined" else f"tfidf_{name}"
                result = next(r for r in runs[name]["results"] if
                              (r["corpus"], r["variant"], r["system"]) == (corpus, variant, system))
                results[name] = result
                predictions[name] = json.loads((directories[name] / result["predictions"]).read_text())
                models[name] = joblib.load(directories[name] / result["checkpoint"])
            for reference in ("char", "word"):
                check_branch_identity(models[reference], models["combined"], reference)
                if (results[reference]["train_rows"], results[reference]["train_labels"]) != (
                        results["combined"]["train_rows"], results["combined"]["train_labels"]):
                    raise ValueError("Different training counts")
                comparisons.append({"corpus": corpus, "variant": variant, "reference_system": reference,
                                    **paired_combination(predictions[reference], predictions["combined"],
                                                         runs["combined"]["config"]["labels"][corpus])})
            scores.append({"corpus": corpus, "variant": variant,
                           "macro_f1": {name: result["metrics"]["macro_f1"] for name, result in results.items()},
                           "combined_errors": sum(r["predicted"] != r["truth"] for r in predictions["combined"]),
                           "combined_zero_feature_train_rows": results["combined"]["zero_feature_train_rows"],
                           "combined_zero_feature_validation_rows": results["combined"]["zero_feature_validation_rows"]})
    data = {
        "manifest_version": runs["combined"]["config"]["manifest_version"], "test_loaded": False,
        "branch_vocabularies_and_idf_match_references": True,
        "scores": scores, "paired_comparisons": comparisons,
        "run_results_sha256": {name: digest(directory / "results.json") for name, directory in directories.items()},
        "configs": {name: run["config"] for name, run in runs.items()},
        "manifests_sha256": runs["combined"]["manifests_sha256"],
        "uv_lock_sha256": runs["combined"]["uv_lock_sha256"],
        "comparison_script_sha256": digest(Path(__file__)),
        "paired_comparison_script_sha256": digest(ROOT / "scripts/compare_baseline_runs.py"),
    }
    lines = ["# TuniScope — Mots, caractères et combinaison", "",
             "**Livrables :** L04, L07 et L12, contributions partielles.", "",
             "Validation v2 commune : mêmes IDs, groupes, labels et configurations de classifieur. "
             "Les trois runs et leurs checkpoints sont vérifiés. Les vocabulaires et IDF des deux branches "
             "combinées sont identiques à ceux des références seules. Aucun test officiel chargé.", "",
             "| Corpus | Variante | Macro-F1 mots | Macro-F1 caractères | Macro-F1 combinaison | Erreurs combinaison |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in scores:
        m = row["macro_f1"]
        lines.append(f"| {row['corpus']} | {row['variant']} | {m['word']:.6f} | {m['char']:.6f} | "
                     f"{m['combined']:.6f} | {row['combined_errors']} |")
    lines += ["", "## Écarts appariés", "",
              "| Corpus | Variante | Référence | Δ macro-F1 combinaison − référence | Référence seule correcte | Combinaison seule correcte | Deux systèmes en erreur |",
              "| --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for row in comparisons:
        lines.append(f"| {row['corpus']} | {row['variant']} | {row['reference_system']} | "
                     f"{row['delta_combined_minus_reference']:+.6f} | {row['reference_correct_combined_wrong']} | "
                     f"{row['combined_correct_reference_wrong']} | {row['both_wrong']} |")
    lines += ["", "## Portée", "",
              "La combinaison est une nouvelle régression logistique sur les colonnes juxtaposées, "
              "pas un vote ou une sélection a posteriori. Blocs de poids 1:1 puis normalisation L2 globale, "
              "C=1. L'échelle de chaque bloc et la régularisation effective changent par rapport aux "
              "références seules ; ce protocole compare des systèmes, pas un effet isolé de feature.", "",
              "Les lignes sans caractéristique restent évaluées. La recherche de familles est heuristique "
              "et les sous-groupes très petits restent fragiles. La validation n'établit ni significativité "
              "ni supériorité générale ; aucune recherche de poids ou de C n'a suivi les scores.", "",
              "Les scores sont recalculés depuis les prédictions. `comparison.json` contient les comptes "
              "et empreintes, sans commentaires ni identifiants privés. Les règles, analyses linguistiques "
              "et humaines d'erreurs, autres baselines et évaluation finale restent à réaliser."]
    report_dir.mkdir(parents=True)
    (report_dir / "comparison.json").write_text(json.dumps(data, ensure_ascii=False, sort_keys=True,
                                                          indent=2) + "\n", encoding="utf-8")
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"scores": scores, "paired_comparisons": comparisons}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("character_run", type=Path)
    parser.add_argument("word_run", type=Path)
    parser.add_argument("combined_run", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(args.character_run, args.word_run, args.combined_run, args.report_dir)
