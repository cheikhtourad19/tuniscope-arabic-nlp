"""Compare verified word/character validation runs, never unaligned datasets."""
import argparse
import json
from pathlib import Path

from baseline_metrics import classification_metrics
from run_baselines import ROOT, digest
from verify_baseline_run import verify


def paired_summary(character, words, labels):
    """Compute aggregate paired evidence, rejecting any missing or shifted row."""
    if not character or len(character) != len(words):
        raise ValueError("Different or empty validation sizes")
    if len({r["id"] for r in character}) != len(character):
        raise ValueError("Repeated validation ID")
    for left, right in zip(character, words):
        if any(left[key] != right[key] for key in ("id", "group_id", "truth", "writing_system")):
            raise ValueError("Validation references, groups or truth are not aligned")
    truth = [r["truth"] for r in character]
    a = classification_metrics(truth, [r["predicted"] for r in character], labels)
    b = classification_metrics(truth, [r["predicted"] for r in words], labels)
    return {
        "validation_rows": len(truth), "character_macro_f1": a["macro_f1"],
        "word_macro_f1": b["macro_f1"], "delta_word_minus_character": b["macro_f1"] - a["macro_f1"],
        "character_errors": sum(r["predicted"] != r["truth"] for r in character),
        "word_errors": sum(r["predicted"] != r["truth"] for r in words),
        "different_predictions": sum(a["predicted"] != b["predicted"] for a, b in zip(character, words)),
        "character_correct_word_wrong": sum(a["predicted"] == a["truth"] and b["predicted"] != b["truth"]
                                             for a, b in zip(character, words)),
        "word_correct_character_wrong": sum(b["predicted"] == b["truth"] and a["predicted"] != a["truth"]
                                             for a, b in zip(character, words)),
        "both_wrong": sum(a["predicted"] != a["truth"] and b["predicted"] != b["truth"]
                          for a, b in zip(character, words)),
    }


def compatible(left, right):
    for data in (left, right):
        if data["mode"] != "full" or data["test_loaded"]:
            raise ValueError("Full validation-only runs are required")
    for key in ("manifests_sha256", "uv_lock_sha256", "packages", "device", "python"):
        if left[key] != right[key]:
            raise ValueError(f"Different run condition: {key}")
    for key in ("manifest_version", "corpora", "labels", "variants", "classifier", "threads", "seed"):
        if left["config"].get(key) != right["config"].get(key):
            raise ValueError(f"Different model/data condition: {key}")
    for key in ("lowercase", "min_df", "sublinear_tf", "norm", "dtype"):
        if left["config"]["tfidf"][key] != right["config"]["tfidf"][key]:
            raise ValueError(f"Different TF-IDF condition: {key}")
    if (left["config"]["tfidf"]["analyzer"], right["config"]["tfidf"]["analyzer"]) != ("char", "word"):
        raise ValueError("Expected character then word run")


def compare(char_dir, word_dir, report_dir):
    verify(char_dir)
    verify(word_dir)
    left = json.loads((char_dir / "results.json").read_text())
    right = json.loads((word_dir / "results.json").read_text())
    compatible(left, right)
    report_dir = report_dir.resolve()
    if (ROOT / "reports").resolve() not in report_dir.parents or report_dir.exists():
        raise ValueError("Choose a new report directory under reports/")
    comparisons = []
    for corpus in left["config"]["corpora"]:
        for variant in left["config"]["variants"]:
            a = next(r for r in left["results"] if (r["corpus"], r["system"], r["variant"]) ==
                     (corpus, "tfidf_char", variant))
            b = next(r for r in right["results"] if (r["corpus"], r["system"], r["variant"]) ==
                     (corpus, "tfidf_word", variant))
            if (a["train_rows"], a["train_labels"]) != (b["train_rows"], b["train_labels"]):
                raise ValueError("Different training label counts")
            paired = paired_summary(json.loads((char_dir / a["predictions"]).read_text()),
                                    json.loads((word_dir / b["predictions"]).read_text()),
                                    left["config"]["labels"][corpus])
            comparisons.append({"corpus": corpus, "variant": variant, **paired,
                                "word_zero_feature_train_rows": b["zero_feature_train_rows"],
                                "word_zero_feature_validation_rows": b["zero_feature_validation_rows"]})
    data = {
        "manifest_version": left["config"]["manifest_version"],
        "test_loaded": False, "comparisons": comparisons,
        "run_results_sha256": {"character": digest(char_dir / "results.json"),
                               "word": digest(word_dir / "results.json")},
        "configs": {"character": left["config"], "word": right["config"]},
        "manifests_sha256": left["manifests_sha256"], "uv_lock_sha256": left["uv_lock_sha256"],
        "comparison_script_sha256": digest(Path(__file__)),
    }
    lines = ["# TuniScope — Comparaison TF-IDF mots et caractères", "",
             "**Livrables :** L04, L07 et L12, contributions partielles.", "",
             f"Manifestes {data['manifest_version']}, mêmes identifiants/groupes/labels de validation. "
             "Prédictions et checkpoints vérifiés, métriques recalculées. Aucun test officiel chargé.", "",
             "| Corpus | Variante | Macro-F1 caractères | Macro-F1 mots | Écart mots − caractères | Erreurs caractères | Erreurs mots |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in comparisons:
        lines.append(f"| {row['corpus']} | {row['variant']} | {row['character_macro_f1']:.6f} | "
                     f"{row['word_macro_f1']:.6f} | {row['delta_word_minus_character']:+.6f} | "
                     f"{row['character_errors']} | {row['word_errors']} |")
    lines += ["", "## Complémentarité des erreurs", "",
              "| Corpus | Variante | Caractères seuls corrects | Mots seuls corrects | Deux systèmes en erreur | Vecteurs mots nuls en validation |",
              "| --- | --- | ---: | ---: | ---: | ---: |"]
    for row in comparisons:
        lines.append(f"| {row['corpus']} | {row['variant']} | {row['character_correct_word_wrong']} | "
                     f"{row['word_correct_character_wrong']} | {row['both_wrong']} | "
                     f"{row['word_zero_feature_validation_rows']} |")
    lines += ["", "## Interprétation et limites", "",
              "Les éléments comparés diffèrent par les unités, les étendues de n-grammes et le traitement "
              "des symboles. Les autres paramètres communs sont contrôlés. Ces scores de validation "
              "ne prouvent pas une supériorité générale ni un écart significatif sur le test final.", "",
              "La baseline mots ignore ponctuation et émojis et ne réalise pas de segmentation "
              "morphologique arabe. Les lignes sans feature après segmentation/min_df restent "
              "dans les métriques. Les vocabulaires sont ajustés sur train seulement.", "",
              "Les cas corrigés par un seul système décrivent une complémentarité observée. Leur "
              "somme ne constitue pas la performance d'une combinaison réelle : il faudrait savoir "
              "à l'avance quel système choisir. Une combinaison doit être entraînée et évaluée séparément.", "",
              "Les comptes et empreintes sont dans `comparison.json`. Les prédictions ligne par ligne "
              "et vocabulaires restent privés. Les analyses d'erreurs humaines, règles linguistiques, "
              "combinaison et évaluation finale restent à réaliser."]
    report_dir.mkdir(parents=True)
    (report_dir / "comparison.json").write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                                          sort_keys=True) + "\n", encoding="utf-8")
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(comparisons, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("character_run", type=Path)
    parser.add_argument("word_run", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    compare(args.character_run, args.word_run, args.report_dir)
