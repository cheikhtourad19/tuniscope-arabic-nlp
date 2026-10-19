"""Publish aggregate-only baseline evidence after independent verification."""

import argparse
import json
from pathlib import Path

from audit_structure import ROOT
from verify_baseline_run import verify
from run_baselines import digest
from baseline_models import tfidf_system


def render(run_dir, report_dir):
    verify(run_dir)
    data = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))
    if data["mode"] != "full":
        raise ValueError("A smoke test is not a full baseline result")
    data["report_generator_sha256"] = digest(Path(__file__))
    manifest_version = data["config"].get("manifest_version", "v1")
    learned_system = tfidf_system(data["config"])
    data["paired_comparisons"] = {}
    for corpus in data["config"]["corpora"]:
        variants = {r["variant"]: r for r in data["results"]
                    if r["corpus"] == corpus and r["system"] == learned_system}
        raw = json.loads((run_dir / variants["raw-v0"]["predictions"]).read_text(encoding="utf-8"))
        normalized = json.loads((run_dir / variants["norm-nfc-space-v1"]["predictions"]).read_text(encoding="utf-8"))
        data["paired_comparisons"][corpus] = {
            "delta_macro_f1": variants["norm-nfc-space-v1"]["metrics"]["macro_f1"] - variants["raw-v0"]["metrics"]["macro_f1"],
            "different_predictions": sum(a["predicted"] != b["predicted"] for a, b in zip(raw, normalized)),
            "raw_errors": sum(a["predicted"] != a["truth"] for a in raw),
            "normalized_errors": sum(a["predicted"] != a["truth"] for a in normalized),
        }
    report_dir = report_dir.resolve()
    if (ROOT / "reports").resolve() not in report_dir.parents:
        raise ValueError("Choose a report directory under reports/")
    if report_dir.exists():
        if {path.name for path in report_dir.iterdir()} != {"metrics.json", "README.md"}:
            raise ValueError("Refusing to overwrite a directory with unrelated files")
        previous = json.loads((report_dir / "metrics.json").read_text(encoding="utf-8"))
        if previous["created_at_utc"] != data["created_at_utc"]:
            raise ValueError("Refusing to replace a different run report")
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(data, ensure_ascii=False, indent=2,
                                                       sort_keys=True) + "\n", encoding="utf-8")
    lines = [
        "# TuniScope — Premières baselines : résultats de validation", "",
        "**Livrables concernés :** L04 et L07, contributions partielles ; L12 pour la traçabilité.", "",
        "Ce rapport est généré par `scripts/render_baseline_report.py` depuis les prédictions vérifiées. "
        f"Il porte uniquement sur la validation des manifestes {manifest_version}. Le test officiel n'a pas été chargé.", "",
        f"Configuration `{data['config']['version']}`, Python {data['python']}, "
        f"scikit-learn {data['packages']['scikit-learn']}, {data['platform']} {data['architecture']}, CPU, un thread.", "",
        f"Commit de départ : `{data['git_commit']}` ; fichiers non commités lors du run : "
        f"`{str(data['git_dirty']).lower()}`. Les empreintes du code, du verrouillage et des manifestes "
        "figurent dans `metrics.json`.", "",
        "## Résultats", "",
        "| Corpus | Système | Texte | Train | Validation | Macro-F1 | Accuracy | Balanced accuracy | Ajustement (s) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in data["results"]:
        m = item["metrics"]
        lines.append(f"| {item['corpus']} | {item['system']} | {item['variant']} | "
                     f"{item['train_rows']} | {item['validation_rows']} | {m['macro_f1']:.6f} | "
                     f"{m['accuracy']:.6f} | {m['balanced_accuracy']:.6f} | {item['fit_seconds']:.3f} |")
    lines += ["", "## Contraste normalisé − brut", "",
              "Ces écarts concernent la même baseline et les mêmes exemples. Ils ne constituent "
              "pas un résultat final sur trois graines ni une preuve de significativité.", ""]
    for corpus in data["config"]["corpora"]:
        pair = data["paired_comparisons"][corpus]
        delta = pair["delta_macro_f1"]
        lines.append(f"- {corpus} : Δ macro-F1 = {delta:+.6f} ({delta * 100:+.4f} point(s) de pourcentage), "
                     f"{pair['different_predictions']} prédiction(s) différente(s), "
                     f"{pair['raw_errors']} erreur(s) en brut et {pair['normalized_errors']} en normalisé.")
    lines += ["", ("L'analyseur de caractères de scikit-learn harmonise déjà les suites d'espaces. "
              "Le contraste mesure l'apport supplémentaire de NFC et du retrait des espaces périphériques."
              if learned_system == "tfidf_char" else
              "La combinaison juxtapose les blocs mots et caractères, à poids égaux, puis normalise "
              "le vecteur global en L2. Les deux vocabulaires sont ajustés sur train seulement. "
              "Les limites de segmentation du bloc mots persistent ; les émojis peuvent être représentés "
              "par le bloc caractères s'ils disposent de caractéristiques retenues."
              if learned_system == "tfidf_word_char" else
              "L'analyseur mots utilise la segmentation Unicode déclarée dans la configuration. Les espaces "
              "et la ponctuation servent déjà de séparateurs ; les émojis ne sont pas représentés. Les lignes "
              "sans caractéristique retenue restent évaluées. Le contraste brut/normalisé porte sur NFC/espaces."), "",
              "## Résultats par classe", "",
              "| Corpus | Système | Texte | Classe | Précision | Rappel | F1 | Support validation |",
              "| --- | --- | --- | --- | ---: | ---: | ---: | ---: |"]
    for item in data["results"]:
        for label in data["config"]["labels"][item["corpus"]]:
            m = item["metrics"]["per_class"][label]
            lines.append(f"| {item['corpus']} | {item['system']} | {item['variant']} | {label} | "
                         f"{m['precision']:.6f} | {m['recall']:.6f} | {m['f1-score']:.6f} | {int(m['support'])} |")
    lines += ["", "## Traçabilité et limites", "",
              "Les matrices de confusion, scores par écriture, tailles des vocabulaires, itérations, temps, "
              "empreintes et tailles des checkpoints sont dans `metrics.json`. Les classes sans prédiction "
              "reçoivent une précision/rappel/F1 nul selon `zero_division=0`. Les sous-groupes trop petits "
              "ou privés d'une classe sont signalés et ne doivent pas servir à une conclusion globale.", "",
              "La mémoire maximale enregistrée est le pic du processus depuis son démarrage, donc cumulé "
              "entre systèmes ; elle n'est pas une mesure isolée du coût de chaque modèle. Le temps "
              "d'ajustement comprend le vectoriseur et le classifieur. Le modèle majoritaire utilise les "
              "fréquences du train uniquement. Les modèles TF-IDF ajustent vocabulaire et pondérations "
              "sur train uniquement ; les checkpoints ont été rechargés et leurs prédictions vérifiées.", "",
              "Les prédictions ligne par ligne et les checkpoints contenant des vocabulaires restent sous "
              "`outputs/baselines/`, ignoré par Git. Le présent rapport ne contient aucun texte de corpus. "
              "Les comparaisons classiques complètes, règles, analyses d'erreurs, modèles neuronaux et résultats finaux restent à réaliser.", "",
              ("La recherche de familles est heuristique. L'audit de réduction des lettres répétées a signalé neuf "
              "groupes de collisions candidates train/validation TSAC avec cette autre règle. Conserver les "
              "répétitions dans v1 ne garantit pas que ces variantes soient indépendantes ; leur sens reste à "
              "examiner avant de considérer cette validation comme définitivement consolidée. Les scores "
              "présentés constituent donc une référence initiale provisoire, sans comparaison directe à un "
              "benchmark publié utilisant d'autres partitions." if manifest_version == "v1" else
              "La v2 intègre les 12 décisions humaines de même sens et conserve ensemble toutes les familles "
              "candidates de la règle de répétitions, y compris celles qui ne traversaient pas les partitions v1. "
              "Cette précaution de partitionnement ne transforme pas les candidats non relus en équivalences "
              "sémantiques validées. Les variantes raw/norm d'un même run restent comparables, mais les scores "
              "TSAC v1 et v2 n'utilisent pas les mêmes exemples de validation : leur différence n'isole pas "
              "l'effet de la correction. Les résultats restent provisoires, sans garantie d'absence de toute "
              "similarité résiduelle ni comparaison directe à des benchmarks utilisant d'autres partitions.")]
    if any("zero_feature_validation_rows" in r for r in data["results"]):
        lines += ["", "## Lignes sans caractéristique retenue", "",
                  "Ces lignes sont conservées dans les prédictions et métriques, sans filtrage.", "",
                  "| Corpus | Système | Texte | Train | Validation |",
                  "| --- | --- | --- | ---: | ---: |"]
        for r in data["results"]:
            if "zero_feature_validation_rows" in r:
                lines.append(f"| {r['corpus']} | {r['system']} | {r['variant']} | "
                             f"{r['zero_feature_train_rows']} | {r['zero_feature_validation_rows']} |")
    (report_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Aggregate report written to {report_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    render(args.run_dir, args.report_dir)
