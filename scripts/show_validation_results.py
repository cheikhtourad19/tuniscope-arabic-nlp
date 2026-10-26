"""Render aggregate validation results for a jury, without training any model.

Use --verify to replay the existing checkpoint verifiers before rendering.
No corpus text, private row identifiers, or test predictions are exported.
"""

import argparse
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import subprocess
import sys

from baseline_metrics import classification_metrics
from run_baselines import digest


ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "characters": "2026-10-17_grouped_v2_full",
    "words": "2026-10-17_words_v2_full",
    "combined": "2026-10-17_combined_v2_full",
    "rules": "2026-10-17_rules_v1_full",
    "augmented": "2026-10-19_character_rules_v1_full",
}
CSS = """*{box-sizing:border-box}body{margin:0;background:#fff;color:#142b44;font:19px Arial,sans-serif}
main{max-width:1280px;margin:auto;padding:30px 42px}h1{font-size:34px;margin:0 0 10px}
h2{font-size:24px;margin:16px 0 10px}.lead{font-size:20px;color:#40566a;margin:0 0 20px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
th,td{padding:13px 16px;text-align:left;border-bottom:1px solid #dce3e6}
th{background:#142b44;color:white;font-weight:500}td:not(:first-child){text-align:right}
tr.best td{background:#e6f4f1;color:#095b57;font-weight:bold}.split{display:grid;grid-template-columns:1fr 1fr;gap:36px}
.matrix td,.matrix th{text-align:center;padding:10px}.matrix th{background:#e8eef0;color:#142b44}
.matrix td.diag{background:#e6f4f1;color:#095b57;font-weight:bold}.small{font-size:17px}
.split table:not(.matrix){font-size:16px}.split th,.split td{padding:9px 8px}
footer{margin-top:22px;padding-top:14px;border-top:2px solid #32958e;font-size:16px;line-height:1.5;color:#40566a}
.finding{font-size:23px;font-weight:bold;margin:22px 0 10px}.context{margin:15px 0 22px;color:#40566a}
"""


def selected_result(data, corpus, system, variant=None):
    rows = [r for r in data["results"] if r["corpus"] == corpus and r["system"] == system
            and (variant is None or r.get("variant") == variant)]
    if len(rows) != 1:
        raise ValueError("Expected one unambiguous full-run result")
    return rows[0]


def recomputed_metrics(directory, result, labels):
    predictions = json.loads((directory / result["predictions"]).read_text(encoding="utf-8"))
    metrics = classification_metrics([r["truth"] for r in predictions],
                                     [r["predicted"] for r in predictions], labels)
    # The historical evaluator also saves writing-system slices. Compare every
    # common/global field, without mistaking those extra slices for a mismatch.
    if any(value != result["metrics"].get(key) for key, value in metrics.items()):
        raise ValueError("Saved metrics differ from recomputed predictions")
    return metrics


def table(headers, rows, highlighted=None):
    head = "".join(f"<th>{escape(str(value))}</th>" for value in headers)
    body = "".join(
        f'<tr class="{"best" if index == highlighted else ""}">'
        + "".join(f"<td>{escape(str(value))}</td>" for value in row) + "</tr>"
        for index, row in enumerate(rows)
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def matrix_html(metrics):
    labels = metrics["labels"]
    head = "<th>Réel / prédit</th>" + "".join(f"<th>{escape(label)}</th>" for label in labels)
    body = "".join(
        f"<tr><th>{escape(labels[i])}</th>" + "".join(
            f'<td class="{"diag" if i == j else ""}">{value}</td>'
            for j, value in enumerate(row)) + "</tr>"
        for i, row in enumerate(metrics["confusion_matrix"])
    )
    return f'<table class="matrix"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>'


def page(title, body, verified):
    state = "Checkpoints rechargés et prédictions vérifiées" if verified else "Métriques recalculées depuis les prédictions sauvegardées"
    return (f'<!doctype html><html lang="fr"><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{escape(title)}</title><style>{CSS}</style><main><h1>{escape(title)}</h1>'
            f'<p class="lead">TuniScope · Validation v2 · {state}</p>{body}'
            '<footer>Résultats des entraînements complets, pas des petits tests sur 100 exemples. '
            'Aucun score sur le test officiel. Données brutes et labels inchangés.<br>'
            'Le macro-F1 est la moyenne des F1 des classes, distincte de l’accuracy. '
            'Ce rapport ne contient aucun texte de corpus.</footer></main></html>')


def display_run(directory, verify=False):
    """Print a single TF-IDF run, including a new run launched by a notebook."""
    directory = directory.resolve()
    if (ROOT / "outputs/baselines").resolve() not in directory.parents:
        raise ValueError("Choose a local run under outputs/baselines/")
    data = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    if data["test_loaded"] or data["config"].get("manifest_version") != "v2":
        raise ValueError("Expected v2 train/validation-only results")
    if verify:
        subprocess.run([sys.executable, str(ROOT / "scripts/verify_baseline_run.py"), str(directory)],
                       cwd=ROOT, check=True)
    print(f"\nVALIDATION — mode {data['mode']} — {directory.name}")
    if data["mode"] == "smoke":
        print("ATTENTION : petit test technique, pas les scores de l'expérience complète.")
    checked = []
    for result in data["results"]:
        if digest(directory / result["predictions"]) != result["predictions_sha256"]:
            raise ValueError("Saved prediction file changed")
        metrics = recomputed_metrics(directory, result, data["config"]["labels"][result["corpus"]])
        checked.append(metrics)
        print(f"\n{result['corpus'].upper()} | {result['system']} | {result['variant']}")
        print(f"Accuracy : {metrics['accuracy']:.6f} ({100 * metrics['accuracy']:.2f} %)"
              f" | Macro-F1 : {metrics['macro_f1']:.6f}"
              f" | Balanced accuracy : {metrics['balanced_accuracy']:.6f}")
        print(f"{'Classe':<9}{'Précision':>12}{'Rappel':>12}{'F1':>12}{'Effectif':>10}")
        for label in metrics["labels"]:
            row = metrics["per_class"][label]
            print(f"{label:<9}{row['precision']:>12.6f}{row['recall']:>12.6f}"
                  f"{row['f1-score']:>12.6f}{int(row['support']):>10}")
        print("Matrice : lignes = labels réels, colonnes = prédictions", metrics["labels"])
        for row in metrics["confusion_matrix"]:
            print(row)
    return checked


def generate(output, verify=False):
    output = output.resolve()
    if (ROOT / "outputs").resolve() not in output.parents or output.exists():
        raise ValueError("Choose a NEW report directory under outputs/")
    data, sources = {}, {}
    for name, folder in RUNS.items():
        directory = ROOT / "outputs/baselines" / folder
        payload = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        if payload["mode"] != "full" or payload["test_loaded"] or payload["config"].get("manifest_version") != "v2":
            raise ValueError("Expected full, v2, validation-only results")
        if verify:
            verifier = ("verify_rule_baselines.py" if name == "rules" else
                        "verify_character_rules.py" if name == "augmented" else "verify_baseline_run.py")
            subprocess.run([sys.executable, str(ROOT / "scripts" / verifier), str(directory)],
                           cwd=ROOT, check=True)
        data[name] = payload
        sources[name] = {"directory": str(directory.relative_to(ROOT)),
                         "results_sha256": digest(directory / "results.json")}
    primary, detail = [], {}
    for title, run, system in [
        ("Classe majoritaire", "characters", "majority"),
        ("Vote lexical", "rules", "lexicon"),
        ("TF-IDF mots", "words", "tfidf_word"),
        ("TF-IDF caractères", "characters", "tfidf_char"),
        ("Mots + caractères", "combined", "tfidf_word_char"),
    ]:
        item = {"system": title}
        for corpus in ("tsac", "madar"):
            result = selected_result(data[run], corpus, system, None if run == "rules" else "raw-v0")
            metrics = recomputed_metrics(ROOT / "outputs/baselines" / RUNS[run], result,
                                         data[run]["config"]["labels"][corpus])
            item[corpus] = metrics["macro_f1"]
            item[corpus + "_accuracy"] = metrics["accuracy"]
            if system == "tfidf_char":
                detail[corpus] = metrics
        primary.append(item)
    def score(run, corpus, system, variant=None):
        return selected_result(data[run], corpus, system, variant)["metrics"]["macro_f1"]
    ablations = [
        ["Caractères, texte brut", primary[3]["tsac"], primary[3]["madar"]],
        ["Caractères, NFC / espaces", score("characters","tsac","tfidf_char","norm-nfc-space-v1"),
         score("characters","madar","tfidf_char","norm-nfc-space-v1")],
        ["Mots + caractères", primary[4]["tsac"], primary[4]["madar"]],
        ["Caractères + lexique", score("augmented","tsac","char_lexicon"), score("augmented","madar","char_lexicon")],
        ["Caractères + lexique / négation", score("augmented","tsac","char_lexicon_negation"), None],
    ]
    fmt = lambda value: "—" if value is None else f"{value:.6f}".replace(".", ",")
    comparison = table(["Système", "TSAC macro-F1", "TSAC accuracy", "MADAR macro-F1", "MADAR accuracy"],
                       [[r["system"], fmt(r["tsac"]), fmt(r["tsac_accuracy"]),
                         fmt(r["madar"]), fmt(r["madar_accuracy"])] for r in primary], 3)
    comparison += '<p class="finding">Les caractères seuls restent la meilleure référence initiale</p>'
    comparison += '<p class="context">TSAC : 10 053 train / 1 774 validation. MADAR : 6 388 train / 800 validation. Texte brut.</p>'
    sections = []
    for corpus, metrics in detail.items():
        short = lambda value: f"{value:.3f}".replace(".", ",")
        classes = [[label, short(metrics["per_class"][label]["precision"]),
                    short(metrics["per_class"][label]["recall"]),
                    short(metrics["per_class"][label]["f1-score"]), int(metrics["per_class"][label]["support"])]
                   for label in metrics["labels"]]
        sections.append(f'<section><h2>{corpus.upper()} — caractères bruts</h2>{matrix_html(metrics)}'
                        '<h2>Métriques par classe</h2>' + table(["Classe", "Précision", "Rappel", "F1", "n"], classes) + '</section>')
    rendered = {
        "01_comparaison.html": page("Résultats des références classiques", comparison, verify),
        "02_classes_confusions.html": page("Résultats par classe et matrices de confusion", '<div class="split">'+"".join(sections)+"</div>", verify),
        "03_ablations.html": page("Ablations : normalisation et indices lexicaux",
            table(["Système", "Macro-F1 TSAC", "Macro-F1 MADAR"], [[row[0], fmt(row[1]), fmt(row[2])] for row in ablations], 0)
            + '<p class="finding">Aucun gain face aux caractères seuls dans les configurations testées</p>', verify),
    }
    output.mkdir(parents=True)
    for filename, html in rendered.items():
        (output / filename).write_text(html, encoding="utf-8")
    summary = {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "manifest_version": "v2",
               "checkpoint_verifiers_passed": verify, "test_evaluated": False, "new_training": False,
               "sources": sources, "primary": primary, "character_details": detail, "ablations": ablations,
               "uv_lock_sha256": digest(ROOT / "uv.lock")}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print("\nRÉSULTATS COMPLETS — VALIDATION v2 — MACRO-F1")
    for row in primary:
        print(f'{row["system"]:<24} TSAC F1 {fmt(row["tsac"])} accuracy {fmt(row["tsac_accuracy"])}'
              f'   MADAR F1 {fmt(row["madar"])} accuracy {fmt(row["madar_accuracy"])}')
    print(f"\nRapport : {output}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output", type=Path, help="Create the historical comparison in a NEW output directory")
    target.add_argument("--run", type=Path, help="Print all metrics from one local v2 TF-IDF run")
    parser.add_argument("--verify", action="store_true", help="Reload saved models and replay existing verifiers")
    args = parser.parse_args()
    if args.run:
        display_run(args.run, args.verify)
    else:
        generate(args.output, args.verify)
