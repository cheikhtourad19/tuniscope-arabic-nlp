"""Rebuild rule lexicons from train, reload predictions, hand-check metrics/report."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from load_working_data import ROOT
from rule_baselines import fit_rules
from run_rule_baselines import coverage, predict_rows, split_rows, systems
from run_baselines import digest, evaluate
from verify_baseline_run import verify as verify_character


def paired(left, right):
    if not left or len(left) != len(right) or len({r["id"] for r in left}) != len(left):
        raise ValueError("Invalid paired rows")
    for a,b in zip(left,right):
        if any(a[k] != b[k] for k in ("id","group_id","truth","writing_system")):
            raise ValueError("Unaligned validation comparisons")
    return {"different_predictions":sum(a["predicted"]!=b["predicted"] for a,b in zip(left,right)),
            "corrected_by_right":sum(a["predicted"]!=a["truth"] and b["predicted"]==b["truth"] for a,b in zip(left,right)),
            "introduced_by_right":sum(a["predicted"]==a["truth"] and b["predicted"]!=b["truth"] for a,b in zip(left,right))}


def verify(run):
    data = json.loads((run/"results.json").read_text())
    config = data["config"]
    if data["mode"] not in {"smoke","full"} or data["test_loaded"] or len(data["results"]) != 5:
        raise ValueError("Expected five validation-only systems")
    if data["uv_lock_sha256"] != digest(ROOT/"uv.lock"):
        raise ValueError("Environment lock changed")
    for name, value in data["code_sha256"].items():
        if digest(ROOT/"scripts"/name) != value:
            raise ValueError(f"Run code changed: {name}")
    loaded = {}
    for corpus in config["corpora"]:
        if digest(ROOT/f"data/manifests/v2/{corpus}.csv") != data["manifests_sha256"][corpus]:
            raise ValueError("Manifest changed")
        train, validation = split_rows(corpus,config,data["mode"])
        saved = data["models"][corpus]
        checkpoint = run/saved["checkpoint"]
        if digest(checkpoint) != saved["checkpoint_sha256"]:
            raise ValueError("Lexicon checksum mismatch")
        model = json.loads(checkpoint.read_text())
        if model != fit_rules(train,config["labels"][corpus],config):
            raise ValueError("Lexicon differs from fresh train-only reconstruction")
        if (saved["train_rows"] != len(train) or saved["train_labels"] != dict(Counter(r["label"] for r in train))
            or saved["lexicon_size"] != len(model["entries"])):
            raise ValueError("Training provenance/count mismatch")
        results = [r for r in data["results"] if r["corpus"]==corpus]
        if len(results)!=len(systems(corpus)) or {r["system"] for r in results} != set(systems(corpus)):
            raise ValueError("Missing/duplicate rule systems")
        for result in results:
            path = run/result["predictions"]
            if digest(path) != result["predictions_sha256"]:
                raise ValueError("Prediction checksum mismatch")
            predictions = json.loads(path.read_text())
            if predictions != predict_rows(validation,model,result["system"]):
                raise ValueError("Rebuilt predictions or decision flags differ")
            if result["coverage"] != coverage(predictions) or result["metrics"] != evaluate(validation,[p["predicted"] for p in predictions],model["labels"]):
                raise ValueError("Metrics, slices or coverage mismatch")
            labels = model["labels"]
            matrix = [[0]*len(labels) for _ in labels]
            for p in predictions:
                matrix[labels.index(p["truth"])][labels.index(p["predicted"])]+=1
            f1s = [2*matrix[i][i]/den if (den:=sum(matrix[i])+sum(row[i] for row in matrix)) else 0 for i in range(len(labels))]
            if matrix!=result["metrics"]["confusion_matrix"] or not math.isclose(sum(f1s)/len(labels),result["metrics"]["macro_f1"],abs_tol=1e-12):
                raise ValueError("Hand-recomputed metrics differ")
            loaded[(corpus,result["system"])]=predictions
    print(f"Verified {data['mode']}: five systems, train-only lexicons rebuilt, predictions/metrics/coverage checked")
    return data,loaded


def render_report(run, report):
    data, predictions = verify(run)
    if data["mode"] != "full":
        raise ValueError("Public comparative report requires full run")
    reference = ROOT/"outputs/baselines/2026-10-17_grouped_v2_full"
    verify_character(reference)
    char = json.loads((reference/"results.json").read_text())
    if char["manifests_sha256"] != data["manifests_sha256"] or char["config"].get("manifest_version") != "v2" or char["config"]["labels"] != data["config"]["labels"]:
        raise ValueError("Character reference uses incompatible data")
    comparisons = {}
    for corpus in data["config"]["corpora"]:
        result = next(r for r in char["results"] if (r["corpus"],r["system"],r["variant"])==(corpus,"tfidf_char","raw-v0"))
        ref_predictions = json.loads((reference/result["predictions"]).read_text())
        for system in systems(corpus):
            comparisons[f"{corpus}:{system}"] = paired(ref_predictions,predictions[(corpus,system)])
    ablation = paired(predictions[("tsac","lexicon")],predictions[("tsac","lexicon_negation")])
    output = {"run_results_sha256":digest(run/"results.json"), "character_run_results_sha256":digest(reference/"results.json"),
        "config":data["config"], "config_sha256":data["config_sha256"],"protocol_sha256_at_run":data["protocol_sha256"],
        "code_sha256_at_run":data["code_sha256"],"manifests_sha256":data["manifests_sha256"],
        "uv_lock_sha256":data["uv_lock_sha256"],"test_loaded":False,
        "models":data["models"],"results":data["results"],"comparisons_to_character_raw":comparisons,
        "negation_minus_lexicon_paired":ablation}
    lines = ["# TuniScope — Baselines lexicales et portée locale de négation", "",
        "**Livrables :** L04, L07, L10 et L12, contributions partielles. Validation v2 uniquement, paramètres fixés avant ce run.", "",
        "## Résultats sur tous les exemples", "",
        "Les métriques incluent le recours majoritaire quand le vote n'est pas décisif. "
        "La couverture est la fraction de votes décisifs, pas la proportion de textes auxquels le système retourne un label.", "",
        "| Corpus | Système | Macro-F1 global | Accuracy | Couverture décisive | Recours majoritaire |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in data["results"]:
        lines.append(f"| {r['corpus']} | {r['system']} | {r['metrics']['macro_f1']:.6f} | {r['metrics']['accuracy']:.6f} | "
                     f"{100*r['coverage']['decisive_fraction']:.2f} % | {r['coverage']['fallback_rows']}/{r['validation_rows']} |")
    lines += ["", "## Effet apparié de la négation sur TSAC", "",
        f"Même lexique et mêmes textes : {ablation['different_predictions']} prédictions changent. "
        f"La variante avec négation corrige {ablation['corrected_by_right']} erreurs du lexique seul "
        f"et introduit {ablation['introduced_by_right']} nouvelles erreurs. Les labels source restent inchangés.", "",
        "## Comparaison descriptive avec les caractères", "",
        "| Corpus | Règles | Erreurs caractères corrigées | Erreurs nouvelles face aux caractères |",
        "| --- | --- | ---: | ---: |"]
    for name, c in comparisons.items():
        corpus,system=name.split(":")
        if system!="majority":
            lines.append(f"| {corpus} | {system} | {c['corrected_by_right']} | {c['introduced_by_right']} |")
    lines += ["", "## Méthode et limites", "",
        "Les lexiques sont statistiquement induits du train : fréquence documentaire lissée par classe, "
        "seuils fixes et au plus 200 termes par classe. Ce ne sont pas des lexiques experts ni des preuves "
        "qu'un terme possède intrinsèquement un sentiment ou un dialecte. Noms propres et effets de domaine sont possibles.", "",
        "Le vote lexical est déterministe. La variante de négation inverse les indices dans une fenêtre "
        "de trois mots, arrêtée aux frontières et adversatifs reconnus. Les marqueurs/ruptures/exceptions "
        "ne sont activés que s'ils sont attestés sur train. La parité des négations est une heuristique "
        "qui ne modélise pas la concordance négative, les cibles multiples ou l'ironie.", "",
        "Émojis ignorés par ce lexique, graphies non reconnues, absence d'indices et égalités aboutissent "
        "parfois au recours majoritaire. Le vote n'est pas une probabilité calibrée. Les cas non couverts "
        "ne sont jamais supprimés du macro-F1 principal. Les petites slices restent signalées dans les métriques.", "",
        "Le contraste avec les caractères aligne les identifiants, groupes et labels, mais compare des "
        "représentations et traitements différents. Aucun gain significatif ou causal, aucune généralisation "
        "au test officiel et aucun résultat d'intégration lexique+TF-IDF ne sont revendiqués.", "",
        "## Vérification et ressources", "",
        "Smoke test préalable sur 100 lignes par partition/tâche, puis run complet CPU. Les lexiques JSON "
        "sont reconstruits intégralement depuis le train, les prédictions et indicateurs rejoués, les matrices "
        "de confusion et macro-F1 recalculés indépendamment. Le checkpoint caractères de référence est "
        "également vérifié. Les lexiques et sorties par exemple restent privés.", "",
        "Les durées et pics RSS cumulatifs au processus figurent dans summary.json. Aucun GPU ou modèle "
        "neuronal utilisé. Les sources et manifestes restent inchangés. La relecture des analyses linguistiques, "
        "l'éventuelle intégration contrôlée au meilleur classifieur, les notebooks et l'évaluation finale restent ouverts."]
    report=report.resolve()
    if (ROOT/"reports").resolve() not in report.parents or report.exists():
        raise ValueError("Choose a new report directory under reports/")
    report.mkdir(parents=True)
    (report/"summary.json").write_text(json.dumps(output,ensure_ascii=False,indent=2)+"\n")
    (report/"README.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    if args.report: render_report(args.run,args.report)
    else: verify(args.run)
