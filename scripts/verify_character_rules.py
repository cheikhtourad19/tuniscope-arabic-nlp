"""Verify sparse augmented models against train and historical baselines."""
import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from load_working_data import ROOT
from character_rule_models import coverage, feature_matrix, predict_rows, rule_features, systems, validate_config
from rule_baselines import fit_rules
from run_baselines import digest, evaluate
from run_rule_baselines import split_rows
from verify_baseline_run import verify as verify_character
from verify_rule_baselines import paired, verify as verify_rules


def verify(run):
    data=json.loads((run/"results.json").read_text())
    if data["test_loaded"] or data["mode"] not in {"smoke","full"} or len(data["results"])!=5:
        raise ValueError("Expected five complete validation-only results")
    for name,sha in data["code_sha256"].items():
        if digest(ROOT/"scripts"/name)!=sha: raise ValueError(f"Run code changed: {name}")
    if digest(ROOT/"uv.lock")!=data["uv_lock_sha256"]: raise ValueError("Environment changed")
    refs={name:ROOT/item["directory"] for name,item in data["reference_runs"].items()}
    for name,directory in refs.items():
        if digest(directory/"results.json")!=data["reference_runs"][name]["results_sha256"]:
            raise ValueError("Reference run changed")
    verify_character(refs["character"])
    verify_rules(refs["rules"])
    char=json.loads((refs["character"]/"results.json").read_text())
    rules=json.loads((refs["rules"]/"results.json").read_text())
    config=data["config"]
    validate_config(config,char["config"])
    if data["rule_config"]!=rules["config"] or data["manifests_sha256"]!=char["manifests_sha256"] or data["manifests_sha256"]!=rules["manifests_sha256"]:
        raise ValueError("Reference resource condition mismatch")
    if data["packages"]!=char["packages"] or data["python"]!=char["python"] or data["uv_lock_sha256"]!=char["uv_lock_sha256"]:
        raise ValueError("Character environment condition mismatch")
    predictions={}
    models={}
    with threadpool_limits(limits=1):
        for corpus in config["corpora"]:
            if digest(ROOT/f"data/manifests/v2/{corpus}.csv")!=data["manifests_sha256"][corpus]:
                raise ValueError("Manifest changed")
            train,validation=split_rows(corpus,config,data["mode"])
            rebuilt_rules=fit_rules(train,config["labels"][corpus],data["rule_config"])
            historical_rules=json.loads((refs["rules"]/rules["models"][corpus]["checkpoint"]).read_text())
            if rebuilt_rules!=historical_rules: raise ValueError("Train-derived lexical resource differs")
            ref=next(r for r in char["results"] if (r["corpus"],r["system"],r["variant"])==(corpus,"tfidf_char","raw-v0"))
            original=joblib.load(refs["character"]/ref["checkpoint"])
            reference_predictions=json.loads((refs["character"]/ref["predictions"]).read_text())
            results=[r for r in data["results"] if r["corpus"]==corpus]
            if len(results)!=len(systems(corpus)) or {r["system"] for r in results}!=set(systems(corpus)):
                raise ValueError("Missing/duplicate augmented systems")
            for result in results:
                checkpoint=run/result["checkpoint"]
                pred_path=run/result["predictions"]
                if digest(checkpoint)!=result["checkpoint_sha256"] or digest(pred_path)!=result["predictions_sha256"]:
                    raise ValueError("Model/prediction hash mismatch")
                model=joblib.load(checkpoint)
                if model["config"]!=config or model["rules"]!=rebuilt_rules or model["system"]!=result["system"] or model["corpus"]!=corpus:
                    raise ValueError("Saved model condition mismatch")
                branch=model["vectorizer"]
                if branch.vocabulary_!=original["tfidf"].vocabulary_ or not np.array_equal(branch.idf_,original["tfidf"].idf_):
                    raise ValueError("Character vocabulary/IDF changed")
                for key,value in config["tfidf"].items():
                    actual=branch.get_params()[key]
                    value=tuple(value) if key=="ngram_range" else value
                    if key=="dtype":
                        if np.dtype(actual)!=np.dtype(value): raise ValueError("Dtype mismatch")
                    elif actual!=value: raise ValueError("Vectorizer parameter mismatch")
                if any(model["classifier"].get_params()[k]!=v for k,v in config["classifier"].items()):
                    raise ValueError("Classifier parameter mismatch")
                if result["train_rows"]!=len(train) or result["validation_rows"]!=len(validation) or result["train_labels"]!=model["train_labels"]:
                    raise ValueError("Training/validation counts changed")
                for name,rows in (("train",train),("validation",validation)):
                    if coverage(model,rows)!=result[f"{name}_coverage"]: raise ValueError("Coverage mismatch")
                matrix=feature_matrix(model,validation)
                if result["feature_count"]!=matrix.shape[1] or model["classifier"].n_features_in_!=matrix.shape[1]:
                    raise ValueError("Feature dimension mismatch")
                # No hidden global normalization: character columns remain the
                # historical values, and new columns are exactly the declared block.
                width=len(branch.vocabulary_)
                if (matrix[:,:width]-original["tfidf"].transform([r["text_raw"] for r in validation])).nnz:
                    raise ValueError("Character values changed on concatenation")
                if result["system"]!="char_control":
                    extra=rule_features(validation,rebuilt_rules,negation=result["system"]=="char_lexicon_negation")
                    if (matrix[:,width:]-extra).nnz: raise ValueError("Lexical block mismatch")
                saved=json.loads(pred_path.read_text())
                paired(reference_predictions,saved)
                if predict_rows(model,validation)!=[p["predicted"] for p in saved]: raise ValueError("Reloaded predictions differ")
                if result["system"]=="char_control" and saved!=reference_predictions:
                    raise ValueError("Control predictions not identical to reference")
                metrics=evaluate(validation,[p["predicted"] for p in saved],config["labels"][corpus])
                if metrics!=result["metrics"]: raise ValueError("Metrics/slices mismatch")
                matrix_counts=metrics["confusion_matrix"]
                f1s=[2*matrix_counts[i][i]/den if (den:=sum(matrix_counts[i])+sum(r[i] for r in matrix_counts)) else 0
                     for i in range(len(config["labels"][corpus]))]
                if not math.isclose(sum(f1s)/len(f1s),metrics["macro_f1"],abs_tol=1e-12):
                    raise ValueError("Hand-recomputed macro-F1 mismatch")
                predictions[(corpus,result["system"])]=saved
                models[(corpus,result["system"])]=model
    print(f"Verified {data['mode']}: five augmented/control models; identical char blocks, train lexicons, predictions and metrics")
    return data,predictions


def report(run,directory):
    data,preds=verify(run)
    if data["mode"]!="full": raise ValueError("Full run required for public report")
    comparisons={}
    for corpus in data["config"]["corpora"]:
        baseline=next(r for r in data["results"] if r["corpus"]==corpus and r["system"]=="char_control")
        for result in data["results"]:
            if result["corpus"]==corpus and result["system"]!="char_control":
                comparisons[f"{corpus}:{result['system']}"]={
                    **paired(preds[(corpus,"char_control")],preds[(corpus,result["system"])]),
                    "delta_macro_f1":result["metrics"]["macro_f1"]-baseline["metrics"]["macro_f1"]}
    negation=paired(preds[("tsac","char_lexicon")],preds[("tsac","char_lexicon_negation")])
    summary={"created_at_utc":data["created_at_utc"],"test_loaded":False,"config":data["config"],
        "rule_config":data["rule_config"],"comparisons":comparisons,"negation_vs_lexicon":negation,
        "results":data["results"],"provenance":{"run_results_sha256":digest(run/"results.json"),
            **{k:data[k] for k in ("config_sha256","rule_config_sha256","protocol_sha256_at_run","reference_runs","manifests_sha256","uv_lock_sha256","code_sha256")}}}
    lines=["# TuniScope — Caractères et indices lexicaux", "",
        "**Livrables :** L04, L07, L10 et L12, contributions partielles. Validation v2, texte brut et configurations fixées avant ce run.", "",
        "Les contrôles reproduisent exactement les prédictions historiques. Tous les exemples sont inclus ; "
        "aucune correction de label, sélection de meilleure graine ou ouverture du test officiel.", "",
        "| Corpus | Système | Macro-F1 | Accuracy | Erreurs | Colonnes ajoutées |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    for r in data["results"]:
        m=r["metrics"]; errors=m["n"]-sum(m["confusion_matrix"][i][i] for i in range(len(m["labels"])))
        lines.append(f"| {r['corpus']} | {r['system']} | {m['macro_f1']:.6f} | {m['accuracy']:.6f} | {errors} | {r['validation_coverage']['additional_features']} |")
    lines += ["", "## Comparaisons appariées avec le contrôle", "",
        "| Corpus / ajout | Écart macro-F1 | Erreurs corrigées | Erreurs introduites | Prédictions différentes |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for name,c in comparisons.items():
        lines.append(f"| {name} | {c['delta_macro_f1']:+.6f} | {c['corrected_by_right']} | {c['introduced_by_right']} | {c['different_predictions']} |")
    lines += ["", "## Négation dans le bloc lexical TSAC", "",
        f"Comparée à l'ajout lexical sans négation : {negation['different_predictions']} prédictions différentes, "
        f"erreurs corrigées : {negation['corrected_by_right']}, erreurs introduites : {negation['introduced_by_right']}.", "",
        "## Interprétation et limites", "",
        "Le bloc ajouté contient les fractions de voix par classe et un indicateur de présence d'indice : "
        "trois colonnes TSAC, quatre MADAR. Ces valeurs ne sont pas des probabilités calibrées. "
        "Le poids du bloc est fixé à 1 ; aucune normalisation globale ne modifie les colonnes caractères.", "",
        "Les lexiques restent ajustés uniquement sur train et identiques aux ressources de la baseline à règles. "
        "Le classifieur apprend à pondérer les nouvelles informations ; aucune règle n'écrase directement sa prédiction. "
        "Les associations lexicales sont supervisées et calculées sur le train qui sert aussi au classifieur : "
        "elles peuvent surajuster celui-ci. Les graphies rares, noms propres, effets de domaine et limites de portée restent présents.", "",
        "Les nouveaux modèles partagent les mêmes vocabulaires/IDF caractères, exemples, paramètres du classifieur "
        "et budget de recherche. L'ajout change néanmoins la dimension et la géométrie de l'espace. "
        "Les écarts de validation ne prouvent ni significativité, ni gain sur le test final. Une amélioration "
        "sur quelques exemples ne justifie pas une généralisation à toutes les variétés ou sources.", "",
        "## Vérification et suite", "",
        "Tests fictifs, smoke test avant run complet, checkpoints rechargés et toutes les prédictions rejouées. "
        "Les blocs sont vérifiés numériquement, les lexiques reconstruits depuis le train, les références historiques "
        "revérifiées, et le macro-F1 recalculé depuis les matrices de confusion. Les coûts et nombres de blocs nuls "
        "figurent dans summary.json ; la mémoire maximale est cumulative au processus. Aucune dépendance ou poids neuronal ajouté.", "",
        "Les notebooks, autres baselines requises, relecture humaine des études linguistiques et évaluation finale "
        "restent à réaliser. Aucun lexique, vocabulaire, checkpoint ou texte de corpus n'est publié dans ce rapport."]
    directory=directory.resolve()
    if (ROOT/"reports").resolve() not in directory.parents or directory.exists():
        raise ValueError("Choose a new report directory under reports/")
    directory.mkdir(parents=True)
    (directory/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    (directory/"README.md").write_text("\n".join(lines)+"\n")
    print(json.dumps({"comparisons":comparisons,"negation_vs_lexicon":negation},indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--report",type=Path)
    args=parser.parse_args()
    if args.report: report(args.run,args.report)
    else: verify(args.run)
