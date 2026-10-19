"""Train raw-v2 character controls and lexical-feature augmentations on CPU."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

from load_working_data import ROOT
from audit_text_composition import writing_system
from character_rule_models import coverage, fit_model, predict_rows, systems, validate_config
from run_baselines import digest, evaluate, peak_rss_bytes
from run_rule_baselines import split_rows


def reference_directories(mode):
    return {"character":ROOT/f"outputs/baselines/2026-10-17_grouped_v2_{mode}",
            "rules":ROOT/f"outputs/baselines/2026-10-17_rules_v1_{mode}"}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/baselines_char_rules_v1.json")
    parser.add_argument("--mode",choices=("smoke","full"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    config=json.loads(args.config.read_text())
    refs=reference_directories(args.mode)
    reference=json.loads((refs["character"]/"results.json").read_text())
    previous_rules=json.loads((refs["rules"]/"results.json").read_text())
    validate_config(config,reference["config"])
    rule_path=(ROOT/config["rule_config"]).resolve()
    if rule_path!=(ROOT/"configs/rule_baselines_v1.json").resolve():
        raise ValueError("Use the frozen rule configuration")
    rule_config=json.loads(rule_path.read_text())
    if rule_config!=previous_rules["config"]:
        raise ValueError("Historical rule configuration differs")
    target=args.output.resolve()
    if (ROOT/"outputs/baselines").resolve() not in target.parents or target.exists():
        raise ValueError("Choose a new run directory under outputs/baselines/")
    target.mkdir(parents=True)
    data={"created_at_utc":datetime.now(timezone.utc).isoformat(),"mode":args.mode,
        "config":config,"rule_config":rule_config,"config_sha256":digest(args.config),
        "rule_config_sha256":digest(rule_path),"test_loaded":False,"device":"cpu",
        "python":platform.python_version(),"platform":platform.platform(),"architecture":platform.machine(),
        "packages":{n:importlib.metadata.version(n) for n in ("numpy","pandas","scipy","scikit-learn","joblib","threadpoolctl")},
        "uv_lock_sha256":digest(ROOT/"uv.lock"),"protocol_sha256_at_run":digest(ROOT/"docs/30_caracteres_et_indices_lexicaux.md"),
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True)),
        "code_sha256":{n:digest(ROOT/"scripts"/n) for n in ("character_rule_models.py","run_character_rules.py",
            "verify_character_rules.py","rule_baselines.py","baseline_models.py","run_baselines.py","run_rule_baselines.py","baseline_metrics.py")},
        "reference_runs":{name:{"directory":str(directory.relative_to(ROOT)),"results_sha256":digest(directory/"results.json")}
                          for name,directory in refs.items()},
        "manifests_sha256":{c:digest(ROOT/f"data/manifests/v2/{c}.csv") for c in config["corpora"]},"results":[]}
    with threadpool_limits(limits=config["threads"]):
        for corpus in config["corpora"]:
            train,validation=split_rows(corpus,config,args.mode)
            historical_rule=json.loads((refs["rules"]/previous_rules["models"][corpus]["checkpoint"]).read_text())
            previous=next(r for r in reference["results"] if (r["corpus"],r["system"],r["variant"])==(corpus,"tfidf_char","raw-v0"))
            original=joblib.load(refs["character"]/previous["checkpoint"])
            for system in systems(corpus):
                started=time.perf_counter()
                with warnings.catch_warnings():
                    warnings.simplefilter("error",ConvergenceWarning)
                    model=fit_model(train,corpus,system,config,rule_config)
                elapsed=time.perf_counter()-started
                if model["rules"]!=historical_rule:
                    raise ValueError("Lexicon is not identical to historical train-derived reference")
                if model["vectorizer"].vocabulary_!=original["tfidf"].vocabulary_ or not np.array_equal(model["vectorizer"].idf_,original["tfidf"].idf_):
                    raise ValueError("Character vocabulary/IDF differ from reference")
                if system=="char_control" and predict_rows(model,validation)!=original.predict([r["text_raw"] for r in validation]).tolist():
                    raise ValueError("Control does not reproduce historical character predictions")
                checkpoint=target/f"{corpus}__{system}.joblib"
                joblib.dump(model,checkpoint)
                reloaded=joblib.load(checkpoint)
                started=time.perf_counter()
                predicted=predict_rows(reloaded,validation)
                predict_seconds=time.perf_counter()-started
                if predicted!=predict_rows(model,validation):
                    raise ValueError("Reloaded predictions changed")
                rows=[{"id":r["id"],"group_id":r["group_id"],"truth":r["label"],"predicted":p,
                       "writing_system":writing_system(r["text_raw"])} for r,p in zip(validation,predicted)]
                pred_path=target/f"{corpus}__{system}.predictions.json"
                pred_path.write_text(json.dumps(rows,ensure_ascii=False,indent=2)+"\n")
                metrics=evaluate(validation,predicted,config["labels"][corpus])
                result={"corpus":corpus,"system":system,"train_rows":len(train),"validation_rows":len(validation),
                    "train_labels":model["train_labels"],"feature_count":model["feature_count"],
                    "train_coverage":coverage(model,train),"validation_coverage":coverage(model,validation),
                    "fit_seconds":elapsed,"predict_seconds":predict_seconds,
                    "process_peak_rss_bytes_cumulative":peak_rss_bytes(),"metrics":metrics,
                    "checkpoint":checkpoint.name,"checkpoint_sha256":digest(checkpoint),
                    "predictions":pred_path.name,"predictions_sha256":digest(pred_path)}
                data["results"].append(result)
                print(f"{corpus} {system}: macro-F1={metrics['macro_f1']:.6f}, features={model['feature_count']}",flush=True)
    (target/"results.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")


if __name__=="__main__":
    main()
