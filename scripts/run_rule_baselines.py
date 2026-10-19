"""Build train-only lexicons and evaluate five validation-only rule systems."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import time

from load_working_data import ROOT, load_records
from audit_text_composition import writing_system
from rule_baselines import fit_rules, predict_rule, validate_config
from run_baselines import digest, evaluate, peak_rss_bytes, sample_rows


def split_rows(corpus, config, mode):
    records = load_records(corpus, ("train", "validation"), manifest_version="v2")
    train = [r for r in records if r["split"] == "train"]
    validation = [r for r in records if r["split"] == "validation"]
    if {r["group_id"] for r in train} & {r["group_id"] for r in validation}:
        raise ValueError("Train/validation group overlap")
    if mode == "smoke":
        train = sample_rows(train, config["smoke_rows_per_partition"], config["seed"])
        validation = sample_rows(validation, config["smoke_rows_per_partition"], config["seed"])
    return train, validation


def systems(corpus):
    return ["majority", "lexicon", "lexicon_negation"] if corpus == "tsac" else ["majority", "lexicon"]


def predict_rows(rows, model, system):
    predictions = []
    for row in rows:
        if system == "majority":
            decision = {"predicted": model["majority"], "decisive": False, "fallback": True,
                        "lexical_hit": False, "lexical_hits": 0, "flipped_cues": 0,
                        "negation_triggers": 0, "scores": {label: 0 for label in model["labels"]}}
        else:
            decision = predict_rule(row["text_raw"], model, negation=system == "lexicon_negation")
        predictions.append({"id":row["id"], "group_id":row["group_id"], "truth":row["label"],
                            "writing_system":writing_system(row["text_raw"]), **decision})
    return predictions


def coverage(predictions):
    total = len(predictions)
    return {"rows": total,
            "lexical_hit_rows": sum(p["lexical_hit"] for p in predictions),
            "decisive_rows": sum(p["decisive"] for p in predictions),
            "decisive_fraction": sum(p["decisive"] for p in predictions) / total,
            "fallback_rows": sum(p["fallback"] for p in predictions),
            "negation_trigger_rows": sum(p["negation_triggers"] > 0 for p in predictions),
            "flipped_cue_rows": sum(p["flipped_cues"] > 0 for p in predictions)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/rule_baselines_v1.json")
    parser.add_argument("--mode", choices=("smoke", "full"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    validate_config(config)
    if config["corpora"] != ["tsac", "madar"] or config["labels"] != {"tsac":["NEG","POS"],"madar":["MSA","TN","EG"]}:
        raise ValueError("Unexpected corpus/label contract")
    target = args.output.resolve()
    if (ROOT / "outputs/baselines").resolve() not in target.parents or target.exists():
        raise ValueError("Choose a new run directory under ignored outputs/baselines/")
    target.mkdir(parents=True)
    data = {"created_at_utc":datetime.now(timezone.utc).isoformat(), "mode":args.mode,
        "config":config, "config_sha256":digest(args.config), "test_loaded":False,
        "git_commit":subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        "git_dirty":bool(subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True)),
        "device":"cpu", "python":platform.python_version(), "platform":platform.platform(),
        "packages":{name:importlib.metadata.version(name) for name in ("numpy","scikit-learn")},
        "uv_lock_sha256":digest(ROOT/"uv.lock"),
        "protocol_sha256":digest(ROOT/"docs/29_baselines_regles_lexicales.md"),
        "code_sha256":{name:digest(ROOT/"scripts"/name) for name in
            ("rule_baselines.py","run_rule_baselines.py","verify_rule_baselines.py","baseline_metrics.py","run_baselines.py")},
        "manifests_sha256":{c:digest(ROOT/f"data/manifests/v2/{c}.csv") for c in config["corpora"]},
        "models":{}, "results":[]}
    for corpus in config["corpora"]:
        train, validation = split_rows(corpus, config, args.mode)
        labels = config["labels"][corpus]
        started = time.perf_counter()
        model = fit_rules(train, labels, config)
        fit_seconds = time.perf_counter()-started
        checkpoint = target / f"{corpus}__lexicon.json"
        checkpoint.write_text(json.dumps(model,ensure_ascii=False,indent=2)+"\n")
        reloaded = json.loads(checkpoint.read_text())
        if model != reloaded:
            raise ValueError("JSON model reload mismatch")
        data["models"][corpus] = {"checkpoint":checkpoint.name,"checkpoint_sha256":digest(checkpoint),
            "fit_seconds":fit_seconds,"train_rows":len(train),"train_labels":dict(Counter(r["label"] for r in train)),
            "lexicon_size":len(model["entries"]),"lexicon_classes":dict(Counter(e["label"] for e in model["entries"])),
            "active_negation_marker_count":len(model["active_negation_markers"]),
            "active_exception_count":len(model["active_exceptions"])}
        for system in systems(corpus):
            started = time.perf_counter()
            predictions = predict_rows(validation, reloaded, system)
            elapsed = time.perf_counter()-started
            metrics = evaluate(validation,[p["predicted"] for p in predictions],labels)
            filename = target / f"{corpus}__{system}.predictions.json"
            filename.write_text(json.dumps(predictions,ensure_ascii=False,indent=2)+"\n")
            result = {"corpus":corpus,"system":system,"validation_rows":len(validation),
                "metrics":metrics,"coverage":coverage(predictions),"predict_seconds":elapsed,
                "process_peak_rss_bytes_cumulative":peak_rss_bytes(),
                "predictions":filename.name,"predictions_sha256":digest(filename)}
            data["results"].append(result)
            print(f"{corpus} {system}: macro-F1={metrics['macro_f1']:.6f}, decisive={result['coverage']['decisive_rows']}/{len(validation)}",flush=True)
    (target/"results.json").write_text(json.dumps(data,ensure_ascii=False,indent=2)+"\n")


if __name__ == "__main__":
    main()
