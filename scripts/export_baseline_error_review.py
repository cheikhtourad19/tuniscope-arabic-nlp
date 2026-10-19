"""Prepare private human error-review inputs from verified v2 validation only.

No semantic causes are inferred, no labels changed, no model fitted. A bounded
qualitative sample balances confusion/script strata, not population prevalence.
"""
from collections import Counter, defaultdict, deque
import hashlib
import json

from load_working_data import ROOT, load_records
from normalize_text import normalize
from run_baselines import digest
from verify_baseline_run import verify


RUN = ROOT / "outputs/baselines/2026-10-17_grouped_v2_full"
DIRECTORY = ROOT / "outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4"
OUTPUT = DIRECTORY / "baseline_error_review_2026-10-17.json"
INDEX = ROOT / "data/interim/baseline_error_review_index_2026-10-17.json"
SEED = 42
PER_TASK = 30
LABEL_OPINIONS = ["Label plausible", "Label discutable", "Incertain", "Non interprétable"]
CATEGORIES = {
    "tsac": ["Négation", "Avis mixte / ambigu", "Ironie / sarcasme", "Orthographe / Arabizi",
             "Mélange de langues", "Expression inconnue", "Manque de contexte", "Émoji / symbole", "Autre", "Incertain"],
    "madar": ["Variétés proches / texte commun", "Texte très court", "Orthographe / morphologie",
              "Lexique partagé", "Expression inconnue", "Manque de contexte", "Autre", "Incertain"],
}


def stratum(row):
    return row["truth"], row["predicted"], row["writing_system"]


def select_errors(rows, count=PER_TASK, seed=SEED):
    """Deterministic rare-stratum-first round robin, unique groups and NFC texts."""
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate review candidate ID")
    if any(r["truth"] == r["predicted"] or r["split"] != "validation" for r in rows):
        raise ValueError("Review candidates must be validation errors")
    if count < 1:
        raise ValueError("Positive review size required")
    buckets = defaultdict(list)
    for row in rows:
        buckets[stratum(row)].append(row)
    queues = {}
    for key, bucket in buckets.items():
        queues[key] = deque(sorted(bucket, key=lambda r: hashlib.sha256(
            f"{seed}|{r['id']}".encode()).hexdigest()))
    order = sorted(queues, key=lambda key: (len(queues[key]), key))
    selected, groups, texts = [], set(), set()
    while len(selected) < count:
        progress = False
        for key in order:
            while queues[key]:
                row = queues[key].popleft()
                text_key = normalize(row["text_raw"])
                if row["group_id"] in groups or text_key in texts:
                    continue
                selected.append(row)
                groups.add(row["group_id"])
                texts.add(text_key)
                progress = True
                break
            if len(selected) == count:
                break
        if not progress:
            raise ValueError("Insufficient distinct groups/texts for requested review size")
    return selected


def counts(rows):
    return {"/".join(key): value for key, value in sorted(Counter(stratum(r) for r in rows).items())}


def build_review():
    verify(RUN)
    metadata = json.loads((RUN / "results.json").read_text())
    if metadata["mode"] != "full" or metadata["config"].get("manifest_version") != "v2":
        raise ValueError("Expected full v2 reference run")
    tasks = {}
    for corpus, expected_errors in (("tsac", 126), ("madar", 39)):
        result = next(r for r in metadata["results"] if
                      (r["corpus"], r["system"], r["variant"]) == (corpus, "tfidf_char", "raw-v0"))
        predictions = json.loads((RUN / result["predictions"]).read_text())
        records = load_records(corpus, ("validation",), manifest_version="v2")
        if [p["id"] for p in predictions] != [r["id"] for r in records]:
            raise ValueError("Prediction/source alignment mismatch")
        candidates = []
        for prediction, record in zip(predictions, records):
            if prediction["truth"] != record["label"] or prediction["group_id"] != record["group_id"]:
                raise ValueError("Reference truth/group mismatch")
            if prediction["predicted"] != prediction["truth"]:
                candidates.append({**record, **prediction})
        if len(candidates) != expected_errors:
            raise ValueError("Reference error count changed")
        chosen = select_errors(candidates)
        cases = [{"case_id": f"{corpus.upper()}-{i:02d}", "id": row["id"],
                  "group_id": row["group_id"], "file": row["source_file"], "line": row["source_line"],
                  "source_id": row["source_id"], "split": "validation", "truth": row["truth"],
                  "predicted": row["predicted"], "writing_system": row["writing_system"],
                  "text_raw": row["text_raw"], "label_opinion": "", "category": "", "hypothesis": ""}
                 for i, row in enumerate(chosen, 1)]
        tasks[corpus] = {
            "source_predictions_sha256": digest(RUN / result["predictions"]),
            "manifest_sha256": metadata["manifests_sha256"][corpus],
            "summary": {"candidate_errors": len(candidates),
                        "candidate_groups": len({r["group_id"] for r in candidates}),
                        "selected_cases": len(cases), "selected_groups": len({r["group_id"] for r in cases}),
                        "population_strata": counts(candidates), "selected_strata": counts(chosen),
                        "max_selected_text_characters": max(len(r["text_raw"]) for r in cases)},
            "categories": CATEGORIES[corpus], "cases": cases,
        }
    return {
        "schema_version": "1.0", "model_system": "tfidf_char", "text_variant": "raw-v0",
        "manifest_version": "v2", "selection_seed": SEED,
        "selection_rule": "rare-stratum-first round robin; SHA256(seed|id) within strata; unique group and NFC-space text",
        "purpose": "qualitative validation diagnosis, not category prevalence or independent relabelling",
        "labels_and_predictions_visible": True, "human_decisions_prefilled": 0,
        "official_test_rows_loaded": 0, "label_opinions": LABEL_OPINIONS,
        "provenance": {"run_results_sha256": digest(RUN / "results.json"),
                       "export_script_sha256": digest(ROOT / "scripts/export_baseline_error_review.py")},
        "tasks": tasks,
    }


def main():
    if OUTPUT.exists() or INDEX.exists():
        raise FileExistsError("Refusing to overwrite existing error-review inputs")
    payload = build_review()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    INDEX.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Omit texts in this private reference ledger; no human decisions are inferred.
    index = {**payload, "payload_sha256": digest(OUTPUT), "tasks": {
        corpus: {**task, "cases": [{k: v for k, v in case.items() if k != "text_raw"}
                                   for case in task["cases"]]}
        for corpus, task in payload["tasks"].items()}}
    INDEX.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({corpus: task["summary"] for corpus, task in payload["tasks"].items()}, indent=2))


if __name__ == "__main__":
    main()
