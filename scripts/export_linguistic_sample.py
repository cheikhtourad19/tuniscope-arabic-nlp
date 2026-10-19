"""Select a private, qualitative 50-text train-only syntax/pragmatics sample.

Candidate markers enrich coverage, never constitute negation annotations.
No model predictions or held-out texts are read by this selector.
"""
from collections import Counter
import hashlib
import json
import re

from load_working_data import ROOT, load_records
from audit_text_composition import writing_system
from normalize_text import normalize

CONFIG = ROOT / "configs/linguistic_sample_v1.json"
OUTPUT = ROOT / "data/interim/linguistic_sample_train_v1.json"
INDEX = ROOT / "data/interim/linguistic_sample_train_v1_index.json"
# This bounded screen is intentionally imperfect (e.g. names can match it).
# Its raw matches are reviewed as text, never accepted as syntactic truth.
MARKERS = re.compile(
    r"\b(?:لا|لم|لن|ليس|ليست|لست|مش|موش|ماهوش|ماهمش|ماكانش|"
    r"ما[^\W\d_]{2,}ش|m[ao]ch|mouch|moush|m[ao][a-z0-9]{2,}(?:ch|sh))\b",
    re.IGNORECASE,
)
WORDS = re.compile(r"[^\W_]+", re.UNICODE)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def marker_candidates(text):
    return [match.group() for match in MARKERS.finditer(text)]


def select_sample(records, config):
    if config["split"] != "train" or config["manifest_version"] != "v2":
        raise ValueError("Only explicit v2 train selection is permitted")
    if any(row["split"] != "train" for row in records):
        raise ValueError("Held-out rows are forbidden")
    if len({row["id"] for row in records}) != len(records):
        raise ValueError("Duplicate source IDs")
    eligible = []
    for row in records:
        raw = row["text_raw"]
        units = len(WORDS.findall(raw))
        if config["min_word_units"] <= units <= config["max_word_units"] and len(raw) <= config["max_characters"]:
            eligible.append({**row, "writing_system": writing_system(raw),
                             "marker_candidates": marker_candidates(raw)})
    seen_groups, seen_texts, selected, counts = set(), set(), [], []
    strata = set()
    for quota in config["quotas"]:
        key = (quota["corpus"], quota["label"], quota["script"], quota["marker_candidate"])
        if key in strata or type(quota["count"]) is not int or quota["count"] < 1:
            raise ValueError("Repeated stratum or invalid quota")
        strata.add(key)
        candidates = [row for row in eligible if (row["corpus"], row["label"], row["writing_system"],
                                                    bool(row["marker_candidates"])) == key]
        candidates.sort(key=lambda row: (hashlib.sha256(f"{config['seed']}|{row['id']}".encode()).hexdigest(), row["id"]))
        chosen = []
        for row in candidates:
            group = (row["corpus"], row["group_id"])
            text = normalize(row["text_raw"])
            if group in seen_groups or text in seen_texts:
                continue
            chosen.append(row)
            seen_groups.add(group)
            seen_texts.add(text)
            if len(chosen) == quota["count"]:
                break
        if len(chosen) != quota["count"]:
            raise ValueError(f"Insufficient distinct train texts for {key}")
        selected.extend(chosen)
        counts.append({**quota, "eligible_candidates": len(candidates), "selected": len(chosen)})
    if len(selected) != sum(q["count"] for q in config["quotas"]):
        raise ValueError("Quota total mismatch")
    return selected, counts


def build_sample():
    config = json.loads(CONFIG.read_text())
    records = []
    provenance = {}
    for corpus in ("tsac", "madar"):
        records.extend({**row, "corpus": corpus} for row in
                       load_records(corpus, ("train",), manifest_version="v2"))
        provenance[corpus] = digest(ROOT / f"data/manifests/v2/{corpus}.csv")
    selected, strata = select_sample(records, config)
    if len(selected) != 50:
        raise ValueError("Expected exactly fifty texts")
    cases = [{"case_id": f"LING-{i:02d}", **row} for i, row in enumerate(selected, 1)]
    summary = {"selected": len(cases), "split_counts": dict(Counter(r["split"] for r in cases)),
               "source_label_counts": dict(Counter(f"{r['corpus']}:{r['label']}" for r in cases)),
               "script_counts": dict(Counter(r["writing_system"] for r in cases)),
               "marker_candidate_cases": sum(bool(r["marker_candidates"]) for r in cases),
               "syntax_annotations_complete": 0, "human_validated_cases": 0}
    return {"schema_version": "1.0", "config": config, "strata": strata, "summary": summary,
            "purpose": "qualitative syntax/pragmatics study; non-proportional train-only sample",
            "marker_matches_are_annotations": False, "validation_or_test_rows_loaded": 0,
            "provenance": {"config_sha256": digest(CONFIG), "selector_sha256": digest(__file_path()),
                           "manifest_sha256": provenance}, "cases": cases}


def __file_path():
    from pathlib import Path
    return Path(__file__)


def main():
    if OUTPUT.exists() or INDEX.exists():
        raise FileExistsError("Refusing to overwrite a linguistic sample")
    payload = build_sample()
    with OUTPUT.open("x") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    index = {"payload_sha256": digest(OUTPUT), "summary": payload["summary"],
             "cases": [{k: row[k] for k in ("case_id", "id", "group_id", "source_file", "source_line", "label", "split")}
                       for row in payload["cases"]]}
    with INDEX.open("x") as stream:
        json.dump(index, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
