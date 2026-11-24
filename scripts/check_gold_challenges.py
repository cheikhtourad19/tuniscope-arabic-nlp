"""Check the structure and quotas of private, human-sourced Gold challenge files."""

import argparse
from collections import Counter
import json
from pathlib import Path

from load_working_data import load_records
from normalize_text import normalize


CATEGORIES = {"sentiment": ("AR", "ARABIZI", "MIXED_LANG"),
              "dialect": ("MSA", "TN", "EG")}
LABELS = {"sentiment": ("POS", "NEG"), "dialect": CATEGORIES["dialect"]}
FIELDS = {"id", "group_id", "text", "label", "category", "origin", "source_ref",
          "permission_ref", "review_ref"}


def key(text):
    return normalize(text).casefold()


def check(path, task, working_keys):
    seen_ids, seen_texts = set(), set()
    categories, labels = Counter(), Counter()
    ambiguous = 0
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict) or set(row) != FIELDS:
                    raise ValueError("champs manquants ou inconnus")
                if not all(isinstance(value, str) for value in row.values()):
                    raise ValueError("tous les champs doivent être des chaînes")
                if not all(row[field].strip() for field in FIELDS - {"review_ref"}):
                    raise ValueError("champ obligatoire vide")
                identifier, text = row["id"], key(row["text"])
                if identifier in seen_ids or text in seen_texts:
                    raise ValueError("identifiant ou texte dupliqué dans le fichier")
                if text in working_keys:
                    raise ValueError("texte déjà présent dans le train ou la validation")
                seen_ids.add(identifier)
                seen_texts.add(text)
                origins = ("human",) if task == "sentiment" else ("human", "authorized_resource")
                if row["origin"] not in origins:
                    raise ValueError("origine incompatible avec le jeu")
                if row["category"] not in CATEGORIES[task]:
                    raise ValueError("catégorie inconnue")
                label = row["label"]
                if label == "AMB" and task == "sentiment":
                    ambiguous += 1
                    continue
                if label not in LABELS[task] or (task == "dialect" and label != row["category"]):
                    raise ValueError("label incompatible avec la tâche")
                if task == "dialect" and label == "EG" and row["origin"] == "human" and not row["review_ref"].strip():
                    raise ValueError("indiquer la revue compétente du label EG")
                categories[row["category"]] += 1
                labels[label] += 1
            except (ValueError, TypeError) as error:
                raise ValueError(f"{task} ligne {line_no} · {error}") from None
    expected = 100 if task == "sentiment" else 50
    target_met = all(categories[c] == expected for c in CATEGORIES[task])
    if task == "sentiment":
        target_met = target_met and all(labels[label] >= 100 for label in LABELS[task])
    return {"main_rows": sum(categories.values()), "ambiguous_reserve": ambiguous,
            "categories": {c: categories[c] for c in CATEGORIES[task]},
            "labels": {label: labels[label] for label in LABELS[task]}, "quota_met": target_met}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--sentiment", required=True, type=Path)
    cli.add_argument("--dialect", required=True, type=Path)
    cli.add_argument("--compare-working", action="store_true", help="Check normalized matches against train/validation")
    options = cli.parse_args()
    # Provenance references are checked for presence, then verified by a reviewer.
    working = set()
    if options.compare_working:
        for corpus in ("tsac", "madar"):
            working.update(key(row["text_raw"]) for row in
                           load_records(corpus, splits=("train", "validation"), manifest_version="v2"))
    report = {task: check(getattr(options, task), task, working) for task in CATEGORIES}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(item["quota_met"] for item in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
