"""Read-only audit of tatweel and repeated-letter normalization candidates.

Only retained train/validation is read. No text, row IDs or text hashes leave
memory. Candidate transformations are not adopted by this audit.
"""

from collections import Counter, defaultdict
import json
import unicodedata

from load_working_data import load_records
from normalize_text import normalize


TATWEEL = "\u0640"


def is_arabic_or_latin_letter(char):
    return char != TATWEEL and unicodedata.category(char).startswith("L") and any(
        script in unicodedata.name(char, "") for script in ("ARABIC", "LATIN")
    )


def repeated_letter_runs(text):
    """Count exact same-letter runs of length >= 3 (not digits or punctuation)."""
    count = 0
    previous = None
    run = 0
    for char in text:
        run = run + 1 if char == previous else 1
        if run == 3 and is_arabic_or_latin_letter(char):
            count += 1
        previous = char
    return count


def collapse_repeated_letters(text):
    """Exploratory only: reduce runs of >= 3 identical letters to two."""
    result = []
    previous = None
    run = 0
    for char in text:
        run = run + 1 if char == previous else 1
        if run <= 2 or not is_arabic_or_latin_letter(char):
            result.append(char)
        previous = char
    return "".join(result)


def candidate_summary(rows, transform):
    counts = Counter()
    forms = defaultdict(list)
    for row in rows:
        split, label = row["split"], row["label"]
        base = normalize(row["text_raw"])
        altered = transform(base)
        counts[(split, "changed_rows")] += altered != base
        counts[(split, label, "changed_rows")] += altered != base
        forms[altered].append((split, label))
    return {
        "changed_rows": {split: counts[(split, "changed_rows")] for split in ("train", "validation")},
        "changed_rows_by_label": {
            f"{split}:{label}": counts[(split, label, "changed_rows")]
            for split in ("train", "validation")
            for label in sorted({row["label"] for row in rows})
        },
        "exact_groups_crossing_train_validation": sum(
            len({split for split, _ in group}) > 1 for group in forms.values()
        ),
        "duplicate_groups_within_splits": sum(
            len(group) > 1 and len({split for split, _ in group}) == 1 for group in forms.values()
        ),
    }


def summarize(corpus):
    rows = load_records(corpus)
    incidence = Counter()
    runs = Counter()
    for row in rows:
        key = (row["split"], row["label"])
        base = normalize(row["text_raw"])
        incidence[key] += TATWEEL in base
        runs[key] += bool(repeated_letter_runs(base))
    return {
        "rows_with_tatweel": {f"{split}:{label}": incidence[(split, label)]
                              for split in ("train", "validation")
                              for label in sorted({row["label"] for row in rows})},
        "rows_with_3plus_identical_letter_run": {
            f"{split}:{label}": runs[(split, label)]
            for split in ("train", "validation")
            for label in sorted({row["label"] for row in rows})
        },
        "remove_tatweel_candidate": candidate_summary(rows, lambda text: text.replace(TATWEEL, "")),
        "collapse_letter_runs_candidate": candidate_summary(rows, collapse_repeated_letters),
    }


def main():
    print(json.dumps({
        "scope": "retained train/validation after NFC/spacing v1; frozen test unopened",
        "candidate_only": True,
        "tsac": summarize("tsac"),
        "madar": summarize("madar"),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
