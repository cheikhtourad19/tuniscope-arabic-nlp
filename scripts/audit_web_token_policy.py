"""Compare optional URL/mention masking with v1 on train/validation only.

Print aggregate counts; never print corpus text or row identifiers. The frozen
test is not opened, and v2 remains a candidate until the protocol is decided.
"""

from collections import Counter, defaultdict
import json

from load_working_data import load_records
from normalize_text import normalize
from normalize_web_tokens import VERSION, mask_web_tokens


def summarize(corpus):
    rows = load_records(corpus)
    counts = Counter()
    forms = defaultdict(list)
    for row in rows:
        split = row["split"]
        base = normalize(row["text_raw"])
        masked, url_count, mention_count = mask_web_tokens(base)
        counts[(split, "rows")] += 1
        counts[(split, "rows_with_url")] += bool(url_count)
        counts[(split, "url_occurrences")] += url_count
        counts[(split, "rows_with_mention")] += bool(mention_count)
        counts[(split, "mention_occurrences")] += mention_count
        counts[(split, "rows_changed_from_v1")] += masked != base
        forms[masked].append(split)
    cross = sum(len(set(splits)) > 1 for splits in forms.values())
    if cross:
        raise ValueError(f"Candidate v2 creates {cross} exact train/validation overlaps in {corpus}")
    return {split: {name: counts[(split, name)] for name in
                    ("rows", "rows_with_url", "url_occurrences", "rows_with_mention",
                     "mention_occurrences", "rows_changed_from_v1")}
            for split in ("train", "validation")} | {
        "candidate_exact_groups_crossing_train_validation": cross,
        "candidate_duplicate_groups_within_splits": sum(len(splits) > 1 for splits in forms.values()),
    }


def main():
    print(json.dumps({"version": VERSION,
                      "scope": "candidate on retained train/validation; official test unopened",
                      "tsac": summarize("tsac"), "madar": summarize("madar")},
                     ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
