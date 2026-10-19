"""Audit norm-nfc-space-v1 on train/validation in RAM, printing aggregates only.

No corpus text, record IDs, or text-derived hashes are emitted. The frozen
test is not loaded; the normalization rule is predeclared in normalize_text.py.
"""

from collections import Counter, defaultdict
import json
import unicodedata

from load_working_data import load_records
from normalize_text import VERSION, normalize_records


def summarize(corpus):
    raw = load_records(corpus)
    normalized = normalize_records(raw)
    if len(raw) != len(normalized):
        raise ValueError("Normalization changed the number of records")
    counts = Counter()
    forms = defaultdict(list)
    for before, after in zip(raw, normalized):
        split = before["split"]
        if after["text_raw"] != before["text_raw"] or before["text_norm"] != before["text_raw"]:
            raise ValueError("Raw text changed")
        if after["text_norm"] != before["text_raw"]:
            counts[(split, "changed_total")] += 1
        if unicodedata.normalize("NFC", before["text_raw"]) != before["text_raw"]:
            counts[(split, "nfc_changed")] += 1
        if " ".join(before["text_raw"].split()) != before["text_raw"]:
            counts[(split, "whitespace_changed")] += 1
        forms[after["text_norm"]].append((split, before["label"], before["group_id"]))
        counts[(split, "rows")] += 1
    cross_split = [items for items in forms.values() if len({split for split, _, _ in items}) > 1]
    if cross_split:
        raise ValueError(f"Normalization creates {len(cross_split)} cross-split exact text groups in {corpus}")
    return {
        split: {key: counts[(split, key)] for key in
                ("rows", "changed_total", "nfc_changed", "whitespace_changed")}
        for split in ("train", "validation")
    } | {
        "normalized_exact_groups_crossing_train_validation": 0,
        "normalized_duplicate_groups_within_splits": sum(len(items) > 1 for items in forms.values()),
        "normalized_groups_with_multiple_labels": sum(
            len({label for _, label, _ in items}) > 1 for items in forms.values()
        ),
        "raw_preserved": True,
    }


def main():
    print(json.dumps({
        "version": VERSION,
        "scope": "train and validation in RAM; frozen test unopened",
        "tsac": summarize("tsac"),
        "madar": summarize("madar"),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
