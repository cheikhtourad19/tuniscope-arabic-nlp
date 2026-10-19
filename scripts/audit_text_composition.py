"""Aggregate writing-system and length audit of retained train/validation text.

No source text, row IDs, or text-derived hashes are printed or written. The
frozen official test is not loaded. This is a script audit, not language ID.
"""

from collections import Counter, defaultdict
from fractions import Fraction
import json
import unicodedata

from load_working_data import load_records


def writing_system(text):
    """Classify presence of Arabic and Latin letters, ignoring digits/punctuation."""
    arabic = latin = False
    for char in text:
        if not unicodedata.category(char).startswith("L"):
            continue
        name = unicodedata.name(char, "")
        arabic |= "ARABIC" in name
        latin |= "LATIN" in name
    if arabic and latin:
        return "MIXED"
    if arabic:
        return "AR"
    if latin:
        return "LATIN"
    return "UNKNOWN"


def percentile(values, fraction):
    """Nearest-rank percentile for a nonempty sorted integer sequence."""
    position = max(0, -(-len(values) * fraction.numerator // fraction.denominator) - 1)
    return values[position]


def audit_corpus(corpus):
    records = load_records(corpus)
    counts = defaultdict(Counter)
    lengths = defaultdict(list)
    whitespace_tokens = defaultdict(list)
    for row in records:
        key = (row["split"], row["label"])
        counts[key][writing_system(row["text_raw"])] += 1
        lengths[key].append(len(row["text_raw"]))
        whitespace_tokens[key].append(len(row["text_raw"].split()))
    result = {}
    for (split, label), scripts in sorted(counts.items()):
        chars = sorted(lengths[(split, label)])
        tokens = sorted(whitespace_tokens[(split, label)])
        result[f"{split}:{label}"] = {
            "rows": len(chars),
            "writing_system": {name: scripts[name] for name in ("AR", "LATIN", "MIXED", "UNKNOWN")},
            "unicode_codepoints": {"min": chars[0], "p50": percentile(chars, Fraction(1, 2)),
                                   "p95": percentile(chars, Fraction(95, 100)), "max": chars[-1]},
            "whitespace_separated_units": {"min": tokens[0], "p50": percentile(tokens, Fraction(1, 2)),
                                            "p95": percentile(tokens, Fraction(95, 100)), "max": tokens[-1]},
        }
    if sum(item["rows"] for item in result.values()) != len(records):
        raise ValueError("Composition counts do not match loaded records")
    return result


def main():
    print(json.dumps({
        "scope": "manifest v1 retained train and validation only; official test unopened",
        "writing_system_rule": "Arabic/Latin Unicode letter presence; not language detection",
        "tsac": audit_corpus("tsac"),
        "madar": audit_corpus("madar"),
    }, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
