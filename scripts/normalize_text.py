"""Minimal, versioned and non-destructive text normalization for in-memory rows.

norm-nfc-space-v1 applies Unicode NFC and collapses whitespace runs to one
ASCII space, trimming only outer whitespace. It intentionally leaves every
other character unchanged. The original text remains in text_raw.
"""

import unicodedata


VERSION = "norm-nfc-space-v1"


def normalize(text):
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    return " ".join(unicodedata.normalize("NFC", text).split())


def normalize_records(records):
    """Return new record dictionaries; never mutate caller rows or source files."""
    result = []
    for row in records:
        if row.get("normalization_version") != "raw-v0" or row.get("text_norm") != row.get("text_raw"):
            raise ValueError("Expected untouched raw-v0 records")
        normalized = normalize(row["text_raw"])
        if not normalized and row["split"] != "test":
            raise ValueError(f"Normalization emptied retained working record {row['id']}")
        result.append({**row, "text_norm": normalized, "normalization_version": VERSION})
    return result
