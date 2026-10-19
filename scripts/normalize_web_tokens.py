"""Optional, versioned URL/mention masking after norm-nfc-space-v1.

This is a candidate v2 for comparison, not a silent replacement of v1.
The original text and v1 output remain available in memory.
"""

import re

from normalize_text import VERSION as BASE_VERSION, normalize


VERSION = "norm-nfc-space-web-v2-candidate"
URL = re.compile(r"(?i)(?<!\w)(?:https?://|www\.)[^\s<>]+")
MENTION = re.compile(r"(?<![\w@])@[\w]+", re.UNICODE)
TRAILING_PUNCTUATION = ".,!?:;،؟؛)]}"


def mask_web_tokens(text):
    """Return (masked_text, URL_count, mention_count), preserving end punctuation."""
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    url_count = 0
    mention_count = 0

    def replace_url(match):
        nonlocal url_count
        body = match.group()
        suffix = body[len(body.rstrip(TRAILING_PUNCTUATION)):]
        core = body[:len(body) - len(suffix)] if suffix else body
        if not core:
            return body
        url_count += 1
        return "<URL>" + suffix

    without_urls = URL.sub(replace_url, text)

    def replace_mention(match):
        nonlocal mention_count
        mention_count += 1
        return "<USER>"

    return MENTION.sub(replace_mention, without_urls), url_count, mention_count


def normalize_candidate(text):
    """Apply the fixed v1 rule, then mask URL and @mention tokens."""
    return mask_web_tokens(normalize(text))[0]


def normalize_records_candidate(records):
    result = []
    for row in records:
        if row.get("normalization_version") != "raw-v0" or row.get("text_norm") != row.get("text_raw"):
            raise ValueError("Expected untouched raw-v0 records")
        base = normalize(row["text_raw"])
        masked, _, _ = mask_web_tokens(base)
        if not masked and row["split"] != "test":
            raise ValueError(f"Candidate normalization emptied {row['id']}")
        result.append({**row, "text_norm": masked, "normalization_version": VERSION})
    return result
