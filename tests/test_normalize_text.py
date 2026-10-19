"""Synthetic-only tests for conservative normalization."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from normalize_text import VERSION, normalize, normalize_records  # noqa: E402


class NormalizationTests(unittest.TestCase):
    def test_nfc_and_whitespace_only(self):
        self.assertEqual(normalize("  cafe\u0301\t\nمَرْحَبًا  "), "café مَرْحَبًا")
        self.assertEqual(normalize("a\u00a0\u00a0b"), "a b")

    def test_semantic_markers_are_preserved(self):
        sample = "موش behi 😄 3andi 7keya service!!! ـ"
        self.assertEqual(normalize(sample), sample)
        self.assertEqual(normalize("لا not no 3 7 😂"), "لا not no 3 7 😂")

    def test_copy_preserves_raw_and_input(self):
        original = {"id": "sample", "split": "train", "text_raw": " a\t b ",
                    "text_norm": " a\t b ", "normalization_version": "raw-v0"}
        result = normalize_records([original])[0]
        self.assertEqual(result["text_raw"], original["text_raw"])
        self.assertEqual(result["text_norm"], "a b")
        self.assertEqual(result["normalization_version"], VERSION)
        self.assertEqual(original["normalization_version"], "raw-v0")

    def test_previously_normalized_input_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_records([{"id": "sample", "split": "train", "text_raw": "x",
                                "text_norm": "x", "normalization_version": VERSION}])


if __name__ == "__main__":
    unittest.main()
