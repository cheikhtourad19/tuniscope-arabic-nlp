"""Synthetic-only tests for writing-system and percentile rules."""

from fractions import Fraction
from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_text_composition import percentile, writing_system  # noqa: E402


class TextCompositionRulesTests(unittest.TestCase):
    def test_arabic_latin_and_mixed_letters(self):
        self.assertEqual(writing_system("مرحبا"), "AR")
        self.assertEqual(writing_system("bonjour"), "LATIN")
        self.assertEqual(writing_system("école"), "LATIN")
        self.assertEqual(writing_system("مرحبا bonjour"), "MIXED")

    def test_digits_symbols_and_emoji_do_not_define_language(self):
        self.assertEqual(writing_system("😀 123 !"), "UNKNOWN")
        self.assertEqual(writing_system("7abibi"), "LATIN")

    def test_nearest_rank_percentiles(self):
        values = [1, 2, 3, 4, 5]
        self.assertEqual(percentile(values, Fraction(1, 2)), 3)
        self.assertEqual(percentile(values, Fraction(95, 100)), 5)


if __name__ == "__main__":
    unittest.main()
