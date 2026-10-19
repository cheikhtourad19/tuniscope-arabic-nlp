"""Synthetic-only checks for exploratory tatweel/repeated-letter rules."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_elongation_policy import (  # noqa: E402
    TATWEEL, collapse_repeated_letters, repeated_letter_runs,
)


class ElongationPolicyTests(unittest.TestCase):
    def test_repeated_letters_but_not_digits_or_punctuation(self):
        self.assertEqual(repeated_letter_runs("باهيـــ جييييد !!! 777"), 1)
        self.assertEqual(repeated_letter_runs("soooo goooood"), 2)
        self.assertEqual(repeated_letter_runs("لا لا"), 0)

    def test_candidate_keeps_two_letters(self):
        self.assertEqual(collapse_repeated_letters("جييييد sooo !!! 777"), "جييد soo !!! 777")
        self.assertEqual(collapse_repeated_letters("باهي"), "باهي")

    def test_tatweel_is_separate_from_letter_repetition(self):
        text = "مــرحبا"
        self.assertIn(TATWEEL, text)
        self.assertEqual(collapse_repeated_letters(text), text)
        self.assertEqual(text.replace(TATWEEL, ""), "مرحبا")


if __name__ == "__main__":
    unittest.main()
