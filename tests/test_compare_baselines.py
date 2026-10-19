"""Paired-comparison checks use only fictitious prediction records."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_baseline_runs import paired_summary, compatible


class PairedComparisonTests(unittest.TestCase):
    def setUp(self):
        self.left = [{"id": str(i), "group_id": str(i), "truth": truth, "predicted": pred,
                      "writing_system": "LATIN"}
                     for i, (truth, pred) in enumerate(zip(["POS", "NEG", "NEG", "POS"],
                                                         ["POS", "POS", "NEG", "NEG"]))]
        self.right = [{**r, "predicted": pred} for r, pred in
                      zip(self.left, ["NEG", "NEG", "NEG", "NEG"])]

    def test_complementarity_and_macro_f1(self):
        summary = paired_summary(self.left, self.right, ["NEG", "POS"])
        self.assertEqual(summary["character_correct_word_wrong"], 1)
        self.assertEqual(summary["word_correct_character_wrong"], 1)
        self.assertEqual(summary["both_wrong"], 1)
        self.assertEqual(summary["different_predictions"], 2)
        self.assertEqual(summary["character_errors"], 2)
        self.assertEqual(summary["word_errors"], 2)
        self.assertAlmostEqual(summary["character_macro_f1"], .5)
        self.assertAlmostEqual(summary["word_macro_f1"], 1 / 3)

    def test_reject_misalignment_missing_and_duplicate(self):
        for right in (self.right[::-1], self.right[:-1],
                      [{**r, "group_id": "changed"} for r in self.right],
                      [{**r, "truth": "POS"} for r in self.right]):
            with self.assertRaises(ValueError):
                paired_summary(self.left, right, ["NEG", "POS"])
        with self.assertRaises(ValueError):
            paired_summary(self.left * 2, self.right * 2, ["NEG", "POS"])

    def test_smoke_is_not_comparable_to_full(self):
        with self.assertRaises(ValueError):
            compatible({"mode": "smoke", "test_loaded": False},
                       {"mode": "full", "test_loaded": False})


if __name__ == "__main__":
    unittest.main()
