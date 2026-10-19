"""Fictitious-data checks for the qualitative error-review sampling rule."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_baseline_error_review import select_errors


class ErrorReviewSelectionTests(unittest.TestCase):
    def setUp(self):
        self.rows = [{"id": str(i), "group_id": f"g{i}", "text_raw": f"fictitious example {i}",
                      "truth": "POS" if i % 2 else "NEG", "predicted": "NEG" if i % 2 else "POS",
                      "writing_system": "LATIN", "split": "validation"} for i in range(16)]

    def test_deterministic_and_input_order_independent(self):
        selected = select_errors(self.rows, 6)
        self.assertEqual(selected, select_errors(self.rows[::-1], 6))
        self.assertEqual(len(selected), 6)
        self.assertEqual({r["truth"] for r in selected}, {"POS", "NEG"})

    def test_distinct_groups_and_normalized_texts(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["group_id"] = rows[0]["group_id"]
        rows[2]["text_raw"] = "  fictitious   example 0  "
        selected = select_errors(rows, 10)
        self.assertEqual(len({r["group_id"] for r in selected}), 10)
        self.assertEqual(len({" ".join(r["text_raw"].split()) for r in selected}), 10)

    def test_non_errors_and_test_rows_rejected(self):
        for key, value in (("predicted", self.rows[0]["truth"]), ("split", "test")):
            rows = copy.deepcopy(self.rows)
            rows[0][key] = value
            with self.assertRaises(ValueError):
                select_errors(rows, 4)

    def test_insufficient_or_repeated_candidates_rejected(self):
        with self.assertRaises(ValueError):
            select_errors(self.rows, 20)
        with self.assertRaises(ValueError):
            select_errors(self.rows + self.rows[:1], 4)
        with self.assertRaises(ValueError):
            select_errors(self.rows, 0)


if __name__ == "__main__":
    unittest.main()
