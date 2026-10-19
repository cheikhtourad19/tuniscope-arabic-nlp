"""Scientific metric contract on a hand-computable confusion matrix."""

from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline_metrics import classification_metrics  # noqa: E402


class BaselineMetricTests(unittest.TestCase):
    def test_unpredicted_class_stays_in_macro_f1(self):
        result = classification_metrics(["NEG", "POS"], ["POS", "POS"], ["NEG", "POS"])
        self.assertAlmostEqual(result["macro_f1"], 1 / 3)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["per_class"]["NEG"]["recall"], 0)
        self.assertEqual(result["confusion_matrix"], [[0, 1], [0, 1]])

    def test_invalid_prediction_is_not_silently_dropped(self):
        with self.assertRaises(ValueError):
            classification_metrics(["POS"], ["OTHER"], ["NEG", "POS"])


if __name__ == "__main__":
    unittest.main()
