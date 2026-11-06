import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from compare_encoder_errors import compare_predictions


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.reference = [dict(id=str(i), group_id=str(i), truth='POS', predicted=p)
                          for i, p in enumerate(['POS', 'NEG', 'POS', 'NEG'])]
        self.encoder = [dict(r, predicted=p) for r, p in
                        zip(self.reference, ['POS', 'POS', 'NEG', 'NEG'])]

    def test_all_four_outcomes_and_error_conservation(self):
        result = compare_predictions(self.reference, self.encoder)
        self.assertEqual(result, dict(both_correct=1, corrected=1, introduced=1,
                                      both_wrong=1, rows=4, baseline_errors=2, encoder_errors=2))

    def test_rejects_misaligned_identity_or_truth(self):
        for key in ('id', 'group_id', 'truth'):
            with self.subTest(key=key):
                rows = copy.deepcopy(self.encoder)
                rows[0][key] = 'different'
                with self.assertRaisesRegex(ValueError, 'mismatch'):
                    compare_predictions(self.reference, rows)

    def test_rejects_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, 'Repeated'):
            compare_predictions(self.reference * 2, self.encoder * 2)

    def test_rejects_missing_predictions(self):
        with self.assertRaises(ValueError):
            compare_predictions(self.reference, self.encoder[:-1])
        with self.assertRaises(ValueError):
            compare_predictions([], [])


if __name__ == '__main__':
    unittest.main()
