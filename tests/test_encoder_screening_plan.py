import copy
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from encoder_screening_plan import effective_config, execution_seeds, select_candidates, validate_effective_config


class ScreeningPlanTests(unittest.TestCase):
    def candidates(self):
        return [dict(learning_rate=rate, mode='full', experiment_scope='screening', test_loaded=False,
                     device='cuda', seed=42, scores=dict(tsac=t, madar=m))
                for rate, t, m in [(1e-5, .90, .92), (2e-5, .91, .93), (5e-5, .91, .94)]]

    def test_initial_full_retains_three_seeds_and_screening_only_one(self):
        initial = effective_config()
        self.assertEqual(execution_seeds('full', 'initial', initial), [13, 42, 2026])
        for rate in [1e-5, 2e-5, 5e-5]:
            c = effective_config('screening', rate)
            self.assertEqual(execution_seeds('full', 'screening', c), [42])
            self.assertEqual({**c, 'learning_rate': initial['learning_rate']}, initial)

    def test_only_rate_can_change_and_no_implicit_initial_override(self):
        with self.assertRaises(ValueError): effective_config('initial', 2e-5)
        with self.assertRaises(ValueError): effective_config('screening', 1e-4)
        for field, value in [('max_length', 256), ('seeds', [42]), ('manifest_version', 'v1')]:
            c = effective_config('screening', 1e-5); c[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): validate_effective_config(c, 'screening')

    def test_task_specific_selection_and_exact_tie_break(self):
        self.assertEqual(select_candidates(self.candidates()), {'tsac': 2e-5, 'madar': 5e-5})
        self.assertEqual(select_candidates(list(reversed(self.candidates()))), {'tsac': 2e-5, 'madar': 5e-5})

    def test_smoke_partial_duplicate_or_mixed_device_cannot_select(self):
        with self.assertRaises(ValueError): select_candidates(self.candidates()[:2])
        rows = self.candidates(); rows[0] = copy.deepcopy(rows[1])
        with self.assertRaises(ValueError): select_candidates(rows)
        for field, value in [('mode', 'smoke'), ('device', 'mps'), ('seed', 13), ('test_loaded', True)]:
            rows = self.candidates(); rows[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError): select_candidates(rows)

    def test_nonfinite_or_missing_scores_rejected(self):
        for scores in [dict(tsac=float('nan'), madar=.9), dict(tsac=.9), dict(tsac=.9, madar=float('inf'))]:
            rows = self.candidates(); rows[0]['scores'] = scores
            with self.assertRaises(ValueError): select_candidates(rows)


if __name__ == '__main__': unittest.main()
