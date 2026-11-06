import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from export_encoder_error_review import select_cases


class SelectionTests(unittest.TestCase):
    def rows(self):
        return [dict(id=str(i), group_id=str(i), text_raw=f'example {i}', split='validation',
                     truth='POS', predicted='NEG', writing_system='LATIN',
                     comparison='Erreur commune' if i % 2 else 'Erreur introduite') for i in range(50)]

    def test_deterministic_and_stratified_with_unique_groups(self):
        rows = self.rows()
        chosen = select_cases(rows)
        self.assertEqual([r['id'] for r in chosen], [r['id'] for r in select_cases(list(reversed(rows)))])
        self.assertEqual(len(chosen), 30)
        self.assertEqual(len({r['group_id'] for r in chosen}), 30)
        self.assertEqual({r['comparison'] for r in chosen}, {'Erreur commune', 'Erreur introduite'})

    def test_repeated_groups_and_texts_cannot_inflate_review(self):
        rows = self.rows()
        for r in rows:
            r['text_raw'] = 'same text'
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_cases(rows)
        rows = self.rows()
        for r in rows:
            r['group_id'] = 'same group'
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            select_cases(rows)

    def test_rejects_test_or_correct_predictions(self):
        for key, value in [('split', 'test'), ('predicted', 'POS')]:
            rows = copy.deepcopy(self.rows()); rows[0][key] = value
            with self.assertRaises(ValueError):
                select_cases(rows)


if __name__ == '__main__':
    unittest.main()
