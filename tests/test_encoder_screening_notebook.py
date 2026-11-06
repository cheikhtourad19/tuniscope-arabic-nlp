import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import encoder_screening_notebook_support as support
from check_notebooks import execute, validate
from check_bigru_notebook import validate_bigru


class ScreeningNotebookTests(unittest.TestCase):
    def test_default_full_and_off_execution_does_not_start_children(self):
        doc = json.loads((ROOT / 'notebooks/07_selection_encodeur.ipynb').read_text())
        validate_bigru(doc, allow_saved_outputs=True)
        with patch.dict('os.environ', {'TUNISCOPE_SCREENING_MODE': 'off'}), patch.object(support, 'run_tool', side_effect=AssertionError('child')):
            result = execute(doc, '07_selection_encodeur.ipynb')
        self.assertIsNone(result['run_result'])
        self.assertIn("'full'", ''.join(doc['cells'][2]['source']))

    def test_full_uses_campaign_tool_and_keeps_requested_mode(self):
        with patch.object(support, 'run_tool') as child, patch.object(support, 'selected', return_value=('chosen', {})) as select:
            support.notebook_run('full', 'cuda')
        self.assertEqual(child.call_args.args[:5], ('run_encoder_screening.py', '--mode', 'full', '--device', 'cuda'))
        self.assertEqual(select.call_args.args[1], 'full')

    def test_replay_checks_campaign_without_training(self):
        with patch.object(support, 'selected', return_value=(Path('chosen'), {})), patch.object(support, 'run_tool') as child:
            support.notebook_run('replay', replay_directory='chosen', replay_mode='smoke')
        child.assert_called_once_with('run_encoder_screening.py', '--verify', Path('chosen'), '--expected-mode', 'smoke')

    def test_invalid_modes_do_not_launch_training(self):
        with patch.object(support, 'run_tool') as child:
            with self.assertRaises(ValueError): support.notebook_run('whatever')
        child.assert_not_called()


if __name__ == '__main__': unittest.main()
