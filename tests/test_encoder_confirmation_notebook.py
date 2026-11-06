import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import encoder_confirmation_notebook_support as support
from check_notebooks import execute,validate
from check_bigru_notebook import validate_bigru


class ConfirmationNotebookTests(unittest.TestCase):
    def test_off_is_model_free_and_notebook_structure_valid(self):
        doc=json.loads((ROOT/'notebooks/08_confirmation_encodeur.ipynb').read_text())
        validate_bigru(doc, allow_saved_outputs=True)
        with patch.dict('os.environ',{'TUNISCOPE_CONFIRMATION_MODE':'off'}),patch.object(support,'run_tool',side_effect=AssertionError('child')):
            state=execute(doc,'08_confirmation_encodeur.ipynb')
        self.assertIsNone(state['run_result'])

    def test_full_uses_confirmation_not_screening(self):
        with patch.object(support,'run_tool') as child,patch.object(support,'selected',return_value=('chosen',{})) as selected:
            support.notebook_run('full','mps')
        self.assertEqual(child.call_args.args[:5],('run_encoder_confirmation.py','--mode','full','--device','mps'))
        self.assertEqual(selected.call_args.args[1],'full')

    def test_replay_verifies_only_and_keeps_explicit_mode(self):
        with patch.object(support,'selected',return_value=(Path('chosen'),{})),patch.object(support,'run_tool') as child:
            support.notebook_run('replay',replay_directory='chosen',replay_mode='smoke')
        child.assert_called_once_with('run_encoder_confirmation.py','--verify',Path('chosen'),'--expected-mode','smoke')

    def test_bad_mode_cannot_start_child(self):
        with patch.object(support,'run_tool') as child:
            with self.assertRaises(ValueError):support.notebook_run('unexpected')
        child.assert_not_called()


if __name__=='__main__':unittest.main()
