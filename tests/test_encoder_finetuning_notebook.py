"""Notebook06 modes and orchestration, without weights/corpus/GPU."""
import json
from io import StringIO
from pathlib import Path
import sys,subprocess,unittest
from unittest.mock import patch,MagicMock
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
import encoder_finetuning_notebook_support as support
from check_bigru_notebook import validate_bigru
from check_notebooks import execute


class FineTuningNotebookTests(unittest.TestCase):
    def test_off_structure_without_children_or_model_import(self):
        doc=json.loads((ROOT/'notebooks/06_fine_tuning_encodeur.ipynb').read_text());validate_bigru(doc,allow_saved_outputs=True)
        with patch.dict('os.environ',{'TUNISCOPE_FINETUNING_MODE':'off'}),patch.object(support,'run_tool',side_effect=AssertionError('child')):
            ns=execute(doc,'06_fine_tuning_encodeur.ipynb')
        self.assertIsNone(ns['run_result'])

    def test_smoke_is_paired_and_verified_before_return(self):
        with patch.object(support,'run_tool') as child,patch.object(support,'selected',return_value=('chosen',{})) as select:
            support.notebook_run('smoke','finetuned')
        self.assertEqual(child.call_args_list[0].args[1:5],('--mode','smoke','--regime','paired'))
        self.assertEqual([c.args[0] for c in child.call_args_list],['run_encoder_finetuning.py','verify_encoder_finetuning.py','encoder_finetuning_feasibility.py'])
        self.assertEqual(select.call_args.args[1],'smoke')

    def test_replay_explicit_expected_mode_does_not_train(self):
        with patch.object(support,'selected',return_value=('chosen',{})) as select,patch.object(support,'run_tool') as child:
            support.notebook_run('replay',replay_directory='chosen',replay_mode='smoke')
        child.assert_called_once_with('verify_encoder_finetuning.py','chosen')
        self.assertTrue(all(c.args[1]=='smoke' for c in select.call_args_list))

    def test_failed_child_raises_and_uses_offline_separate_profile(self):
        child=MagicMock();child.stdout=StringIO('invented failure\n');child.wait.return_value=1
        with patch.dict('os.environ',{'VIRTUAL_ENV':'invented','MPLBACKEND':'inline'}),patch.object(support.subprocess,'Popen',return_value=child) as launch:
            with self.assertRaises(subprocess.CalledProcessError):support.run_tool('verify_encoder_finetuning.py','invented')
        self.assertTrue(launch.call_args.args[0][1].endswith('encoder_uv.sh'))
        self.assertIn('--offline',launch.call_args.args[0]);self.assertNotIn('VIRTUAL_ENV',launch.call_args.kwargs['env'])
        self.assertEqual(launch.call_args.kwargs['env']['MPLBACKEND'],'Agg')

    def test_unexpected_script_or_mode_rejected(self):
        with self.assertRaises(ValueError):support.run_tool('run_bigru.py')
        with self.assertRaises(ValueError):support.notebook_run('whatever')


if __name__=='__main__':unittest.main()
