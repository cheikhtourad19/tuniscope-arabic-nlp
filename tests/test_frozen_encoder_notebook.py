"""Notebook orchestration tests without pretrained weights or protected data."""
import json
from io import StringIO
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import frozen_encoder_notebook_support as bridge
from check_bigru_notebook import validate_bigru
from check_notebooks import execute


class FrozenNotebookTests(unittest.TestCase):
    def test_structure_and_off_never_loads_or_launches(self):
        document = json.loads((ROOT / 'notebooks/05_encodeur_gele.ipynb').read_text())
        validate_bigru(document, allow_saved_outputs=True)
        with patch.dict('os.environ', {'TUNISCOPE_FROZEN_MODE': 'off'}), patch.object(bridge, 'run_tool', side_effect=AssertionError('child launched')):
            namespace = execute(document, '05_encodeur_gele.ipynb')
        self.assertIsNone(namespace['run_result'])

    def test_full_requests_smoke_verification_then_full_no_fallback(self):
        with patch.object(bridge, 'run_tool') as tool, patch.object(bridge, 'select_run', return_value=('chosen', {})) as select:
            bridge.notebook_run('full')
        calls = tool.call_args_list
        self.assertEqual([c.args[0] for c in calls], ['run_frozen_encoder.py', 'verify_frozen_encoder.py', 'run_frozen_encoder.py', 'verify_frozen_encoder.py'])
        self.assertEqual(calls[0].args[1:3], ('--mode', 'smoke'))
        self.assertEqual(calls[2].args[1:3], ('--mode', 'full'))
        self.assertIn('--smoke-run', calls[2].args)
        self.assertEqual(select.call_args.args[1], 'full')

    def test_failed_subprocess_visible_and_kernel_env_removed(self):
        process = MagicMock()
        process.stdout = StringIO('invented failure\n')
        process.wait.return_value = 1
        with patch.dict('os.environ', {'VIRTUAL_ENV': 'invented', 'MPLBACKEND': 'module://inline'}), patch.object(bridge.subprocess, 'Popen', return_value=process) as launch:
            with self.assertRaises(subprocess.CalledProcessError):
                bridge.run_tool('verify_frozen_encoder.py', 'invented')
        self.assertTrue(launch.call_args.args[0][1].endswith('encoder_uv.sh'))
        environment = launch.call_args.kwargs['env']
        self.assertNotIn('VIRTUAL_ENV', environment)
        self.assertEqual(environment['MPLBACKEND'], 'Agg')

    def test_replay_never_trains(self):
        with patch.object(bridge, 'select_run', return_value=('chosen', {})), patch.object(bridge, 'run_tool') as tool:
            bridge.notebook_run('replay', 'chosen')
        tool.assert_called_once_with('verify_frozen_encoder.py', 'chosen')

    def test_wrong_modes_and_script_fail_before_execution(self):
        with self.assertRaises(ValueError):
            bridge.notebook_run('train-whatever')
        with self.assertRaises(ValueError):
            bridge.run_tool('run_bigru.py')


if __name__ == '__main__':
    unittest.main()
