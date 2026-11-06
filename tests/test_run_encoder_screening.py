"""Campaign orchestration and failure boundaries using invented child outputs only."""
import json
from io import StringIO
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_encoder_screening as workflow


class CampaignBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'docs').mkdir()
        shutil.copyfile(ROOT / 'docs/52_protocole_selection_encodeur.md', self.root / 'docs/52_protocole_selection_encodeur.md')
        self.calls = []

    def child(self, script, *args):
        args = list(map(str, args)); self.calls.append((script, args))
        if script == 'run_encoder_finetuning.py':
            target = Path(args[args.index('--output') + 1]); target.mkdir()
            # Fictional scores for checking the wrapper, not a scientific run.
            result = {'device': 'cpu', 'results': [
                {'corpus': c, 'metrics': {'macro_f1': .5}} for c in ('tsac', 'madar')]}
            (target / 'results.json').write_text(json.dumps(result))
        elif script == 'verify_encoder_finetuning.py':
            (Path(args[0]) / 'verification.json').write_text('{}')

    def test_full_checks_smoke_before_each_long_run(self):
        target = self.root / 'outputs/encoder_finetuning/campaign'
        proof = {'fixture_only': True}
        with patch.object(workflow, 'ROOT', self.root), patch.object(workflow, 'child', side_effect=self.child), \
             patch.object(workflow, 'verify_campaign', return_value=({}, proof)), patch('sys.stdout', StringIO()):
            workflow.campaign('full', target, 'cpu')
        self.assertEqual(len(self.calls), 15)
        for offset in (0, 5, 10):
            block = self.calls[offset:offset+5]
            self.assertEqual([x[0] for x in block], ['run_encoder_finetuning.py', 'verify_encoder_finetuning.py',
                'run_encoder_finetuning.py', 'verify_encoder_finetuning.py', 'plot_encoder_finetuning.py'])
            self.assertEqual(block[0][1][block[0][1].index('--mode')+1], 'smoke')
            self.assertEqual(block[2][1][block[2][1].index('--mode')+1], 'full')
            self.assertIn('--smoke-run', block[2][1])
            self.assertIn('screening', block[2][1])

    def test_smoke_verification_failure_cannot_launch_full_or_select(self):
        target = self.root / 'outputs/encoder_finetuning/failed'
        def failed(script, *args):
            self.child(script, *args)
            if script == 'verify_encoder_finetuning.py':
                raise subprocess.CalledProcessError(1, ['invented-verification'])
        with patch.object(workflow, 'ROOT', self.root), patch.object(workflow, 'child', side_effect=failed), patch('sys.stdout', StringIO()):
            with self.assertRaises(subprocess.CalledProcessError): workflow.campaign('full', target, 'cpu')
        self.assertEqual(len(self.calls), 2)
        self.assertFalse((target / 'screening.json').exists())
        self.assertFalse(json.loads((target / 'failure.json').read_text())['selection_completed'])


if __name__ == '__main__': unittest.main()
