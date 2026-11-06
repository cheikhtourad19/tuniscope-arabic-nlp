"""Exercise confirmation ordering and failure boundaries with invented outputs only."""
from io import StringIO
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_encoder_confirmation as workflow


class ConfirmationWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / 'docs').mkdir()
        shutil.copyfile(ROOT / 'docs/55_protocole_confirmation_encodeur.md',
                        self.root / 'docs/55_protocole_confirmation_encodeur.md')
        self.calls = []

    def child(self, script, *args):
        args = list(map(str, args))
        self.calls.append((script, args))
        if script == 'run_encoder_finetuning.py':
            target = Path(args[args.index('--output') + 1]); target.mkdir()
            corpus = args[args.index('--corpus') + 1]
            # Fictional metrics for wrapper tests, never scientific evidence.
            (target / 'results.json').write_text(json.dumps({'results': [
                {'corpus': corpus, 'seed': seed, 'metrics': {
                    'macro_f1': .5, 'accuracy': .5, 'balanced_accuracy': .5}}
                for seed in (13, 42, 2026)]}))
        elif script == 'verify_encoder_finetuning.py':
            (Path(args[0]) / 'verification.json').write_text('{}')

    def test_full_runs_two_task_preflights_then_all_seeds(self):
        target = self.root / 'outputs/encoder_finetuning/example'
        with patch.object(workflow, 'ROOT', self.root), \
             patch.object(workflow, 'child', side_effect=self.child), \
             patch.object(workflow, 'verify_campaign', return_value=({}, {'fixture_only': True})), \
             patch.object(workflow, 'export_evidence', return_value=target / 'fixture.zip'), \
             patch('sys.stdout', StringIO()):
            # Only supply the summary consumed by the wrapper, not a result claim.
            def summary(folder, metadata):
                (folder / 'summary.md').write_text('Fictional wrapper fixture')
            with patch.object(workflow, 'write_report', side_effect=summary):
                workflow.campaign('full', target, 'cpu')
        self.assertEqual(len(self.calls), 10)
        for offset, corpus in ((0, 'tsac'), (5, 'madar')):
            block = self.calls[offset:offset + 5]
            self.assertEqual([x[0] for x in block], [
                'run_encoder_finetuning.py', 'verify_encoder_finetuning.py',
                'run_encoder_finetuning.py', 'verify_encoder_finetuning.py',
                'plot_encoder_finetuning.py'])
            self.assertEqual(block[0][1][block[0][1].index('--mode') + 1], 'smoke')
            self.assertEqual(block[2][1][block[2][1].index('--mode') + 1], 'full')
            self.assertEqual(block[2][1][block[2][1].index('--corpus') + 1], corpus)
            self.assertIn('--smoke-run', block[2][1])
            self.assertIn('confirmation', block[2][1])
            self.assertNotIn('--learning-rate', block[2][1])

    def test_failed_preflight_stops_before_full_or_final_report(self):
        target = self.root / 'outputs/encoder_finetuning/failure'
        def failed(script, *args):
            self.child(script, *args)
            if script == 'verify_encoder_finetuning.py':
                raise subprocess.CalledProcessError(1, ['invented-verifier'])
        with patch.object(workflow, 'ROOT', self.root), \
             patch.object(workflow, 'child', side_effect=failed), \
             patch.object(workflow, 'write_report') as report, \
             patch.object(workflow, 'export_evidence') as archive, \
             patch('sys.stdout', StringIO()):
            with self.assertRaises(subprocess.CalledProcessError):
                workflow.campaign('full', target, 'cpu')
        self.assertEqual(len(self.calls), 2)
        report.assert_not_called(); archive.assert_not_called()
        self.assertFalse((target / 'confirmation.json').exists())
        self.assertFalse(json.loads((target / 'failure.json').read_text())['confirmation_completed'])

    def test_full_cards_and_archive_keep_all_seeds_and_exclude_weights(self):
        directory = self.root / 'report'; directory.mkdir()
        stage = directory / 'tsac_full'; stage.mkdir()
        rows = [{'seed': seed, 'corpus': 'tsac', 'parameters': 100, 'trainable_parameters': 100,
                 'best_epoch': 1, 'checkpoint': f'example_{seed}',
                 'metrics': {'macro_f1': .5, 'accuracy': .5, 'balanced_accuracy': .5}}
                for seed in (13, 42, 2026)]
        (stage / 'results.json').write_text(json.dumps({'results': rows, 'config': {
            'learning_rate': 5e-5, 'model_id': 'invented-test-only', 'revision': 'invented'}}))
        metadata = {'mode': 'full', 'entries': [{'corpus': 'tsac', 'full': {'directory': 'tsac_full'}}],
                    'aggregates': {'tsac': workflow.aggregate(rows)}}
        workflow.write_report(directory, metadata)
        card = json.loads((directory / 'model_card_tsac.json').read_text())
        self.assertEqual(len(card['checkpoint_references']), 3)
        self.assertFalse(card['test_evaluated'])
        self.assertEqual(card['validation_aggregates']['macro_f1']['sample_std'], 0)
        for seed in (13, 42, 2026):
            self.assertIn(f'| {seed} |', (directory / 'summary.md').read_text())
        (stage / 'model.safetensors').write_bytes(b'invented-weight-placeholder')
        (stage / 'head.pt').write_bytes(b'invented-weight-placeholder')
        archive = workflow.export_evidence(directory)
        with ZipFile(archive) as contents:
            names = contents.namelist()
            self.assertIn('model_card_tsac.json', names)
            self.assertIn('tsac_full/results.json', names)
            self.assertIn('project/configs/camelbert_confirmation_v1.json', names)
            self.assertFalse(any(n.endswith(('.pt', '.safetensors')) for n in names))


if __name__ == '__main__':
    unittest.main()
