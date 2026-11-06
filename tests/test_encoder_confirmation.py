import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from encoder_confirmation_plan import confirmation_plan,confirmation_config
from encoder_screening_plan import effective_config,execution_seeds,validate_effective_config
from encoder_source_provenance import digest,verify_recorded_sources
from run_encoder_confirmation import aggregate


class ConfirmationTests(unittest.TestCase):
    def test_fixed_task_rates_and_three_seeds(self):
        for corpus,rate in [('tsac',5e-5),('madar',2e-5)]:
            c=confirmation_config(corpus)
            self.assertEqual(c['learning_rate'],rate);self.assertEqual(c['corpora'],[corpus])
            self.assertEqual(execution_seeds('full','confirmation',c),[13,42,2026])
            self.assertEqual(execution_seeds('smoke','confirmation',c),[42])
            baseline=effective_config()
            self.assertEqual({**c,'learning_rate':baseline['learning_rate'],'corpora':baseline['corpora']},baseline)

    def test_no_new_rate_or_task_search_in_confirmation(self):
        with self.assertRaises(ValueError):effective_config('confirmation',1e-5,'tsac')
        with self.assertRaises(ValueError):effective_config('confirmation')
        with self.assertRaises(ValueError):effective_config('initial',corpus='tsac')
        c=confirmation_config('tsac');c['learning_rate']=2e-5
        with self.assertRaises(ValueError):validate_effective_config(c,'confirmation')
        c=confirmation_config('tsac');c['corpora']=['tsac','madar']
        with self.assertRaises(ValueError):validate_effective_config(c,'confirmation')

    def test_mean_and_sample_std_include_all_three_seeds(self):
        rows=[{'seed':seed,'metrics':dict(macro_f1=x,accuracy=x,balanced_accuracy=x)}
              for seed,x in [(13,.5),(42,.7),(2026,.9)]]
        values=aggregate(rows)
        self.assertAlmostEqual(values['macro_f1']['mean'],.7)
        self.assertAlmostEqual(values['macro_f1']['sample_std'],.2)
        with self.assertRaises(ValueError):aggregate(rows[:2])
        rows[2]['seed']=42
        with self.assertRaises(ValueError):aggregate(rows)

    def test_original_source_report_is_pinned_and_not_console_only(self):
        p=confirmation_plan();report=json.loads((ROOT/p['selection_report']).read_text())
        self.assertEqual(digest(ROOT/p['selection_report']),p['selection_report_sha256'])
        self.assertEqual(report['evidence_type'],'retained-local-full-artifacts-and-replay-proofs')
        self.assertFalse(p['reuse_screening_checkpoints'])

    def test_historical_source_bundle_survives_current_code_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'scripts').mkdir();current=root/'scripts/example.py'
            current.write_text('original = True\n');record={'example.py':digest(current)}
            self.assertEqual(verify_recorded_sources(record,root),'current')
            snapshot=root/'outputs/encoder_finetuning/source_snapshots/example'
            (snapshot/'scripts').mkdir(parents=True);old=snapshot/'scripts/example.py';old.write_bytes(current.read_bytes())
            (snapshot/'manifest.json').write_text(json.dumps({'files':{'scripts/example.py':record['example.py']}}))
            current.write_text('different = True\n')
            self.assertIn('source_snapshots',verify_recorded_sources(record,root))
            old.write_text('forged = True\n')
            with self.assertRaises(ValueError):verify_recorded_sources(record,root)
            with self.assertRaises(ValueError):verify_recorded_sources({'../example.py':'x'},root)


if __name__=='__main__':unittest.main()
