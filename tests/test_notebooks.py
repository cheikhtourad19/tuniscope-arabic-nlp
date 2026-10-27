"""Fictitious-only tests for notebook safeguards and aggregate reporting."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import notebook_support as support
from check_notebooks import NOTEBOOKS, ROOT, validate


class NotebookTests(unittest.TestCase):
    def test_clean_notebook_structure(self):
        for name in NOTEBOOKS:
            validate(json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8")))

    def test_outputs_are_rejected(self):
        document = json.loads((ROOT / "notebooks" / NOTEBOOKS[0]).read_text(encoding="utf-8"))
        cell = next(c for c in document["cells"] if c["cell_type"] == "code")
        cell["outputs"] = [{"text": "not publishable"}]
        with self.assertRaises(ValueError):
            validate(document)

    def test_synthetic_overview_does_not_load_corpora(self):
        with patch("load_working_data.load_records", side_effect=AssertionError("corpus read")):
            result = support.data_overview("synthetic")
        self.assertFalse(result["corpus_read"])
        summary = result["corpora"]["invented"]
        self.assertEqual(summary["dataframe_shape"], [4, 17])
        self.assertEqual(summary["partitions"]["train"]["normalization_changed_rows"], 1)
        self.assertEqual(summary["shared_train_validation_groups"], 0)
        serialized = json.dumps(result, ensure_ascii=False)
        for row in support.synthetic_records():
            self.assertNotIn(row["text_raw"], serialized)
            self.assertNotIn(row["id"], serialized)

    def test_raw_copy_and_group_guard(self):
        records = support.synthetic_records()
        before = copy.deepcopy(records)
        self.assertTrue(support.summarize_records(records)["raw_input_preserved"])
        self.assertEqual(records, before)
        records[2]["group_id"] = records[0]["group_id"]
        with self.assertRaises(ValueError):
            support.summarize_records(records)

    def test_test_rows_are_rejected(self):
        records = support.synthetic_records()
        records[0]["split"] = "test"
        with self.assertRaises(ValueError):
            support.summarize_records(records)

    def test_real_mode_uses_explicit_v2_and_safe_partitions(self):
        with patch("load_working_data.load_records", side_effect=lambda *a, **k: support.synthetic_records()) as loader:
            support.data_overview("real")
        self.assertEqual(loader.call_count, 2)
        for call in loader.call_args_list:
            self.assertEqual(call.kwargs, {"splits": ("train", "validation"), "manifest_version": "v2"})

    def test_unknown_modes_do_not_launch(self):
        with patch.object(support, "run_project", side_effect=AssertionError("unexpected run")):
            for family, mode in [("neural", "smoke"), ("characters", "off"), ("characters", "unknown")]:
                with self.assertRaises(ValueError):
                    support.train_baselines(family, mode)
        with self.assertRaises(ValueError):
            support.data_overview("unknown")

    def test_locked_offline_subprocess(self):
        with patch.object(support.subprocess, "run") as run:
            support.run_project("check_environment.py", capture=True)
        command = run.call_args.args[0]
        self.assertEqual(command[2:6], ["run", "--locked", "--offline", "python"])
        self.assertTrue(run.call_args.kwargs["check"])
        self.assertTrue(run.call_args.kwargs["capture_output"])

    def test_failed_smoke_stops_full_and_uses_allowed_directory(self):
        with patch.object(support, "run_project", side_effect=RuntimeError("smoke failed")) as run:
            with self.assertRaises(RuntimeError):
                support.train_baselines("characters", "full")
        self.assertEqual(run.call_count, 1)
        arguments = run.call_args.args
        self.assertEqual(arguments[arguments.index("--mode") + 1], "smoke")
        target = arguments[arguments.index("--output") + 1]
        self.assertIn(ROOT / "outputs/baselines", target.parents)
        self.assertFalse(target.exists())

    def test_final_selection_uses_full_not_first_smoke(self):
        runs = [{"mode": mode, "directory": f"outputs/baselines/invented_{mode}",
                 "validation": [{}] * 6} for mode in ("smoke", "full")]
        self.assertIs(support.select_training_run(runs, "full"), runs[1])
        for invalid in (runs[:1], list(reversed(runs)), runs + runs[1:]):
            with self.assertRaises(ValueError):
                support.select_training_run(invalid, "full")

    def test_smoke_selection_stays_explicitly_smoke(self):
        runs = [{"mode": "smoke", "validation": [{}] * 6}]
        self.assertIs(support.select_training_run(runs, "smoke"), runs[0])
        with self.assertRaises(ValueError):
            support.select_training_run(runs, "off")
        runs[0]["validation"] = []
        with self.assertRaises(ValueError):
            support.select_training_run(runs, "smoke")

    def test_final_summary_has_full_counts_and_no_smoke_array(self):
        row = {"corpus": "tsac", "system": "tfidf_char", "variant": "raw-v0",
               "train_rows": 10053, "validation_rows": 1774, "accuracy": .9, "macro_f1": .89}
        run = {"mode": "full", "directory": "outputs/baselines/invented_full", "validation": [row]}
        rendered = support.training_summary(run)
        for value in ("FULL", "10053", "1774", "0.900000", "0.890000"):
            self.assertIn(value, rendered)
        self.assertNotIn("SMOKE", rendered)

    def test_active_kernel_environment_is_not_used_for_training(self):
        with patch.dict("os.environ", {"VIRTUAL_ENV": "fiction-kernel"}), \
                patch.object(support.subprocess, "run") as process:
            support.run_project("check_environment.py")
            self.assertNotIn("VIRTUAL_ENV", process.call_args.kwargs["env"])
            self.assertEqual(__import__("os").environ["VIRTUAL_ENV"], "fiction-kernel")

    def test_captured_subprocess_failure_is_visible(self):
        from contextlib import redirect_stdout
        from io import StringIO
        import subprocess
        error = subprocess.CalledProcessError(1, ["fiction"], output="fictitious output\n", stderr="fictitious error\n")
        output = StringIO()
        with patch.object(support.subprocess, "run", side_effect=error), redirect_stdout(output):
            with self.assertRaises(subprocess.CalledProcessError):
                support.run_project("check_environment.py", capture=True)
        self.assertIn("fictitious error", output.getvalue())

    def test_export_uses_selected_run_and_never_overwrites_figures(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(support, "ROOT", Path(directory)), \
                patch("plot_validation_results.export_figures", return_value={}) as export:
            source = Path(directory) / "outputs/baselines/new_full"
            source.mkdir(parents=True)
            (source / "results.json").write_text(json.dumps({"mode": "full"}))
            run = {"mode": "full", "directory": "outputs/baselines/new_full"}
            support.export_training_figures(run)
            self.assertEqual(export.call_args.args, (source, source / "figures"))
            self.assertTrue(export.call_args.kwargs["verify"])
            (source / "figures").mkdir()
            support.export_training_figures(run)
            self.assertNotEqual(export.call_args.args[1], source / "figures")
            self.assertEqual(export.call_args.args[1].parent, source)
            run["mode"] = "smoke"
            with self.assertRaises(ValueError):
                support.export_training_figures(run)


if __name__ == "__main__":
    unittest.main()
