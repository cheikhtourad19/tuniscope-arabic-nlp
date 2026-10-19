"""Fictitious-only tests for notebook safeguards and aggregate reporting."""

import copy
import json
from pathlib import Path
import sys
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


if __name__ == "__main__":
    unittest.main()
