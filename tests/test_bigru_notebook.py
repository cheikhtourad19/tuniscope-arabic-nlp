"""Notebook orchestration tests; no real corpus or fitting in these checks."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bigru_notebook_support as support
from check_notebooks import validate, execute
from check_bigru_notebook import validate_bigru

ROOT = Path(__file__).resolve().parents[1]


def fake_metadata(mode="full"):
    seeds = [13, 42, 2026] if mode == "full" else [42]
    return {"mode": mode, "test_loaded": False,
            "config": {"manifest_version": "v2", "seeds": [13, 42, 2026], "smoke_seed": 42},
            "results": [{"corpus": c, "seed": s, "variant": "raw-v0", "best_epoch": 2,
                         "train": {"rows": 10053 if c == "tsac" else 6388},
                         "validation": {"rows": 1774 if c == "tsac" else 800},
                         "metrics": {"accuracy": .85, "macro_f1": .84 + i * .001}}
                        for c in ("tsac", "madar") for i, s in enumerate(seeds)]}


class BiGRUNotebookTests(unittest.TestCase):
    def test_structure_default_full_and_no_torch_import(self):
        notebook = json.loads((ROOT / "notebooks/03_bigru.ipynb").read_text())
        validate_bigru(notebook, allow_saved_outputs=True)
        code = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        self.assertIn("os.environ.get('TUNISCOPE_BIGRU_MODE', 'full')", code)
        self.assertNotIn("import torch", code)
        self.assertNotIn(".fit(", code)

    def test_saved_output_check_never_mutates_the_notebook(self):
        import copy
        notebook = json.loads((ROOT / "notebooks/03_bigru.ipynb").read_text())
        cell = next(c for c in notebook["cells"] if c["cell_type"] == "code")
        cell["outputs"] = [{"output_type": "stream", "name": "stdout", "text": ["fictitious saved output"]}]
        cell["execution_count"] = 1
        before = copy.deepcopy(notebook)
        validate_bigru(notebook, allow_saved_outputs=True)
        self.assertEqual(notebook, before)
        with self.assertRaises(ValueError):
            validate_bigru(notebook)

    def test_off_executes_without_any_subprocess(self):
        notebook = json.loads((ROOT / "notebooks/03_bigru.ipynb").read_text())
        with patch.dict("os.environ", {"TUNISCOPE_BIGRU_MODE": "off"}), patch.object(support, "run_neural", side_effect=AssertionError("unexpected run")):
            result = execute(notebook, "03_bigru.ipynb")
        self.assertIsNone(result["final_run"])

    def test_child_uses_isolated_locked_offline_profile(self):
        completed = subprocess.CompletedProcess([], 0, "verified\n", "")
        with patch.dict("os.environ", {"VIRTUAL_ENV": "fictitious-kernel", "MPLBACKEND": "module://matplotlib_inline.backend_inline"}), patch.object(support.subprocess, "run", return_value=completed) as run:
            support.run_neural("verify_bigru.py", "fake")
            args = run.call_args.args[0]
            self.assertTrue(args[1].endswith("scripts/bigru_uv.sh"))
            self.assertEqual(args[2:6], ["run", "--locked", "--offline", "python"])
            self.assertNotIn("VIRTUAL_ENV", run.call_args.kwargs["env"])
            self.assertEqual(run.call_args.kwargs["env"]["MPLBACKEND"], "Agg")
            self.assertEqual(__import__("os").environ["MPLBACKEND"], "module://matplotlib_inline.backend_inline")
            self.assertEqual(__import__("os").environ["VIRTUAL_ENV"], "fictitious-kernel")
        with self.assertRaises(ValueError):
            support.run_neural("run_baselines.py")

    def test_child_failure_displays_error(self):
        result = subprocess.CompletedProcess([], 2, "fictitious output\n", "fictitious error\n")
        output = StringIO()
        with patch.object(support.subprocess, "run", return_value=result), redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError):
            support.run_neural("verify_bigru.py")
        self.assertIn("fictitious error", output.getvalue())

    def test_failed_smoke_prevents_full(self):
        with patch.object(support, "run_neural", side_effect=RuntimeError("failed smoke")) as run:
            with self.assertRaises(RuntimeError):
                support.train_bigru("full")
            self.assertEqual(run.call_count, 1)
            self.assertEqual(run.call_args.args[2], "smoke")

    def test_full_selects_full_and_passes_smoke_guard(self):
        with patch.object(support, "run_neural") as run, patch.object(support, "selected_run", side_effect=[{"mode": "smoke"}, {"mode": "full"}]) as select:
            result = support.train_bigru("full")
            self.assertEqual(result["mode"], "full")
            self.assertEqual(run.call_count, 4)
            full_call = run.call_args_list[2]
            self.assertIn("--smoke-run", full_call.args)
            self.assertEqual(select.call_args.args[1], "full")

    def test_replay_only_verifies_no_training(self):
        with patch.object(support, "selected_run", return_value={"directory": Path("fictitious"), "mode": "full"}), patch.object(support, "run_neural") as run:
            support.replay_bigru("fictitious")
            run.assert_called_once_with("verify_bigru.py", Path("fictitious"))

    def test_missing_seed_wrong_mode_or_test_selection_rejected(self):
        with tempfile.TemporaryDirectory() as temporary, patch.object(support, "ROOT", Path(temporary)):
            source = Path(temporary) / "outputs/neural_baselines/fictitious"
            source.mkdir(parents=True)
            missing_seed = fake_metadata()
            missing_seed["results"].pop()
            opened_test = fake_metadata()
            opened_test["test_loaded"] = True
            for data in [fake_metadata("smoke"), missing_seed, opened_test]:
                (source / "results.json").write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    support.selected_run(source, "full")

    def test_summary_displays_full_counts_and_three_seed_spread(self):
        result = {"directory": "fictitious", **fake_metadata()}
        rendered = support.summary(result)
        for expected in ("FULL", "TROIS GRAINES", "10053", "1774", "6388", "800", "±", "2026"):
            self.assertIn(expected, rendered)
        self.assertNotIn("SMOKE", rendered)

    def test_figures_are_from_selected_run_and_existing_exports_are_preserved(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary, patch.object(support, "ROOT", Path(temporary)):
            source = Path(temporary) / "outputs/neural_baselines/fictitious"
            figure_dir = source / "figures"
            figure_dir.mkdir(parents=True)
            metadata = fake_metadata()
            (source / "results.json").write_text(json.dumps(metadata))
            expected = {f"{r['corpus']}_seed_{r['seed']}.png": b"invented png" for r in metadata["results"]}
            provenance = {"source_results_sha256": hashlib.sha256((source / "results.json").read_bytes()).hexdigest(),
                          "files": {f: hashlib.sha256(v).hexdigest() for f, v in expected.items()}}
            for name, contents in expected.items():
                (figure_dir / name).write_bytes(contents)
            (figure_dir / "provenance.json").write_text(json.dumps(provenance))
            run = support.selected_run(source, "full")
            with patch.object(support, "run_neural", side_effect=AssertionError("must not regenerate")):
                result = support.figures(run)
                self.assertEqual(len(result["images"]), 6)
                (figure_dir / next(iter(expected))).write_bytes(b"changed")
                with self.assertRaises(ValueError):
                    support.figures(run)


if __name__ == "__main__":
    unittest.main()
