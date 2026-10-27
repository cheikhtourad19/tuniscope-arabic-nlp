"""Fictitious-only checks; plotting arithmetic does not need Matplotlib/sklearn."""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import plot_validation_results as plots
from baseline_metrics import classification_metrics
from check_notebooks import validate


class PlotValidationTests(unittest.TestCase):
    def fixture(self, root):
        run = root / "outputs/baselines/fictitious"
        run.mkdir(parents=True)
        data = {"created_at_utc": "2000-01-01T00:00:00Z", "mode": "full", "test_loaded": False,
                "config": {"manifest_version": "v2", "corpora": ["tsac", "madar"],
                           "tfidf": {"analyzer": "char"},
                           "labels": {"tsac": ["NEG", "POS"], "madar": ["MSA", "TN", "EG"]}},
                "results": []}
        for corpus in ("tsac", "madar"):
            labels = data["config"]["labels"][corpus]
            for index, (system, variant) in enumerate([
                    ("majority", "raw-v0"), ("tfidf_char", "raw-v0"),
                    ("tfidf_char", "norm-nfc-space-v1")]):
                predictions = [{"id": f"private-id-{i}", "group_id": f"private-group-{i}",
                                "truth": label, "predicted": labels[0] if index == 0 else label,
                                "writing_system": "AR", "secret_text": "not-exported"}
                               for i, label in enumerate(labels)]
                path = run / f"{corpus}-{index}.json"
                path.write_text(json.dumps(predictions), encoding="utf-8")
                metrics = classification_metrics([p["truth"] for p in predictions],
                                                 [p["predicted"] for p in predictions], labels)
                data["results"].append({"corpus": corpus, "system": system, "variant": variant,
                    "train_rows": 12, "validation_rows": len(predictions), "metrics": metrics,
                    "predictions": path.name, "predictions_sha256": plots.digest(path),
                    "fit_seconds": .01, "predict_seconds": .002})
        (run / "results.json").write_text(json.dumps(data), encoding="utf-8")
        return run, data

    def save(self, run, data):
        (run / "results.json").write_text(json.dumps(data), encoding="utf-8")

    def test_independent_metrics_match_sklearn_including_missing_classes(self):
        for truth, predicted in [(["NEG", "POS", "POS"], ["NEG", "NEG", "POS"]),
                                  (["POS"], ["POS"]), (["NEG"], ["POS"])]:
            rows = [dict(truth=a, predicted=b) for a, b in zip(truth, predicted)]
            actual = plots.metrics_from_predictions(rows, ["NEG", "POS"])
            saved = classification_metrics(truth, predicted, ["NEG", "POS"])
            plots.check_metrics(actual, saved)

    def test_empty_predictions_and_unknown_labels_rejected(self):
        for rows in ([], [{"truth": "other", "predicted": "POS"}]):
            with self.assertRaises(ValueError):
                plots.metrics_from_predictions(rows, ["NEG", "POS"])

    def test_collect_exports_only_aggregates(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(plots, "ROOT", Path(temp)):
            run, _ = self.fixture(Path(temp))
            data = plots.collect(run)
            self.assertEqual(len(data["results"]), 6)
            serialized = json.dumps(data)
            for forbidden in ("private-id", "private-group", "secret_text", "not-exported"):
                self.assertNotIn(forbidden, serialized)
            self.assertTrue(data["results"][0]["by_writing_system"]["AR"]["small_or_missing_class_slice"])

    def test_test_v1_incomplete_or_duplicate_results_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(plots, "ROOT", Path(temp)):
            run, original = self.fixture(Path(temp))
            for change in ("test", "v1", "missing", "duplicate"):
                data = copy.deepcopy(original)
                if change == "test": data["test_loaded"] = True
                if change == "v1": data["config"]["manifest_version"] = "v1"
                if change == "missing": data["results"].pop()
                if change == "duplicate": data["results"][1] = data["results"][0]
                self.save(run, data)
                with self.assertRaises(ValueError): plots.collect(run)

    def test_tampered_prediction_checksum_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(plots, "ROOT", Path(temp)):
            run, data = self.fixture(Path(temp))
            path = run / data["results"][0]["predictions"]
            path.write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError): plots.collect(run)

    def test_modified_metrics_or_support_rejected(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(plots, "ROOT", Path(temp)):
            run, original = self.fixture(Path(temp))
            for key in ("macro_f1", "per_class"):
                data = copy.deepcopy(original)
                if key == "macro_f1": data["results"][0]["metrics"][key] = .9
                else: data["results"][0]["metrics"][key]["NEG"]["support"] = 999
                self.save(run, data)
                with self.assertRaises(ValueError): plots.collect(run)

    def test_different_validation_ids_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(plots, "ROOT", Path(temp)):
            run, data = self.fixture(Path(temp))
            row = data["results"][1]
            path = run / row["predictions"]
            predictions = json.loads(path.read_text())
            predictions[0]["id"] = "different-example"
            path.write_text(json.dumps(predictions))
            row["predictions_sha256"] = plots.digest(path)
            self.save(run, data)
            with self.assertRaises(ValueError): plots.collect(run)

    def test_artifact_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(ValueError): plots.local_path(Path(temp), "../outside.json")

    def test_smoke_label_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(plots, "ROOT", Path(temp)):
            run, data = self.fixture(Path(temp))
            data["mode"] = "smoke"
            self.save(run, data)
            self.assertEqual(plots.collect(run)["mode"], "smoke")

    def test_output_overwrite_and_outside_output_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(plots, "ROOT", Path(temp)):
            root = Path(temp)
            existing = root / "outputs/existing"
            existing.mkdir(parents=True)
            for path in (existing, root / "reports/not-allowed"):
                with self.assertRaises(ValueError): plots.render({}, path)

    def test_checkpoint_verification_explicit_and_uses_locked_environment(self):
        with patch.object(plots, "collect", return_value={}), patch.object(plots, "render") as render, \
                patch.object(plots.subprocess, "run") as process:
            plots.export_figures("local-run", "output", verify=False)
            process.assert_not_called()
            self.assertFalse(render.call_args.kwargs["checkpoint_verified"])
            plots.export_figures("local-run", "output", verify=True)
            command = process.call_args.args[0]
            self.assertIn("--locked", command)
            self.assertIn("--offline", command)
            self.assertIn("verify_baseline_run.py", " ".join(command))
            self.assertTrue(render.call_args.kwargs["checkpoint_verified"])

    def test_visualization_notebook_is_clean_and_never_calls_fit(self):
        path = Path(__file__).resolve().parents[1] / "notebooks/03_visualisation_validation.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        validate(notebook)
        source = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code")
        self.assertNotIn(".fit(", source)
        self.assertNotIn("train_baselines(", source)
        self.assertIn("SOURCE_RUN", source)


if __name__ == "__main__":
    unittest.main()
