"""Synthetic checks for the aggregate jury report, without any corpus reads."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline_metrics import classification_metrics
from show_validation_results import display_run, matrix_html, page, recomputed_metrics, selected_result, table


class PresentationResultsTests(unittest.TestCase):
    def test_variant_selection_must_be_unambiguous(self):
        data = {"results": [{"corpus": "tsac", "system": "char", "variant": v} for v in ("raw", "norm")]}
        self.assertEqual(selected_result(data, "tsac", "char", "raw")["variant"], "raw")
        with self.assertRaises(ValueError):
            selected_result(data, "tsac", "char")

    def test_metrics_recomputed_even_with_extra_slices(self):
        predictions = [{"truth": "POS", "predicted": "POS"}, {"truth": "NEG", "predicted": "POS"}]
        metrics = classification_metrics(["POS", "NEG"], ["POS", "POS"], ["NEG", "POS"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "pred.json").write_text(json.dumps(predictions), encoding="utf-8")
            saved = dict(metrics, by_writing_system={})
            self.assertEqual(recomputed_metrics(path, {"predictions": "pred.json", "metrics": saved}, ["NEG", "POS"]), metrics)

    def test_tampered_accuracy_is_rejected(self):
        predictions = [{"truth": "POS", "predicted": "POS"}]
        metrics = classification_metrics(["POS"], ["POS"], ["NEG", "POS"])
        metrics["accuracy"] = 0
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "pred.json").write_text(json.dumps(predictions), encoding="utf-8")
            with self.assertRaises(ValueError):
                recomputed_metrics(path, {"predictions": "pred.json", "metrics": metrics}, ["NEG", "POS"])

    def test_html_escapes_table_content(self):
        rendered = table(["<script>"], [["<img>"]], 0)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;img&gt;", rendered)
        self.assertIn('class="best"', rendered)

    def test_confusion_axes_and_diagonal(self):
        html = matrix_html({"labels": ["NEG", "POS"], "confusion_matrix": [[8, 2], [1, 9]]})
        self.assertIn("Réel / prédit", html)
        self.assertEqual(html.count('class="diag"'), 2)
        self.assertIn('class="diag">9', html)

    def test_page_does_not_claim_checkpoint_check_without_flag(self):
        self.assertNotIn("Checkpoints rechargés et prédictions vérifiées", page("Fictif", "", False))
        self.assertIn("Checkpoints rechargés et prédictions vérifiées", page("Fictif", "", True))

    def test_single_run_rejects_test_access(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "outputs/baselines/invented"
            run.mkdir(parents=True)
            (run / "results.json").write_text(json.dumps({"test_loaded": True}), encoding="utf-8")
            with patch("show_validation_results.ROOT", root), self.assertRaises(ValueError):
                display_run(run)

    def test_single_run_prints_recomputed_metrics(self):
        from run_baselines import digest
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / "outputs/baselines/invented"
            run.mkdir(parents=True)
            predictions = run / "pred.json"
            predictions.write_text(json.dumps([{"truth": "POS", "predicted": "POS"}]), encoding="utf-8")
            metrics = classification_metrics(["POS"], ["POS"], ["NEG", "POS"])
            data = {"test_loaded": False, "mode": "smoke", "config": {
                "manifest_version": "v2", "labels": {"tsac": ["NEG", "POS"]}}, "results": [{
                    "corpus": "tsac", "system": "tfidf_char", "variant": "raw-v0", "metrics": metrics,
                    "predictions": "pred.json", "predictions_sha256": digest(predictions)}]}
            (run / "results.json").write_text(json.dumps(data), encoding="utf-8")
            output = StringIO()
            with patch("show_validation_results.ROOT", root), redirect_stdout(output):
                self.assertEqual(display_run(run), [metrics])
            self.assertIn("petit test technique", output.getvalue())
            self.assertIn("Accuracy : 1.000000", output.getvalue())
            self.assertIn("Précision", output.getvalue())


if __name__ == "__main__":
    unittest.main()
