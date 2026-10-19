"""Synthetic tests: feature concatenation, train-only vocabularies and scaling."""
import copy
import io
import json
from pathlib import Path
import sys
import unittest

import joblib
import numpy as np
from scipy.sparse import isspmatrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline_models import (build_tfidf_model, tfidf_system, vectorizer_branches,
                             vocabulary_sizes, zero_feature_rows, verify_model_config)
from compare_combined_baselines import compatible_combination, paired_combination, check_branch_identity

ROOT = Path(__file__).resolve().parents[1]


class CombinedBaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/baselines_combined_v2.json").read_text())
        self.train = ["behi service", "behi service", "khayeb service", "khayeb service"]
        self.model = build_tfidf_model(self.config).fit(self.train, ["POS", "POS", "NEG", "NEG"])

    def test_branches_match_existing_configs_and_dimensions_add_up(self):
        for branch, filename in (("word", "baselines_words_v2.json"), ("char", "baselines_grouped_v2.json")):
            previous = json.loads((ROOT / "configs" / filename).read_text())
            self.assertEqual(self.config["tfidf"][branch], previous["tfidf"])
        self.assertEqual(tfidf_system(self.config), "tfidf_word_char")
        matrix = self.model[:-1].transform(self.train)
        self.assertTrue(isspmatrix(matrix))
        self.assertEqual(matrix.shape[1], sum(vocabulary_sizes(self.model).values()))

    def test_equal_blocks_and_final_unit_norm(self):
        matrix = self.model[:-1].transform(["behi service"])
        word_size = vocabulary_sizes(self.model)["word"]
        self.assertAlmostEqual(float(matrix.multiply(matrix).sum()), 1.0)
        self.assertAlmostEqual(float(matrix[:, :word_size].multiply(matrix[:, :word_size]).sum()), .5)
        self.assertAlmostEqual(float(matrix[:, word_size:].multiply(matrix[:, word_size:]).sum()), .5)

    def test_no_validation_fitting_and_zero_rows_kept(self):
        vocabularies = {name: dict(v.vocabulary_) for name, v in vectorizer_branches(self.model).items()}
        validation = ["zzzzzz", "😀", "behi"]
        self.assertEqual(len(self.model.predict(validation)), 3)
        self.assertEqual(zero_feature_rows(self.model, validation), 2)
        self.assertEqual({name: v.vocabulary_ for name, v in vectorizer_branches(self.model).items()}, vocabularies)
        self.assertTrue(all("zzzzzz" not in v for v in vocabularies.values()))

    def test_character_only_evidence_is_retained(self):
        words = vectorizer_branches(self.model)["word"]
        self.assertEqual(words.transform(["behiXYZ"]).getnnz(), 0)
        matrix = self.model[:-1].transform(["behiXYZ"])
        self.assertGreater(matrix.getnnz(), 0)
        self.assertAlmostEqual(float(matrix.multiply(matrix).sum()), 1.0)

    def test_reload_and_config_validation(self):
        buffer = io.BytesIO()
        joblib.dump(self.model, buffer)
        buffer.seek(0)
        reloaded = joblib.load(buffer)
        self.assertTrue(np.array_equal(reloaded.predict(self.train), self.model.predict(self.train)))
        verify_model_config(reloaded, self.config)
        bad = copy.deepcopy(self.config)
        bad["tfidf"]["word"]["token_pattern"] = "changed"
        with self.assertRaises(ValueError):
            verify_model_config(reloaded, bad)

    def test_unregistered_scaling_rejected(self):
        self.config["tfidf"]["block_weights"]["word"] = 2
        with self.assertRaises(ValueError):
            build_tfidf_model(self.config)

    def test_comparison_rejects_changed_partitions_or_branch_parameters(self):
        def metadata(config):
            return {"mode": "full", "test_loaded": False, "manifests_sha256": {"fake": "fixed"},
                    "uv_lock_sha256": "fixed", "packages": {}, "device": "cpu", "python": "synthetic",
                    "config": config}

        char = metadata(json.loads((ROOT / "configs/baselines_grouped_v2.json").read_text()))
        word = metadata(json.loads((ROOT / "configs/baselines_words_v2.json").read_text()))
        combined = metadata(copy.deepcopy(self.config))
        compatible_combination(char, word, combined)
        bad = copy.deepcopy(combined)
        bad["manifests_sha256"]["fake"] = "changed"
        with self.assertRaises(ValueError):
            compatible_combination(char, word, bad)
        bad = copy.deepcopy(combined)
        bad["config"]["tfidf"]["word"]["min_df"] = 3
        with self.assertRaises(ValueError):
            compatible_combination(char, word, bad)

    def test_branch_vocabulary_and_idf_must_match(self):
        config = json.loads((ROOT / "configs/baselines_words_v2.json").read_text())
        standalone = build_tfidf_model(config).fit(self.train, ["POS", "POS", "NEG", "NEG"])
        check_branch_identity(standalone, self.model, "word")
        standalone["tfidf"].idf_ = standalone["tfidf"].idf_ + 1
        with self.assertRaises(ValueError):
            check_branch_identity(standalone, self.model, "word")

    def test_combination_pairing_names_and_identity(self):
        rows = [{"id": "fictional", "group_id": "g", "truth": "POS", "predicted": "POS",
                 "writing_system": "LATIN"}]
        result = paired_combination(rows, rows, ["NEG", "POS"])
        self.assertEqual(result["delta_combined_minus_reference"], 0)
        self.assertEqual(result["combined_errors"], 0)
        self.assertEqual(result["reference_correct_combined_wrong"], 0)


if __name__ == "__main__":
    unittest.main()
