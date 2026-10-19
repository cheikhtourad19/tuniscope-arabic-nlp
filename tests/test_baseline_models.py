"""Fictitious-text tests for the preregistered word representation."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from baseline_models import build_tfidf_model, tfidf_system, zero_feature_rows

ROOT = Path(__file__).resolve().parents[1]


class BaselineModelTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/baselines_words_v2.json").read_text())

    def test_unicode_one_letter_and_arabizi_preserved(self):
        analyzer = build_tfidf_model(self.config)["tfidf"].build_analyzer()
        tokens = analyzer("و لا 7لو 3lech A")
        for value in ("و", "لا", "7لو", "3lech", "A", "و لا", "لا 7لو"):
            self.assertIn(value, tokens)

    def test_emoji_and_punctuation_limitation_explicit(self):
        analyzer = build_tfidf_model(self.config)["tfidf"].build_analyzer()
        self.assertEqual(analyzer("😀 !!!"), [])
        self.assertEqual(analyzer("moch, behi"), ["moch", "behi", "moch behi"])

    def test_vocabulary_fit_on_training_only_and_empty_features_kept(self):
        model = build_tfidf_model(self.config)
        model.fit(["behi service", "behi service", "khayeb service", "khayeb service"],
                  ["POS", "POS", "NEG", "NEG"])
        vocabulary = dict(model["tfidf"].vocabulary_)
        validation = ["unseenword", "😀", "behi"]
        self.assertEqual(len(model.predict(validation)), len(validation))
        self.assertEqual(zero_feature_rows(model, validation), 2)
        self.assertEqual(model["tfidf"].vocabulary_, vocabulary)
        self.assertNotIn("unseenword", vocabulary)

    def test_legacy_char_configs_and_unknown_analyzer(self):
        for name in ("baselines_initial.json", "baselines_grouped_v2.json"):
            config = json.loads((ROOT / "configs" / name).read_text())
            self.assertEqual(tfidf_system(config), "tfidf_char")
            self.assertEqual(build_tfidf_model(config)["tfidf"].ngram_range, (3, 5))
        self.assertEqual(tfidf_system(self.config), "tfidf_word")
        self.config["tfidf"]["analyzer"] = "unsupported"
        with self.assertRaises(ValueError):
            build_tfidf_model(self.config)


if __name__ == "__main__":
    unittest.main()
