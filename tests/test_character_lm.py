"""Hand-calculated and synthetic integration checks; no protected corpus read."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from character_lm import CharacterLM, aggregate, UNK, BOS, EOS
from character_lm_experiment import ROOT, digest
from run_character_lm import run
from verify_character_lm import independent_score, verify


class CharacterLMTests(unittest.TestCase):
    def test_unigram_probability_by_hand(self):
        model = CharacterLM(1, 1, ["a", "b"]).fit(["ab", "a"])
        # counts a=2,b=1,EOS=2,UNK=0 ; denominator 5+4=9.
        self.assertAlmostEqual(model.probability((), model.ids["a"]), 3 / 9)
        self.assertAlmostEqual(model.probability((), EOS), 3 / 9)
        self.assertAlmostEqual(model.probability((), UNK), 1 / 9)
        expected = -math.log(3 / 9) - math.log(2 / 9) - math.log(3 / 9)
        self.assertAlmostEqual(model.score("ab")["nll"], expected)

    def test_probabilities_sum_to_one_for_seen_and_unseen_contexts(self):
        for order in (1, 3):
            model = CharacterLM(order, .1, ["a", "b"]).fit(["ab", "a"])
            contexts = [()] if order == 1 else [(BOS, BOS), (UNK, UNK), (model.ids["a"], model.ids["b"])]
            for history in contexts:
                self.assertAlmostEqual(sum(model.probability(history, c) for c in (UNK, EOS, *model.ids.values())), 1)

    def test_documents_do_not_share_contexts_and_eos_counted_once(self):
        model = CharacterLM(3, .1, ["a", "b"]).fit(["a", "b"])
        self.assertNotIn((model.ids["a"], model.ids["b"]), model.counts)
        self.assertEqual(sum(model.totals.values()), 4)
        self.assertEqual(model.counts[(BOS, BOS)][model.ids["a"]], 1)
        self.assertEqual(model.score("a")["targets"], 2)

    def test_unknowns_finite_without_mutating_training_vocab_or_counts(self):
        model = CharacterLM(3, .1, ["a"]).fit(["aa"])
        before = copy.deepcopy(model.to_dict())
        result = model.score("a🙂ب")
        self.assertEqual(result["unknown_characters"], 2)
        self.assertEqual(result["targets"], 4)
        self.assertTrue(math.isfinite(result["nll"]))
        self.assertEqual(model.to_dict(), before)

    def test_weighted_aggregation_not_mean_of_document_perplexities(self):
        scores = [{"nll": math.log(2), "targets": 1, "characters": 0, "unknown_characters": 0},
                  {"nll": 3 * math.log(8), "targets": 3, "characters": 2, "unknown_characters": 1}]
        result = aggregate(scores)
        self.assertAlmostEqual(result["perplexity"], (2 * 8 ** 3) ** .25)
        self.assertNotAlmostEqual(result["perplexity"], (2 + 8) / 2)
        self.assertEqual(result["unknown_character_rate"], .5)

    def test_empty_document_has_only_eos_and_literal_markers_do_not_collide(self):
        text = "<BOS> <EOS> <UNK>"
        model = CharacterLM(3, .1, sorted(set(text))).fit([text])
        self.assertEqual(model.score("")["targets"], 1)
        self.assertEqual(model.score(text)["unknown_characters"], 0)
        self.assertTrue(all(i >= 3 for i in model.ids.values()))

    def test_unicode_codepoints_not_utf8_bytes_or_graphemes(self):
        text = "بَ🙂"
        model = CharacterLM(3, .1, sorted(set(text))).fit([text])
        self.assertEqual(model.score(text)["characters"], 3)
        self.assertEqual(model.score(text)["targets"], 4)

    def test_json_roundtrip_and_independent_probability_formula(self):
        for order in (1, 3):
            model = CharacterLM(order, .1, ["a", "b"]).fit(["ab", "ba"])
            clone = CharacterLM.from_dict(json.loads(json.dumps(model.to_dict())))
            self.assertEqual(model.to_dict(), clone.to_dict())
            for text in ["", "aba", "🙂"]:
                self.assertEqual(model.score(text), clone.score(text))
                self.assertAlmostEqual(model.score(text)["nll"], independent_score(clone, text)["nll"])

    def test_invalid_protocol_and_checkpoint_rejected(self):
        for order, alpha, chars in [(2, .1, ["a"]), (True, .1, ["a"]), (3, 0, ["a"]), (3, float("nan"), ["a"]),
                                    (3, .1, []), (3, .1, ["a", "a"]), (3, .1, ["ab"])]:
            with self.assertRaises(ValueError):
                CharacterLM(order, alpha, chars)
        model = CharacterLM(3, .1, ["a"]).fit(["a"])
        data = model.to_dict()
        data["counts"][0][1][0][1] = -1
        with self.assertRaises(ValueError):
            CharacterLM.from_dict(data)
        with self.assertRaises(ValueError):
            model.probability((BOS, BOS), BOS)
        with self.assertRaises(ValueError):
            CharacterLM(3, .1, ["a"]).fit(["b"])


class CharacterLMRunnerTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "outputs/language_models"
        base.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-test-", dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.config = ROOT / "configs/character_lm_v1.json"
        self.train = [{"id": "train-a", "group_id": "group-a", "text_raw": "aba", "label": "POS"},
                      {"id": "train-b", "group_id": "group-b", "text_raw": "bab", "label": "NEG"}]
        self.validation = [{"id": "validation-a", "group_id": "group-c", "text_raw": "a🙂", "label": "POS"}]
        for target in ("run_character_lm.partitions", "verify_character_lm.partitions"):
            mocked = patch(target, return_value=(self.train, self.validation))
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_synthetic_smoke_full_reload_and_refuse_overwrite(self):
        smoke = run(self.config, "smoke", self.directory / "smoke")
        self.assertEqual(len(verify(smoke)["results"]), 4)
        full = run(self.config, "full", self.directory / "full", smoke)
        self.assertEqual(verify(full)["mode"], "full")
        with self.assertRaises(ValueError):
            run(self.config, "smoke", smoke)

    def test_full_requires_smoke(self):
        with self.assertRaises(ValueError):
            run(self.config, "full", self.directory / "full")

    def test_verifier_rejects_wrong_aggregate_and_rehashed_wrong_row_score(self):
        directory = run(self.config, "smoke", self.directory / "smoke")
        path = directory / "results.json"
        metadata = json.loads(path.read_text())
        metadata["results"][0]["metrics"]["perplexity"] += 1
        path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            verify(directory)
        metadata["results"][0]["metrics"]["perplexity"] -= 1
        score_path = directory / metadata["results"][0]["scores"]
        scores = json.loads(score_path.read_text())
        scores[0]["nll"] += 1
        score_path.write_text(json.dumps(scores))
        metadata["results"][0]["scores_sha256"] = digest(score_path)
        path.write_text(json.dumps(metadata))
        with self.assertRaises(ValueError):
            verify(directory)


if __name__ == "__main__":
    unittest.main()
