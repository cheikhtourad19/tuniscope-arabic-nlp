"""Tokenizer-only profile tests, using invented vocabulary and texts."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from transformers import BertTokenizerFast

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import encoder_tokenizer_audit as module
from encoder_tokenizer_audit import features, stats, choose_length, parameter_counts
from fetch_camelbert_tokenizer import FILES

ROOT = Path(__file__).resolve().parents[2]


class TokenizerAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "a", "b", "موش", "باهي", "e", "##l", "3", "##lech", "!"]
        path = self.directory / "vocab.txt"
        path.write_text("\n".join(vocabulary) + "\n")
        self.tokenizer = BertTokenizerFast(vocab_file=str(path), do_lower_case=False, strip_accents=False, model_max_length=512)
        self.config = json.loads((ROOT / "configs/camelbert_tokenizer_audit_v1.json").read_text())

    def rows(self, texts):
        return [{"id": f"invented-{i}", "group_id": f"group-{i}", "text_raw": text} for i, text in enumerate(texts)]

    def test_known_words_wordpieces_and_unknown(self):
        self.assertEqual(self.tokenizer.tokenize("el"), ["e", "##l"])
        self.assertEqual(self.tokenizer.tokenize("موش باهي"), ["موش", "باهي"])
        self.assertEqual(self.tokenizer.tokenize("🙂"), ["[UNK]"])

    def test_special_tokens_and_unknown_denominators(self):
        values = features(self.rows(["a 🙂", "موش باهي"]), self.tokenizer)
        self.assertEqual(values[0]["length_with_special"], 4)
        self.assertEqual(values[0]["content_tokens"], 2)
        self.assertEqual(values[0]["unknown_tokens"], 1)
        result = stats(values, [64, 128], 100)
        self.assertEqual(result["unknown_token_rate"], .25)
        self.assertEqual(result["rows_with_unknown_rate"], .5)
        self.assertEqual(result["subwords_per_whitespace_field"], 1)

    def test_lengths_are_measured_without_silent_truncation(self):
        values = features(self.rows([" ".join(["a"] * 127)]), self.tokenizer)
        result = stats(values, [64, 128, 256, 512], 100)
        self.assertEqual(result["length_with_special"]["max"], 129)
        self.assertEqual(result["candidate_truncation"]["128"], {"rows": 1, "row_rate": 1, "removed_tokens": 1})
        self.assertEqual(result["candidate_truncation"]["256"]["rows"], 0)

    def test_empty_content_and_small_groups_remain_visible(self):
        result = stats(features(self.rows([" \t", "a"]), self.tokenizer), [64], 100)
        self.assertEqual(result["rows"], 2)
        self.assertEqual(result["empty_content_rows"], 1)
        self.assertFalse(result["minimum_100_reached"])
        self.assertEqual(result["length_with_special"]["p50"], 2)

    def test_choice_uses_train_groups_not_validation(self):
        train = {"rows": 100, "candidate_truncation": {str(n): {"row_rate": .02 if n == 64 else .01 if n == 128 else 0} for n in [64, 128, 256, 512]}}
        tiny = {"rows": 1, "candidate_truncation": {str(n): {"row_rate": 1} for n in [64, 128, 256, 512]}}
        corpora = {"invented": {"train": {"by_writing_system": {"AR": train, "MIXED": tiny}},
                                "validation": {"by_writing_system": {"AR": tiny}}}}
        chosen = choose_length(corpora, self.config)
        self.assertEqual(chosen["max_length"], 128)
        self.assertEqual(chosen["chosen_from"], "train_only")
        self.assertEqual(chosen["small_training_groups_not_used_for_selection"], ["invented:MIXED"])

    def test_fallback_reports_unmet_coverage(self):
        group = {"rows": 100, "candidate_truncation": {str(n): {"row_rate": .5} for n in [64, 128, 256, 512]}}
        result = choose_length({"invented": {"train": {"by_writing_system": {"AR": group}}}}, self.config)
        self.assertEqual(result["max_length"], 512)
        self.assertFalse(result["coverage_target_met"])

    def test_parameter_count_for_standard_bert_architecture(self):
        config = {"model_type": "bert", "hidden_size": 768, "num_attention_heads": 12, "num_hidden_layers": 12,
                  "intermediate_size": 3072, "vocab_size": 30522, "max_position_embeddings": 512, "type_vocab_size": 2}
        counts = parameter_counts(config)
        self.assertEqual(counts["encoder_with_pooler"], 109482240)
        self.assertEqual(counts["sequence_classification_2_labels"] - counts["encoder_with_pooler"], 1538)
        self.assertFalse(counts["checkpoint_loaded"])

    def test_aggregate_audit_never_requests_test_or_serializes_corpus_text(self):
        secret = "invented-private-field-that-must-not-appear"
        records = [{"id": f"r{i}", "group_id": f"g{i}", "text_raw": f"a {secret}", "label": "POS", "split": split}
                   for i, split in enumerate(["train"] * 100 + ["validation"] * 2)]
        def loader(corpus, **kwargs):
            self.assertEqual(kwargs, {"splits": ("train", "validation"), "manifest_version": "v2"})
            return records
        architecture = {"model_type": "bert", "hidden_size": 4, "num_attention_heads": 2, "num_hidden_layers": 1,
                        "intermediate_size": 8, "vocab_size": len(self.tokenizer), "max_position_embeddings": 512, "type_vocab_size": 2}
        config = {**self.config, "corpora": ["invented"]}
        with patch.object(module, "load_tokenizer", return_value=(self.tokenizer, architecture, {"model_license_declared": "apache-2.0", "files": {}})):
            result = module.audit(config, loader=loader)
        serialized = json.dumps(result, ensure_ascii=False)
        self.assertNotIn(secret, serialized)
        self.assertNotIn('"r0"', serialized)
        self.assertFalse(result["test_loaded"])
        self.assertFalse(result["weights_downloaded"])

    def test_fetch_allowlist_excludes_all_weight_formats(self):
        self.assertEqual(set(FILES), {"README.md", "config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"})
        self.assertTrue(all(not name.endswith((".bin", ".pt", ".safetensors", ".h5", ".msgpack")) for name in FILES))


if __name__ == "__main__":
    unittest.main()
