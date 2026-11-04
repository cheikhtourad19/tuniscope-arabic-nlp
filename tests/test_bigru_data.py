"""Dependency-free tests for the neural input contract; fictitious data only."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bigru_data import audit, build_vocabulary, encode, partitions, tokenize, UNK


class BiGRUDataTests(unittest.TestCase):
    def test_arabic_arabizi_emoji_negation_and_case_preserved(self):
        tokens = tokenize("موش behi 3lech 🙂!!! A a")
        self.assertEqual(tokens, ["موش", "behi", "3lech", "🙂", "!", "!", "!", "A", "a"])

    def test_vocab_train_only_min_frequency_and_stable_ties(self):
        vocabulary = build_vocabulary(["behi moch behi", "moch 🙂 🙂 rare"], 6, 2)
        self.assertEqual(list(vocabulary), ["behi", "moch", "🙂"])
        self.assertNotIn("rare", vocabulary)
        before = dict(vocabulary)
        self.assertEqual(encode("newword rare", vocabulary, 128), [UNK, UNK])
        self.assertEqual(vocabulary, before)
        self.assertEqual(build_vocabulary(["b a b a c c"], 4, 2), {"a": 2, "b": 3})

    def test_empty_encoding_and_truncation_audited_without_exclusion(self):
        vocabulary = {"a": 2, "b": 3}
        self.assertEqual(encode("  ", vocabulary, 4), [UNK])
        self.assertEqual(encode("a b c", vocabulary, 2), [2, 3])
        result = audit(["a b c", "🙂", " "], vocabulary, 2)
        self.assertEqual(result["rows"], 3)
        self.assertEqual(result["truncated_rows"], 1)
        self.assertEqual(result["truncated_tokens"], 1)
        self.assertEqual(result["unknown_tokens_before_truncation"], 2)
        self.assertEqual(result["empty_token_rows"], 1)

    def test_partition_loader_explicit_v2_and_never_requests_test(self):
        rows = [{"id": "a", "group_id": "one", "split": "train"},
                {"id": "b", "group_id": "two", "split": "validation"}]
        before = copy.deepcopy(rows)
        with patch("bigru_data.load_records", return_value=rows) as loader:
            train, validation = partitions("tsac", "full", {})
            loader.assert_called_once_with("tsac", splits=("train", "validation"), manifest_version="v2")
        self.assertEqual(train, rows[:1])
        self.assertEqual(validation, rows[1:])
        self.assertEqual(rows, before)

    def test_shared_group_rejected(self):
        rows = [{"id": "a", "group_id": "same", "split": "train"},
                {"id": "b", "group_id": "same", "split": "validation"}]
        with patch("bigru_data.load_records", return_value=rows), self.assertRaises(ValueError):
            partitions("tsac", "full", {})


if __name__ == "__main__":
    unittest.main()
