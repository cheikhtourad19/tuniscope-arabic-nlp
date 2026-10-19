"""Local integration checks; they require the authorized raw corpora and v1 manifests."""

from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import load_working_data as loader  # noqa: E402


class WorkingDataLoaderTests(unittest.TestCase):
    def test_default_tsac_load_is_train_validation_only(self):
        observed_files = []
        original = loader.decoded_lines

        def track(path):
            observed_files.append(path.name)
            yield from original(path)

        with patch.object(loader, "decoded_lines", track):
            records = loader.load_records("tsac")
        self.assertEqual(len(records), 11827)
        self.assertEqual(Counter(row["split"] for row in records),
                         {"train": 10053, "validation": 1774})
        self.assertEqual(set(observed_files), {"train_pos.txt", "train_neg.txt"})
        self.assertTrue(all(row["text_raw"] == row["text_norm"] for row in records))
        self.assertTrue(all(row["normalization_version"] == "raw-v0" for row in records))
        self.assertTrue(all(row["script"] == "UNKNOWN" for row in records))

    def test_default_madar_load_preserves_source_text(self):
        records = loader.load_records("madar")
        self.assertEqual(len(records), 7188)
        self.assertEqual(Counter(row["split"] for row in records),
                         {"train": 6388, "validation": 800})
        self.assertTrue(all(row["text_raw"] == row["text_norm"] for row in records))
        self.assertTrue(all(row["group_id"] == f"madar:{row['source_id']}" for row in records))

    def test_test_requires_explicit_opt_in(self):
        with self.assertRaises(PermissionError):
            loader.load_records("tsac", ("test",))
        with self.assertRaises(PermissionError):
            loader.load_records("madar", ("test",))

    def test_invalid_selection_rejected(self):
        with self.assertRaises(ValueError):
            loader.load_records("tsac", ("train", "train"))
        with self.assertRaises(ValueError):
            loader.load_records("other")


if __name__ == "__main__":
    unittest.main()
