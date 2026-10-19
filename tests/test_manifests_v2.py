"""Private-source integration checks for the revised, train-only grouping."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import load_working_data as loader
from audit_elongation_policy import collapse_repeated_letters
from build_manifests_v2 import ORIGINAL, OUTPUT, LEDGER, read_csv, plan, sha256
from normalize_text import normalize


class RevisedManifestTests(unittest.TestCase):
    def test_rebuild_is_deterministic_and_v1_preserved(self):
        rows, _, summary = plan()
        self.assertEqual(rows, read_csv(OUTPUT / "tsac.csv"))
        self.assertEqual(summary, json.loads((OUTPUT / "summary.json").read_text()))
        self.assertEqual(sha256(ORIGINAL / "tsac.csv"), summary["original_manifest_sha256"])
        for name in ("madar.csv", "madar_exclusions.csv", "tsac_exclusions.csv"):
            self.assertEqual((ORIGINAL / name).read_bytes(), (OUTPUT / name).read_bytes())

    def test_only_assignments_and_groups_can_change(self):
        before = read_csv(ORIGINAL / "tsac.csv")
        after = read_csv(OUTPUT / "tsac.csv")
        self.assertEqual(len(before), len(after))
        for old, new in zip(before, after):
            self.assertEqual({k: v for k, v in old.items() if k not in {"split", "group_id"}},
                             {k: v for k, v in new.items() if k not in {"split", "group_id"}})
            if old["split"] == "test":
                self.assertEqual(old, new)
        self.assertEqual(Counter((r["split"], r["label"]) for r in before),
                         Counter((r["split"], r["label"]) for r in after))

    def test_all_candidate_families_and_human_pairs_are_together(self):
        families = defaultdict(set)
        groups = defaultdict(set)
        rows = loader.load_records("tsac", manifest_version="v2")
        by_id = {r["id"]: r for r in rows}
        for row in rows:
            families[collapse_repeated_letters(normalize(row["text_raw"]))].add(row["split"])
            groups[row["group_id"]].add(row["split"])
        self.assertTrue(all(len(splits) == 1 for splits in families.values()))
        self.assertTrue(all(len(splits) == 1 for splits in groups.values()))
        old_groups = defaultdict(set)
        for old in read_csv(ORIGINAL / "tsac.csv"):
            if old["split"] != "test":
                old_groups[old["group_id"]].add(by_id[old["id"]]["group_id"])
        self.assertTrue(all(len(new_ids) == 1 for new_ids in old_groups.values()))
        decisions = json.loads(LEDGER.read_text(encoding="utf-8"))
        for pair in decisions["pairs"]:
            a, b = (by_id[pair[side]["id"]] for side in ("a", "b"))
            self.assertEqual(a["group_id"], b["group_id"])
            self.assertEqual(a["split"], b["split"])

    def test_v2_does_not_open_official_test_and_checks_hashes(self):
        files = []
        original = loader.decoded_lines

        def track(path):
            files.append(path.name)
            yield from original(path)

        with patch.object(loader, "decoded_lines", track):
            loader.load_records("tsac", manifest_version="v2")
        self.assertEqual(set(files), {"train_pos.txt", "train_neg.txt"})
        with self.assertRaises(PermissionError):
            loader.load_records("tsac", ("test",), manifest_version="v2")
        with self.assertRaises(ValueError):
            loader.load_records("tsac", manifest_version="../../other")


if __name__ == "__main__":
    unittest.main()
