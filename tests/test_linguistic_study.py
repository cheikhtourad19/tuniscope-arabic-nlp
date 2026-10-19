"""Fictitious-data safeguards for the train-only linguistic study."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_linguistic_sample import marker_candidates, select_sample
from render_linguistic_study import validate_annotations


class LinguisticSampleTests(unittest.TestCase):
    def setUp(self):
        self.config = {"split":"train", "manifest_version":"v2", "seed":42,
                       "min_word_units":4, "max_word_units":50, "max_characters":320,
                       "quotas":[{"corpus":"tsac", "label":"POS", "script":"LATIN",
                                  "marker_candidate":False, "count":3}]}
        self.rows = [{"id":str(i), "group_id":f"g{i}", "split":"train", "corpus":"tsac",
                      "label":"POS", "text_raw":f"fictional example number {i}"} for i in range(6)]

    def test_deterministic_and_order_independent(self):
        self.assertEqual(select_sample(self.rows, self.config), select_sample(self.rows[::-1], self.config))

    def test_no_held_out_selection(self):
        rows = copy.deepcopy(self.rows); rows[0]["split"] = "validation"
        with self.assertRaises(ValueError): select_sample(rows, self.config)
        config = copy.deepcopy(self.config); config["split"] = "test"
        with self.assertRaises(ValueError): select_sample(self.rows, config)

    def test_duplicate_ids_and_insufficient_quota(self):
        with self.assertRaises(ValueError): select_sample(self.rows + self.rows[:1], self.config)
        with self.assertRaises(ValueError): select_sample(self.rows[:2], self.config)

    def test_group_and_normalized_text_uniqueness(self):
        rows = copy.deepcopy(self.rows)
        rows[1]["group_id"] = rows[0]["group_id"]
        rows[2]["text_raw"] = " fictional   example number 0 "
        chosen, _ = select_sample(rows, self.config)
        self.assertEqual(len({r["group_id"] for r in chosen}),3)
        self.assertEqual(len({" ".join(r["text_raw"].split()) for r in chosen}),3)

    def test_markers_remain_candidates(self):
        self.assertEqual(marker_candidates("mannouch"), ["mannouch"])
        self.assertEqual(marker_candidates("لا بد أن نذهب"), ["لا"])
        self.assertEqual(marker_candidates("MAY3JBK CHAY"), [])


class LinguisticAnnotationTests(unittest.TestCase):
    def setUp(self):
        self.cases = [{"case_id":"FAKE", "text_raw":"fictional text", "split":"train"}]
        self.annotations = [{"case_id":"FAKE", "cue":"text", "cue_role":"NOM",
            "negation_status":"absente", "negation_markers":[], "negation_scope":"None identified",
            "target":"Fiction", "reading":"Fictitious study", "irony":"Unknown"}]

    def test_analysis_axis_and_evidence(self):
        self.assertEqual(len(validate_annotations(self.cases, self.annotations)),1)
        bad = copy.deepcopy(self.annotations); bad[0]["cue"] = "not observed"
        with self.assertRaises(ValueError): validate_annotations(self.cases, bad)
        bad = copy.deepcopy(self.annotations); bad[0]["target"] = " "
        with self.assertRaises(ValueError): validate_annotations(self.cases, bad)

    def test_negation_requires_observed_marker(self):
        bad = copy.deepcopy(self.annotations); bad[0]["negation_status"] = "présente"
        with self.assertRaises(ValueError): validate_annotations(self.cases, bad)

    def test_case_and_split_guards(self):
        with self.assertRaises(ValueError): validate_annotations(self.cases, self.annotations * 2)
        cases = copy.deepcopy(self.cases); cases[0]["split"] = "test"
        with self.assertRaises(ValueError): validate_annotations(cases, self.annotations)


if __name__ == "__main__":
    unittest.main()
