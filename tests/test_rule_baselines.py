"""Fictitious examples for lexical induction, coverage and bounded negation."""
import copy
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from rule_baselines import fit_rules, predict_rule, tokenize, validate_config
from verify_rule_baselines import paired

CONFIG=json.loads((Path(__file__).resolve().parents[1]/"configs/rule_baselines_v1.json").read_text())


class RuleInductionTests(unittest.TestCase):
    def setUp(self):
        self.config=copy.deepcopy(CONFIG)
        self.config["lexicon"].update(min_class_document_frequency=1,min_log_rate_ratio=0.1,max_terms_per_class=20)
        self.rows=[{"id":str(i),"split":"train","label":"POS" if i<4 else "NEG",
                    "text_raw":"good light" if i<4 else "bad dark not"} for i in range(8)]

    def test_lexicon_train_only(self):
        rows=copy.deepcopy(self.rows); rows[0]["split"]="validation"
        with self.assertRaises(ValueError): fit_rules(rows,["NEG","POS"],self.config)
        rows[0]["split"]="test"
        with self.assertRaises(ValueError): fit_rules(rows,["NEG","POS"],self.config)

    def test_document_frequency_not_occurrence_frequency(self):
        model=fit_rules(self.rows,["NEG","POS"],self.config)
        repeated=copy.deepcopy(self.rows); repeated[0]["text_raw"]="good good good light"
        self.assertEqual(model,fit_rules(repeated,["NEG","POS"],self.config))

    def test_stable_reconstruction_and_json(self):
        model=fit_rules(self.rows,["NEG","POS"],self.config)
        self.assertEqual(model,fit_rules(self.rows[::-1],["NEG","POS"],self.config))
        self.assertEqual(model,json.loads(json.dumps(model)))

    def test_controls_are_not_cues_and_must_be_attested(self):
        model=fit_rules(self.rows,["NEG","POS"],self.config)
        self.assertNotIn("not",[e["token"] for e in model["entries"]])
        self.assertIn("not",model["active_negation_markers"])
        self.assertNotIn("لا",model["active_negation_markers"])
        self.assertEqual(model["active_exceptions"],[])

    def test_min_frequency_and_size(self):
        self.config["lexicon"]["min_class_document_frequency"]=5
        self.assertEqual(fit_rules(self.rows,["NEG","POS"],self.config)["entries"],[])
        self.config["lexicon"].update(min_class_document_frequency=1,max_terms_per_class=1)
        self.assertEqual(len(fit_rules(self.rows,["NEG","POS"],self.config)["entries"]),2)

    def test_duplicate_ids_or_missing_class_rejected(self):
        with self.assertRaises(ValueError): fit_rules(self.rows+self.rows[:1],["NEG","POS"],self.config)
        with self.assertRaises(ValueError): fit_rules(self.rows[:4],["NEG","POS"],self.config)

    def test_invalid_config_rejected(self):
        self.config["manifest_version"]="v1"
        with self.assertRaises(ValueError): validate_config(self.config)


class NegationRuleTests(unittest.TestCase):
    def setUp(self):
        self.model={"labels":["NEG","POS"],"majority":"NEG","config":CONFIG,
            "entries":[{"token":"good","label":"POS"},{"token":"bad","label":"NEG"}],
            "active_negation_markers":["not","لا"],"active_clause_breaks":["but"],
            "active_exceptions":[["لا","بد"]]}

    def predict(self,text,negation=True):
        return predict_rule(text,self.model,negation=negation)

    def test_plain_and_negated_polarities(self):
        self.assertEqual(self.predict("good")["predicted"],"POS")
        self.assertEqual(self.predict("not good")["predicted"],"NEG")
        self.assertEqual(self.predict("not bad")["predicted"],"POS")
        self.assertEqual(self.predict("not good",False)["predicted"],"POS")

    def test_window_is_three_words(self):
        self.assertEqual(self.predict("not x x good")["predicted"],"NEG")
        self.assertEqual(self.predict("not x x x good")["predicted"],"POS")

    def test_punctuation_newline_and_adversative_end_scope(self):
        for separator in (".","،","\n","but"):
            self.assertEqual(self.predict(f"not {separator} good")["predicted"],"POS")

    def test_obligation_exception(self):
        self.assertEqual(self.predict("لا بد good")["negation_triggers"],0)
        self.assertEqual(self.predict("لا بد good")["predicted"],"POS")
        self.assertEqual(self.predict("لا good")["predicted"],"NEG")

    def test_double_marker_parity_is_explicit(self):
        self.assertEqual(self.predict("not not good")["predicted"],"POS")

    def test_repetition_not_extra_votes(self):
        decision=self.predict("good good good bad")
        self.assertTrue(decision["fallback"])
        self.assertEqual(decision["scores"],{"NEG":1,"POS":1})

    def test_unknown_and_emoji_stay_in_fallback(self):
        for text in ("", "unknown", "😍"):
            self.assertEqual(self.predict(text)["predicted"],"NEG")
            self.assertFalse(self.predict(text)["decisive"])
            self.assertFalse(self.predict(text)["lexical_hit"])

    def test_false_shape_does_not_trigger_generic_negation(self):
        self.assertEqual(self.predict("mannouch good")["negation_triggers"],0)
        self.assertEqual(self.predict("mannouch good")["predicted"],"POS")

    def test_tokenization_preserves_arabizi_and_technical_boundary(self):
        self.assertEqual(tokenize("BEHI 3lik _x000D_"),["behi","3lik","_x000d_"])
        self.assertEqual(self.predict("not _x000D_ good")["predicted"],"POS")

    def test_dialect_cannot_flip_polarity(self):
        self.model["labels"]=["MSA","TN","EG"]
        with self.assertRaises(ValueError): self.predict("anything")


class RulePairingTests(unittest.TestCase):
    def test_pairing_and_misalignment(self):
        left=[{"id":"fake","group_id":"g","truth":"POS","predicted":"NEG","writing_system":"LATIN"}]
        right=[{**left[0],"predicted":"POS"}]
        self.assertEqual(paired(left,right),{"different_predictions":1,"corrected_by_right":1,"introduced_by_right":0})
        with self.assertRaises(ValueError): paired(left,[{**right[0],"id":"other"}])


if __name__=="__main__":
    unittest.main()
