"""Fictitious-data tests for controlled sparse lexical augmentation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from character_rule_models import coverage, feature_matrix, fit_model, predict_rows, rule_features, validate_config

ROOT=Path(__file__).resolve().parents[1]
CONFIG=json.loads((ROOT/"configs/baselines_char_rules_v1.json").read_text())
RULES=json.loads((ROOT/"configs/rule_baselines_v1.json").read_text())


class RuleFeatureTests(unittest.TestCase):
    def setUp(self):
        self.rules={"labels":["NEG","POS"],"majority":"NEG","config":RULES,
            "entries":[{"token":"good","label":"POS"},{"token":"bad","label":"NEG"}],
            "active_negation_markers":["not"],"active_clause_breaks":["but"],"active_exceptions":[]}

    def test_schema_and_unknown_zero_block(self):
        rows=[{"text_raw":"good"},{"text_raw":"unknown"},{"text_raw":"good bad"}]
        self.assertTrue(np.array_equal(rule_features(rows,self.rules).toarray(),[[0,1,1],[0,0,0],[0.5,0.5,1]]))

    def test_negation_changes_values_not_shape_or_resource(self):
        rows=[{"text_raw":"not good"}]
        original=copy.deepcopy(self.rules)
        plain=rule_features(rows,self.rules)
        neg=rule_features(rows,self.rules,negation=True)
        self.assertEqual(plain.shape,neg.shape)
        self.assertTrue(np.array_equal(neg.toarray(),[[1,0,1]]))
        self.assertEqual(original,self.rules)

    def test_labels_are_not_used_to_transform(self):
        a=rule_features([{"text_raw":"good","label":"NEG","split":"validation"}],self.rules)
        b=rule_features([{"text_raw":"good","label":"POS","split":"validation"}],self.rules)
        self.assertEqual((a-b).nnz,0)

    def test_no_majority_class_injected_when_uncovered(self):
        self.assertEqual(rule_features([{"text_raw":"?"}],self.rules).nnz,0)


class CharacterAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.config=copy.deepcopy(CONFIG)
        self.config["tfidf"]["min_df"]=1
        self.rule_config=copy.deepcopy(RULES)
        self.rule_config["lexicon"].update(min_class_document_frequency=1,min_log_rate_ratio=0.1)
        self.rows=[{"id":str(i),"split":"train","label":"POS" if i<4 else "NEG",
                    "text_raw":"good light" if i<4 else "bad dark"} for i in range(8)]

    def fit(self,system):
        with threadpool_limits(limits=1):
            return fit_model(self.rows,"tsac",system,self.config,self.rule_config)

    def test_train_only_and_unique_ids(self):
        rows=copy.deepcopy(self.rows); rows[0]["split"]="validation"
        with self.assertRaises(ValueError): fit_model(rows,"tsac","char_control",self.config,self.rule_config)
        with self.assertRaises(ValueError): fit_model(self.rows+self.rows[:1],"tsac","char_control",self.config,self.rule_config)

    def test_character_block_and_idf_unchanged(self):
        control=self.fit("char_control"); augmented=self.fit("char_lexicon")
        self.assertEqual(control["vectorizer"].vocabulary_,augmented["vectorizer"].vocabulary_)
        self.assertTrue(np.array_equal(control["vectorizer"].idf_,augmented["vectorizer"].idf_))
        left=feature_matrix(control,self.rows); right=feature_matrix(augmented,self.rows)
        self.assertEqual((left-right[:,:left.shape[1]]).nnz,0)
        self.assertEqual(right.shape[1],left.shape[1]+3)

    def test_no_global_normalization(self):
        model=self.fit("char_lexicon")
        matrix=feature_matrix(model,self.rows)
        self.assertTrue(np.all(np.asarray(matrix.multiply(matrix).sum(axis=1)).ravel()>1))

    def test_reload_is_faithful(self):
        model=self.fit("char_lexicon")
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"fiction.joblib"
            joblib.dump(model,path)
            self.assertEqual(predict_rows(model,self.rows),predict_rows(joblib.load(path),self.rows))

    def test_empty_lexicon_keeps_character_input(self):
        self.rule_config["lexicon"]["min_class_document_frequency"]=100
        model=self.fit("char_lexicon")
        counts=coverage(model,self.rows)
        self.assertEqual(counts["zero_rule_block_rows"],8)
        self.assertEqual(counts["zero_combined_rows"],0)

    def test_configuration_rejects_confounding_changes(self):
        ref=json.loads((ROOT/"configs/baselines_grouped_v2.json").read_text())
        validate_config(CONFIG,ref)
        bad=copy.deepcopy(CONFIG); bad["global_normalization"]="l2"
        with self.assertRaises(ValueError): validate_config(bad,ref)
        bad=copy.deepcopy(CONFIG); bad["classifier"]["C"]=2
        with self.assertRaises(ValueError): validate_config(bad,ref)


if __name__=="__main__":
    unittest.main()
