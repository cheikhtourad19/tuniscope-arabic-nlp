"""Checks for citation grounding and out-of-vocabulary queries."""
import math
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'rag'))
from evaluate import evaluate, fit_threshold
from search import BM25, extract, load_cards, load_questions


class RagSearchTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cards = load_cards()
        cls.bm25 = BM25(cls.cards)

    def test_no_overlap_abstains(self):
        question = 'zzzxquu'
        hits = self.bm25.rank(question, 'fr')
        self.assertEqual(hits, [])
        self.assertEqual(extract(question, 'fr', hits, 0)['sources'], [])

    def test_citation_matches_literal_passage_and_language(self):
        text = 'تطبيع يونيكود NFC'
        hits = self.bm25.rank(text, 'ar')
        self.assertTrue(all(card.lang == 'ar' and score > 0 for card, score in hits))
        result = extract(text, 'ar', hits, 0)
        self.assertEqual(result['sources'][0]['id'], 'normalisation.ar')
        self.assertEqual(result['answer'], hits[0][0].text)

    def test_bilingual_validation_counts_and_rejection(self):
        questions = [q for q in load_questions(self.cards) if q['split'] == 'validation']
        ranks = [self.bm25.rank(q['question'], q['lang']) for q in questions]
        threshold = fit_threshold(questions, ranks, math.nextafter(0.0, math.inf))
        result = evaluate(questions, ranks, threshold)
        self.assertEqual(result['all']['questions'], 20)
        self.assertEqual(result['all']['unanswerable'], 4)
        self.assertEqual(result['by_language']['fr']['answerable'], 8)
        self.assertEqual(result['by_language']['ar']['answerable'], 8)


if __name__ == '__main__':
    unittest.main()
