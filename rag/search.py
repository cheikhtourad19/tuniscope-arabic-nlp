"""Rank and cite versioned TuniScope documentation cards."""
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
import unicodedata

ROOT = Path(__file__).resolve().parents[1]
CARDS = ROOT / 'rag/cards_v1.json'
QUESTIONS = ROOT / 'rag/questions_v1.json'
LANGS = ('fr', 'ar')
ABSTAIN = {'fr': 'Les fiches disponibles ne suffisent pas pour répondre.',
           'ar': 'لا تكفي البطاقات المتاحة للإجابة عن هذا السؤال.'}


@dataclass(frozen=True)
class Card:
    id: str
    lang: str
    version: int
    title: str
    text: str
    author: str
    references: list[str]

    @property
    def passage(self):
        return f'{self.title}\n{self.text}'


def load_cards(path=CARDS):
    cards = [Card(**row) for row in json.loads(path.read_text())]
    if len({card.id for card in cards}) != len(cards):
        raise ValueError('Duplicate card identifier')
    for card in cards:
        if (card.lang not in LANGS or not re.fullmatch(r'[a-z][a-z0-9-]*\.' + card.lang, card.id)
                or type(card.version) is not int or card.version < 1
                or any(not field.strip() for field in (card.title, card.text, card.author))
                or not card.references):
            raise ValueError(f'Invalid card {card.id}')
        for reference in card.references:
            target = (ROOT / reference).resolve()
            if not target.is_relative_to(ROOT) or not target.is_file() or target.suffix != '.md':
                raise ValueError(f'Invalid reference in {card.id}: {reference}')
    return cards


def load_questions(cards, path=QUESTIONS):
    questions = json.loads(path.read_text())
    card_lang = {card.id: card.lang for card in cards}
    if len({q['id'] for q in questions}) != len(questions):
        raise ValueError('Duplicate question identifier')
    if len({q['question'] for q in questions}) != len(questions):
        raise ValueError('Duplicate question')
    expected = {('validation', 'fr'): 10, ('validation', 'ar'): 10, ('test', 'fr'): 20, ('test', 'ar'): 20}
    if Counter((q['split'], q['lang']) for q in questions) != expected:
        raise ValueError('Expected 20 validation and 40 test questions, bilingual')
    if sum(not q['relevant_ids'] for q in questions) < 10:
        raise ValueError('At least ten questions must have no answer')
    for q in questions:
        ids = q['relevant_ids']
        if not q['question'].strip() or len(set(ids)) != len(ids) or any(card_lang.get(i) != q['lang'] for i in ids):
            raise ValueError(f'Invalid question or relevance labels: {q["id"]}')
    return questions


def terms(text):
    text = unicodedata.normalize('NFD', text.casefold())
    text = ''.join(ch for ch in text if unicodedata.category(ch) != 'Mn' and ch != 'ـ')
    return re.findall(r'[^\W_]+', unicodedata.normalize('NFC', text))


def ordered(scores, k):
    return sorted(scores, key=lambda item: (-item[1], item[0].id))[:k]


class BM25:
    def __init__(self, cards):
        self.index = {}
        for lang in LANGS:
            docs = [(card, Counter(terms(card.passage))) for card in cards if card.lang == lang]
            if not docs:
                raise ValueError(f'Empty index for {lang}')
            frequencies = Counter(word for _, counts in docs for word in counts)
            average = sum(sum(counts.values()) for _, counts in docs) / len(docs)
            self.index[lang] = (docs, frequencies, average)

    def rank(self, text, lang, k=3):
        documents, frequencies, average = self.index[lang]
        tokens = set(terms(text))
        scores = []
        for card, counts in documents:
            norm = 1.2 * (0.25 + 0.75 * sum(counts.values()) / average)
            score = sum(math.log1p((len(documents) - frequencies[word] + 0.5) / (frequencies[word] + 0.5))
                        * counts[word] * 2.2 / (counts[word] + norm)
                        for word in sorted(tokens & counts.keys()))
            if score > 0:
                scores.append((card, score))
        return ordered(scores, k)


def extract(text, lang, hits, threshold):
    if not hits or not terms(text) or hits[0][1] < threshold:
        return {'status': 'insufficient_sources', 'answer': ABSTAIN[lang], 'sources': []}
    card, _ = hits[0]
    return {'status': 'answered', 'answer': card.text,
            'sources': [{'id': card.id, 'title': card.title, 'references': card.references}]}


if __name__ == '__main__':
    import argparse
    import hashlib

    parser = argparse.ArgumentParser(description='Réponse documentaire extractive BM25.')
    parser.add_argument('--question', required=True)
    parser.add_argument('--lang', choices=LANGS, required=True)
    parser.add_argument('--calibration', type=Path, required=True, help='Rapport BM25 de validation documentaire')
    args = parser.parse_args()
    proof = json.loads(args.calibration.read_text())
    if proof['split'] != 'validation' or proof['cards_sha256'] != hashlib.sha256(CARDS.read_bytes()).hexdigest():
        parser.error('Choisir un rapport de validation du même catalogue')
    threshold = proof['systems']['bm25']['threshold']
    hits = BM25(load_cards()).rank(args.question, args.lang)
    print(json.dumps(extract(args.question, args.lang, hits, threshold), ensure_ascii=False, indent=2))
