"""Documentary search, with source links confined to the versioned catalogue."""
import hashlib
import json
import math
import sys
from threading import Lock
from urllib.parse import quote

from .models import ROOT

sys.path.insert(0, str(ROOT / 'rag'))
from search import BM25, CARDS, extract, load_cards

CALIBRATION = ROOT / 'reports/rag_validation_v1_2026-11-09/retrieval.json'


class Documents:
    def __init__(self):
        self.bm25 = self.e5 = None
        self.errors = {}
        self.references = {}
        self.thresholds = {}
        self.metrics = None
        self.lock = Lock()

    def load(self, e5_directory=None):
        try:
            cards = load_cards()
            proof = json.loads(CALIBRATION.read_text())
            if proof['split'] != 'validation' or proof['cards_sha256'] != hashlib.sha256(CARDS.read_bytes()).hexdigest():
                raise ValueError('Seuils RAG incompatibles avec les fiches.')
            for name in ('bm25', 'e5'):
                score = proof['systems'][name]['threshold']
                if not math.isfinite(score):
                    raise ValueError('Seuil documentaire non fini.')
                self.thresholds[name] = score
            self.bm25 = BM25(cards)
            self.references = {path: ROOT / path for card in cards for path in card.references}
            self.metrics = {"split": "validation", "top_k": proof["top_k"],
                            "results": {method: proof["systems"][method]["all"] for method in ("bm25", "e5")}}
        except (OSError, ValueError, KeyError) as error:
            self.errors['bm25'] = f'Catalogue documentaire indisponible ({type(error).__name__}).'
            return self
        if e5_directory is not None:
            try:
                from dense import DenseE5
                from fetch_e5 import MODEL
                if (proof['systems']['e5']['model'], proof['systems']['e5']['revision']) != (
                        MODEL['model_id'], MODEL['revision']):
                    raise ValueError('Calibration E5 incompatible.')
                self.e5 = DenseE5(cards, 'cpu', directory=e5_directory)
            except Exception as error:
                self.errors['e5'] = f'Recherche dense indisponible ({type(error).__name__}).'
        return self

    def status(self):
        return {method: {'available': getattr(self, method) is not None, 'reason': self.errors.get(method)}
                for method in ('bm25', 'e5')}

    def ask(self, question, lang, method):
        engine = getattr(self, method)
        if engine is None:
            return {'status': 'unavailable', 'answer': None, 'sources': [], 'source_ids': [], 'excerpts': [],
                    'reason': self.errors.get(method) or 'Moteur documentaire indisponible.',
                    'method': method, 'lang': lang}
        if method == 'e5':
            with self.lock:
                hits, = engine.rank_many([{'question': question, 'lang': lang}])
        else:
            hits = engine.rank(question, lang)
        result = extract(question, lang, hits, self.thresholds[method])
        for source in result['sources']:
            source['links'] = [{'label': path, 'url': f'/api/documentation/{quote(path, safe="/")}'}
                               for path in source['references']]
        return {**result, 'method': method, 'lang': lang, 'source_ids': [s['id'] for s in result['sources']],
                'excerpts': [result['answer']] if result['sources'] else [], 'reason': None}
