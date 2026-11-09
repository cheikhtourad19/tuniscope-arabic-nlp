"""Evaluate BM25 and E5 on the v1 documentary validation questions."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

from search import BM25, CARDS, QUESTIONS, ROOT, extract, load_cards, load_questions

PROTOCOL = ROOT / 'docs/66_protocole_rag_documentaire.md'
K = 3


def fit_threshold(questions, ranks, min_score):
    eligible = [hits[0][1] for hits in ranks if hits and hits[0][1] >= min_score]
    if not eligible:
        return min_score
    candidates = {min_score, *eligible, math.nextafter(max(eligible), math.inf)}
    def objective(threshold):
        accepts = [bool(hits) and hits[0][1] >= threshold for hits in ranks]
        acceptance = statistics.mean(flag for flag, q in zip(accepts, questions, strict=True) if q['relevant_ids'])
        rejection = statistics.mean(not flag for flag, q in zip(accepts, questions, strict=True) if not q['relevant_ids'])
        return ((acceptance + rejection) / 2, rejection, threshold)
    return max(candidates, key=objective)


def retrieve_scores(rows):
    answerable = [row for row in rows if row['relevant_ids']]
    unsupported = [row for row in rows if not row['relevant_ids']]
    returned = [row for row in rows if row['response']['status'] == 'answered']
    return {'questions': len(rows), 'answerable': len(answerable), 'unanswerable': len(unsupported),
            'recall_at_3': statistics.mean(row['recall'] for row in answerable),
            'mrr_at_3': statistics.mean(row['reciprocal_rank'] for row in answerable),
            'answered': len(returned), 'answerable_accepted': sum(row in returned for row in answerable),
            'unanswerable_rejected': sum(row not in returned for row in unsupported),
            'answered_with_relevant_citation': sum(
                row['response']['sources'][0]['id'] in row['relevant_ids'] for row in returned)}


def evaluate(questions, ranks, threshold):
    rows = []
    for question, hits in zip(questions, ranks, strict=True):
        gold = set(question['relevant_ids'])
        positions = [position for position, (card, _) in enumerate(hits, 1) if card.id in gold]
        rows.append({**question, 'ranking': [{'id': card.id, 'score': score} for card, score in hits],
                     'recall': len(positions) / len(gold) if gold else None,
                     'reciprocal_rank': (1 / positions[0] if positions else 0.0) if gold else None,
                     'response': extract(question['question'], question['lang'], hits, threshold)})
    return {'threshold': threshold, 'all': retrieve_scores(rows),
            'by_language': {lang: retrieve_scores([row for row in rows if row['lang'] == lang])
                            for lang in ('fr', 'ar')}, 'rows': rows}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(methods, device):
    cards = load_cards()
    questions = [q for q in load_questions(cards) if q['split'] == 'validation']
    results = {}
    if 'bm25' in methods:
        ranker = BM25(cards)
        ranks = [ranker.rank(q['question'], q['lang'], K) for q in questions]
        results['bm25'] = evaluate(questions, ranks, fit_threshold(questions, ranks, math.nextafter(0.0, math.inf)))
    if 'e5' in methods:
        import torch
        from dense import DenseE5
        from fetch_e5 import MODEL, PLAN
        torch.set_num_threads(2)
        ranks = DenseE5(cards, device).rank_many(questions, K)
        results['e5'] = {**evaluate(questions, ranks, fit_threshold(questions, ranks, -1.0)),
                         'model': MODEL['model_id'], 'revision': MODEL['revision'],
                         'plan_sha256': sha(PLAN), 'device': device, 'torch': torch.__version__}
    return {'split': 'validation', 'classification_test_loaded': False, 'human_review': 'pending',
            'top_k': K, 'cards': len(cards), 'cards_sha256': sha(CARDS), 'questions_sha256': sha(QUESTIONS),
            'protocol_sha256': sha(PROTOCOL),
            'code_sha256': {name: sha(ROOT / 'rag' / name) for name in ('search.py', 'dense.py', 'evaluate.py')},
            'systems': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=['bm25', 'e5', 'both'], default='both')
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choose a new report path')
    result = run(('bm25', 'e5') if args.method == 'both' else (args.method,), args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    for name, metrics in result['systems'].items():
        print(name, metrics['all'], flush=True)
