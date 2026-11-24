"""Freeze and evaluate documentary test retrieval with explicit reference limitations."""
import argparse
from collections import Counter
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from App.backend.service.documents import (
    BM25_MIN_RELATIVE_MARGIN, BM25_MIN_SCORE_WITH_MARGIN, CALIBRATION, Documents,
)
from App.backend.service.lexical import MIN_RELATIVE_LEAD, MIN_TERM_COVERAGE, SHORT_QUERY_MAX_TERMS
from evaluate import K, retrieve_scores, sha
from evaluate_application import CODE, verify as verify_validation
from fetch_e5 import CACHE, MODEL, PLAN
from search import CARDS, QUESTIONS, extract, load_cards, load_questions

VALIDATION = ROOT / 'reports/rag_application_validation_2026-11-11/results.json'
PROTOCOL_DOCUMENT = ROOT / 'docs/70_protocole_gel_documentaire.md'


def policies():
    calibration = json.loads(CALIBRATION.read_text())
    return {
        'top_k': K, 'response': 'literal_first_card_or_abstention', 'card_language': 'question_language',
        'thresholds': {method: calibration['systems'][method]['threshold'] for method in ('bm25', 'e5')},
        'bm25_short_max_terms': SHORT_QUERY_MAX_TERMS, 'bm25_short_min_coverage': MIN_TERM_COVERAGE,
        'bm25_short_min_relative_lead': MIN_RELATIVE_LEAD,
        'bm25_long_min_score_with_margin': BM25_MIN_SCORE_WITH_MARGIN,
        'bm25_long_min_relative_margin': BM25_MIN_RELATIVE_MARGIN,
        'e5_model': MODEL['model_id'], 'e5_revision': MODEL['revision'], 'device': 'cpu',
    }


def check_fingerprints(fingerprints, root=ROOT):
    for name, digest in fingerprints.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha(path) != digest:
            raise ValueError(f'Fichier différent de la version gelée — {name}')


def freeze(path):
    if path.exists():
        raise FileExistsError('Un protocole existant doit être conservé.')
    verify_validation(json.loads(VALIDATION.read_text()))
    cards = load_cards()
    questions = load_questions(cards)
    selected = [q for q in questions if q['split'] == 'test']
    sample = ROOT / 'rag/review_sample_v1.md'
    sample_ids = set(re.findall(r'^\| (Q\d+) \|', sample.read_text(), flags=re.M))
    files = set(CODE) | {
        'rag/final_evaluation.py', 'App/backend/service/models.py', 'scripts/normalize_text.py',
        str(CARDS.relative_to(ROOT)), str(QUESTIONS.relative_to(ROOT)), str(PLAN.relative_to(ROOT)),
        str(CALIBRATION.relative_to(ROOT)), str(VALIDATION.relative_to(ROOT)),
        'App/backend/uv.lock', 'rag/review_sample_v1.md',
        str(PROTOCOL_DOCUMENT.relative_to(ROOT)),
        'reports/rag_card_review_2026-11-11/summary.json',
    } | {reference for card in cards for reference in card.references}
    protocol = {
        'version': 1, 'frozen_on': '2026-11-12', 'split': 'test',
        'scope': 'documentary_retrieval_and_extractive_responses',
        'policies': policies(), 'questions': len(selected),
        'question_ids': [q['id'] for q in selected],
        'questions_by_language': dict(Counter(q['lang'] for q in selected)),
        'answerable': sum(bool(q['relevant_ids']) for q in selected),
        'unanswerable': sum(not q['relevant_ids'] for q in selected),
        'previously_disclosed_sample_test_ids': [q['id'] for q in selected if q['id'] in sample_ids],
        'editorial_card_decisions': '44 Conforme received, reviewer metadata incomplete',
        'relevance_origin': 'AI proposals', 'independent_human_review': 'pending',
        'pedagogical_waiver': 'not_confirmed', 'interpretation': 'provisional_reference_labels',
        'classification_test_in_scope': False,
        'files_sha256': {name: sha(ROOT / name) for name in sorted(files)},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return check(path)


def check(path):
    protocol = json.loads(path.read_text())
    if (protocol['version'] != 1 or protocol['split'] != 'test'
            or protocol['policies'] != policies() or protocol['questions'] != 40
            or protocol['questions_by_language'] != {'fr': 20, 'ar': 20}
            or protocol['answerable'] + protocol['unanswerable'] != 40):
        raise ValueError('Protocole documentaire incompatible.')
    required = set(CODE) | {'rag/final_evaluation.py', str(CARDS.relative_to(ROOT)),
                            str(QUESTIONS.relative_to(ROOT)), str(PLAN.relative_to(ROOT)),
                            str(CALIBRATION.relative_to(ROOT)), str(PROTOCOL_DOCUMENT.relative_to(ROOT))}
    if not required.issubset(protocol['files_sha256']):
        raise ValueError('Empreintes obligatoires absentes du protocole.')
    check_fingerprints(protocol['files_sha256'])
    return protocol


def verify_rows(rows, questions, cards):
    if len(rows) != len(questions):
        raise ValueError('Nombre de questions différent du protocole.')
    for row, question in zip(rows, questions, strict=True):
        if {key: row[key] for key in question} != question:
            raise ValueError('Question ou référence pertinente désalignée.')
        ranking = row['ranking']
        ids = [hit['id'] for hit in ranking]
        if (len(ids) > K or len(set(ids)) != len(ids)
                or any(identifier not in cards or cards[identifier].lang != row['lang'] for identifier in ids)
                or any(not math.isfinite(hit['score']) for hit in ranking)):
            raise ValueError('Classement documentaire invalide.')
        positions = [i for i, identifier in enumerate(ids, 1) if identifier in question['relevant_ids']]
        recall = len(positions) / len(question['relevant_ids']) if question['relevant_ids'] else None
        reciprocal = 1 / positions[0] if positions else (0.0 if question['relevant_ids'] else None)
        if row['recall'] != recall or row['reciprocal_rank'] != reciprocal:
            raise ValueError('Métrique par question incorrecte.')
        response = row['response']
        if response['status'] == 'answered':
            if (not ids or response['sources'] != [{'id': ids[0], 'title': cards[ids[0]].title,
                                                    'references': cards[ids[0]].references}]
                    or response['answer'] != cards[ids[0]].text):
                raise ValueError('Réponse différente du passage cité.')
        elif response != extract(question['question'], question['lang'], [], 0):
            raise ValueError('Abstention documentaire invalide.')


def metrics(rows):
    result = retrieve_scores(rows)
    return {
        **result,
        'answer_coverage': result['answered'] / result['questions'],
        'relevant_citation_rate_among_answers': (result['answered_with_relevant_citation'] / result['answered']
                                                if result['answered'] else None),
        'answers_with_irrelevant_citation': result['answered'] - result['answered_with_relevant_citation'],
        'answerable_abstained': result['answerable'] - result['answerable_accepted'],
        'unanswerable_answered': result['unanswerable'] - result['unanswerable_rejected'],
    }


def aggregate(rows):
    return {'all': metrics(rows),
            'by_language': {lang: metrics([row for row in rows if row['lang'] == lang]) for lang in ('fr', 'ar')}}


def evaluate(protocol_path, output):
    if output.exists():
        raise FileExistsError('Choisir un nouveau dossier, sans remplacer un test existant.')
    protocol = check(protocol_path)
    protocol_digest = sha(protocol_path)
    cards = load_cards()
    questions = [q for q in load_questions(cards) if q['split'] == 'test']
    if [q['id'] for q in questions] != protocol['question_ids']:
        raise ValueError('Identifiants différents de la sélection gelée.')
    import torch
    torch.set_num_threads(2)
    started = time.perf_counter()
    documents = Documents().load(e5_directory=CACHE)
    if documents.errors or any(not status['available'] for status in documents.status().values()):
        raise RuntimeError(f'Moteur indisponible — {documents.errors}')
    systems = {}
    for method in ('bm25', 'e5'):
        rows = []
        for question in questions:
            hits, threshold = documents.retrieve(question['question'], question['lang'], method, k=K)
            response = extract(question['question'], question['lang'], hits, threshold)
            replay = documents.ask(question['question'], question['lang'], method)
            if ((response['status'], response['answer'], [s['id'] for s in response['sources']])
                    != (replay['status'], replay['answer'], replay['source_ids'])):
                raise ValueError('Le rejeu de la réponse applicative diverge.')
            expected = set(question['relevant_ids'])
            positions = [i for i, (card, _) in enumerate(hits, 1) if card.id in expected]
            rows.append({**question, 'ranking': [{'id': card.id, 'score': score} for card, score in hits],
                         'recall': len(positions) / len(expected) if expected else None,
                         'reciprocal_rank': 1 / positions[0] if positions else (0.0 if expected else None),
                         'response': response})
        verify_rows(rows, questions, {card.id: card for card in cards})
        systems[method] = {**aggregate(rows), 'rows': rows}
        print(f'{method.upper()} — {systems[method]["all"]}', flush=True)
    result = {
        'split': 'test', 'scope': 'documentary_only', 'protocol_sha256': protocol_digest,
        'top_k': K, 'thresholds_refitted': False, 'classification_test_evaluated': False,
        'reference_review': protocol['independent_human_review'],
        'reference_origin': protocol['relevance_origin'], 'interpretation': protocol['interpretation'],
        'previously_disclosed_sample_test_ids': protocol['previously_disclosed_sample_test_ids'],
        'application_responses_replayed': 2 * len(questions), 'device': 'cpu', 'torch': torch.__version__,
        'python': platform.python_version(), 'platform': platform.platform(),
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'scikit-learn', 'numpy')},
        'torch_threads': torch.get_num_threads(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)),
        'seconds_including_load_and_replay': time.perf_counter() - started, 'systems': systems,
    }
    check(protocol_path)
    if sha(protocol_path) != protocol_digest:
        raise ValueError('Le protocole a changé pendant le calcul.')
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    saved = json.loads(path.read_text())
    for system in saved['systems'].values():
        verify_rows(system['rows'], questions, {card.id: card for card in cards})
        if aggregate(system['rows']) != {key: system[key] for key in ('all', 'by_language')}:
            raise ValueError('Agrégats enregistrés incorrects.')
    (output / 'verification.json').write_text(json.dumps({
        'results_sha256': sha(path), 'protocol_sha256': protocol_digest,
        'questions_rankings_and_metrics_verified': True, 'literal_citations_verified': True,
        'request_replays': 80, 'human_factual_review': 'pending',
    }, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    freeze_parser = sub.add_parser('freeze')
    freeze_parser.add_argument('--output', required=True, type=Path)
    check_parser = sub.add_parser('check')
    check_parser.add_argument('--protocol', required=True, type=Path)
    evaluate_parser = sub.add_parser('evaluate')
    evaluate_parser.add_argument('--protocol', required=True, type=Path)
    evaluate_parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'freeze':
        result = freeze(args.output)
        print(f'Protocole gelé — {args.output.resolve()}')
    elif args.command == 'check':
        result = check(args.protocol)
        print(f'Protocole vérifié — {result["questions"]} questions, revue indépendante en attente.')
    else:
        evaluate(args.protocol, args.output)
        print(f'Évaluation documentaire vérifiée — {args.output.resolve()}')
