"""Evaluate the deployed search policy on documentary validation only."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from App.backend.service.documents import CALIBRATION, Documents
from evaluate import K, retrieve_scores
from fetch_e5 import CACHE, MODEL
from search import CARDS, QUESTIONS, extract, load_cards, load_questions

CODE = (
    'App/backend/service/documents.py', 'App/backend/service/lexical.py',
    'rag/search.py', 'rag/dense.py', 'rag/fetch_e5.py', 'rag/evaluate.py',
    'rag/evaluate_application.py',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    import torch

    torch.set_num_threads(2)
    started = time.perf_counter()
    cards = load_cards()
    questions = [q for q in load_questions(cards) if q['split'] == 'validation']
    if len(questions) != 20:
        raise ValueError('La validation doit contenir exactement 20 questions.')
    documents = Documents().load(e5_directory=CACHE)
    if documents.errors or any(not row['available'] for row in documents.status().values()):
        raise RuntimeError(f'Moteur documentaire indisponible — {documents.errors}')
    systems = {}
    for method in ('bm25', 'e5'):
        rows = []
        for question in questions:
            hits, threshold = documents.retrieve(question['question'], question['lang'], method, k=K)
            response = extract(question['question'], question['lang'], hits, threshold)
            # Exercise the actual application entry point as a separate replay.
            replay = documents.ask(question['question'], question['lang'], method)
            if ((response['status'], response['answer'], [s['id'] for s in response['sources']])
                    != (replay['status'], replay['answer'], replay['source_ids'])):
                raise ValueError(f'Réponse applicative divergente pour {method}/{question["id"]}.')
            expected = set(question['relevant_ids'])
            positions = [rank for rank, (card, _) in enumerate(hits, 1) if card.id in expected]
            rows.append({
                **question,
                'ranking': [{'id': card.id, 'score': score} for card, score in hits],
                'recall': len(positions) / len(expected) if expected else None,
                'reciprocal_rank': 1 / positions[0] if positions else (0.0 if expected else None),
                'response': response,
            })
        systems[method] = {
            'all': retrieve_scores(rows),
            'by_language': {lang: retrieve_scores([r for r in rows if r['lang'] == lang])
                            for lang in ('fr', 'ar')},
            'rows': rows,
        }
        print(f'{method.upper()} — {systems[method]["all"]}', flush=True)
    return {
        'split': 'validation', 'top_k': K, 'cards': len(cards),
        'classification_test_evaluated': False, 'documentary_test_evaluated': False,
        'editorial_review': '44 decisions Conforme received, signature metadata pending',
        'independent_human_question_review': 'pending; prefilled forms are AI proposals',
        'relevance_annotation_origin': 'AI proposals, independent human review pending',
        'thresholds_refitted': False, 'calibration_sha256': sha(CALIBRATION),
        'thresholds': documents.thresholds,
        'cards_sha256': sha(CARDS), 'questions_sha256': sha(QUESTIONS),
        'code_sha256': {name: sha(ROOT / name) for name in CODE},
        'environment_sha256': sha(ROOT / 'App/backend/uv.lock'),
        'e5': {'model': MODEL['model_id'], 'revision': MODEL['revision'], 'device': 'cpu'},
        'packages': {name: importlib.metadata.version(name) for name in
                     ('torch', 'transformers', 'scikit-learn', 'numpy')},
        'python': platform.python_version(), 'platform': platform.platform(),
        'torch_threads': torch.get_num_threads(),
        'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'worktree_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)),
        'application_responses_replayed': 2 * len(questions),
        'seconds_including_load_and_replay': time.perf_counter() - started,
        'systems': systems,
    }


def verify(result):
    """Recalculate scores directly from saved rankings and literal responses."""
    cards = {card.id: card for card in load_cards()}
    expected = [q for q in load_questions(list(cards.values())) if q['split'] == 'validation']
    if sha(CARDS) != result['cards_sha256'] or sha(QUESTIONS) != result['questions_sha256']:
        raise ValueError('Catalogue ou annotations modifiés depuis cette exécution.')
    if any(sha(ROOT / name) != digest for name, digest in result['code_sha256'].items()):
        raise ValueError('Le code diffère de la version évaluée.')
    if sha(CALIBRATION) != result['calibration_sha256']:
        raise ValueError('La calibration diffère de la version évaluée.')
    for method, system in result['systems'].items():
        rows = system['rows']
        if [{key: row[key] for key in question} for row, question in zip(rows, expected, strict=True)] != expected:
            raise ValueError(f'Questions ou annotations désalignées pour {method}.')
        for row in rows:
            ranking = row['ranking']
            assert len(ranking) <= K and len({r['id'] for r in ranking}) == len(ranking)
            assert all(cards[r['id']].lang == row['lang'] for r in ranking)
            positions = [i for i, hit in enumerate(ranking, 1) if hit['id'] in row['relevant_ids']]
            assert row['recall'] == (len(positions) / len(row['relevant_ids']) if row['relevant_ids'] else None)
            assert row['reciprocal_rank'] == (1 / positions[0] if positions else
                                               (0.0 if row['relevant_ids'] else None))
            response = row['response']
            if response['status'] == 'answered':
                assert response['sources'][0]['id'] == ranking[0]['id']
                assert response['answer'] == cards[ranking[0]['id']].text
            else:
                assert response['status'] == 'insufficient_sources' and response['sources'] == []
        for lang, subset in [('all', rows), ('fr', [r for r in rows if r['lang'] == 'fr']),
                             ('ar', [r for r in rows if r['lang'] == 'ar'])]:
            supported = [r for r in subset if r['relevant_ids']]
            unsupported = [r for r in subset if not r['relevant_ids']]
            answered = [r for r in subset if r['response']['status'] == 'answered']
            metrics = {
                'questions': len(subset), 'answerable': len(supported), 'unanswerable': len(unsupported),
                'recall_at_3': statistics.mean(r['recall'] for r in supported),
                'mrr_at_3': statistics.mean(r['reciprocal_rank'] for r in supported),
                'answered': len(answered),
                'answerable_accepted': sum(r['response']['status'] == 'answered' for r in supported),
                'unanswerable_rejected': sum(r['response']['status'] == 'insufficient_sources' for r in unsupported),
                'answered_with_relevant_citation': sum(r['response']['sources'][0]['id'] in r['relevant_ids']
                                                     for r in answered),
            }
            assert metrics == (system['all'] if lang == 'all' else system['by_language'][lang])
    return {'rankings_and_metrics_recalculated': True, 'responses_literal_and_same_language': True,
            'questions_aligned': True, 'human_question_review': 'pending'}


def write_readme(output, result):
    lines = ['# Validation de la recherche documentaire applicative', '',
             'Comparaison sur les mêmes 20 questions de validation, dont 16 avec réponse et '
             '4 sans réponse dans les fiches. Le classement porte sur trois fiches dans '
             'la langue de la question. Les critères applicatifs sont conservés sans nouveau réglage.', '',
             '| Méthode | Recall@3 | MRR@3 | Réponses avec citation pertinente | Sans réponse correctement rejetées |',
             '| --- | ---: | ---: | ---: | ---: |']
    for method, system in result['systems'].items():
        metrics = system['all']
        lines.append(f'| {method.upper()} | {metrics["recall_at_3"]:.3f} | {metrics["mrr_at_3"]:.3f} | '
                     f'{metrics["answered_with_relevant_citation"]}/{metrics["answerable"]} | '
                     f'{metrics["unanswerable_rejected"]}/{metrics["unanswerable"]} |')
    lines.extend(['', 'Recall@3 et MRR@3 mesurent le classement avant abstention. '
                  'Les réponses mesurent les passages effectivement affichés après application des critères. '
                  'Les 40 réponses applicatives ont été rejouées et comparées aux décisions évaluées.', '',
                  'Les annotations de pertinence restent des propositions IA. Les questions de validation '
                  'ont déjà servi à la mise au point. Leurs scores ne constituent pas une évaluation indépendante. '
                  'La double revue humaine est en attente. Les tests documentaire et de classification '
                  'restent sans nouvelle évaluation.', '',
                  '[Résultats détaillés et empreintes](results.json) · [Vérification](verification.json)', ''])
    (output / 'README.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Nouveau dossier de résultats')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choisir un nouveau dossier de résultats.')
    result = run()
    verification = verify(result)
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'results.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    verification = {**verify(json.loads(path.read_text())), 'results_sha256': sha(path)}
    (args.output / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    write_readme(args.output, result)
    print(f'Rapport vérifié — {args.output.resolve()}')
