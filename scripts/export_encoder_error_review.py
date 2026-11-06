"""Prepare private, blank qualitative review of saved encoder validation errors."""
import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path

from audit_text_composition import writing_system
from baseline_metrics import classification_metrics
from compare_encoder_errors import checked_predictions, compare_predictions, digest
from export_baseline_error_review import CATEGORIES, LABEL_OPINIONS
from load_working_data import ROOT, load_records
from normalize_text import normalize

BASELINE = ROOT / 'outputs/baselines/2026-10-17_grouped_v2_full'
ENCODER = ROOT / 'outputs/encoder_finetuning/2026-11-05_cls_paired_mps_full'
DIRECTORY = ROOT / 'outputs/01a0fda5-3c22-72f3-99cd-53dca78016f4/encoder_error_review_2026-11-06'
REVIEW_SEED = 42


def select_cases(candidates, count=30):
    """Rare-stratum-first round robin, unique groups and normalized text."""
    if len({r['id'] for r in candidates}) != len(candidates):
        raise ValueError('Repeated candidate ID')
    if any(r['split'] != 'validation' or r['truth'] == r['predicted'] for r in candidates):
        raise ValueError('Only validation errors can enter the review')
    if count < 1:
        raise ValueError('Positive review count required')
    buckets = defaultdict(list)
    for row in candidates:
        key = (row['comparison'], row['truth'], row['predicted'], row['writing_system'])
        buckets[key].append(row)
    queues = {key: deque(sorted(rows, key=lambda r: hashlib.sha256(
        f"{REVIEW_SEED}|{r['id']}".encode()).hexdigest())) for key, rows in buckets.items()}
    order = sorted(queues, key=lambda key: (len(queues[key]), key))
    chosen, groups, texts = [], set(), set()
    while len(chosen) < count:
        progressed = False
        for key in order:
            while queues[key]:
                row = queues[key].popleft()
                text = normalize(row['text_raw'])
                if row['group_id'] in groups or text in texts:
                    continue
                chosen.append(row); groups.add(row['group_id']); texts.add(text)
                progressed = True
                break
            if len(chosen) == count:
                break
        if not progressed:
            raise ValueError('Insufficient distinct groups/texts; do not silently duplicate cases')
    return chosen


def build_review():
    base = json.loads((BASELINE / 'results.json').read_text())
    enc = json.loads((ENCODER / 'results.json').read_text())
    proof = json.loads((ENCODER / 'verification.json').read_text())
    for meta in (base, enc):
        if meta['mode'] != 'full' or meta['test_loaded'] is not False or meta['config']['manifest_version'] != 'v2':
            raise ValueError('Expected FULL v2 validation-only selection')
    if enc['config']['variant'] != 'raw-v0' or proof['results_sha256'] != digest(ENCODER / 'results.json'):
        raise ValueError('Wrong encoder representation or prior verification')
    if proof['models_replayed'] != 12:
        raise ValueError('Expected prior replay evidence for all initial checkpoints')
    tasks = {}
    for corpus in ('tsac', 'madar'):
        manifest_hash = digest(ROOT / f'data/manifests/v2/{corpus}.csv')
        if any(m['manifests_sha256'][corpus] != manifest_hash for m in (base, enc)):
            raise ValueError('Manifest changed')
        records = load_records(corpus, splits=('validation',), manifest_version='v2')
        ref, = [r for r in base['results'] if
                (r['corpus'], r['system'], r['variant']) == (corpus, 'tfidf_char', 'raw-v0')]
        baseline = checked_predictions(BASELINE, ref['predictions'], ref['predictions_sha256'])
        compare_predictions([{**r, 'truth': r['label'], 'predicted': r['label']} for r in records], baseline)
        adapted = [r for r in enc['results'] if r['corpus'] == corpus and r['regime'] == 'finetuned']
        if sorted(r['seed'] for r in adapted) != [13, 42, 2026]:
            raise ValueError('Missing or repeated encoder seeds')
        predictions, hashes = {}, {}
        for result in adapted:
            name = result['checkpoint'] + '/predictions.json'
            hashes[str(result['seed'])] = enc['artifacts'][name]
            rows = checked_predictions(ENCODER, name, enc['artifacts'][name])
            compare_predictions(baseline, rows)
            metrics = classification_metrics([r['truth'] for r in rows], [r['predicted'] for r in rows], enc['config']['labels'][corpus])
            if metrics != {k: result['metrics'][k] for k in metrics}:
                raise ValueError('Encoder metrics do not match saved predictions')
            predictions[result['seed']] = rows
        candidates = []
        for i, (row, b, p) in enumerate(zip(records, baseline, predictions[REVIEW_SEED])):
            if p['predicted'] == p['truth']:
                continue
            candidates.append({**row, **p, 'baseline_predicted': b['predicted'],
                'comparison': 'Erreur introduite' if b['predicted'] == b['truth'] else 'Erreur commune',
                'writing_system': writing_system(row['text_raw']),
                'predictions_by_seed': {str(seed): predictions[seed][i]['predicted'] for seed in (13, 42, 2026)}})
        selected = select_cases(candidates)
        cases = [{**r, 'case_id': f'ENC-{corpus.upper()}-{i:02d}',
                  'label_opinion': '', 'category': '', 'hypothesis': ''}
                 for i, r in enumerate(selected, 1)]
        tasks[corpus] = {'categories': CATEGORIES[corpus], 'manifest_sha256': manifest_hash,
            'baseline_predictions_sha256': ref['predictions_sha256'], 'encoder_predictions_sha256': hashes,
            'summary': {'candidate_errors': len(candidates), 'selected_cases': len(cases),
                'selected_groups': len({r['group_id'] for r in cases}),
                'selected_comparisons': dict(Counter(r['comparison'] for r in cases)),
                'selected_writing_systems': dict(Counter(r['writing_system'] for r in cases))}, 'cases': cases}
    return {'schema': 'encoder-error-review-v1', 'reference_seed': REVIEW_SEED,
        'selection_rule': 'SHA256(42|id), rare-stratum-first round robin by comparison/truth/prediction/writing; unique groups and NFC-space text',
        'purpose': 'Qualitative diagnosis, not population prevalence or independent reannotation',
        'official_test_rows_selected': 0, 'human_fields_prefilled': 0, 'label_opinions': LABEL_OPINIONS,
        'baseline_results_sha256': digest(BASELINE / 'results.json'),
        'encoder_results_sha256': digest(ENCODER / 'results.json'), 'exporter_sha256': digest(Path(__file__)), 'tasks': tasks}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DIRECTORY / 'review_payload.json')
    destination = p.parse_args().output
    if destination.exists():
        raise FileExistsError('Preserve existing review payload and human work')
    payload = build_review()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({c: t['summary'] for c, t in payload['tasks'].items()}, ensure_ascii=False, indent=2))
