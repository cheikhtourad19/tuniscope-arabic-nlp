"""Verify Qwen validation predictions and publish aggregate metrics."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from audit_text_composition import writing_system
from baseline_metrics import classification_metrics
from encoder_tokenizer_audit import object_digest
from fetch_qwen3 import MODEL_ID, REVISION, ROOT, load_plan, sha256
from load_working_data import load_records
from qwen_prompting import PROMPT_CONFIG, parse_label

INVALID = 'INVALID'
MODES = ('zero_shot', 'few_shot')
AUDIT = ROOT / 'reports/qwen_tokenizer_audit_v2_2026-11-07/summary.json'


def score(rows, labels):
    truth = [row['truth'] for row in rows]
    predicted = [row['prediction'] if row['prediction'] is not None else INVALID for row in rows]
    metrics = classification_metrics(truth, predicted, [*labels, INVALID])
    metrics['labels'] = labels
    metrics['prediction_columns'] = [*labels, INVALID]
    metrics['confusion_matrix'] = metrics['confusion_matrix'][:len(labels)]
    del metrics['per_class'][INVALID]
    metrics['macro_f1'] = statistics.mean(metrics['per_class'][label]['f1-score'] for label in labels)
    invalid = predicted.count(INVALID)
    return {**metrics, 'correct_rows': sum(y == p for y, p in zip(truth, predicted, strict=True)),
            'invalid_rows': invalid, 'valid_rows': len(rows) - invalid, 'valid_rate': 1 - invalid / len(rows)}


def aggregate(run):
    config = json.loads(PROMPT_CONFIG.read_text())
    audit = json.loads(AUDIT.read_text())
    plan_path = run / 'plan.json'
    plan = json.loads(plan_path.read_text())
    completion = json.loads((run / 'completion.json').read_text())
    stages = {f'{corpus}_{mode}' for corpus in config['tasks'] for mode in MODES}
    if completion['plan_sha256'] != sha256(plan_path) or set(completion['stages']) != stages:
        raise ValueError('Expected a complete four-stage validation run')
    expected_generation = {**config['initial_generation'], 'max_new_tokens': config['max_new_tokens'], 'use_cache': True}
    official_weight = load_plan()['files']['model.safetensors']['sha256']
    if (plan['model_id'] != MODEL_ID or plan['revision'] != REVISION or plan['split'] != 'validation'
            or plan['demonstration_split'] != 'train' or plan['training'] is not False
            or plan['variant'] != 'raw-v0' or plan['prompt_config_sha256'] != sha256(PROMPT_CONFIG)
            or plan['audit_sha256'] != sha256(AUDIT) or plan['model_sha256'] != official_weight
            or plan['generation'] != expected_generation or plan['manifests_sha256'] != audit['manifests_sha256']):
        raise ValueError('Validation provenance differs from the pinned protocol')
    for filename, expected in plan['code_sha256'].items():
        if sha256(ROOT / 'scripts' / filename) != expected:
            raise ValueError(f'Generation source differs from the run: {filename}')
    results = {}
    for corpus, task in config['tasks'].items():
        if sha256(ROOT / f'data/manifests/v2/{corpus}.csv') != plan['manifests_sha256'][corpus]:
            raise ValueError('Current validation manifest differs from the run')
        validation = load_records(corpus, splits=('validation',), manifest_version='v2')
        expected_rows = [(row['id'], row['group_id'], row['label'], writing_system(row['text_raw'])) for row in validation]
        audit_partition = audit['corpora'][corpus]['partitions']['validation']
        if object_digest([row['id'] for row in validation]) != audit_partition['ids_sha256']:
            raise ValueError('Validation identifiers differ from the audit')
        for mode in MODES:
            name = f'{corpus}_{mode}'
            path = run / f'{name}.json'
            if sha256(path) != completion['stages'][name]:
                raise ValueError(f'Prediction file differs from the completion record: {name}')
            stage = json.loads(path.read_text())
            rows = stage['predictions']
            if (stage['plan_sha256'] != completion['plan_sha256'] or (stage['corpus'], stage['mode']) != (corpus, mode)
                    or [(row['id'], row['group_id'], row['truth'], row['writing']) for row in rows] != expected_rows):
                raise ValueError(f'Validation alignment differs: {name}')
            groups = defaultdict(list)
            for row in rows:
                if (parse_label(row['response'], task['labels']) != row['prediction']
                        or not 0 < row['prompt_tokens'] <= audit['length_decision']['max_prompt_tokens']
                        or not 0 < row['generated_tokens'] <= config['max_new_tokens']
                        or (not row['ended_with_eos'] and row['generated_tokens'] != config['max_new_tokens'])):
                    raise ValueError(f'Invalid parser or length evidence: {name}')
                groups[row['writing']].append(row)
            if max(row['prompt_tokens'] for row in rows) != audit_partition['all']['lengths'][mode]['max']:
                raise ValueError(f'Prompt maximum differs from the audit: {name}')
            results[name] = {'corpus': corpus, 'mode': mode, 'seconds': stage['seconds'],
                             **score(rows, task['labels']),
                             'mean_generated_tokens': statistics.mean(row['generated_tokens'] for row in rows),
                             'rows_without_eos': sum(not row['ended_with_eos'] for row in rows),
                             'invalid_rows_without_eos': sum(row['prediction'] is None and not row['ended_with_eos'] for row in rows),
                             'by_writing': {name: score(items, task['labels']) for name, items in sorted(groups.items())}}
    return {'run': str(run.relative_to(ROOT)), 'plan': plan, 'completion': completion,
            'total_predictions': sum(stage['n'] for stage in results.values()), 'stages': results,
            'report_script_sha256': sha256(Path(__file__)), 'metrics_code_sha256': sha256(ROOT / 'scripts/baseline_metrics.py'),
            'metrics_lock_sha256': sha256(ROOT / 'uv.lock')}


def markdown(result):
    lines = ['# Référence Qwen3-0.6B sur la validation v2', '',
             'Inférence avec consignes v2 et décodage glouton. Les métriques couvrent toutes les lignes. '
             'Une sortie invalide compte comme une erreur. Le macro-F1 est la moyenne des F1 des labels réels.', '',
             '| Corpus | Mode | Lignes | Sorties valides | Exactitude | Macro-F1 | Durée de génération |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for stage in result['stages'].values():
        lines.append(f'| {stage["corpus"].upper()} | {stage["mode"]} | {stage["n"]} | '
                     f'{stage["valid_rows"]}/{stage["n"]} ({stage["valid_rate"]:.2%}) | '
                     f'{stage["accuracy"]:.4f} | {stage["macro_f1"]:.4f} | {stage["seconds"]:.2f} s |')
    lines += ['', f'**{result["total_predictions"]:,} prédictions**. '
              f'Exécution sur {result["plan"]["device"]} en {result["plan"]["dtype"]}, '
              f'avec des lots de {result["plan"]["batch_size"]} et un tri par longueur des prompts. '
              f'Durée de la session mesurée à **{result["completion"]["seconds_this_execution"]:.2f} s**.', '',
              'Le JSON contient les matrices avec une colonne `INVALID`, les métriques par classe, '
              'les groupes d’écriture et les empreintes de vérification.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--verify', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    target = (args.output or args.verify).resolve()
    if (ROOT / 'outputs/qwen').resolve() not in run.parents or (ROOT / 'reports').resolve() not in target.parents:
        parser.error('Use a private run under outputs/qwen/ and an aggregate report under reports/')
    if args.output and target.exists():
        raise FileExistsError(target)
    result = aggregate(run)
    if args.verify:
        if json.loads((target / 'summary.json').read_text()) != result or (target / 'README.md').read_text() != markdown(result):
            raise ValueError('Saved report differs from recalculated metrics')
        print('Rapport Qwen recalculé à l’identique.')
        return
    target.mkdir(parents=True)
    (target / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (target / 'README.md').write_text(markdown(result))
    print(markdown(result))


if __name__ == '__main__':
    main()
