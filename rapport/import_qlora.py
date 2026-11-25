"""Verify the private QLoRA run and export only aggregate results for the report."""

import argparse
from contextlib import redirect_stdout
from io import StringIO
import json
from math import isclose
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))

from prepare_qwen4b import digest, load_plan
from run_qwen4b_few_shot import report, save_json
from run_qwen4b_qlora import BASELINE, verify_adapter


def read(path):
    return json.loads(path.read_text())


def export(run):
    plan = load_plan()
    training, completion = verify_adapter(run, plan)
    if training['smoke'] or training['split'] != 'train':
        raise ValueError('Le rapport exige le run QLoRA complet sur le train.')
    for name, sha in training['code_sha256'].items():
        if digest(ROOT / 'scripts' / name) != sha:
            raise ValueError(f'Le script source diffère du run enregistré : {name}')
    comparison_path = run / 'comparison.json'
    validation_path = run / 'validation_results.json'
    comparison = read(comparison_path)
    if (comparison['baseline_sha256'] != digest(BASELINE)
            or comparison['validation_sha256'] != digest(validation_path)
            or comparison['training_sha256'] != digest(run / 'completion.json')):
        raise ValueError('Les empreintes de la comparaison ne correspondent pas.')

    with TemporaryDirectory() as temp, redirect_stdout(StringIO()):
        base_output = Path(temp) / 'base.json'
        adapted_output = Path(temp) / 'adapted.json'
        report(plan, ROOT / 'outputs/qwen/qwen3_4b_few_shot_v1', base_output)
        report(plan, run / 'validation', adapted_output)
        baseline, adapted = read(base_output), read(adapted_output)
    if baseline != read(BASELINE) or adapted != read(validation_path):
        raise ValueError('Les rapports sauvegardés diffèrent du recalcul à partir des prédictions.')
    if ({k: v for k, v in baseline['run'].items() if k != 'code_sha256'}
            != {k: v for k, v in adapted['run'].items() if k not in {'code_sha256', 'adapter_sha256'}}
            or adapted['run']['adapter_sha256'] != completion['adapter_sha256']
            or training['manifest_hashes'] != adapted['data_manifests_sha256']):
        raise ValueError('Les deux conditions de la comparaison ne sont pas appariées.')
    results = {}
    for corpus, before in baseline['results'].items():
        after = adapted['results'][corpus]
        score = comparison['scores'][corpus]
        if (before['n'] != after['n'] or score['n'] != after['n']
                or before['demonstrations_sha256'] != after['demonstrations_sha256']
                or not isclose(after['macro_f1'] - before['macro_f1'], score['macro_f1_gain_points'] / 100)):
            raise ValueError(f'Contraste incohérent pour {corpus}.')
        keys = ('accuracy', 'macro_f1', 'invalid_rows')
        for field, metrics in [('base', before), ('qlora', after)]:
            if score[field] != {key: metrics[key] for key in keys}:
                raise ValueError(f'Scores incohérents pour {corpus}/{field}.')
        visible = (*keys, 'n', 'correct_rows', 'per_class', 'confusion_matrix', 'labels', 'seconds')
        results[corpus] = {'base': {key: before[key] for key in visible},
                           'qlora': {key: after[key] for key in visible}}

    log = (run / 'train.log').read_text()
    elapsed = re.findall(r'100\.0%.*écoulé (\d{2}:\d{2}:\d{2})', log)
    if not elapsed:
        raise ValueError('Le journal ne contient pas le temps d’apprentissage complet.')
    seconds = sum(int(part) * factor for part, factor in zip(elapsed[-1].split(':'), (3600, 60, 1), strict=True))
    return {'partition': 'validation', 'mode': 'few_shot', 'manifest_version': 'v2', 'training': training,
            'completion': completion, 'training_seconds_from_log': seconds,
            'adapter_bytes': (run / 'adapters.safetensors').stat().st_size, 'results': results,
            'source_sha256': {'comparison': digest(comparison_path), 'training_log': digest(run / 'train.log'),
                              'baseline': digest(BASELINE), 'validation': digest(validation_path)}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT / 'reports').resolve() not in output.parents:
        parser.error('Enregistrer les agrégats sous reports/.')
    result = export(args.run.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    save_json(output, result)
    print(f'Validation QLoRA recalculée et agrégats enregistrés dans {output}')
