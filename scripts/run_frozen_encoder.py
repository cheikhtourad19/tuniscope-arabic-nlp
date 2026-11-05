"""Extract fixed CAMeLBERT vectors and train separate linear heads; never test."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time

import numpy as np
import torch

from bigru_data import partitions
from bigru_model import resolve_device, set_seed
from bigru_training import add_writing_slices
from character_lm_experiment import ROOT, object_digest, partition_summary
from fetch_camelbert_encoder import sha256
from frozen_encoder import extract_features, fit_head, load_encoder, state_digest

CONFIG = ROOT / 'configs/camelbert_frozen_v1.json'
CODE = ('frozen_encoder.py', 'run_frozen_encoder.py', 'verify_frozen_encoder.py', 'fetch_camelbert_encoder.py')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def validate_config(config):
    expected = json.loads(CONFIG.read_text())
    if config != expected or config['manifest_version'] != 'v2' or config['variant'] != 'raw-v0':
        raise ValueError('Use the fixed initial frozen-encoder protocol')


def row_keys(rows):
    return [{'id': r['id'], 'group_id': r['group_id'], 'label': r['label'],
             'text_raw_sha256': object_digest(r['text_raw'])} for r in rows]


def memory(device):
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = {'process_peak_rss_bytes': rss if platform.system() == 'Darwin' else rss * 1024,
              'note': 'Process RSS and device allocations are different, not additive peak measures.'}
    if device.type == 'mps':
        result.update(mps_current_allocated_bytes=torch.mps.current_allocated_memory(),
                      mps_driver_allocated_bytes=torch.mps.driver_allocated_memory(),
                      mps_recommended_max_bytes=torch.mps.recommended_max_memory())
    elif device.type == 'cuda':
        result.update(cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    return result


def summary(result):
    lines = [f'CAMeLBERT gelé — {result["mode"].upper()} — encodeur {result["device"]}, tête CPU',
             'Train/validation v2 ; test officiel non évalué.',
             f'Backbone : {result["encoder"]["backbone_parameters"]:,} paramètres, 0 entraînable.',
             'Corpus    Graine   Accuracy   Macro-F1   Époque retenue']
    for r in result['results']:
        lines.append(f'{r["corpus"].upper():<10}{r["seed"]:<9}{r["metrics"]["accuracy"]:<11.6f}'
                     f'{r["metrics"]["macro_f1"]:<11.6f}{r["best_epoch"]}')
    for corpus, item in result['aggregates'].items():
        lines.append(f'{corpus.upper()} macro-F1 : {item["mean_macro_f1"]:.6f}'
                     + (f' ± {item["sample_std_macro_f1"]:.6f}' if item['sample_std_macro_f1'] is not None else ' (essai, une graine)'))
    lines.append(f'Durée totale : {result["total_seconds"]:.1f} s ; encodeur inchangé : {result["encoder_unchanged"]}')
    return '\n'.join(lines)


def run(mode, output, device_request=None, smoke_run=None):
    config = json.loads(CONFIG.read_text())
    validate_config(config)
    if mode not in {'smoke', 'full'}:
        raise ValueError('Expected smoke/full')
    output = output.resolve()
    if (ROOT / 'outputs/frozen_encoder').resolve() not in output.parents or output.exists():
        raise ValueError('Choose a new directory under outputs/frozen_encoder/')
    if mode == 'full':
        if smoke_run is None:
            raise ValueError('Full run requires a verified smoke run')
        preliminary = json.loads((smoke_run / 'results.json').read_text())
        verified = json.loads((smoke_run / 'verification.json').read_text())
        if (preliminary['mode'] != 'smoke' or preliminary['config'] != config
                or verified['results_sha256'] != sha256(smoke_run / 'results.json')):
            raise ValueError('Smoke verification/configuration mismatch')
    set_seed(config['smoke_seed'], config['cpu_threads'])
    device = resolve_device(device_request or config['device'])
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    tick = time.perf_counter()
    print(f'Chargement local CAMeLBERT sur {device} — aucun téléchargement implicite.', flush=True)
    tokenizer, model, encoder_info = load_encoder(device)
    initial_digest = state_digest(model)
    output.mkdir(parents=True)
    result = {'mode': mode, 'config': config, 'test_loaded': False, 'device': str(device),
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
              'python': platform.python_version(), 'encoder': encoder_info, 'features': {}, 'results': [],
              'environment_lock_sha256': sha256(ROOT / 'environments/encoder/uv.lock'),
              'classical_lock_sha256': sha256(ROOT / 'uv.lock'),
              'bigru_lock_sha256': sha256(ROOT / 'environments/bigru/uv.lock'),
              'tokenizer_lock_sha256': sha256(ROOT / 'environments/tokenizer/uv.lock'),
              'packages': {p: importlib.metadata.version(p) for p in ('torch', 'transformers', 'tokenizers', 'numpy', 'scikit-learn')},
              'git_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'git_dirty': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)),
              'config_sha256': sha256(CONFIG), 'protocol_sha256': sha256(ROOT / 'docs/44_protocole_encodeur_gele.md'),
              'code_sha256': {name: sha256(ROOT / 'scripts' / name) for name in CODE},
              'manifests_sha256': {c: sha256(ROOT / f'data/manifests/v2/{c}.csv') for c in config['corpora']},
              'artifacts': {}, 'smoke_run': str(smoke_run) if smoke_run else None,
              'backbone_state_before_sha256': initial_digest}
    seeds = [config['smoke_seed']] if mode == 'smoke' else config['seeds']
    for corpus in config['corpora']:
        train, validation = partitions(corpus, mode, config)
        labels = config['labels'][corpus]
        arrays = []
        result['features'][corpus] = {}
        for split, rows in (('train', train), ('validation', validation)):
            print(f'{corpus.upper()} {split} : {len(rows)} textes', flush=True)
            vectors, timing = extract_features(model, tokenizer, rows, config, device, progress=True)
            name = f'{corpus}_{split}_features.npy'
            np.save(output / name, vectors, allow_pickle=False)
            keys = f'{corpus}_{split}_rows.json'
            save(output / keys, row_keys(rows))
            result['features'][corpus][split] = {'array': name, 'rows': keys,
                'shape': list(vectors.shape), 'partition': partition_summary(rows), 'timing': timing}
            arrays.append(vectors)
        mapping = {label: i for i, label in enumerate(labels)}
        for seed in seeds:
            head, evaluation, history, fitted = fit_head(*arrays,
                [mapping[r['label']] for r in train], [mapping[r['label']] for r in validation],
                labels, config, seed, mode, progress=True)
            stem = f'{corpus}_seed_{seed}'
            checkpoint, history_file, predictions_file = stem + '.pt', stem + '_history.json', stem + '_predictions.json'
            torch.save({'state_dict': head.state_dict(), 'label2id': mapping,
                        'id2label': {i: label for label, i in mapping.items()},
                        'feature_dim': arrays[0].shape[1], 'config': config, 'seed': seed}, output / checkpoint)
            save(output / history_file, history)
            save(output / predictions_file, [{'id': row['id'], 'group_id': row['group_id'], 'truth': row['label'],
                  'predicted': pred, 'probabilities': probability}
                  for row, pred, probability in zip(validation, evaluation['predicted'], evaluation['probabilities'])])
            metrics = add_writing_slices(evaluation['metrics'], validation, evaluation['predicted'], labels)
            result['results'].append({'corpus': corpus, 'seed': seed, 'metrics': metrics, **fitted,
                                      'checkpoint': checkpoint, 'history': history_file, 'predictions': predictions_file,
                                      'train': partition_summary(train), 'validation': partition_summary(validation)})
    result['backbone_state_after_sha256'] = state_digest(model)
    result['encoder_unchanged'] = initial_digest == result['backbone_state_after_sha256']
    if not result['encoder_unchanged'] or any(p.grad is not None or p.requires_grad for p in model.parameters()):
        raise ValueError('Frozen backbone changed or received gradients')
    result['aggregates'] = {}
    for corpus in config['corpora']:
        values = [r['metrics']['macro_f1'] for r in result['results'] if r['corpus'] == corpus]
        result['aggregates'][corpus] = {'mean_macro_f1': statistics.mean(values),
                                        'sample_std_macro_f1': statistics.stdev(values) if len(values) > 1 else None}
    result['memory'] = memory(device)
    result['total_seconds'] = time.perf_counter() - tick
    result['artifacts'] = {p.name: sha256(p) for p in output.iterdir() if p.is_file()}
    save(output / 'results.json', result)
    (output / 'summary.txt').write_text(summary(result) + '\n')
    print(summary(result), flush=True)
    print(f'Run enregistré : {output}', flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', required=True, choices=['smoke', 'full'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'])
    parser.add_argument('--smoke-run', type=Path)
    args = parser.parse_args()
    run(args.mode, args.output, args.device, args.smoke_run)
