"""Replay every linear head, validate provenance, and reencode ten dev rows/task."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics

import numpy as np
import torch
from torch import nn

from bigru_data import partitions
from bigru_model import resolve_device, set_seed
from bigru_training import add_writing_slices
from character_lm_experiment import ROOT, partition_summary, read_artifact
from fetch_camelbert_encoder import sha256
from frozen_encoder import extract_features, head_output, load_encoder, state_digest
from run_frozen_encoder import row_keys, save, summary, validate_config
from verify_bigru import compare_nested


def verify(directory, reencode=True, reference_run=None):
    directory = directory.resolve()
    saved = json.loads((directory / 'results.json').read_text())
    config, mode = saved['config'], saved['mode']
    validate_config(config)
    if mode not in {'smoke', 'full'} or saved['test_loaded'] is not False:
        raise ValueError('Expected train/validation-only completed run')
    for label, path in [('environment', ROOT / 'environments/encoder/uv.lock'), ('classical', ROOT / 'uv.lock'),
                        ('bigru', ROOT / 'environments/bigru/uv.lock'), ('tokenizer', ROOT / 'environments/tokenizer/uv.lock')]:
        if sha256(path) != saved[f'{label}_lock_sha256']:
            raise ValueError('Environment lock changed')
    for p, version in saved['packages'].items():
        if importlib.metadata.version(p) != version:
            raise ValueError(f'Package changed: {p}')
    for name, checksum in saved['artifacts'].items():
        if sha256(read_artifact(directory, name)) != checksum:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    if (saved['encoder_unchanged'] is not True or saved['encoder']['backbone_trainable_parameters'] != 0
            or saved['backbone_state_before_sha256'] != saved['backbone_state_after_sha256']):
        raise ValueError('Backbone was not frozen')
    seeds = [config['smoke_seed']] if mode == 'smoke' else config['seeds']
    if (len(saved['results']) != 2 * len(seeds) or {(r['corpus'], r['seed']) for r in saved['results']}
            != {(c, s) for c in config['corpora'] for s in seeds}):
        raise ValueError('Missing/duplicate task or seed')
    set_seed(config['smoke_seed'], config['cpu_threads'])
    # Replaying CPU heads / plotting cached vectors does not require the original GPU.
    device = resolve_device(saved['device']) if reencode else torch.device('cpu')
    model = tokenizer = None
    if reencode:
        tokenizer, model, encoder_info = load_encoder(device)
        compare_nested(saved['encoder'], encoder_info)
        if state_digest(model) != saved['backbone_state_before_sha256']:
            raise ValueError('Reloaded backbone differs')
    reference = None
    if reference_run:
        reference = json.loads((reference_run / 'results.json').read_text())
        if mode != 'full' or reference['mode'] != 'full' or reference['test_loaded'] is not False:
            raise ValueError('Compare full validation runs only')
    probes = {}
    for corpus in config['corpora']:
        if sha256(ROOT / f'data/manifests/v2/{corpus}.csv') != saved['manifests_sha256'][corpus]:
            raise ValueError('Manifest changed')
        train, validation = partitions(corpus, mode, config)
        arrays = {}
        for split, rows in [('train', train), ('validation', validation)]:
            feature = saved['features'][corpus][split]
            if partition_summary(rows) != feature['partition']:
                raise ValueError('Partition selection changed')
            keys = json.loads(read_artifact(directory, feature['rows']).read_text())
            if keys != row_keys(rows):
                raise ValueError('Cached feature order/text alignment changed')
            array = np.load(read_artifact(directory, feature['array']), allow_pickle=False)
            if array.dtype != np.float32 or array.shape != (len(rows), 768) or not np.isfinite(array).all():
                raise ValueError('Invalid feature matrix')
            arrays[split] = array
        if reencode:
            fresh, _ = extract_features(model, tokenizer, validation[:10], config, device)
            delta = float(np.max(np.abs(fresh - arrays['validation'][:10])))
            if not np.allclose(fresh, arrays['validation'][:10], rtol=0, atol=config['feature_reload_atol']):
                raise ValueError(f'Reencoded vectors differ: {delta}')
            probes[corpus] = {'rows': 10, 'maximum_absolute_feature_difference': delta}
        labels = config['labels'][corpus]
        truth = torch.tensor([labels.index(r['label']) for r in validation], dtype=torch.long)
        if reference:
            candidates = [r for r in reference['results'] if r['corpus'] == corpus
                          and r['system'] == 'tfidf_char' and r['variant'] == 'raw-v0']
            if len(candidates) != 1:
                raise ValueError('Missing unique TF-IDF character reference')
            ref = candidates[0]
            path = read_artifact(reference_run, ref['predictions'])
            if sha256(path) != ref['predictions_sha256']:
                raise ValueError('Reference predictions changed')
            predictions = json.loads(path.read_text())
            if [(p['id'], p['group_id'], p['truth']) for p in predictions] != [(r['id'], r['group_id'], r['label']) for r in validation]:
                raise ValueError('Different validation examples from TF-IDF reference')
        for result in [r for r in saved['results'] if r['corpus'] == corpus]:
            if result['train'] != partition_summary(train) or result['validation'] != partition_summary(validation):
                raise ValueError('Head partition summaries differ')
            checkpoint = torch.load(read_artifact(directory, result['checkpoint']), map_location='cpu', weights_only=True)
            if (checkpoint['config'] != config or checkpoint['seed'] != result['seed'] or checkpoint['feature_dim'] != 768
                    or checkpoint['label2id'] != {label: i for i, label in enumerate(labels)}
                    or checkpoint['id2label'] != dict(enumerate(labels))):
                raise ValueError('Head checkpoint mapping/configuration mismatch')
            head = nn.Linear(768, len(labels))
            head.load_state_dict(checkpoint['state_dict'], strict=True)
            if state_digest(head) != result['selected_head_sha256'] or state_digest(head) == result['initial_head_sha256']:
                raise ValueError('Head weights unchanged or inconsistent')
            if result['trainable_parameters'] != (768 + 1) * len(labels):
                raise ValueError('Wrong trainable parameter count')
            output = head_output(head, arrays['validation'], truth, labels)
            metrics = add_writing_slices(output['metrics'], validation, output['predicted'], labels)
            compare_nested(metrics, result['metrics'])
            predictions = json.loads(read_artifact(directory, result['predictions']).read_text())
            expected = [{'id': r['id'], 'group_id': r['group_id'], 'truth': r['label'], 'predicted': pred, 'probabilities': p}
                        for r, pred, p in zip(validation, output['predicted'], output['probabilities'])]
            compare_nested(predictions, expected)
            history = json.loads(read_artifact(directory, result['history']).read_text())
            best = max(history, key=lambda h: h['validation_macro_f1'])
            if (best['epoch'] != result['best_epoch'] or len(history) != result['epochs_completed']
                    or best['validation_macro_f1'] != result['metrics']['macro_f1']):
                raise ValueError('Best epoch selection differs')
        values = [r['metrics']['macro_f1'] for r in saved['results'] if r['corpus'] == corpus]
        compare_nested(saved['aggregates'][corpus], {'mean_macro_f1': statistics.mean(values),
                       'sample_std_macro_f1': statistics.stdev(values) if len(values) > 1 else None})
    if reencode and state_digest(model) != saved['backbone_state_before_sha256']:
        raise ValueError('Backbone mutated during replay')
    proof = {'results_sha256': sha256(directory / 'results.json'), 'reencoded': reencode,
             'probes': probes, 'heads_replayed': len(saved['results']), 'test_loaded': False,
             'reference_validation_ids_checked': reference is not None,
             'verifier_code_sha256': sha256(Path(__file__))}
    if reencode:
        save(directory / 'verification.json', proof)
    print(f'Verified {mode}: {len(saved["results"])} heads replayed, metrics recalculated; ten-row reencoding/task: {reencode}', flush=True)
    return saved


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--reference-run', type=Path)
    args = parser.parse_args()
    print(summary(verify(args.run, reference_run=args.reference_run)))
