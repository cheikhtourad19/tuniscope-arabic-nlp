"""Freeze classification choices and check saved artifacts without reading test texts."""
import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

from fetch_camelbert_encoder import ENCODER_DIR, verify_encoder
from fetch_qwen3 import MODEL_DIR, verify_model
from load_working_data import _manifest_rows

ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION = ROOT / 'outputs/encoder_finetuning/confirmation/notebooks/177d9f5ad78b4ef4907e97d97abe96f8_full'
LORA = ROOT / 'outputs/qwen/2026-11-07_cuda_lora_c10d02666e154607ac9d7faddbd47838_full'
LABELS = {'tsac': ['NEG', 'POS'], 'madar': ['MSA', 'TN', 'EG']}
MODEL_FILES = {'head.pt', 'model.safetensors', 'tuniscope.json', 'config.json',
               'tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.txt'}
CONFIGS = ['baselines_grouped_v2', 'baselines_words_v2', 'bigru_v1', 'camelbert_finetuning_v1',
           'camelbert_confirmation_v1', 'qwen3_base_v1', 'qwen_prompting_v2', 'qwen_sft_v1', 'qwen_lora_v1']


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def validate_structure(protocol):
    if (protocol['version'] != 1 or protocol['manifest_version'] != 'v2'
            or protocol['split'] != 'test' or protocol['labels'] != LABELS
            or protocol['metrics']['macro_f1_labels'] != 'fixed_task_labels_excluding_INVALID'
            or protocol['test_texts_loaded'] is not False):
        raise ValueError('Contrat de classification incompatible.')
    evaluations, models = protocol['evaluations'], protocol['models']
    if len(evaluations) != 48 or len({row['id'] for row in evaluations}) != 48 or len(models) != 36:
        raise ValueError('Les 48 évaluations et 36 modèles uniques du protocole v1 sont requis.')
    if Counter(row['tier'] for row in evaluations) != {'primary': 32, 'controlled_comparison': 16}:
        raise ValueError('Groupes principaux et comparaisons complémentaires différents.')
    for row in evaluations:
        if row['model'] not in models or row['corpus'] not in LABELS:
            raise ValueError('Modèle ou tâche inconnu dans une évaluation.')
        model = models[row['model']]
        if (model['family'] != row['family']
                or model.get('corpus', row['corpus']) != row['corpus']):
            raise ValueError('Modèle associé à la mauvaise famille ou tâche.')
        if row['variant'] not in ('raw-v0', 'norm-nfc-space-v1'):
            raise ValueError('Transformation non prévue.')
    if any(model.get('total_parameters', 1) > 1_000_000_000 for model in models.values()):
        raise ValueError('Un modèle dépasse le plafond de paramètres.')
    if {corpus: values['rows'] for corpus, values in protocol['datasets'].items()} != {'tsac': 3400, 'madar': 800}:
        raise ValueError('Effectifs différents du test officiel retenu.')
    for family in ('bigru', 'camelbert_confirmed', 'qwen_lora'):
        for corpus in LABELS:
            modes = ('zero_shot', 'few_shot') if family == 'qwen_lora' else (None,)
            for mode in modes:
                seeds = [models[row['model']]['seed'] for row in evaluations
                         if row['family'] == family and row['corpus'] == corpus and row['prompt_mode'] == mode]
                if sorted(seeds) != [13, 42, 2026]:
                    raise ValueError('Trois graines distinctes requises pour chaque condition finale.')


def freeze(path):
    if path.exists():
        raise FileExistsError('Un protocole existant doit être conservé.')
    files, models, evaluations = {}, {}, []

    def pin(target, expected=None):
        target = target.resolve()
        if not target.is_relative_to(ROOT) or not target.is_file():
            raise ValueError(f'Artefact manquant ou hors dépôt — {target}')
        name = str(target.relative_to(ROOT))
        if name not in files:
            files[name] = sha(target)
        if expected is not None and files[name] != expected:
            raise ValueError(f'Empreinte incompatible — {name}')
        return name

    def read(target):
        pin(target)
        return json.loads(target.read_text())

    def register(identifier, metadata, artifacts):
        if identifier in models:
            raise ValueError('Modèle final dupliqué.')
        models[identifier] = {**metadata, 'artifact_files': [pin(p, expected) for p, expected in artifacts]}

    def evaluate_later(identifier, corpus, tier, variant='raw-v0', prompt_mode=None):
        evaluations.append({'id': f'{identifier}__{corpus}__{prompt_mode or variant}',
                            'model': identifier, 'family': models[identifier]['family'],
                            'corpus': corpus, 'tier': tier, 'variant': variant, 'prompt_mode': prompt_mode})

    for config in CONFIGS:
        pin(ROOT / f'configs/{config}.json')
    for profile in ('uv.lock', 'environments/bigru/uv.lock', 'environments/encoder/uv.lock',
                    'environments/llm/uv.lock', 'environments/lora/uv.lock'):
        pin(ROOT / profile)
    pending_sources = {'classification_protocol', 'load_working_data', 'normalize_text', 'baseline_models',
                       'baseline_metrics', 'bigru_model', 'bigru_data', 'bigru_training',
                       'encoder_finetuning_inference', 'encoder_finetuning', 'frozen_encoder',
                       'qwen_prompting', 'report_qwen_validation', 'fetch_qwen3', 'fetch_camelbert_encoder'}
    seen_sources = set()
    while pending_sources:
        script = min(pending_sources)
        pending_sources.remove(script)
        source = ROOT / f'scripts/{script}.py'
        pin(source)
        seen_sources.add(script)
        for node in ast.walk(ast.parse(source.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom) and node.module:
                modules = [node.module]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            for module in modules:
                dependency = module.split('.')[0]
                if dependency not in seen_sources and (ROOT / f'scripts/{dependency}.py').is_file():
                    pending_sources.add(dependency)
    pin(ROOT / 'docs/73_protocole_classification_finale.md')
    pin(ROOT / 'reports/classification_traceability_2026-11-12/summary.json')
    pin(ROOT / 'reports/silver_evidence_2026-11-12/summary.json')

    for directory, system in [('2026-10-17_grouped_v2_full', None), ('2026-10-17_words_v2_full', 'tfidf_word')]:
        folder = ROOT / 'outputs/baselines' / directory
        data = read(folder / 'results.json')
        for row in data['results']:
            selected = ((row['system'] in ('majority', 'tfidf_char')) if system is None
                        else (row['system'] == system and row['variant'] == 'raw-v0'))
            if not selected:
                continue
            identifier = f'{row["corpus"]}__{row["system"]}__{row["variant"]}'
            register(identifier, {'family': row['system'], 'corpus': row['corpus'], 'seed': data['config']['seed'],
                                  'checkpoint': str((folder / row['checkpoint']).relative_to(ROOT)),
                                  'source_results': str((folder / 'results.json').relative_to(ROOT)),
                                  'effective_config': data['config'], 'selection_split': 'validation'},
                     [(folder / row['checkpoint'], row['checkpoint_sha256'])])
            tier = ('primary' if row['system'] == 'majority' or
                    (row['system'] == 'tfidf_char' and row['variant'] == 'raw-v0') else 'controlled_comparison')
            evaluate_later(identifier, row['corpus'], tier, variant=row['variant'])

    bigru_folder = ROOT / 'outputs/neural_baselines/2026-11-04_bigru_v1_cpu_full'
    bigru = read(bigru_folder / 'results.json')
    for row in bigru['results']:
        identifier = f'{row["corpus"]}__bigru__{row["seed"]}'
        register(identifier, {'family': 'bigru', 'corpus': row['corpus'], 'seed': row['seed'],
                              'checkpoint': str((bigru_folder / row['checkpoint']).relative_to(ROOT)),
                              'source_results': str((bigru_folder / 'results.json').relative_to(ROOT)),
                              'effective_config': bigru['config'], 'selection_split': 'validation',
                              'total_parameters': row['parameters']},
                 [(bigru_folder / row['checkpoint'], row['checkpoint_sha256'])])
        evaluate_later(identifier, row['corpus'], 'primary')

    campaign = read(CONFIRMATION / 'confirmation.json')
    folders = [('camelbert_confirmed', CONFIRMATION / entry['full']['directory'], 'primary')
               for entry in campaign['entries']]
    folders.append(('camelbert_paired', ROOT / 'outputs/encoder_finetuning/2026-11-05_cls_paired_mps_full',
                    'controlled_comparison'))
    for family, folder, tier in folders:
        data = read(folder / 'results.json')
        for row in data['results']:
            identifier = f'{row["corpus"]}__{family}__{row["regime"]}__{row["seed"]}'
            checkpoint = folder / row['checkpoint']
            artifacts = [(p, data['artifacts'][str(p.relative_to(folder))])
                         for p in sorted(checkpoint.iterdir()) if p.name in MODEL_FILES]
            register(identifier, {'family': family, 'regime': row['regime'], 'corpus': row['corpus'],
                                  'seed': row['seed'], 'checkpoint': str(checkpoint.relative_to(ROOT)),
                                  'source_results': str((folder / 'results.json').relative_to(ROOT)),
                                  'effective_config': data['config'], 'selection_split': 'validation',
                                  'total_parameters': row['parameters']}, artifacts)
            evaluate_later(identifier, row['corpus'], tier)

    verify_encoder()
    encoder_assets = [pin(p) for p in sorted(ENCODER_DIR.iterdir()) if p.is_file()]
    verify_model()
    qwen_plan = read(ROOT / 'outputs/qwen/2026-11-07_validation_v2_mps_b16/plan.json')
    qwen_config = read(ROOT / 'configs/qwen_prompting_v2.json')
    register('qwen_base', {'family': 'qwen_base', 'seed': None, 'checkpoint': str(MODEL_DIR.relative_to(ROOT)),
                           'effective_config': qwen_config, 'generation': qwen_plan['generation'],
                           'selection_split': 'validation', 'total_parameters': 596049920},
             [(p, None) for p in sorted(MODEL_DIR.iterdir()) if p.is_file()])
    lora_report = read(ROOT / 'reports/qwen_lora_cuda_2026-11-07/summary.json')
    for seed in (13, 42, 2026):
        folder = LORA / f'seed_{seed}'
        read(folder / 'result.json')
        plan = read(folder / 'plan.json')
        identifier = f'qwen_lora__{seed}'
        register(identifier, {'family': 'qwen_lora', 'seed': seed, 'base_model': 'qwen_base',
                              'checkpoint': str((folder / 'adapter').relative_to(ROOT)),
                              'effective_config': plan['config'], 'selection_split': 'validation',
                              'total_parameters': 597196800},
                 [(folder / name, digest) for name, digest in lora_report['runs'][str(seed)]['training']['adapter_files'].items()])
    for identifier in ('qwen_base', 'qwen_lora__13', 'qwen_lora__42', 'qwen_lora__2026'):
        for corpus in LABELS:
            for mode in ('zero_shot', 'few_shot'):
                evaluate_later(identifier, corpus, 'primary', prompt_mode=mode)

    datasets = {}
    for corpus in LABELS:
        rows = [row for row in _manifest_rows(corpus, 'v2') if row['split'] == 'test']
        pin(ROOT / f'data/manifests/v2/{corpus}.csv')
        datasets[corpus] = {'rows': len(rows), 'groups': len({row['group_id'] for row in rows}),
                            'labels': dict(Counter(row['label'] for row in rows)),
                            'selection': 'all manifest v2 test rows without filtering'}
    pin(ROOT / 'data/manifests/v2/file_hashes.csv')
    protocol = {
        'version': 1, 'frozen_on': '2026-11-12', 'manifest_version': 'v2', 'split': 'test', 'labels': LABELS,
        'selection_basis': 'classification train/validation only; documentary scores not used',
        'test_texts_loaded': False, 'test_predictions_generated': False, 'datasets': datasets,
        'models': models, 'evaluations': evaluations, 'camelbert_base_assets': encoder_assets,
        'metrics': {'primary': 'macro_f1', 'macro_f1_labels': 'fixed_task_labels_excluding_INVALID',
                    'invalid_policy': 'count as classification errors and retain INVALID column',
                    'zero_division': 0, 'secondary': ['accuracy', 'balanced_accuracy', 'per_class_precision_recall_f1',
                                                   'confusion_matrix', 'valid_label_rate'],
                    'stochastic_summary': 'mean and sample standard deviation over 13/42/2026'},
        'inference': {'training': False, 'model_mode': 'eval', 'dtype_neural': 'float32',
                      'attention': 'eager', 'quantization': None,
                      'device': 'auto', 'require_same_device_for_paired_qwen_conditions': True,
                      'qwen_prompt_limit': 480, 'qwen_output_limit': 32,
                      'qwen_overflow_policy': 'INVALID with prompt_too_long reason, included in scores',
                      'sklearn_zero_vector_policy': 'retain model prediction, flag diagnostically',
                      'scientific_filtering_by_predicted_dialect': False},
        'bootstrap': {'resamples': 1000, 'seed': 42, 'unit': 'manifest group_id',
                      'same_resampled_groups_across_conditions_and_seeds': True,
                      'statistic': 'difference of mean macro_f1 over seeds, paired with fixed reference',
                      'interval': 'percentile 95%', 'primary_contrast': 'Qwen LoRA minus base in each prompt condition',
                      'secondary_contrasts': ['confirmed CAMeLBERT minus raw character TF-IDF',
                                              'paired CLS adapted minus frozen',
                                              'normalized minus raw character TF-IDF',
                                              'word minus character TF-IDF']},
        'slices': {'writing': ['AR', 'LATIN', 'MIXED', 'UNKNOWN'],
                   'character_length_upper_bounds': [20, 50, 100], 'small_slice_warning_below': 100},
        'resource_protocol': {'one_process_per_model': True, 'warmup_batches': 2,
                              'latency_batch_size': 1, 'latency_examples': 100,
                              'latency_sample_selection': 'SHA256(42|id), first 100 per corpus',
                              'latency_summary': ['median', 'p95'], 'synchronize_device': True,
                              'apple_memory_policy': 'report CPU RSS and MPS snapshots separately'},
        'outputs': {'private': 'outputs/final_classification/<run>/',
                    'public': 'reports/final_classification/<run>/',
                    'individual_prediction_fields': ['id', 'group_id', 'truth', 'prediction', 'model', 'seed',
                                                     'variant', 'prompt_mode', 'truncated', 'invalid_reason'],
                    'required_provenance': ['protocol_sha256', 'git_commit', 'git_dirty', 'code_sha256',
                                            'model_sha256', 'data_sha256', 'config', 'packages', 'hardware', 'runtime']},
        'git_commit_at_freeze': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'worktree_dirty_at_freeze': bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True)),
        'files_sha256': files,
    }
    validate_structure(protocol)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(protocol, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    return protocol


def check(path):
    protocol = json.loads(path.read_text())
    validate_structure(protocol)
    for name, digest in protocol['files_sha256'].items():
        target = (ROOT / name).resolve()
        if not target.is_relative_to(ROOT) or not target.is_file() or sha(target) != digest:
            raise ValueError(f'Artefact différent du gel — {name}')
    for model in protocol['models'].values():
        if not model['artifact_files'] or any(name not in protocol['files_sha256'] for name in model['artifact_files']):
            raise ValueError('Artefact de modèle non lié au gel.')
    return protocol


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('freeze')
    create.add_argument('--output', type=Path, required=True)
    verify = sub.add_parser('check')
    verify.add_argument('--protocol', type=Path, required=True)
    args = parser.parse_args()
    result = freeze(args.output) if args.command == 'freeze' else check(args.protocol)
    print(f'Protocole {args.command} vérifié — {len(result["models"])} modèles, '
          f'{len(result["evaluations"])} conditions tâche/modèle, sans lecture des textes du test.')
