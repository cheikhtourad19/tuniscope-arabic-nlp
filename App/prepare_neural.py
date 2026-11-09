"""Prepare the private web inference bundle from explicit verified experiments."""
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from prepare_models import ROOT, digest

sys.path.insert(0, str(ROOT / 'scripts'))
from fetch_qwen3 import MODEL_DIR, verify_model
sys.path.insert(0, str(ROOT / 'rag'))
from fetch_e5 import MODEL as E5, assets as e5_assets

DEST = ROOT / 'App/models/neural'
ENCODER_FILES = ('config.json', 'model.safetensors', 'tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'vocab.txt', 'tuniscope.json')


def read(path):
    return json.loads(path.read_text())


def copy_checked(source, target, expected):
    if digest(source) != expected:
        raise ValueError(f'Empreinte incorrecte pour {source.name}')
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return expected


def prepare(encoder, lora, seed, replace):
    campaign = read(encoder / 'confirmation.json')
    proof = read(lora / 'import_verified.json')
    if (campaign['mode'] != 'full' or campaign['test_loaded'] is not False
            or proof['scope'] != 'validation_v2' or proof['test_loaded'] is not False):
        raise ValueError('Campagnes complètes sur validation requises.')
    if (DEST / 'manifest.json').exists() and not replace:
        raise ValueError('Le bundle neural existe déjà. Utiliser --replace pour changer la copie.')
    run = proof['runs'][str(seed)]
    adapter = lora / f'seed_{seed}' / 'adapter'
    base = verify_model()
    if (proof['plan']['revision'], proof['plan']['model_sha256']) != (
            base['revision'], base['files']['model.safetensors']['sha256']):
        raise ValueError('Révision de base Qwen incompatible avec l’adaptateur.')
    base_report_path = ROOT / 'reports/qwen_validation_v2_2026-11-07/summary.json'
    if digest(base_report_path) != proof['plan']['base_report_sha256']:
        raise ValueError('Rapport Qwen de base incompatible.')
    if digest(ROOT / 'configs/qwen_prompting_v2.json') != proof['plan']['prompt_config_sha256']:
        raise ValueError('Configuration de prompt incompatible.')
    base_report = read(base_report_path)
    manifest = {'version': 1, 'split': 'validation', 'test_loaded': False, 'seed': seed, 'encoder': {}, 'qwen': {}}
    cache = e5_assets()
    with tempfile.TemporaryDirectory(prefix='.staging-', dir=DEST.parent) as staging:
        stage = Path(staging) / 'bundle'
        stage.mkdir()
        (stage / '.gitkeep').touch()
        for item in campaign['entries']:
            corpus = item['corpus']
            directory = encoder / item['full']['directory']
            result_path = directory / 'results.json'
            if digest(result_path) != item['full']['results_sha256']:
                raise ValueError('Le stage encodeur ne correspond pas à la campagne.')
            data = read(result_path)
            if data['mode'] != 'full' or data['test_loaded'] is not False or data['regime'] != 'finetuned':
                raise ValueError('Stage encodeur adapté de validation requis.')
            selected, = (r for r in data['results'] if r['seed'] == seed and r['corpus'] == corpus)
            checkpoint = directory / selected['checkpoint']
            target = stage / 'encoder' / corpus
            files = {name: copy_checked(checkpoint / name, target / name,
                                       data['artifacts'][f'{checkpoint.name}/{name}']) for name in ENCODER_FILES}
            manifest['encoder'][corpus] = {
                'files': files, 'metrics': selected['metrics'], 'revision': data['config']['revision'],
                'aggregate': campaign['aggregates'][corpus], 'source_run': encoder.name,
                'train_rows': selected['train']['rows'], 'validation_rows': selected['validation']['rows'],
            }
        qwen = manifest['qwen']
        qwen.update(revision=base['revision'], source_run=lora.name, base_source_run=Path(base_report['run']).name,
                    base_files={name: copy_checked(MODEL_DIR / name, stage / 'qwen/base' / name, info['sha256'])
                                for name, info in base['files'].items()},
                    adapter_files={name: copy_checked(adapter / name, stage / 'qwen/adapter' / name,
                                                       run['training']['adapter_files'][f'adapter/{name}'])
                                   for name in ('adapter_model.safetensors', 'adapter_config.json')},
                    metrics={key: value for key, value in run['validation'].items() if key.endswith('_zero_shot')},
                    base_metrics={key: value for key, value in base_report['stages'].items() if key.endswith('_zero_shot')},
                    aggregates={key: value for key, value in proof['aggregates'].items() if key.endswith('_zero_shot')})
        for name in E5['files']:
            target = stage / 'e5' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(cache / name, target)
        manifest['e5_revision'] = E5['revision']
        (stage / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        backup = Path(staging) / 'previous'
        if DEST.exists():
            os.rename(DEST, backup)
        try:
            os.rename(stage, DEST)
        except OSError:
            if backup.exists():
                os.rename(backup, DEST)
            raise
    return {'bundle': str(DEST), 'seed': seed, 'encoder_tasks': sorted(manifest['encoder']), 'shared_qwen_base': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--encoder-run', type=Path, required=True)
    parser.add_argument('--lora-run', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=[13, 42, 2026], default=42)
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    print(json.dumps(prepare(args.encoder_run.resolve(), args.lora_run.resolve(), args.seed, args.replace), indent=2))
