"""Reconcile original LoRA archives and Qwen source fingerprints offline."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
LORA = ROOT / 'outputs/qwen/2026-11-07_cuda_lora_c10d02666e154607ac9d7faddbd47838_full'
LORA_REPORT = ROOT / 'reports/qwen_lora_cuda_2026-11-07/summary.json'
BASE_PLAN = ROOT / 'outputs/qwen/2026-11-07_validation_v2_mps_b16/plan.json'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def git_compatible_revisions(fingerprints):
    commits = subprocess.check_output(['git', 'rev-list', '--all'], cwd=ROOT, text=True).splitlines()
    compatible = []
    for commit in commits:
        matches = True
        for filename, digest in fingerprints.items():
            content = subprocess.run(['git', 'show', f'{commit}:scripts/{filename}'], cwd=ROOT,
                                     capture_output=True, check=False)
            if content.returncode or hashlib.sha256(content.stdout).hexdigest() != digest:
                matches = False
                break
        if matches:
            compatible.append(commit)
    return compatible


def build():
    report = json.loads(LORA_REPORT.read_text())
    plan = json.loads(BASE_PLAN.read_text())
    archives = {}
    missing = {}
    for seed in (13, 42, 2026):
        run = report['runs'][str(seed)]
        folder = LORA / f'seed_{seed}'
        zip_path, = folder.glob('*.zip')
        if sha(zip_path) != run['zip_sha256']:
            raise ValueError('Archive originale différente de celle importée.')
        expected = {'plan.json', 'training.json', 'smoke.json', 'result.json',
                    'adapter/adapter_config.json', 'adapter/adapter_model.safetensors'}
        with ZipFile(zip_path) as archive:
            if set(archive.namelist()) != expected or archive.testzip() is not None:
                raise ValueError('Contenu ou intégrité de l’archive inattendu.')
            for member in expected:
                if archive.read(member) != (folder / member).read_bytes():
                    raise ValueError('Extraction locale différente de l’archive originale.')
            members = sorted(archive.namelist())
        download = Path.home() / 'Downloads' / zip_path.name
        same_download = sha(download) == run['zip_sha256'] if download.is_file() else None
        if same_download is False:
            raise ValueError('La copie Downloads diffère de l’archive importée.')
        archives[str(seed)] = {'archive': str(zip_path.relative_to(ROOT)), 'sha256': run['zip_sha256'],
                               'download_copy_identical': same_download, 'members': members,
                               'adapter_files': run['training']['adapter_files']}
        for stage, metrics in run['validation'].items():
            missing[f'{seed}/{stage}'] = metrics['predictions_sha256']

    candidates = [path for path in (ROOT / 'outputs/qwen').rglob('*.json')
                  if path.name in {'tsac_zero_shot.json', 'tsac_few_shot.json',
                                   'madar_zero_shot.json', 'madar_few_shot.json'}]
    found = {}
    wanted = set(missing.values())
    for path in candidates:
        digest = sha(path)
        if digest in wanted:
            found[digest] = str(path.relative_to(ROOT))
    recovery = {name: {'expected_sha256': digest, 'matching_local_file': found.get(digest)}
                for name, digest in missing.items()}
    for filename, digest in plan['code_sha256'].items():
        if sha(ROOT / 'scripts' / filename) != digest:
            raise ValueError('Une source Qwen diffère de l’empreinte enregistrée.')
    return {
        'scope': 'offline historical traceability reconciliation',
        'lora_source_report_sha256': sha(LORA_REPORT), 'qwen_plan_sha256': sha(BASE_PLAN),
        'original_lora_archives': archives,
        'cuda_prediction_files': recovery, 'candidate_json_files_checked': len(candidates),
        'original_cuda_prediction_files_found': len(found),
        'original_campaign_summary_received': report['original_campaign_summary_received'],
        'original_campaign_wall_seconds': report['campaign_wall_seconds'],
        'derived_campaign_index': {'status': 'reconstructed_from_original_seed_archives',
                                   'source_report': str(LORA_REPORT.relative_to(ROOT)),
                                   'training_commit': report['plan']['commit'], 'seeds': [13, 42, 2026]},
        'qwen_base_source_provenance': {
            'recorded_execution_commit': plan.get('git_commit'),
            'source_fingerprints_verified': plan['code_sha256'],
            'compatible_repository_revisions': git_compatible_revisions(plan['code_sha256']),
            'compatibility_identifies_execution_commit': False,
        },
        'new_training_or_inference': False,
        'classification_test_evaluated': False,
        'reporter_sha256': sha(Path(__file__)),
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choisir une nouvelle destination.')
    result = build()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'summary.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    if json.loads(path.read_text()) != result:
        raise ValueError('Le bilan sauvegardé diffère du bilan vérifié.')
    print(f'Trois archives originales vérifiées, {result["original_cuda_prediction_files_found"]} '
          f'fichier de prédictions CUDA retrouvé parmi {result["candidate_json_files_checked"]} candidats.')
    print(f'Rapport de traçabilité — {args.output.resolve()}')
