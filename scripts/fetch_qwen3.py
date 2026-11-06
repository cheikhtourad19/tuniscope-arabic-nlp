"""Explicit pinned Qwen download; default operation verifies local files offline."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'configs/qwen3_base_v1.json'
MODEL_ID = 'Qwen/Qwen3-0.6B'
REVISION = 'c1899de289a04d12100db370d81485cdf75e47ca'
MODEL_DIR = ROOT / '.tools/llms' / f'qwen3_0_6b_{REVISION}'
FILES = {'LICENSE', 'README.md', 'config.json', 'generation_config.json', 'merges.txt',
         'model.safetensors', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json'}


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def load_plan():
    p = json.loads(PLAN.read_text())
    if (p['version'] != 'qwen3-base-assets-v1' or p['model_id'] != MODEL_ID or p['revision'] != REVISION
            or p['license_declared'] != 'apache-2.0' or p['model_type'] != 'qwen3'
            or set(p['files']) != FILES or p['enable_thinking'] is not False
            or p['training_started'] is not False or p['corpus_read'] is not False):
        raise ValueError('Unexpected Qwen download plan')
    for name, info in p['files'].items():
        if not isinstance(info['bytes'], int) or not 0 < info['bytes'] < 2_000_000_000:
            raise ValueError('Unexpected official file size')
        key = 'sha256' if 'sha256' in info else 'git_blob_sha1'
        if set(info) != {'bytes', key} or not re.fullmatch('[0-9a-f]{64}' if key == 'sha256' else '[0-9a-f]{40}', info[key]):
            raise ValueError(f'Invalid official file digest: {name}')
    return p


def verify_file(path, info):
    if path.stat().st_size != info['bytes']:
        raise ValueError(f'Wrong size: {path.name}')
    actual_sha256 = hashlib.sha256()
    git_blob = hashlib.sha1(f'blob {info["bytes"]}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            actual_sha256.update(block); git_blob.update(block)
    expected = info.get('sha256', info.get('git_blob_sha1'))
    actual = actual_sha256.hexdigest() if 'sha256' in info else git_blob.hexdigest()
    if actual != expected:
        raise ValueError(f'Official file checksum mismatch: {path.name}')
    return {'bytes': info['bytes'], 'sha256': actual_sha256.hexdigest()}


def verify_model(directory=MODEL_DIR):
    p = load_plan()
    source = json.loads((directory / 'qwen_source.json').read_text())
    if (source['model_id'] != MODEL_ID or source['revision'] != REVISION
            or source['plan_sha256'] != sha256(PLAN) or set(source['files']) != FILES
            or source['license_declared'] != p['license_declared']):
        raise ValueError('Qwen source manifest mismatch')
    if {f.name for f in directory.iterdir()} != FILES | {'qwen_source.json'}:
        raise ValueError('Unexpected files in Qwen cache')
    for name, info in p['files'].items():
        if verify_file(directory / name, info) != source['files'][name]:
            raise ValueError('Qwen local manifest mismatch')
    config = json.loads((directory / 'config.json').read_text())
    if config['model_type'] != 'qwen3' or config['architectures'] != ['Qwen3ForCausalLM']:
        raise ValueError('Wrong architecture')
    header = (directory / 'README.md').read_text().split('---', 2)[1]
    if not re.search(r'^license:\s*apache-2.0\s*$', header, re.MULTILINE):
        raise ValueError('Model card licence mismatch')
    if 'Apache License' not in (directory / 'LICENSE').read_text():
        raise ValueError('Missing Apache licence text')
    return source


def fetch():
    p = load_plan(); directory = MODEL_DIR
    if (directory / 'qwen_source.json').exists():
        return verify_model()
    directory.mkdir(parents=True, exist_ok=True)
    allowed = FILES | {n + '.part' for n in FILES}
    if any(f.name not in allowed for f in directory.iterdir()):
        raise ValueError('Unexpected partial download contents')
    source = {'model_id': MODEL_ID, 'revision': REVISION, 'plan_sha256': sha256(PLAN),
              'license_declared': p['license_declared'], 'files': {},
              'fetched_at_utc': datetime.now(timezone.utc).isoformat(), 'quantized': False}
    for name, info in p['files'].items():
        target = directory / name
        if not target.exists():
            partial = directory / (name + '.part')
            url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
            print(f'Téléchargement : {name} ({info["bytes"] / 1e6:.1f} Mo)', flush=True)
            total, last = 0, 0
            with urllib.request.urlopen(url, timeout=90) as response, partial.open('wb') as stream:
                while block := response.read(4 * 1024 * 1024):
                    total += len(block)
                    if total > info['bytes']:
                        raise ValueError('Download exceeds pinned size')
                    stream.write(block)
                    if total - last >= 100_000_000:
                        print(f'  {total / 1e6:.0f}/{info["bytes"] / 1e6:.0f} Mo', flush=True); last = total
            verify_file(partial, info)
            partial.replace(target)
        source['files'][name] = verify_file(target, info)
    (directory / 'qwen_source.json').write_text(json.dumps(source, indent=2) + '\n')
    return verify_model()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true', help='Explicitly download the nine pinned official files')
    args = parser.parse_args()
    manifest = fetch() if args.download else verify_model()
    print(json.dumps({'directory': str(MODEL_DIR), 'model_id': MODEL_ID, 'revision': REVISION,
                      'files': len(manifest['files']), 'bytes': sum(f['bytes'] for f in manifest['files'].values()),
                      'quantized': False, 'checksums_verified': True, 'corpus_read': False,
                      'training_started': False}, indent=2))
