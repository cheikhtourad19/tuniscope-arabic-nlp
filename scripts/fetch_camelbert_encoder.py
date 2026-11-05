"""Explicit download of one pinned official checkpoint into an ignored cache."""
import argparse
import hashlib
import json
import shutil
import urllib.request
from datetime import datetime, timezone

from fetch_camelbert_tokenizer import ROOT, MODEL_ID, REVISION, ASSET_DIR, FILES, verify_assets

ENCODER_DIR = ROOT / '.tools/encoders' / f'camelbert_mix_{REVISION}'
WEIGHT_NAME = 'pytorch_model.bin'
WEIGHT_BYTES = 438873130
WEIGHT_SHA256 = 'd234e4fe0193e74036b1db72ba9b41a2e4921d8cf0c3c421d1f6a6d8816af729'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_encoder(directory=ENCODER_DIR):
    source = verify_assets()
    manifest = json.loads((directory / 'encoder_source.json').read_text())
    if manifest['model_id'] != MODEL_ID or manifest['revision'] != REVISION:
        raise ValueError('Wrong encoder revision')
    expected = {WEIGHT_NAME, 'encoder_source.json', *FILES}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError('Unexpected encoder cache contents')
    weight = directory / WEIGHT_NAME
    if weight.stat().st_size != WEIGHT_BYTES or sha256(weight) != WEIGHT_SHA256:
        raise ValueError('Official checkpoint checksum mismatch')
    if manifest['weight'] != {'filename': WEIGHT_NAME, 'bytes': WEIGHT_BYTES, 'sha256': WEIGHT_SHA256}:
        raise ValueError('Checkpoint manifest mismatch')
    for name in FILES:
        if sha256(directory / name) != source['files'][name]['sha256']:
            raise ValueError(f'Copied asset checksum mismatch: {name}')
    return manifest


def fetch():
    source = verify_assets()
    if (ENCODER_DIR / 'encoder_source.json').exists():
        return verify_encoder()
    ENCODER_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        shutil.copyfile(ASSET_DIR / name, ENCODER_DIR / name)
    url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{WEIGHT_NAME}'
    part = ENCODER_DIR / (WEIGHT_NAME + '.part')
    h, total, last = hashlib.sha256(), 0, 0
    with urllib.request.urlopen(url, timeout=90) as response, part.open('wb') as stream:
        while block := response.read(4 * 1024 * 1024):
            total += len(block)
            if total > WEIGHT_BYTES:
                raise ValueError('Download exceeds official size')
            h.update(block)
            stream.write(block)
            if total - last >= 50_000_000:
                print(f'Checkpoint : {total / 1e6:.0f}/{WEIGHT_BYTES / 1e6:.0f} Mo', flush=True)
                last = total
    if total != WEIGHT_BYTES or h.hexdigest() != WEIGHT_SHA256:
        raise ValueError('Downloaded bytes do not match pinned official LFS SHA256')
    part.replace(ENCODER_DIR / WEIGHT_NAME)
    manifest = {'model_id': MODEL_ID, 'revision': REVISION, 'url': url,
                'fetched_at_utc': datetime.now(timezone.utc).isoformat(),
                'model_license_declared': source['model_license_declared'],
                'weight': {'filename': WEIGHT_NAME, 'bytes': WEIGHT_BYTES, 'sha256': WEIGHT_SHA256},
                'lfs_metadata_url': f'https://huggingface.co/api/models/{MODEL_ID}/revision/{REVISION}?blobs=true'}
    (ENCODER_DIR / 'encoder_source.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return verify_encoder()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    args = parser.parse_args()
    print(json.dumps(fetch() if args.download else verify_encoder(), indent=2))
