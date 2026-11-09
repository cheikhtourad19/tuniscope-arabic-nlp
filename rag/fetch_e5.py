"""Acquire or verify the pinned E5 assets in the ignored local cache."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'configs/rag_e5_small_v1.json'
MODEL = json.loads(PLAN.read_text())
CACHE = ROOT / '.tools/rag' / MODEL['revision']


def verify(path, info):
    size = info['bytes']
    if path.stat().st_size != size:
        raise ValueError(f'Unexpected file size for {path.name}')
    if 'sha256' in info:
        digest = hashlib.sha256()
        expected = info['sha256']
    else:
        digest = hashlib.sha1(f'blob {size}\0'.encode())
        expected = info['git_blob_sha1']
    with path.open('rb') as stream:
        while block := stream.read(4 * 1024 * 1024):
            digest.update(block)
    if digest.hexdigest() != expected:
        raise ValueError(f'Unexpected file digest for {path.name}')


def assets(download=False, directory=CACHE):
    if download:
        directory.mkdir(parents=True, exist_ok=True)
    for name, info in MODEL['files'].items():
        path = directory / name
        if download and not path.exists():
            partial = directory / f'{name}.part'
            print(f'Téléchargement E5 {name} ({info["bytes"]:,} octets)', flush=True)
            url = f'https://huggingface.co/{MODEL["model_id"]}/resolve/{MODEL["revision"]}/{name}?download=true'
            with urlopen(url, timeout=90) as source, partial.open('wb') as sink:
                size = 0
                while block := source.read(4 * 1024 * 1024):
                    size += len(block)
                    if size > info['bytes']:
                        raise ValueError(f'Download too large for {name}')
                    sink.write(block)
            verify(partial, info)
            partial.replace(path)
        else:
            verify(path, info)
    return directory


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--download', action='store_true')
    print(assets(parser.parse_args().download))
