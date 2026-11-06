"""Check recorded source bytes against current code or one preserved snapshot; never execute snapshots."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_recorded_sources(recorded, root=ROOT):
    if not recorded or any(Path(n).name != n or not n.endswith('.py') for n in recorded):
        raise ValueError('Invalid recorded source names')
    if all((root/'scripts'/n).is_file() and digest(root/'scripts'/n) == h for n,h in recorded.items()):
        return 'current'
    parent = root/'outputs/encoder_finetuning/source_snapshots'
    for snapshot in sorted(parent.iterdir()) if parent.exists() else []:
        manifest = snapshot/'manifest.json'
        if not manifest.is_file(): continue
        files = json.loads(manifest.read_text()).get('files', {})
        if all(files.get('scripts/'+n) == h and (snapshot/'scripts'/n).is_file()
               and digest(snapshot/'scripts'/n) == h for n,h in recorded.items()):
            return str(snapshot.relative_to(root))
    raise ValueError('No matching current or archived source bundle; preserve historical code')
