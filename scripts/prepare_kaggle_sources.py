"""Link manually uploaded corpus folders and verify v2 counts; never download data."""
import argparse
from collections import Counter
import json
from pathlib import Path

from audit_structure import TSAC_DIR, MADAR_DIR
from load_working_data import load_records


def link_directory(destination, source):
    if not source.is_dir():
        raise FileNotFoundError(source)
    if destination.exists() or destination.is_symlink():
        if destination.resolve() != source.resolve():
            raise FileExistsError(f'Preserve existing source directory: {destination}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.symlink_to(source.resolve(), target_is_directory=True)


def unique_parent(root, filename):
    matches = sorted(p for p in root.rglob(filename) if p.is_file())
    if len(matches) != 1:
        raise ValueError(f'Expected one extracted {filename}; found {len(matches)}. Choose a narrower --input-root.')
    return matches[0].parent


def prepare(root):
    if not root.is_dir():
        raise FileNotFoundError(root)
    tsac = unique_parent(root, 'train_pos.txt')
    madar = unique_parent(root, 'MADAR.corpus.MSA.tsv')
    if madar.name != 'MADAR_Corpus':
        raise ValueError('Preserve the original MADAR_Corpus folder structure')
    link_directory(TSAC_DIR, tsac)
    link_directory(MADAR_DIR.parent, madar.parent)
    result = {'manifest_version': 'v2', 'selected_test_rows': 0, 'source_file_hashes_verified': True, 'corpora': {}}
    for corpus, expected in [('tsac', {'train': 10053, 'validation': 1774}), ('madar', {'train': 6388, 'validation': 800})]:
        rows = load_records(corpus, ('train', 'validation'), manifest_version='v2')
        counts = dict(Counter(r['split'] for r in rows))
        groups = {s: {r['group_id'] for r in rows if r['split'] == s} for s in expected}
        if counts != expected or groups['train'] & groups['validation']:
            raise ValueError(f'Unexpected partitions/groups: {corpus}')
        result['corpora'][corpus] = {'partitions': counts, 'shared_train_validation_groups': 0}
        print(f'{corpus.upper()} : train={counts["train"]}, validation={counts["validation"]} | empreintes OK | groupes communs=0')
    print('Contrôle réussi — aucune ligne de test sélectionnée.')
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root', type=Path, required=True)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if a.output and a.output.exists(): p.error('Preserve previous data verification output')
    result = prepare(a.input_root)
    if a.output:
        a.output.parent.mkdir(parents=True, exist_ok=True)
        a.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
