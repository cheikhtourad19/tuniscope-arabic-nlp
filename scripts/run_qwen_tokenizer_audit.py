"""Audit the pinned Qwen tokenizer and complete prompts on train/validation v2."""
import argparse
from collections import defaultdict
from fractions import Fraction
import importlib.metadata
import json
from pathlib import Path

from audit_text_composition import percentile, writing_system
from encoder_tokenizer_audit import PROBES, object_digest
from fetch_qwen3 import MODEL_ID, REVISION, ROOT, sha256
from load_working_data import load_records
from qwen_prompting import PROMPT_CONFIG, load_tokenizer, render_prompts, select_demonstrations

SPLITS = ('train', 'validation')
BATCH_SIZE = 256
PROTOCOL = ROOT / 'docs/59_protocole_tokenizer_et_consignes_qwen.md'


def encode(tokenizer, texts):
    return tokenizer(texts, add_special_tokens=False, truncation=False, padding=False,
                     return_attention_mask=False, verbose=False)['input_ids']


def measure(rows, tokenizer, task, demonstrations):
    result = []
    markers = (*tokenizer.all_special_tokens, '<think>', '</think>')
    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]
        texts = [row['text_raw'] for row in batch]
        raw = encode(tokenizer, texts)
        zero = encode(tokenizer, render_prompts(tokenizer, task, texts))
        few = encode(tokenizer, render_prompts(tokenizer, task, texts, demonstrations))
        for value, raw_ids, zero_ids, few_ids in zip(texts, raw, zero, few, strict=True):
            result.append({
                'writing': writing_system(value), 'fields': len(value.split()),
                'raw': len(raw_ids), 'zero_shot': len(zero_ids), 'few_shot': len(few_ids),
                'unknown': raw_ids.count(tokenizer.unk_token_id) if tokenizer.unk_token_id is not None else None,
                'contains_reserved_marker': any(marker in value for marker in markers),
            })
    return result


def summarize(rows, prompt_limits, minimum):
    fields = sum(row['fields'] for row in rows)
    raw_tokens = sum(row['raw'] for row in rows)
    unknown = sum(row['unknown'] for row in rows) if rows[0]['unknown'] is not None else None
    lengths = {}
    for mode in ('raw', 'zero_shot', 'few_shot'):
        counts = sorted(row[mode] for row in rows)
        lengths[mode] = {
            f'p{p}': percentile(counts, Fraction(p, 100)) for p in (50, 95, 99)
        }
        lengths[mode]['max'] = counts[-1]
        if mode != 'raw':
            lengths[mode]['over_limit_rows'] = {
                str(limit): sum(count > limit for count in counts) for limit in prompt_limits}
    return {'rows': len(rows), 'minimum_rows_reached': len(rows) >= minimum,
            'raw_tokens': raw_tokens, 'whitespace_fields': fields,
            'tokens_per_whitespace_field': raw_tokens / fields if fields else None,
            'unknown_tokens': unknown, 'unknown_token_rate': unknown / raw_tokens if unknown is not None and raw_tokens else None,
            'rows_with_reserved_marker': sum(row['contains_reserved_marker'] for row in rows),
            'lengths': lengths}


def audit():
    config = json.loads(PROMPT_CONFIG.read_text())
    if (config['manifest_version'], config['variant'], config['enable_thinking']) != ('v2', 'raw-v0', False):
        raise ValueError('Expected the v2 raw-v0 non-thinking protocol')
    tokenizer, fingerprints = load_tokenizer()
    reserve = config['max_new_tokens']
    limits = [total - reserve for total in config['candidate_total_tokens']]
    corpora = {}
    for corpus, task in config['tasks'].items():
        records = load_records(corpus, splits=SPLITS, manifest_version='v2')
        partitions = {split: [row for row in records if row['split'] == split] for split in SPLITS}
        train_groups = {row['group_id'] for row in partitions['train']}
        validation_groups = {row['group_id'] for row in partitions['validation']}
        if train_groups & validation_groups or sum(map(len, partitions.values())) != len(records):
            raise ValueError('Expected disjoint train/validation groups')
        demonstrations = select_demonstrations(partitions['train'], task['labels'], config['few_shot_seed'])
        corpora[corpus] = {'demonstrations': {
            'split': 'train', 'labels': [row['label'] for row in demonstrations],
            'ids_sha256': object_digest([row['id'] for row in demonstrations]),
            'groups_sha256': object_digest([row['group_id'] for row in demonstrations]),
            'writing_systems': [writing_system(row['text_raw']) for row in demonstrations]}, 'partitions': {}}
        for split, rows in partitions.items():
            measured = measure(rows, tokenizer, task, demonstrations)
            by_writing = defaultdict(list)
            for row in measured:
                by_writing[row['writing']].append(row)
            minimum = config['minimum_writing_group_rows']
            corpora[corpus]['partitions'][split] = {
                'ids_sha256': object_digest([row['id'] for row in rows]),
                'all': summarize(measured, limits, minimum),
                'by_writing_system': {group: summarize(items, limits, minimum)
                                      for group, items in sorted(by_writing.items())},
            }
    train_max = max(corpus['partitions']['train']['all']['lengths'][mode]['max']
                    for corpus in corpora.values() for mode in ('zero_shot', 'few_shot'))
    chosen = next((total for total in config['candidate_total_tokens'] if train_max + reserve <= total), None)
    probes = []
    for (name, value), ids in zip(PROBES, encode(tokenizer, [value for _, value in PROBES]), strict=True):
        probes.append({'name': name, 'invented_text': value, 'tokens': tokenizer.convert_ids_to_tokens(ids),
                       'round_trip_exact': tokenizer.decode(ids, clean_up_tokenization_spaces=False) == value})
    return {'model_id': MODEL_ID, 'revision': REVISION, 'config': config,
            'tokenizer_class': type(tokenizer).__name__, 'tokenizer_length': len(tokenizer),
            'unknown_token_id': tokenizer.unk_token_id, 'tokenizer_files': fingerprints,
            'partitions': list(SPLITS), 'weights_loaded': False, 'test_selected': False,
            'corpora': corpora, 'invented_probes': probes,
            'length_decision': {'selection_split': 'train', 'train_max_prompt_tokens': train_max,
                                'total_tokens': chosen, 'max_prompt_tokens': chosen - reserve if chosen else None,
                                'reserved_output_tokens': reserve},
            'packages': {name: importlib.metadata.version(name) for name in ('transformers', 'tokenizers')},
            'config_sha256': sha256(PROMPT_CONFIG), 'protocol_sha256': sha256(PROTOCOL),
            'environment_lock_sha256': sha256(ROOT / 'environments/llm/uv.lock'),
            'code_sha256': {name: sha256(ROOT / 'scripts' / name) for name in (
                'run_qwen_tokenizer_audit.py', 'qwen_prompting.py', 'fetch_qwen3.py',
                'load_working_data.py', 'audit_text_composition.py', 'encoder_tokenizer_audit.py')},
            'manifests_sha256': {corpus: sha256(ROOT / f'data/manifests/v2/{corpus}.csv') for corpus in corpora}}


def report(result):
    decision = result['length_decision']
    lines = ['# Audit du tokenizer Qwen3', '', 'Train et validation v2, texte brut.', '',
             '| Corpus | Partition | Lignes | Texte seul p99 / max | Zero-shot p99 / max | Few-shot p99 / max |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for corpus, data in result['corpora'].items():
        for split, partition in data['partitions'].items():
            s = partition['all']
            lengths = [f"{s['lengths'][key]['p99']} / {s['lengths'][key]['max']}" for key in ('raw', 'zero_shot', 'few_shot')]
            lines.append(f'| {corpus.upper()} | {split} | {s["rows"]} | ' + ' | '.join(lengths) + ' |')
    lines += ['', '| Corpus | Partition | Écriture | Lignes | Tokens par champ | Au moins 100 lignes |',
              '| --- | --- | --- | ---: | ---: | --- |']
    for corpus, data in result['corpora'].items():
        for split, partition in data['partitions'].items():
            for group, stats in partition['by_writing_system'].items():
                lines.append(f'| {corpus.upper()} | {split} | {group} | {stats["rows"]} | '
                             f'{stats["tokens_per_whitespace_field"]:.2f} | '
                             f'{"Oui" if stats["minimum_rows_reached"] else "Non"} |')
    if decision['total_tokens'] is None:
        budget = 'Les budgets candidats couvrent seulement une partie des prompts du train.'
    else:
        budget = (f'Budget retenu sur le train **{decision["total_tokens"]}** tokens au total, '
                  f'avec **{decision["max_prompt_tokens"]}** tokens d’entrée et **{decision["reserved_output_tokens"]}** en réserve.')
    lines += ['', budget,
              'Les dépassements par budget, les longueurs par écriture et les sondes inventées sont consignés dans `summary.json`.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--verify', type=Path)
    args = parser.parse_args()
    target = (args.output or args.verify).resolve()
    if (ROOT / 'reports').resolve() not in target.parents:
        parser.error('Choose a directory inside reports/')
    if args.output and target.exists():
        raise FileExistsError(target)
    result = audit()
    if args.verify:
        if json.loads((target / 'summary.json').read_text()) != result or (target / 'README.md').read_text() != report(result):
            raise ValueError('Saved Qwen tokenizer audit differs from recomputed report')
        print('Audit Qwen reproduit à l’identique.')
        return
    target.mkdir(parents=True)
    (target / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (target / 'README.md').write_text(report(result))
    print(report(result))


if __name__ == '__main__':
    main()
