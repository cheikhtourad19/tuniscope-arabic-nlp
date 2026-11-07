"""Construct and verify the fixed 4,000-row train-only Qwen SFT dataset."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from encoder_tokenizer_audit import object_digest
from fetch_qwen3 import ROOT, REVISION, sha256
from load_working_data import load_records
from qwen_prompting import PROMPT_CONFIG, load_tokenizer, messages
from run_qwen_tokenizer_audit import encode

CONFIG = ROOT / 'configs/qwen_sft_v1.json'
PROTOCOL = ROOT / 'docs/61_protocole_sft_qwen.md'
AUDIT = ROOT / 'reports/qwen_tokenizer_audit_v2_2026-11-07/summary.json'
IGNORE = -100


def select_rows(corpus, train, counts, seed):
    selected = []
    for label, quota in counts.items():
        ranked = sorted((row for row in train if row['label'] == label),
                        key=lambda row: hashlib.sha256(f"{seed}|{corpus}|{row['id']}".encode()).digest())
        if len(ranked) < quota:
            raise ValueError(f'Insufficient training rows for {corpus} {label}')
        selected.extend(ranked[:quota])
    return selected


def tokenize_sft(tokenizer, task, rows, limit, prompt_limit):
    conversations = [messages(task, row['text_raw']) for row in rows]
    prefixes = tokenizer.apply_chat_template(conversations, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    completed = [conversation + [{'role': 'assistant', 'content': row['label']}]
                 for conversation, row in zip(conversations, rows, strict=True)]
    rendered = tokenizer.apply_chat_template(completed, tokenize=False, add_generation_prompt=False, enable_thinking=False)
    for prefix, full, row in zip(prefixes, rendered, rows, strict=True):
        if full != prefix + row['label'] + tokenizer.eos_token + '\n':
            raise ValueError('Supervised chat template differs from the inference prefix')
    prefix_ids = encode(tokenizer, prefixes)
    full_ids = encode(tokenizer, [value.removesuffix('\n') for value in rendered])
    for row, prompt, sequence in zip(rows, prefix_ids, full_ids, strict=True):
        boundary = len(prompt)
        target = sequence[boundary:]
        if sequence[:boundary] != prompt or len(target) < 2 or target[-1] != tokenizer.eos_token_id:
            raise ValueError('Invalid prompt boundary or missing target EOS')
        if tokenizer.decode(target[:-1], skip_special_tokens=False, clean_up_tokenization_spaces=False) != row['label']:
            raise ValueError('Target tokens do not decode to the source label')
        if len(sequence) > limit or boundary > prompt_limit:
            raise ValueError('Supervised sequence exceeds the audited budget')
        yield {'id': row['id'], 'group_id': row['group_id'], 'label': row['label'],
               'input_ids': sequence, 'labels': [IGNORE] * boundary + target, 'prompt_tokens': boundary}


def build():
    config = json.loads(CONFIG.read_text())
    prompting = json.loads(PROMPT_CONFIG.read_text())
    audit = json.loads(AUDIT.read_text())
    manifests = {c: sha256(ROOT / f'data/manifests/v2/{c}.csv') for c in config['label_counts']}
    if (audit['config_sha256'] != sha256(PROMPT_CONFIG) or audit['revision'] != REVISION
            or audit['manifests_sha256'] != manifests or audit['length_decision']['total_tokens'] != config['max_sequence_tokens']):
        raise ValueError('SFT configuration differs from the pinned tokenizer audit')
    tokenizer, tokenizer_files = load_tokenizer()
    limit = config['max_sequence_tokens']
    prompt_limit = audit['length_decision']['max_prompt_tokens']
    dataset, corpus_summary, inspections = [], {}, {}
    for corpus, quotas in config['label_counts'].items():
        task = prompting['tasks'][corpus]
        if set(quotas) != set(task['labels']) or sum(quotas.values()) != 2000:
            raise ValueError('Expected 2,000 supervised rows with the task labels')
        train = load_records(corpus, splits=('train',), manifest_version='v2')
        if any(row['split'] != 'train' for row in train):
            raise ValueError('SFT requires train rows')
        selected = select_rows(corpus, train, quotas, config['selection_seed'])
        encoded = []
        for start in range(0, len(selected), 256):
            encoded.extend(tokenize_sft(tokenizer, task, selected[start:start + 256], limit, prompt_limit))
        counts = Counter(row['label'] for row in encoded)
        if counts != quotas or len({row['id'] for row in encoded}) != 2000:
            raise ValueError('Unexpected counts or duplicate supervised rows')
        dataset.extend({'corpus': corpus, **row} for row in encoded)
        corpus_summary[corpus] = {'rows': len(encoded), 'label_counts': dict(counts),
                                  'selected_ids_sha256': object_digest([row['id'] for row in encoded]),
                                  'max_sequence_tokens': max(len(row['input_ids']) for row in encoded),
                                  'supervised_tokens': sum(len(row['input_ids']) - row['prompt_tokens'] for row in encoded)}
        # Invented text supplies a shareable inspection of each real task mask.
        sample = {'id': 'invented', 'group_id': 'invented', 'text_raw': 'خدمة باهية 🙂' if corpus == 'tsac' else 'فين المحطة؟',
                  'label': 'POS' if corpus == 'tsac' else 'MSA'}
        record = next(tokenize_sft(tokenizer, task, [sample], limit, prompt_limit))
        boundary = record['prompt_tokens']
        inspections[corpus] = {'invented_text': sample['text_raw'], 'expected_label': sample['label'],
                               'prompt_tokens_masked': boundary, 'target_ids': record['labels'][boundary:],
                               'target_tokens': tokenizer.convert_ids_to_tokens(record['labels'][boundary:])}
    seed = config['selection_seed']
    dataset.sort(key=lambda row: hashlib.sha256(f"mix|{seed}|{row['corpus']}|{row['id']}".encode()).digest())
    summary = {'revision': REVISION, 'rows': len(dataset), 'split': 'train', 'corpora': corpus_summary,
               'invented_mask_inspections': inspections, 'eos_token_id': tokenizer.eos_token_id, 'pad_token_id': tokenizer.pad_token_id,
               'loss_ignore_index': IGNORE, 'tokenizer_files': tokenizer_files, 'config_sha256': sha256(CONFIG),
               'prompt_config_sha256': sha256(PROMPT_CONFIG), 'audit_sha256': sha256(AUDIT), 'protocol_sha256': sha256(PROTOCOL),
               'manifests_sha256': manifests, 'code_sha256': sha256(Path(__file__)),
               'dataset_sha256': object_digest(dataset)}
    return dataset, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--output', type=Path)
    mode.add_argument('--verify', type=Path)
    args = parser.parse_args()
    target = (args.output or args.verify).resolve()
    if (ROOT / 'outputs/qwen').resolve() not in target.parents:
        parser.error('Use a private directory under outputs/qwen/')
    if args.output and target.exists():
        raise FileExistsError(target)
    dataset, summary = build()
    if args.verify:
        if (json.loads((target / 'dataset.json').read_text()) != dataset
                or json.loads((target / 'summary.json').read_text()) != summary):
            raise ValueError('Saved SFT dataset differs from the reproduced selection or masks')
        print('Corpus SFT, sélections et masques reproduits à l’identique.')
    else:
        target.mkdir(parents=True)
        (target / 'dataset.json').write_text(json.dumps(dataset, ensure_ascii=False) + '\n')
        (target / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps({'rows': summary['rows'], 'corpora': summary['corpora'], 'output': str(target)}, indent=2))


if __name__ == '__main__':
    main()
