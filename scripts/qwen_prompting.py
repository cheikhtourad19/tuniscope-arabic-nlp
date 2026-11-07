"""Shared Qwen classification messages, train demonstrations and label parser."""
import hashlib
import json

from fetch_qwen3 import MODEL_DIR, ROOT, load_plan, verify_file

PROMPT_CONFIG = ROOT / 'configs/qwen_prompting_v1.json'
TOKENIZER_FILES = ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')


def load_tokenizer():
    from transformers import AutoTokenizer

    plan = load_plan()
    fingerprints = {name: verify_file(MODEL_DIR / name, plan['files'][name]) for name in TOKENIZER_FILES}
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, use_fast=True, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast or not tokenizer.chat_template:
        raise ValueError('Expected a fast Qwen tokenizer with the official chat template')
    return tokenizer, fingerprints


def select_demonstrations(train, labels, seed):
    """Choose one row per label, in label order, from distinct training groups."""
    if any(row['split'] != 'train' for row in train):
        raise ValueError('Demonstrations require training rows only')
    ranked = sorted(train, key=lambda row: hashlib.sha256(f"{seed}|{row['id']}".encode()).digest())
    selected, used_groups = [], set()
    for label in labels:
        example = next((row for row in ranked if row['label'] == label and row['group_id'] not in used_groups), None)
        if example is None:
            raise ValueError(f'Missing independent training example for {label}')
        selected.append(example)
        used_groups.add(example['group_id'])
    return selected


def text_message(text):
    # JSON escapes preserve the text while preventing embedded Qwen role delimiters.
    return json.dumps(text, ensure_ascii=False).replace('<', r'\u003c').replace('>', r'\u003e')


def messages(task, text, demonstrations=()):
    conversation = [{'role': 'system', 'content': task['instruction']}]
    for row in demonstrations:
        if row['split'] != 'train' or row['label'] not in task['labels']:
            raise ValueError('Invalid demonstration provenance or label')
        conversation.extend([
            {'role': 'user', 'content': text_message(row['text_raw'])},
            {'role': 'assistant', 'content': row['label']},
        ])
    conversation.append({'role': 'user', 'content': text_message(text)})
    return conversation


def render_prompts(tokenizer, task, texts, demonstrations=()):
    return tokenizer.apply_chat_template(
        [messages(task, text, demonstrations) for text in texts],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def parse_label(response, labels):
    value = response.strip()
    return value if value in labels else None


def decode_label(tokenizer, generated_ids, stop_ids, labels):
    """Decode continuation IDs and discard a single terminal generation delimiter."""
    ids = list(generated_ids)
    if ids and ids[-1] in stop_ids:
        ids.pop()
    response = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    return parse_label(response, labels)
