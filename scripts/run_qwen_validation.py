"""Generate and checkpoint the complete local Qwen validation reference."""
import argparse
import json
from pathlib import Path
import platform
import time

import torch
from transformers import AutoModelForCausalLM, GenerationConfig, __version__ as transformers_version

from audit_text_composition import writing_system
from benchmark_qwen_validation import AUDIT, synchronize
from check_qwen_loading import resolve_device
from encoder_tokenizer_audit import object_digest
from fetch_qwen3 import MODEL_DIR, MODEL_ID, REVISION, ROOT, sha256, verify_model
from load_working_data import load_records
from qwen_prompting import PROMPT_CONFIG, load_tokenizer, parse_label, render_prompts, select_demonstrations
from run_qwen_tokenizer_audit import encode

BATCH_SIZE = 16


def save_json(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def prepare(tokenizer, config, audit):
    for corpus, task in config['tasks'].items():
        rows = load_records(corpus, splits=('train', 'validation'), manifest_version='v2')
        train = [row for row in rows if row['split'] == 'train']
        validation = [row for row in rows if row['split'] == 'validation']
        demos = select_demonstrations(train, task['labels'], config['few_shot_seed'])
        audited = audit['corpora'][corpus]
        if (object_digest([row['id'] for row in validation]) != audited['partitions']['validation']['ids_sha256']
                or object_digest([row['id'] for row in demos]) != audited['demonstrations']['ids_sha256']
                or {row['group_id'] for row in train} & {row['group_id'] for row in validation}):
            raise ValueError('Validation or demonstration provenance differs from the audit')
        for mode, examples in (('zero_shot', ()), ('few_shot', demos)):
            pairs = []
            for start in range(0, len(validation), 256):
                block = validation[start:start + 256]
                rendered = render_prompts(tokenizer, task, [row['text_raw'] for row in block], examples)
                pairs.extend(zip(block, encode(tokenizer, rendered), strict=True))
            pairs.sort(key=lambda pair: len(pair[1]))
            if not pairs or len(pairs[-1][1]) > audit['length_decision']['max_prompt_tokens']:
                raise ValueError('Prompt exceeds the audited input budget')
            yield corpus, mode, validation, pairs


def generate_batch(model, tokenizer, generation, options, device, pairs, labels):
    inputs = tokenizer.pad({'input_ids': [ids for _, ids in pairs]}, padding=True,
                           return_attention_mask=True, return_tensors='pt').to(device)
    width = inputs['input_ids'].shape[1]
    generated = model.generate(**inputs, generation_config=generation, **options)[:, width:].cpu().tolist()
    stops = set(generation.eos_token_id)
    for (row, input_ids), continuation in zip(pairs, generated, strict=True):
        stop = next((i for i, token in enumerate(continuation) if token in stops), None)
        # Stop at the first EOS so right padding on shorter responses never reaches the parser.
        content = continuation if stop is None else continuation[:stop]
        count = len(continuation) if stop is None else stop + 1
        if not 0 < count <= options['max_new_tokens']:
            raise ValueError('Generated length outside the protocol')
        response = tokenizer.decode(content, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        yield {'id': row['id'], 'group_id': row['group_id'], 'truth': row['label'],
               'writing': writing_system(row['text_raw']), 'prompt_tokens': len(input_ids),
               'generated_tokens': count, 'ended_with_eos': stop is not None,
               'response': response, 'prediction': parse_label(response, labels)}


def run(output, device_request, resume):
    output = output.resolve()
    if (ROOT / 'outputs/qwen').resolve() not in output.parents:
        raise ValueError('Store private predictions under outputs/qwen/')
    if output.exists() != resume:
        raise ValueError('Use a new output directory or explicitly resume an existing run')
    start = time.perf_counter()
    device = resolve_device(device_request)
    torch.set_num_threads(2)
    config = json.loads(PROMPT_CONFIG.read_text())
    audit = json.loads(AUDIT.read_text())
    if audit['config_sha256'] != sha256(PROMPT_CONFIG) or audit['revision'] != REVISION:
        raise ValueError('Prompt configuration and audit differ')
    manifest_hashes = {corpus: sha256(ROOT / f'data/manifests/v2/{corpus}.csv') for corpus in config['tasks']}
    if manifest_hashes != audit['manifests_sha256']:
        raise ValueError('The validation manifests differ from the audit')
    source = verify_model()
    tokenizer, _ = load_tokenizer()
    tokenizer.padding_side = 'left'
    generation = GenerationConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token_id != generation.pad_token_id:
        raise ValueError('Tokenizer and generation padding disagree')
    options = {**config['initial_generation'], 'max_new_tokens': config['max_new_tokens'], 'use_cache': True}
    plan = {'model_id': MODEL_ID, 'revision': REVISION, 'device': str(device), 'platform': platform.platform(),
            'dtype': 'torch.float32', 'attention': 'eager', 'batch_size': BATCH_SIZE, 'batch_order': 'prompt_length',
            'prompt_config_sha256': sha256(PROMPT_CONFIG), 'audit_sha256': sha256(AUDIT),
            'model_sha256': source['files']['model.safetensors']['sha256'], 'manifests_sha256': manifest_hashes,
            'environment_lock_sha256': sha256(ROOT / 'environments/llm/uv.lock'),
            'code_sha256': {name: sha256(ROOT / 'scripts' / name) for name in (
                'run_qwen_validation.py', 'qwen_prompting.py', 'run_qwen_tokenizer_audit.py', 'load_working_data.py')},
            'torch': torch.__version__, 'transformers': transformers_version, 'generation': options,
            'split': 'validation', 'demonstration_split': 'train', 'variant': 'raw-v0', 'training': False}
    plan_path = output / 'plan.json'
    if resume:
        if json.loads(plan_path.read_text()) != plan or (output / 'completion.json').exists():
            raise ValueError('Expected an incomplete run with identical settings')
    else:
        output.mkdir(parents=True)
        save_json(plan_path, plan)
    plan_hash = sha256(plan_path)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation='eager')
    model.requires_grad_(False).eval().to(device)
    synchronize(device)
    stages = {}
    with torch.inference_mode():
        for corpus, mode, validation, pairs in prepare(tokenizer, config, audit):
            name = f'{corpus}_{mode}'
            path = output / f'{name}.json'
            if path.exists():
                saved = json.loads(path.read_text())
                if (saved['plan_sha256'] != plan_hash or saved['corpus'] != corpus or saved['mode'] != mode
                        or [row['id'] for row in saved['predictions']] != [row['id'] for row in validation]):
                    raise ValueError(f'Invalid saved stage: {name}')
                print(f'{name}: étape enregistrée relue', flush=True)
            else:
                tick = time.perf_counter()
                generated = {}
                print(f'{name}: génération de {len(pairs)} prédictions sur {device}', flush=True)
                for start_index in range(0, len(pairs), BATCH_SIZE):
                    batch = pairs[start_index:start_index + BATCH_SIZE]
                    for prediction in generate_batch(model, tokenizer, generation, options, device, batch,
                                                     config['tasks'][corpus]['labels']):
                        generated[prediction['id']] = prediction
                    done = start_index + len(batch)
                    if done % 256 == 0 or done == len(pairs):
                        print(f'{name}: {done}/{len(pairs)}', flush=True)
                synchronize(device)
                save_json(path, {'corpus': corpus, 'mode': mode, 'plan_sha256': plan_hash,
                                 'seconds': time.perf_counter() - tick,
                                 'predictions': [generated[row['id']] for row in validation]})
            stages[name] = sha256(path)
    save_json(output / 'completion.json', {'plan_sha256': plan_hash, 'stages': stages,
                                          'resumed': resume, 'seconds_this_execution': time.perf_counter() - start})
    print(f'Prédictions complètes sauvegardées dans {output}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    run(args.output, args.device, args.resume)
