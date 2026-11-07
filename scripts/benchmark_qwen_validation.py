"""Time a length-stratified sample of the two Qwen validation modes."""
import argparse
import json
import math
from pathlib import Path
import platform
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, GenerationConfig

from check_qwen_loading import resolve_device
from fetch_qwen3 import MODEL_DIR, REVISION, ROOT, sha256, verify_model
from load_working_data import load_records
from qwen_prompting import PROMPT_CONFIG, load_tokenizer, render_prompts, select_demonstrations

BATCH_SIZES = (8, 16)
SAMPLE_BLOCKS = 6
AUDIT = ROOT / 'reports/qwen_tokenizer_audit_v2_2026-11-07/summary.json'


def synchronize(device):
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def prepare_cases(tokenizer, config, budget):
    cases = []
    for corpus, task in config['tasks'].items():
        rows = load_records(corpus, splits=('train', 'validation'), manifest_version='v2')
        train = [row for row in rows if row['split'] == 'train']
        validation = [row['text_raw'] for row in rows if row['split'] == 'validation']
        demonstrations = select_demonstrations(train, task['labels'], config['few_shot_seed'])
        for mode, examples in (('zero_shot', ()), ('few_shot', demonstrations)):
            inputs = []
            for start in range(0, len(validation), 256):
                rendered = render_prompts(tokenizer, task, validation[start:start + 256], examples)
                inputs.extend(tokenizer(rendered, add_special_tokens=False, truncation=False,
                                        return_attention_mask=False)['input_ids'])
            inputs.sort(key=len)
            if len(inputs) != len(validation) or not inputs or len(inputs[-1]) > budget:
                raise ValueError('Unexpected prompt count or length')
            # Draw the same complete blocks for both batch sizes across the length distribution.
            width = max(BATCH_SIZES)
            blocks = len(inputs) // width
            if blocks < SAMPLE_BLOCKS:
                raise ValueError('Insufficient validation blocks')
            positions = [(2 * index + 1) * blocks // (2 * SAMPLE_BLOCKS) for index in range(SAMPLE_BLOCKS)]
            sample = [item for pos in positions for item in inputs[pos * width:(pos + 1) * width]]
            cases.append({'corpus': corpus, 'mode': mode, 'population': len(inputs),
                          'sample_block_positions': positions, 'inputs': sample})
    return cases


def timed_batch(model, tokenizer, device, generation_config, options, stops, inputs):
    synchronize(device)
    start = time.perf_counter()
    padded = tokenizer.pad({'input_ids': inputs}, padding=True, return_attention_mask=True, return_tensors='pt').to(device)
    input_width = padded['input_ids'].shape[1]
    generated = model.generate(**padded, generation_config=generation_config, **options)
    continuation = generated[:, input_width:].cpu().tolist()
    synchronize(device)
    seconds = time.perf_counter() - start
    counts = [next((index + 1 for index, token in enumerate(row) if token in stops), len(row))
              for row in continuation]
    if len(counts) != len(inputs) or not all(0 < n <= options['max_new_tokens'] for n in counts):
        raise ValueError('Unexpected generation shape or token count')
    return {'seconds': seconds, 'rows': len(inputs), 'padded_input_tokens_per_row': input_width,
            'generated_tokens_total': sum(counts), 'generated_tokens_max': max(counts)}


def benchmark(output, device_request):
    output = output.resolve()
    if (ROOT / 'outputs/qwen').resolve() not in output.parents or output.exists():
        raise ValueError('Choose a new file under outputs/qwen/')
    start = time.perf_counter()
    device = resolve_device(device_request)
    torch.set_num_threads(2)
    config = json.loads(PROMPT_CONFIG.read_text())
    audit = json.loads(AUDIT.read_text())
    if audit['config_sha256'] != sha256(PROMPT_CONFIG) or audit['revision'] != REVISION:
        raise ValueError('Prompt audit and benchmark configuration differ')
    source = verify_model()
    tokenizer, _ = load_tokenizer()
    tokenizer.padding_side = 'left'
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation='eager')
    model.requires_grad_(False).eval().to(device)
    generation = GenerationConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    options = {**config['initial_generation'], 'max_new_tokens': config['max_new_tokens'], 'use_cache': True}
    stops = set(generation.eos_token_id)
    if tokenizer.pad_token_id != generation.pad_token_id:
        raise ValueError('Tokenizer and generation padding disagree')
    synchronize(device)
    setup_seconds = time.perf_counter() - start
    prep_start = time.perf_counter()
    cases = prepare_cases(tokenizer, config, audit['length_decision']['max_prompt_tokens'])
    prep_seconds = time.perf_counter() - prep_start
    result = {'revision': REVISION, 'device': str(device), 'platform': platform.platform(),
              'dtype': str(next(model.parameters()).dtype), 'torch': torch.__version__,
              'prompt_config_sha256': sha256(PROMPT_CONFIG), 'audit_sha256': sha256(AUDIT),
              'script_sha256': sha256(Path(__file__)), 'lock_sha256': sha256(ROOT / 'environments/llm/uv.lock'),
              'model_sha256': source['files']['model.safetensors']['sha256'],
              'setup_seconds': setup_seconds, 'prompt_preparation_seconds_all_validation': prep_seconds,
              'validation_predictions_full': sum(case['population'] for case in cases),
              'sampling': 'six evenly spaced complete blocks of 16 rows after stable prompt-length sorting',
              'generation': options, 'measurements': {}}
    print(f'{device}, FP32. Préparation complète des prompts en {prep_seconds:.1f} s.', flush=True)
    with torch.inference_mode():
        for batch_size in BATCH_SIZES:
            warmup = timed_batch(model, tokenizer, device, generation, options, stops, cases[0]['inputs'][:batch_size])
            measurements = []
            for case in cases:
                inputs = case['inputs']
                timings = [timed_batch(model, tokenizer, device, generation, options, stops, inputs[i:i + batch_size])
                           for i in range(0, len(inputs), batch_size)]
                batch_seconds = [row['seconds'] for row in timings]
                estimated = statistics.mean(batch_seconds) * math.ceil(case['population'] / batch_size)
                measurements.append({**{key: value for key, value in case.items() if key != 'inputs'},
                                     'sample_rows': len(inputs), 'batches': timings,
                                     'estimated_generation_seconds': estimated})
                print(f'{case["corpus"]} {case["mode"]} batch {batch_size}: '
                      f'{sum(batch_seconds):.2f} s / {len(inputs)} lignes', flush=True)
            measurement = {'cases': measurements, 'warmup': warmup,
                           'estimated_full_seconds': setup_seconds + prep_seconds + warmup['seconds'] +
                           sum(case['estimated_generation_seconds'] for case in measurements)}
            if device.type == 'mps':
                measurement['mps_driver_allocated_bytes_at_end'] = torch.mps.driver_allocated_memory()
            result['measurements'][str(batch_size)] = measurement
            print(f'Estimation complète, batch {batch_size}: {measurement["estimated_full_seconds"] / 60:.2f} min', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    benchmark(args.output, args.device)
