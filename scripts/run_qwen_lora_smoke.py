"""Train and reload a small shared Qwen LoRA adapter on local SFT rows."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

import torch
from peft import LoraConfig, PeftModel, get_peft_model, __version__ as peft_version
from transformers import AutoModelForCausalLM, GenerationConfig, __version__ as transformers_version

from benchmark_qwen_validation import synchronize
from build_qwen_sft import CONFIG, IGNORE, PROTOCOL
from check_qwen_loading import resolve_device
from encoder_tokenizer_audit import object_digest
from fetch_qwen3 import MODEL_DIR, MODEL_ID, REVISION, ROOT, sha256, verify_model
from qwen_prompting import PROMPT_CONFIG, load_tokenizer, render_prompts
from run_qwen_tokenizer_audit import encode


def sample_rows(dataset, config):
    rows = []
    count = config['smoke']['examples_per_task']
    for corpus in config['label_counts']:
        ranked = sorted((row for row in dataset if row['corpus'] == corpus), key=lambda row: len(row['input_ids']))
        rows.extend(ranked[(2 * index + 1) * len(ranked) // (2 * count)] for index in range(count))
    random.Random(config['smoke']['seed']).shuffle(rows)
    if {(row['corpus'], row['label']) for row in rows} != {
        (corpus, label) for corpus, quotas in config['label_counts'].items() for label in quotas
    }:
        raise ValueError('Expected all task labels in the smoke subset')
    return rows


def collate(rows, tokenizer, device):
    width = max(len(row['input_ids']) for row in rows)
    ids, masks, targets = [], [], []
    for row in rows:
        sequence, labels = row['input_ids'], row['labels']
        boundary = row['prompt_tokens']
        if (not 0 < boundary < len(sequence) or labels != [IGNORE] * boundary + sequence[boundary:]
                or labels[-1] != tokenizer.eos_token_id):
            raise ValueError('Invalid SFT loss mask')
        extra = width - len(sequence)
        ids.append(sequence + [tokenizer.pad_token_id] * extra)
        masks.append([1] * len(sequence) + [0] * extra)
        targets.append(labels + [IGNORE] * extra)
    return {name: torch.tensor(values, dtype=torch.long, device=device) for name, values in (
        ('input_ids', ids), ('attention_mask', masks), ('labels', targets))}


def digest_parameters(named):
    digest = hashlib.sha256()
    for name, value in named:
        digest.update(name.encode() + b'\0')
        digest.update(memoryview(value.detach().cpu().contiguous().numpy()).cast('B'))
    return digest.hexdigest()


def load_base():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation='eager').requires_grad_(False)


def memory_snapshot(device):
    if device.type == 'mps':
        return {'sampled_mps_tensor_bytes': torch.mps.current_allocated_memory(),
                'sampled_mps_driver_bytes': torch.mps.driver_allocated_memory()}
    if device.type == 'cuda':
        return {'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    return {}


def train(model, tokenizer, rows, config, device):
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    frozen = [(name, p) for name, p in model.named_parameters() if not p.requires_grad]
    before_adapter = digest_parameters(trainable)
    before_base = digest_parameters(frozen)
    parameters = [p for _, p in trainable]
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=0.01)
    physical = config['batch_size']
    effective = physical * config['accumulation_steps']
    if len(rows) % effective:
        raise ValueError('The smoke sample must fill complete accumulation windows')
    steps, memory, manual_difference = [], {}, None
    model.train()
    for start in range(0, len(rows), effective):
        synchronize(device)
        tick = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        window = rows[start:start + effective]
        total_tokens = sum(len(row['input_ids']) - row['prompt_tokens'] for row in window)
        window_loss = 0.0
        for offset in range(0, effective, physical):
            subset = window[offset:offset + physical]
            batch = collate(subset, tokenizer, device)
            output = model(**batch, use_cache=False)
            loss = output.loss
            if not torch.isfinite(loss).item():
                raise ValueError('Non-finite SFT loss')
            if manual_difference is None:
                labels = batch['labels'][:, 1:]
                mask = labels != IGNORE
                manual = torch.nn.functional.cross_entropy(output.logits.detach()[:, :-1][mask], labels[mask])
                manual_difference = float((manual - loss.detach()).abs())
                if manual_difference > 1e-5:
                    raise ValueError('Model loss disagrees with answer-only shifted cross entropy')
            for key, value in memory_snapshot(device).items():
                memory[key] = max(memory.get(key, 0), value)
            weight = sum(len(row['input_ids']) - row['prompt_tokens'] for row in subset) / total_tokens
            (loss * weight).backward()
            window_loss += float(loss.detach()) * weight
            del output, loss
        if any(p.grad is not None for _, p in frozen):
            raise ValueError('A frozen base parameter received a gradient')
        norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
        if norm <= 0:
            raise ValueError('Adapter gradient is zero')
        optimizer.step()
        synchronize(device)
        for key, value in memory_snapshot(device).items():
            memory[key] = max(memory.get(key, 0), value)
        step = {'step': len(steps) + 1, 'loss': window_loss, 'gradient_norm': norm,
                'seconds': time.perf_counter() - tick, 'rows': len(window), 'supervised_tokens': total_tokens}
        steps.append(step)
        print(f"LoRA étape {step['step']} : loss {window_loss:.4f}, gradient {norm:.4f}, {step['seconds']:.2f} s", flush=True)
    base_after = digest_parameters(frozen)
    adapter_after = digest_parameters(trainable)
    if base_after != before_base or adapter_after == before_adapter:
        raise ValueError('Expected unchanged frozen weights and a changed adapter')
    return {'steps': steps, 'memory': memory, 'manual_loss_absolute_difference': manual_difference,
            'frozen_base_sha256_before_and_after': base_after, 'adapter_sha256_before': before_adapter,
            'adapter_sha256_after': adapter_after, 'trainable_parameters': sum(p.numel() for p in parameters),
            'frozen_parameters': sum(p.numel() for _, p in frozen)}


def probe(model, tokenizer, device):
    prompting = json.loads(PROMPT_CONFIG.read_text())
    generation = GenerationConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    options = {**prompting['initial_generation'], 'max_new_tokens': prompting['max_new_tokens'], 'use_cache': True}
    ids = []
    for corpus, text in (('tsac', 'الخدمة باهية برشا'), ('madar', 'فين المحطة؟')):
        ids.extend(encode(tokenizer, render_prompts(tokenizer, prompting['tasks'][corpus], [text])))
    tokenizer.padding_side = 'left'
    batch = tokenizer.pad({'input_ids': ids}, padding=True, return_attention_mask=True, return_tensors='pt').to(device)
    width = batch['input_ids'].shape[1]
    model.eval()
    with torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[:, -1].cpu()
        continuations = model.generate(**batch, generation_config=generation, **options)[:, width:].cpu().tolist()
    return logits, continuations


def run(dataset_dir, output, device_request, batch_size):
    start = time.perf_counter()
    dataset_dir, output = dataset_dir.resolve(), output.resolve()
    private = (ROOT / 'outputs/qwen').resolve()
    if private not in dataset_dir.parents or private not in output.parents or output.exists():
        raise ValueError('Use an existing private SFT dataset and a new private output directory')
    config = json.loads(CONFIG.read_text())
    smoke = config['smoke'].copy()
    if smoke['dtype'] != 'float32':
        raise ValueError('This smoke protocol uses FP32')
    if batch_size is not None:
        effective = smoke['batch_size'] * smoke['accumulation_steps']
        if effective % batch_size:
            raise ValueError('Batch size must divide the effective batch size')
        smoke['batch_size'] = batch_size
        smoke['accumulation_steps'] = effective // batch_size
    dataset = json.loads((dataset_dir / 'dataset.json').read_text())
    proof = json.loads((dataset_dir / 'summary.json').read_text())
    if (proof['split'] != 'train' or proof['revision'] != REVISION or len(dataset) != proof['rows'] or len(dataset) != 4000
            or proof['dataset_sha256'] != object_digest(dataset) or proof['config_sha256'] != sha256(CONFIG)
            or proof['protocol_sha256'] != sha256(PROTOCOL) or proof['prompt_config_sha256'] != sha256(PROMPT_CONFIG)):
        raise ValueError('Unexpected SFT dataset or provenance')
    source = verify_model()
    device = resolve_device(device_request)
    torch.set_num_threads(2)
    torch.manual_seed(smoke['seed'])
    tokenizer, _ = load_tokenizer()
    rows = sample_rows(dataset, config)
    settings = config['lora']
    base = load_base()
    targets = [(name, layer) for name, layer in base.named_modules() if name.rsplit('.', 1)[-1] in settings['target_modules']]
    if Counter(name.rsplit('.', 1)[-1] for name, _ in targets) != {
        name: base.config.num_hidden_layers for name in settings['target_modules']
    }:
        raise ValueError('Unexpected LoRA projection inventory')
    expected = sum(settings['r'] * (layer.in_features + layer.out_features) for _, layer in targets)
    model = get_peft_model(base, LoraConfig(task_type='CAUSAL_LM', r=settings['r'], lora_alpha=settings['alpha'],
                          lora_dropout=settings['dropout'], target_modules=settings['target_modules'], bias='none')).to(device)
    del base, targets
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if sum(p.numel() for _, p in trainable) != expected or any('.lora_' not in name for name, _ in trainable):
        raise ValueError('Unexpected trainable parameters')
    del trainable
    output.mkdir(parents=True)
    print(f'{device}: {len(rows)} exemples SFT, {expected} paramètres entraînables', flush=True)
    measured = train(model, tokenizer, rows, smoke, device)
    before, continuations = probe(model, tokenizer, device)
    adapter_path = output / 'adapter'
    model.peft_config['default'].base_model_name_or_path = MODEL_ID
    model.peft_config['default'].revision = REVISION
    model.save_pretrained(adapter_path, safe_serialization=True, save_embedding_layers=False)
    del model
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    restored = PeftModel.from_pretrained(load_base(), adapter_path, is_trainable=False, local_files_only=True).to(device)
    after, reloaded = probe(restored, tokenizer, device)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
    if reloaded != continuations:
        raise ValueError('Reloaded adapter changed probe generations')
    step_seconds = statistics.mean(step['seconds'] for step in measured['steps'][1:])
    effective = smoke['batch_size'] * smoke['accumulation_steps']
    result = {'scope': 'smoke', 'device': str(device), 'platform': platform.platform(), 'dtype': smoke['dtype'],
              'revision': REVISION, 'model_sha256': source['files']['model.safetensors']['sha256'],
              'dataset_sha256': proof['dataset_sha256'], 'dataset_summary_sha256': sha256(dataset_dir / 'summary.json'),
              'config_sha256': sha256(CONFIG), 'script_sha256': sha256(Path(__file__)),
              'lock_sha256': sha256(ROOT / 'environments/lora/uv.lock'), 'torch': torch.__version__,
              'transformers': transformers_version, 'peft': peft_version,
              'sample_rows': dict(Counter(row['corpus'] for row in rows)),
              'max_sample_sequence_tokens': max(len(row['input_ids']) for row in rows),
              'physical_batch_size': smoke['batch_size'], 'accumulation_steps': smoke['accumulation_steps'],
              'effective_batch_size': effective, **measured, 'adapter_sha256': sha256(adapter_path / 'adapter_model.safetensors'),
              'mps_memory_limits': {key: os.environ.get(key) for key in (
                  'PYTORCH_MPS_HIGH_WATERMARK_RATIO', 'PYTORCH_MPS_LOW_WATERMARK_RATIO')} if device.type == 'mps' else {},
              'reload_max_abs_logit_difference': float((before - after).abs().max()), 'reload_generation_ids_equal': True,
              'post_warmup_step_seconds': step_seconds, 'estimated_epoch_seconds': step_seconds * len(dataset) / effective,
              'seconds_full_smoke': time.perf_counter() - start}
    (output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f"Rechargement vérifié. Estimation par époque : {result['estimated_epoch_seconds'] / 60:.2f} min.", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--batch-size', type=int, choices=[2, 4], help='Keep the same effective batch with more accumulation')
    args = parser.parse_args()
    run(args.dataset, args.output, args.device, args.batch_size)
