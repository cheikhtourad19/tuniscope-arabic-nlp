"""Offline Qwen loading and invented-input generation only; no corpus or fitting."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import struct
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from fetch_qwen3 import ROOT, PLAN, MODEL_DIR, MODEL_ID, REVISION, sha256, verify_model


def resolve_device(request):
    if request == 'auto':
        request = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    if request == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS requested but unavailable')
    return torch.device(request)


def check(output, device_request='auto'):
    output = output.resolve()
    if (ROOT / 'outputs').resolve() not in output.parents or output.exists():
        raise ValueError('Use a new private output file')
    source = verify_model(); device = resolve_device(device_request); torch.set_num_threads(2)
    tick = time.perf_counter()
    with (MODEL_DIR / 'model.safetensors').open('rb') as stream:
        size = struct.unpack('<Q', stream.read(8))[0]
        if not 0 < size < 16_000_000:
            raise ValueError('Unexpected safetensors header size')
        header = json.loads(stream.read(size))
    stored = sum(math.prod(info['shape']) for name, info in header.items() if name != '__metadata__')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, trust_remote_code=False)
    model, loading = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        dtype=torch.float32, attn_implementation='eager', output_loading_info=True)
    if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
        raise ValueError(f'Incomplete Qwen loading: {loading}')
    model.requires_grad_(False); model.eval(); model.to(device)
    count = sum(p.numel() for p in model.parameters())
    if not 0 < count < 1_000_000_000:
        raise ValueError('Parameter ceiling violated')
    tied = model.get_input_embeddings().weight is model.get_output_embeddings().weight
    if model.config.tie_word_embeddings and not tied:
        raise ValueError('Expected tied input/output embeddings')
    gen = GenerationConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    gen.do_sample = False; gen.temperature = None; gen.top_p = None; gen.top_k = None
    gen.max_new_tokens = 16
    # Explicit kwargs prevent Transformers from replacing False/default-valued fields
    # with the pretrained model's sampling defaults during configuration merging.
    decoding = {'do_sample': False, 'max_new_tokens': 16, 'temperature': None,
                'top_p': None, 'top_k': None}
    probes = []
    with torch.inference_mode():
        for prompt in ('Réponds uniquement par le mot OK.', 'أجب فقط بكلمة مرحبا.'):
            rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': prompt}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False)
            if not rendered.endswith('<think>\n\n</think>\n\n'):
                raise ValueError('Official non-thinking template did not produce expected prefix')
            inputs = tokenizer([rendered], return_tensors='pt').to(device)
            if inputs['input_ids'].shape[1] >= 512:
                raise ValueError('Unexpectedly long invented probe')
            logits = model(**inputs).logits[:, -1, :]
            if not torch.isfinite(logits).all().item():
                raise ValueError('Non-finite forward output')
            generated = model.generate(**inputs, generation_config=gen, use_model_defaults=False, **decoding)
            response = generated[0, inputs['input_ids'].shape[1]:]
            if not 0 < len(response) <= 16:
                raise ValueError('Generation budget violated')
            probes.append({'invented_prompt': prompt, 'input_tokens': inputs['input_ids'].shape[1],
                           'generated_token_ids': response.cpu().tolist(),
                           'generated_text': tokenizer.decode(response, skip_special_tokens=True),
                           'finite_last_token_logits': True, 'non_thinking_template_verified': True})
    names = [name for name, _ in model.named_modules()]
    target_counts = {suffix: sum(n.endswith('.' + suffix) for n in names) for suffix in ('q_proj', 'v_proj')}
    memory = {}
    if device.type == 'mps':
        torch.mps.synchronize()
        memory = {'allocated_bytes_at_end': torch.mps.current_allocated_memory(),
                  'driver_bytes_at_end': torch.mps.driver_allocated_memory(), 'absolute_peak_measured': False}
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)
        memory = {'allocated_bytes_at_end': torch.cuda.memory_allocated(device),
                  'peak_allocated_bytes_this_process': torch.cuda.max_memory_allocated(device)}
    result = {'model_id': MODEL_ID, 'revision': REVISION, 'device': str(device),
              'python': platform.python_version(), 'platform': platform.platform(),
              'packages': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'tokenizers', 'numpy', 'safetensors')},
              'dtype_in_memory': str(next(model.parameters()).dtype),
              'stored_tensor_dtypes': sorted({v['dtype'] for k, v in header.items() if k != '__metadata__'}),
              'stored_tensor_elements': stored, 'unique_loaded_parameters': count,
              'input_output_embeddings_tied': tied, 'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
              'model_type': model.config.model_type, 'model_class': type(model).__name__,
              'loaded_in_4bit': bool(getattr(model, 'is_loaded_in_4bit', False)),
              'loaded_in_8bit': bool(getattr(model, 'is_loaded_in_8bit', False)),
              'quantization_config_present': getattr(model.config, 'quantization_config', None) is not None,
              'enable_thinking': False, 'max_new_tokens': 16, 'do_sample': False,
              'explicit_generation_kwargs': decoding, 'use_model_defaults': False,
              'decoding_scope': 'technical-probe-not-final-task-protocol', 'probes': probes,
              'possible_lora_target_counts_inspected_only': target_counts,
              'lora_created': False, 'training_started': False, 'corpus_read': False, 'test_loaded': False,
              'seconds_excluding_asset_checksum_verification': time.perf_counter() - tick,
              'memory': memory, 'memory_note': 'MPS end snapshots are not a peak; do not add unified CPU/GPU allocations.',
              'plan_sha256': sha256(PLAN), 'model_weight_sha256': source['files']['model.safetensors']['sha256'],
              'source_manifest_sha256': sha256(MODEL_DIR / 'qwen_source.json'),
              'environment_lock_sha256': sha256(ROOT / 'environments/llm/uv.lock'),
              'script_sha256': sha256(Path(__file__)), 'loading_info': loading}
    if result['loaded_in_4bit'] or result['loaded_in_8bit'] or result['quantization_config_present']:
        raise ValueError('Technical probe must use the standard unquantized base')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'device': str(device), 'unique_loaded_parameters': count, 'dtype': result['dtype_in_memory'],
                      'quantized': False, 'invented_probes': len(probes), 'training_started': False,
                      'seconds': result['seconds_excluding_asset_checksum_verification'], 'output': str(output)}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    check(args.output, args.device)
