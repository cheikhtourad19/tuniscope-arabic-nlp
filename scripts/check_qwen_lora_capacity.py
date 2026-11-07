"""Run one forward/backward pass on the longest SFT rows with a saved adapter."""
import argparse
import json
from pathlib import Path
import time

import torch
from peft import PeftModel

from benchmark_qwen_validation import synchronize
from check_qwen_loading import resolve_device
from encoder_tokenizer_audit import object_digest
from fetch_qwen3 import MODEL_ID, REVISION, ROOT, sha256, verify_model
from qwen_prompting import load_tokenizer
from run_qwen_lora_smoke import collate, load_base, memory_snapshot


def check(dataset, smoke, output, device_request):
    private = (ROOT / 'outputs/qwen').resolve()
    dataset, smoke, output = (path.resolve() for path in (dataset, smoke, output))
    if any(private not in path.parents for path in (dataset, smoke, output)) or output.exists():
        raise ValueError('Choose private inputs and a new private output file')
    proof = json.loads((smoke / 'summary.json').read_text())
    examples = json.loads((dataset / 'dataset.json').read_text())
    adapter = smoke / 'adapter'
    settings = json.loads((adapter / 'adapter_config.json').read_text())
    adapter_hash = sha256(adapter / 'adapter_model.safetensors')
    source = verify_model()
    device = resolve_device(device_request)
    if (proof['scope'] != 'smoke' or proof['revision'] != REVISION or proof['device'] != str(device)
            or proof['dtype'] != 'float32' or proof['dataset_sha256'] != object_digest(examples)
            or proof['dataset_summary_sha256'] != sha256(dataset / 'summary.json')
            or proof['model_sha256'] != source['files']['model.safetensors']['sha256']
            or proof['adapter_sha256'] != adapter_hash or settings['base_model_name_or_path'] != MODEL_ID
            or settings['revision'] != REVISION):
        raise ValueError('Adapter, base and dataset must match the completed smoke run')
    size = proof['physical_batch_size']
    batch = sorted(examples, key=lambda row: len(row['input_ids']), reverse=True)[:size]
    torch.set_num_threads(2)
    tokenizer, _ = load_tokenizer()
    report = {'device': str(device), 'torch': torch.__version__, 'cuda_runtime': torch.version.cuda,
              'revision': REVISION, 'batch_size': size, 'dataset_sha256': proof['dataset_sha256'],
              'adapter_sha256': adapter_hash, 'sequence_lengths': [len(row['input_ids']) for row in batch]}
    if device.type == 'cuda':
        gpu = torch.cuda.get_device_properties(device)
        report.update(gpu_name=gpu.name, gpu_memory_bytes=gpu.total_memory,
                      visible_cuda_devices=torch.cuda.device_count(), cuda_device_index=torch.cuda.current_device())
        torch.cuda.reset_peak_memory_stats(device)
    try:
        model = PeftModel.from_pretrained(load_base(), adapter, is_trainable=True, local_files_only=True).to(device).train()
        inputs = collate(batch, tokenizer, device)
        synchronize(device)
        tick = time.perf_counter()
        loss = model(**inputs, use_cache=False).loss
        if not torch.isfinite(loss).item():
            raise ValueError('Non-finite longest-batch loss')
        report['after_forward'] = memory_snapshot(device)
        loss.backward()
        synchronize(device)
        report['after_backward'] = memory_snapshot(device)
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError('A frozen parameter received a gradient')
        norm = float(torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True))
        if norm <= 0:
            raise ValueError('Adapter gradient is zero')
        report.update(forward_backward_passed=True, loss=float(loss.detach()), gradient_norm=norm,
                      seconds=time.perf_counter() - tick)
    except RuntimeError as error:
        if 'out of memory' not in str(error).lower():
            raise
        report.update(forward_backward_passed=False, reason=str(error))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--smoke', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    args = parser.parse_args()
    check(args.dataset, args.smoke, args.output, args.device)
