"""Run a fixed three-seed shared LoRA campaign and keep its evidence."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import fcntl
import gc
import json
import math
from pathlib import Path
import platform
from queue import Empty, Queue
import random
import statistics
import subprocess
import sys
import time
from zipfile import ZIP_DEFLATED, ZipFile

import torch
from peft import LoraConfig, PeftModel, get_peft_model, __version__ as peft_version
from transformers import GenerationConfig, __version__ as transformers_version

from audit_text_composition import writing_system
from build_qwen_sft import AUDIT, CONFIG as SFT_CONFIG, IGNORE, PROTOCOL as SFT_PROTOCOL
from check_qwen_loading import resolve_device
from encoder_tokenizer_audit import object_digest
from fetch_qwen3 import MODEL_DIR, MODEL_ID, REVISION, ROOT, sha256, verify_model
from load_working_data import load_records
from qwen_prompting import PROMPT_CONFIG, load_tokenizer, parse_label
from run_qwen_lora_smoke import collate, digest_parameters, load_base, memory_snapshot, probe, sample_rows
from run_qwen_validation import generate_batch, prepare, save_json, synchronize

CONFIG = ROOT / 'configs/qwen_lora_v1.json'
PROTOCOL = ROOT / 'docs/64_protocole_campagne_lora.md'
BASE_REPORT = ROOT / 'reports/qwen_validation_v2_2026-11-07/summary.json'
PRIVATE = ROOT / 'outputs/qwen'
SCRIPT = Path(__file__).resolve()
ADAPTER_FILES = ('adapter/adapter_config.json', 'adapter/adapter_model.safetensors')
MODES = ('zero_shot', 'few_shot')


def read_json(path):
    return json.loads(path.read_text())


def load_dataset(directory, audit):
    if PRIVATE.resolve() not in directory.parents:
        raise ValueError('Use the private SFT directory')
    rows = read_json(directory / 'dataset.json')
    proof = read_json(directory / 'summary.json')
    sft = read_json(SFT_CONFIG)
    expected = {(corpus, label): count for corpus, quotas in sft['label_counts'].items()
                for label, count in quotas.items()}
    if (len(rows) != 4000 or proof['rows'] != len(rows) or proof['split'] != 'train'
            or proof['revision'] != REVISION or proof['dataset_sha256'] != object_digest(rows)
            or len({(row['corpus'], row['id']) for row in rows}) != len(rows)
            or Counter((row['corpus'], row['label']) for row in rows) != expected
            or proof['manifests_sha256'] != audit['manifests_sha256']
            or any(proof[key] != sha256(path) for key, path in (
                ('config_sha256', SFT_CONFIG), ('protocol_sha256', SFT_PROTOCOL),
                ('prompt_config_sha256', PROMPT_CONFIG), ('audit_sha256', AUDIT)))):
        raise ValueError('SFT dataset or provenance does not match the fixed protocol')
    return rows, proof, sft


def new_model(settings, seed, device):
    torch.manual_seed(seed)
    base = load_base()
    targets = Counter(name.rsplit('.', 1)[-1] for name, _ in base.named_modules()
                      if name.rsplit('.', 1)[-1] in settings['target_modules'])
    if targets != dict.fromkeys(settings['target_modules'], base.config.num_hidden_layers):
        raise ValueError('Unexpected LoRA projection inventory')
    model = get_peft_model(base, LoraConfig(
        task_type='CAUSAL_LM', r=settings['r'], lora_alpha=settings['alpha'],
        lora_dropout=settings['dropout'], target_modules=settings['target_modules'], bias='none'))
    return model.to(device)


def release(device):
    gc.collect()
    if device.type == 'mps':
        torch.mps.empty_cache()
    elif device.type == 'cuda':
        torch.cuda.empty_cache()


def fit(model, tokenizer, rows, config, seed, epochs, device):
    trainable, frozen = [], []
    for name, parameter in model.named_parameters():
        (trainable if parameter.requires_grad else frozen).append((name, parameter))
    parameters = [parameter for _, parameter in trainable]
    counts = [sum(p.numel() for _, p in group) for group in (trainable, frozen)]
    if counts != [1_146_880, 596_049_920] or any('.lora_' not in name for name, _ in trainable):
        raise ValueError('Unexpected trainable or frozen parameters')
    base_hash = digest_parameters(frozen)
    initial = digest_parameters(trainable)
    optimizer = torch.optim.AdamW(parameters, lr=config['learning_rate'], weight_decay=config['weight_decay'])
    physical = config['batch_size']
    effective = physical * config['accumulation_steps']
    randomizer = random.Random(seed)
    ordered = list(rows)
    history, memory = [], {}
    manual_difference = None
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    model.train()
    for epoch in range(1, epochs + 1):
        synchronize(device)
        tick = time.perf_counter()
        randomizer.shuffle(ordered)
        loss_sum, tokens, steps, min_gradient = 0.0, 0, 0, math.inf
        for start in range(0, len(ordered), effective):
            optimizer.zero_grad(set_to_none=True)
            window = ordered[start:start + effective]
            window_tokens = sum(len(row['input_ids']) - row['prompt_tokens'] for row in window)
            for offset in range(0, len(window), physical):
                subset = window[offset:offset + physical]
                batch = collate(subset, tokenizer, device)
                output = model(**batch, use_cache=False)
                loss = output.loss
                value = float(loss.detach())
                if not math.isfinite(value):
                    raise ValueError('Non-finite SFT loss')
                if manual_difference is None:
                    labels = batch['labels'][:, 1:]
                    mask = labels != IGNORE
                    manual = torch.nn.functional.cross_entropy(output.logits.detach()[:, :-1][mask], labels[mask])
                    manual_difference = abs(float(manual) - value)
                    if manual_difference > 1e-5:
                        raise ValueError('Loss differs from answer-only cross entropy')
                count = sum(len(row['input_ids']) - row['prompt_tokens'] for row in subset)
                (loss * (count / window_tokens)).backward()
                loss_sum += value * count
                tokens += count
                del output, loss
            if any(p.grad is not None for _, p in frozen):
                raise ValueError('A frozen parameter received a gradient')
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, config['gradient_clip'], error_if_nonfinite=True))
            if norm <= 0:
                raise ValueError('Adapter gradient is zero')
            min_gradient = min(min_gradient, norm)
            optimizer.step()
            steps += 1
            for key, value in memory_snapshot(device).items():
                memory[key] = max(memory.get(key, 0), value)
            if steps % 50 == 0:
                print(f'[{seed}] Époque {epoch}, mises à jour {steps}, perte moyenne {loss_sum / tokens:.4f}', flush=True)
        synchronize(device)
        seconds = time.perf_counter() - tick
        history.append({'epoch': epoch, 'rows': len(ordered), 'steps': steps, 'supervised_tokens': tokens,
                        'loss': loss_sum / tokens, 'min_gradient_norm': min_gradient,
                        'seconds': seconds, 'rows_per_second': len(ordered) / seconds})
        print(f'[{seed}] Époque {epoch}/{epochs} terminée, perte {loss_sum / tokens:.4f}, {seconds:.1f} s', flush=True)
    final = digest_parameters(trainable)
    if base_hash != digest_parameters(frozen) or initial == final:
        raise ValueError('Expected frozen base and changed adapter')
    return {'epochs': history, 'memory': memory, 'manual_loss_absolute_difference': manual_difference,
            'frozen_base_sha256_before_and_after': base_hash, 'adapter_parameters_before_sha256': initial,
            'adapter_parameters_after_sha256': final, 'trainable_parameters': counts[0], 'frozen_parameters': counts[1]}


def metrics(rows, labels):
    columns = [*labels, 'INVALID']
    counts = Counter((row['truth'], row['prediction'] or 'INVALID') for row in rows)
    matrix = [[counts[label, predicted] for predicted in columns] for label in labels]
    correct = sum(matrix[index][index] for index in range(len(labels)))
    invalid = sum(row[-1] for row in matrix)
    per_class = {}
    for index, label in enumerate(labels):
        tp = matrix[index][index]
        support = sum(matrix[index])
        predicted = sum(row[index] for row in matrix)
        per_class[label] = {'precision': tp / predicted if predicted else 0.0,
                            'recall': tp / support if support else 0.0,
                            'f1-score': 2 * tp / (support + predicted) if support + predicted else 0.0,
                            'support': support}
    return {'n': len(rows), 'labels': labels, 'prediction_columns': columns, 'confusion_matrix': matrix,
            'per_class': per_class, 'accuracy': correct / len(rows), 'invalid_rows': invalid,
            'valid_rate': (len(rows) - invalid) / len(rows),
            'balanced_accuracy': statistics.mean(item['recall'] for item in per_class.values() if item['support']),
            'macro_f1': statistics.mean(item['f1-score'] for item in per_class.values())}


def validation_sample(rows, mode, batch_size):
    if mode == 'full':
        return rows
    return [rows[(2 * index + 1) * len(rows) // (2 * batch_size)] for index in range(batch_size)]


def score_stage(rows, labels):
    groups = defaultdict(list)
    for row in rows:
        groups[row['writing']].append(row)
    return {**metrics(rows, labels),
            'by_writing': {name: metrics(items, labels) for name, items in sorted(groups.items())}}


def validate(model, tokenizer, device, config, prompting, audit, folder, args):
    tokenizer.padding_side = 'left'
    generation = GenerationConfig.from_pretrained(MODEL_DIR, local_files_only=True)
    if tokenizer.pad_token_id != generation.pad_token_id:
        raise ValueError('Tokenizer and generation padding disagree')
    options = {**prompting['initial_generation'], 'max_new_tokens': prompting['max_new_tokens'], 'use_cache': True}
    batch_size = config['validation_batch_size']
    results = {}
    model.eval()
    with torch.inference_mode():
        for corpus, mode, validation, pairs in prepare(tokenizer, prompting, audit):
            tick = time.perf_counter()
            name = f'{corpus}_{mode}'
            labels = prompting['tasks'][corpus]['labels']
            validation = validation_sample(validation, args.mode, batch_size)
            selected = {row['id'] for row in validation}
            pairs = [(row, ids) for row, ids in pairs if row['id'] in selected]
            generated = {}
            for start in range(0, len(pairs), batch_size):
                for row in generate_batch(model, tokenizer, generation, options, device,
                                          pairs[start:start + batch_size], labels):
                    generated[row['id']] = row
            rows = [generated[row['id']] for row in validation]
            path = folder / f'{name}.json'
            save_json(path, rows)
            results[name] = {**score_stage(rows, labels), 'seconds': time.perf_counter() - tick,
                             'predictions_sha256': sha256(path)}
            print(f'[{args.seed}] {name}, {len(rows)} prédictions ({args.mode})', flush=True)
    return results


def adapter_hashes(folder):
    return {name: sha256(folder / name) for name in ADAPTER_FILES}


def reload_adapter(folder, device):
    return PeftModel.from_pretrained(load_base(), folder / 'adapter', is_trainable=False, local_files_only=True).to(device)


def worker(args, config, device):
    folder = args.output / f'seed_{args.seed}'
    plan = read_json(args.output / 'plan.json')
    if args.seed not in plan['seeds'] or plan['mode'] != args.mode or plan['device_type'] != device.type:
        raise ValueError('Worker differs from the campaign plan')
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    torch.set_num_threads(2)
    source = verify_model()
    audit = read_json(AUDIT)
    dataset, proof, sft = load_dataset(args.dataset, audit)
    if (proof['dataset_sha256'] != plan['dataset_sha256']
            or source['files']['model.safetensors']['sha256'] != plan['model_sha256']):
        raise ValueError('Worker sources differ from the campaign plan')
    tokenizer, _ = load_tokenizer()
    settings = sft['lora']
    seed = args.seed
    trained_path = folder / 'training.json'
    if trained_path.is_file():
        training = read_json(trained_path)
        if training['adapter_files'] != adapter_hashes(folder) or training['plan_sha256'] != sha256(args.output / 'plan.json'):
            raise ValueError('Saved adapter or training plan differs')
        print(f'[{seed}] Adaptateur sauvegardé réutilisé', flush=True)
        restored = reload_adapter(folder, device)
    else:
        folder.mkdir()
        smoke = {**sft['smoke'], 'examples_per_task': config['smoke_examples_per_task'], 'seed': seed}
        subset = sample_rows(dataset, {'smoke': smoke, 'label_counts': sft['label_counts']})
        properties = torch.cuda.get_device_properties(device) if device.type == 'cuda' else None
        print(f'[{seed}] {device}, {properties.name if properties else platform.machine()}, contrôle de {len(subset)} lignes', flush=True)
        model = new_model(settings, seed, device)
        preflight = fit(model, tokenizer, subset, config, seed, 1, device)
        save_json(folder / 'smoke.json', preflight)
        if args.mode == 'full':
            del model
            release(device)
            model = new_model(settings, seed, device)
            fit_result = fit(model, tokenizer, dataset, config, seed, config['epochs'], device)
        else:
            fit_result = preflight
        before, continuations = probe(model, tokenizer, device)
        model.peft_config['default'].base_model_name_or_path = MODEL_ID
        model.peft_config['default'].revision = REVISION
        model.save_pretrained(folder / 'adapter', safe_serialization=True, save_embedding_layers=False)
        del model
        release(device)
        restored = reload_adapter(folder, device)
        after, reloaded = probe(restored, tokenizer, device)
        torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-5)
        if reloaded != continuations:
            raise ValueError('Reloaded adapter changed probe generations')
        training = {'seed': seed, 'mode': args.mode, 'plan_sha256': sha256(args.output / 'plan.json'),
                    'device': str(device), 'platform': platform.platform(), 'torch': torch.__version__,
                    'transformers': transformers_version, 'peft': peft_version,
                    'cuda_version': torch.version.cuda if properties else None,
                    'gpu_name': properties.name if properties else None,
                    'gpu_total_bytes': properties.total_memory if properties else None,
                    'reload_max_abs_logit_difference': float((before - after).abs().max()),
                    'reload_generation_ids_equal': True, 'adapter_files': adapter_hashes(folder),
                    'smoke_sha256': sha256(folder / 'smoke.json'), **fit_result}
        save_json(trained_path, training)
    results = validate(restored, tokenizer, device, config, read_json(PROMPT_CONFIG), audit, folder, args)
    save_json(folder / 'result.json', {'seed': seed, 'mode': args.mode, 'test_loaded': False,
                                       'training_sha256': sha256(trained_path), 'validation': results})


def verify_seed(folder, plan_path, expected):
    plan, training, result = map(read_json, (plan_path, folder / 'training.json', folder / 'result.json'))
    seed = int(folder.name.removeprefix('seed_'))
    stages = {f'{corpus}_{mode}' for corpus in expected for mode in MODES}
    if (seed not in plan['seeds'] or training['seed'] != seed or result['seed'] != seed
            or training['mode'] != plan['mode'] or result['mode'] != plan['mode'] or result['test_loaded'] is not False
            or training['plan_sha256'] != sha256(plan_path) or result['training_sha256'] != sha256(folder / 'training.json')
            or training['adapter_files'] != adapter_hashes(folder)
            or training['smoke_sha256'] != sha256(folder / 'smoke.json')
            or set(result['validation']) != stages or training['reload_generation_ids_equal'] is not True
            or training['manual_loss_absolute_difference'] > 1e-5
            or training['adapter_parameters_before_sha256'] == training['adapter_parameters_after_sha256']):
        raise ValueError(f'Incomplete or altered run for seed {seed}')
    preflight = read_json(folder / 'smoke.json')
    config = plan['config']
    effective = config['batch_size'] * config['accumulation_steps']
    smoke_count = 2 * config['smoke_examples_per_task']
    full = plan['mode'] == 'full'
    for entry, count, epochs in ((preflight, smoke_count, 1),
                                 (training, 4000 if full else smoke_count, config['epochs'] if full else 1)):
        if (len(entry['epochs']) != epochs or entry['trainable_parameters'] != 1_146_880
                or entry['frozen_parameters'] != 596_049_920
                or any(row['epoch'] != index or row['rows'] != count or row['steps'] != math.ceil(count / effective)
                       or not math.isfinite(row['loss']) or row['min_gradient_norm'] <= 0
                       for index, row in enumerate(entry['epochs'], start=1))):
            raise ValueError(f'Training evidence differs for seed {seed}')
    audit = read_json(AUDIT)
    prompting = read_json(PROMPT_CONFIG)
    for name, stage in result['validation'].items():
        corpus, _ = name.split('_', 1)
        labels = prompting['tasks'][corpus]['labels']
        path = folder / f'{name}.json'
        rows = read_json(path)
        if (stage['predictions_sha256'] != sha256(path)
                or [(row['id'], row['group_id'], row['truth'], row['writing']) for row in rows] != expected[corpus]
                or any(parse_label(row['response'], labels) != row['prediction']
                       or not 0 < row['prompt_tokens'] <= audit['length_decision']['max_prompt_tokens']
                       or not 0 < row['generated_tokens'] <= prompting['max_new_tokens']
                       or (not row['ended_with_eos'] and row['generated_tokens'] != prompting['max_new_tokens'])
                       for row in rows)
                or any(stage[key] != value for key, value in score_stage(rows, labels).items())):
            raise ValueError(f'Validation evidence differs for seed {seed}, {name}')
    return result


def export_seed(folder, plan_path, target):
    if target is None:
        return
    archive = target / f'tuniscope_lora_{plan_path.parent.name}_{folder.name}.zip'
    temp = archive.with_suffix('.part')
    with ZipFile(temp, 'w', compression=ZIP_DEFLATED) as bundle:
        bundle.write(plan_path, 'plan.json')
        for name in ('training.json', 'smoke.json', 'result.json', *ADAPTER_FILES):
            bundle.write(folder / name, name)
    with ZipFile(temp) as bundle:
        if bundle.testzip() is not None:
            raise ValueError('Corrupted seed archive')
    temp.replace(archive)
    print(f'Archive à télécharger {archive}', flush=True)


def campaign(args, config, device):
    start = time.perf_counter()
    audit = read_json(AUDIT)
    _, proof, _ = load_dataset(args.dataset, audit)
    if ({corpus: sha256(ROOT / f'data/manifests/v2/{corpus}.csv') for corpus in proof['corpora']}
            != audit['manifests_sha256']):
        raise ValueError('Current manifests differ from the audit')
    expected = {}
    for corpus in proof['corpora']:
        rows = load_records(corpus, splits=('validation',), manifest_version='v2')
        if object_digest([row['id'] for row in rows]) != audit['corpora'][corpus]['partitions']['validation']['ids_sha256']:
            raise ValueError('Current validation IDs differ from the audit')
        expected[corpus] = [(row['id'], row['group_id'], row['label'], writing_system(row['text_raw']))
                            for row in validation_sample(rows, args.mode, config['validation_batch_size'])]
    source = verify_model()
    if args.mode == 'full' and device.type == 'cpu':
        raise ValueError('Full mode needs MPS or CUDA')
    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
    # Indexes refer to CUDA devices visible to this process, including any external mask.
    devices = [f'cuda:{slot}' for slot in range(min(2, torch.cuda.device_count()))] if device.type == 'cuda' else [str(device)]
    seeds = config['seeds'] if args.mode == 'full' else [42]
    if args.mode == 'smoke':
        devices = devices[:1]
    baseline = read_json(BASE_REPORT)
    reference = baseline['plan']
    if (reference['revision'] != REVISION or reference['model_sha256'] != source['files']['model.safetensors']['sha256']
            or reference['prompt_config_sha256'] != sha256(PROMPT_CONFIG)
            or reference['manifests_sha256'] != proof['manifests_sha256']):
        raise ValueError('Base reference does not match the campaign')
    plan_path = args.output / 'plan.json'
    plan = {'config': config, 'config_sha256': sha256(CONFIG), 'protocol_sha256': sha256(PROTOCOL),
            'code_sha256': {path.name: sha256(path) for path in (
                SCRIPT, SCRIPT.with_name('run_qwen_lora_smoke.py'), SCRIPT.with_name('run_qwen_validation.py'),
                SCRIPT.with_name('qwen_prompting.py'), SCRIPT.with_name('build_qwen_sft.py'))},
            'lock_sha256': sha256(ROOT / 'environments/lora/uv.lock'),
            'commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'dataset_sha256': proof['dataset_sha256'], 'sft_summary_sha256': sha256(args.dataset / 'summary.json'),
            'model_id': MODEL_ID, 'revision': REVISION, 'model_sha256': source['files']['model.safetensors']['sha256'],
            'manifests_sha256': proof['manifests_sha256'], 'audit_sha256': sha256(AUDIT),
            'base_report_sha256': sha256(BASE_REPORT),
            'prompt_config_sha256': sha256(PROMPT_CONFIG), 'mode': args.mode,
            'seeds': seeds, 'device_type': device.type, 'worker_devices': devices, 'test_loaded': False}
    if args.output.exists():
        if not args.resume or read_json(plan_path) != plan:
            raise ValueError('Use a new output directory or resume the identical campaign')
    else:
        if args.resume:
            raise ValueError('No existing campaign to resume')
        args.output.mkdir(parents=True)
        save_json(plan_path, plan)

    def run_seed(seed, worker_device):
        folder = args.output / f'seed_{seed}'
        if not (folder / 'result.json').exists():
            if folder.exists() and not (folder / 'training.json').exists():
                # Preserve interrupted work separately before restarting the same seed.
                folder.rename(folder.with_name(f'{folder.name}_interrupted_{time.time_ns()}'))
            subprocess.run([sys.executable, '-u', str(SCRIPT), '--mode', args.mode,
                            '--device', worker_device, '--dataset', str(args.dataset), '--output', str(args.output),
                            '--seed', str(seed)], cwd=ROOT, check=True)
        result = verify_seed(folder, plan_path, expected)
        export_seed(folder, plan_path, args.export_dir)
        return result

    print(f'Graines {seeds}, appareils {devices}', flush=True)
    # A worker always owns its slot until it finishes. Keep other seeds running if one fails.
    jobs = Queue()
    for seed in seeds:
        jobs.put(seed)

    def use_slot(worker_device):
        results, failures = {}, []
        while True:
            try:
                seed = jobs.get_nowait()
            except Empty:
                return results, failures
            try:
                results[seed] = run_seed(seed, worker_device)
            except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
                failures.append((seed, str(error)))
                print(f'[{seed}] Échec : {error}', flush=True)

    results, failures = {}, []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        for partial, errors in pool.map(use_slot, devices):
            results.update(partial)
            failures.extend(errors)
    if failures:
        raise RuntimeError(f'Campagne incomplète : {failures}')
    base = baseline['stages'] if args.mode == 'full' else {}
    aggregates = {}
    for stage in base:
        scores = [results[seed]['validation'][stage]['macro_f1'] for seed in seeds]
        reference = base[stage]['macro_f1']
        aggregates[stage] = {'macro_f1_by_seed': dict(zip(map(str, seeds), scores, strict=True)),
                             'mean_macro_f1': statistics.mean(scores), 'stdev_macro_f1': statistics.stdev(scores),
                             'base_macro_f1': reference, 'delta_vs_base': statistics.mean(scores) - reference}
    summary = {'plan_sha256': sha256(plan_path), 'mode': args.mode, 'seeds': seeds, 'test_loaded': False,
               'validation': aggregates, 'base_report_sha256': sha256(BASE_REPORT) if base else None}
    path = args.output / 'campaign.json'
    if path.exists():
        previous = read_json(path)
        if any(previous[key] != value for key, value in summary.items()):
            raise ValueError('Saved campaign summary differs from verified seed results')
        summary = previous
    else:
        summary['seconds_this_execution'] = time.perf_counter() - start
        save_json(path, summary)
    if args.export_dir:
        save_json(args.export_dir / f'tuniscope_lora_{args.output.name}_campaign.json', summary)
    print(f'Synthèse {path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['smoke', 'full'], required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--export-dir', type=Path)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seed', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.dataset, args.output = args.dataset.resolve(), args.output.resolve()
    if PRIVATE.resolve() not in args.output.parents:
        parser.error('Use a private run directory under outputs/qwen/')
    config = read_json(CONFIG)
    if args.seed is not None and args.device in {'cuda:0', 'cuda:1'}:
        device = torch.device(args.device)
    else:
        if args.device not in {'auto', 'cpu', 'mps', 'cuda'}:
            parser.error('Choose auto, cpu, mps or cuda')
        device = resolve_device(args.device)
    if args.seed is None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.with_name(f'.{args.output.name}.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            campaign(args, config, device)
    else:
        worker(args, config, device)


if __name__ == '__main__':
    main()
