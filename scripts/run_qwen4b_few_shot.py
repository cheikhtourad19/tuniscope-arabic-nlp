"""Convert the pinned Qwen3-4B locally and evaluate it in few-shot on validation v2."""

import argparse
from collections import defaultdict
from importlib.metadata import version
from itertools import takewhile
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

from audit_text_composition import writing_system
from encoder_tokenizer_audit import object_digest
from load_working_data import load_records
from prepare_qwen4b import ROOT, check_files, digest, load_plan
from qwen_prompting import parse_label, render_prompts, select_demonstrations


AUDIT = ROOT / 'reports/qwen_tokenizer_audit_v2_2026-11-07/summary.json'
LOCK = ROOT / 'environments/qwen4b/uv.lock'
QUANTIZATION = {'bits': 4, 'group_size': 64, 'mode': 'affine'}
BATCH = {'prefill_batch_size': 8, 'completion_batch_size': 16}
CHECKPOINT_ROWS = 64


def save_json(path, value):
    temporary = path.with_suffix('.part')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def model_paths(plan):
    model = plan['model']
    source = ROOT / '.tools/llms' / f"qwen3_4b_{model['revision']}"
    return source, source.with_name(f'{source.name}_mlx_4bit')


def hashes(directory):
    return {file.name: digest(file) for file in sorted(directory.iterdir())
            if file.is_file() and file.name != 'conversion.json'}


def convert_model(plan):
    from mlx_lm import convert

    source, target = model_paths(plan)
    receipt = source / 'qwen_source.json'
    if check_files(source, plan['model']) != json.loads(receipt.read_text())['files']:
        raise ValueError('Le cache amont diffère du reçu SHA-256.')
    with TemporaryDirectory(dir=target.parent) as staging:
        output = Path(staging) / 'converted'
        convert(str(source), mlx_path=output, quantize=True, q_bits=QUANTIZATION['bits'],
                q_group_size=QUANTIZATION['group_size'], q_mode=QUANTIZATION['mode'])
        model_config = json.loads((output / 'config.json').read_text())
        if model_config['quantization'] != QUANTIZATION:
            raise ValueError('Quantification inattendue.')
        save_json(output / 'conversion.json', {
            'upstream_receipt_sha256': digest(receipt), 'environment_lock_sha256': digest(LOCK),
            'mlx_lm': version('mlx-lm'), 'quantization': QUANTIZATION, 'files': hashes(output),
        })
        output.rename(target)
    print(f'Conversion MLX vérifiée et enregistrée dans {target}', flush=True)


def inputs(plan):
    comparison = plan['comparison']
    config = json.loads((ROOT / comparison['prompt_config']).read_text())
    audit = json.loads(AUDIT.read_text())
    if config['enable_thinking'] or config['few_shot_seed'] != comparison['seed']:
        raise ValueError('Les consignes few-shot diffèrent du plan.')
    for corpus in comparison['corpora']:
        task = config['tasks'][corpus]
        manifest = ROOT / f"data/manifests/{comparison['manifest_version']}/{corpus}.csv"
        if digest(manifest) != audit['manifests_sha256'][corpus]:
            raise ValueError(f'Manifeste différent de la référence pour {corpus}.')
        rows = load_records(corpus, splits=('train', 'validation'), manifest_version=comparison['manifest_version'])
        train = [row for row in rows if row['split'] == 'train']
        validation = [row for row in rows if row['split'] == 'validation']
        demos = select_demonstrations(train, task['labels'], config['few_shot_seed'])
        audited = audit['corpora'][corpus]
        if ({row['group_id'] for row in train} & {row['group_id'] for row in validation}
                or object_digest([row['id'] for row in demos]) != audited['demonstrations']['ids_sha256']
                or object_digest([row['id'] for row in validation]) != audited['partitions']['validation']['ids_sha256']):
            raise ValueError(f'Provenance few-shot ou validation inattendue pour {corpus}.')
        yield corpus, task, validation, demos, config['max_new_tokens'], audit['length_decision']['max_prompt_tokens']


def prepare_cache(model, prompts):
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    same_tokens = takewhile(lambda tokens: len(set(tokens)) == 1, zip(*prompts))
    prefix = [tokens[0] for tokens in same_tokens][:min(map(len, prompts)) - 1]
    cache = make_prompt_cache(model)
    if prefix:
        model(mx.array([prefix]), cache=cache)
        mx.eval([layer.state for layer in cache])
    return len(prefix), cache


def validate_rows(predictions, validation, labels, max_new_tokens, max_prompt_tokens):
    for prediction, row in zip(predictions, validation, strict=True):
        if ((prediction['id'], prediction['truth'], prediction['writing'])
                != (row['id'], row['label'], writing_system(row['text_raw']))
                or prediction['prediction'] != parse_label(prediction['response'], labels)
                or not 0 < prediction['prompt_tokens'] <= max_prompt_tokens
                or not 0 <= prediction['output_tokens'] <= max_new_tokens):
            raise ValueError('Prédictions incompatibles avec la validation ou le protocole.')


def evaluate(plan, output, *, adapter=None, validation_limit=None):
    import mlx.core as mx
    from mlx_lm import batch_generate, load
    from mlx_lm.sample_utils import make_sampler

    if not mx.metal.is_available():
        raise RuntimeError('GPU Metal indisponible.')
    mx.set_default_device(mx.gpu)
    source, target = model_paths(plan)
    receipt_file = target / 'conversion.json'
    receipt = json.loads(receipt_file.read_text())
    if (receipt['files'] != hashes(target) or receipt['environment_lock_sha256'] != digest(LOCK)
            or receipt['upstream_receipt_sha256'] != digest(source / 'qwen_source.json')):
        raise ValueError('La base MLX diffère du reçu de conversion.')
    model, tokenizer = load(str(target), adapter_path=str(adapter) if adapter else None)
    expected_stops = json.loads((source / 'generation_config.json').read_text())['eos_token_id']
    if set(tokenizer.eos_token_ids) != set(expected_stops):
        raise ValueError('Les marqueurs de fin MLX diffèrent de ceux de Qwen3-4B.')
    run = {'model_id': plan['model']['id'], 'revision': plan['model']['revision'], 'quantization': QUANTIZATION,
           'conversion_sha256': digest(receipt_file), 'environment_lock_sha256': digest(LOCK),
           'code_sha256': {path: digest(ROOT / path) for path in (
               'scripts/run_qwen4b_few_shot.py', 'scripts/qwen_prompting.py', 'scripts/load_working_data.py')},
           'prompt_config_sha256': digest(ROOT / plan['comparison']['prompt_config']), 'audit_sha256': digest(AUDIT),
           'platform': 'macOS / Metal', 'mlx_lm': version('mlx-lm'), 'mlx': version('mlx'),
           'mode': 'few_shot', 'split': 'validation', 'demonstration_split': 'train', 'training': False,
           'thinking': False, 'sampling': 'greedy', 'batch': BATCH, 'checkpoint_rows': CHECKPOINT_ROWS}
    if adapter:
        run['adapter_sha256'] = {
            name: digest(adapter / name) for name in ('adapters.safetensors', 'adapter_config.json')}
    if validation_limit is not None:
        run['validation_limit'] = validation_limit
    output.mkdir(parents=True, exist_ok=True)
    run_file = output / 'run.json'
    if run_file.exists() and json.loads(run_file.read_text()) != run:
        raise ValueError('Impossible de reprendre avec des paramètres différents.')
    save_json(run_file, run)
    sampler = make_sampler(temp=0.0)
    for corpus, task, validation, demos, max_new, max_prompt in inputs(plan):
        if validation_limit is not None:
            validation = validation[:validation_limit]
        stage_file = output / f'{corpus}.json'
        stage = json.loads(stage_file.read_text()) if stage_file.exists() else {
            'run_sha256': digest(run_file), 'predictions': [], 'seconds': 0, 'peak_mlx_bytes': 0,
        }
        predictions = stage['predictions']
        if stage['run_sha256'] != digest(run_file):
            raise ValueError(f'Version de prédictions inattendue pour {corpus}.')
        validate_rows(predictions, validation[:len(predictions)], task['labels'], max_new, max_prompt)
        rendered = render_prompts(tokenizer, task, [row['text_raw'] for row in validation], demos)
        prompts = [tokenizer.encode(prompt, add_special_tokens=False) for prompt in rendered]
        if not prompts or max(map(len, prompts)) > max_prompt:
            raise ValueError(f'Budget de contexte dépassé pour {corpus}.')
        prefix_size, cache = prepare_cache(model, prompts)
        stage['shared_prompt_tokens'] = prefix_size
        print(f'{corpus.upper()} | few-shot | {len(demos)} exemples du train | '
              f'{len(predictions)}/{len(validation)} | préfixe réutilisé de {prefix_size} tokens', flush=True)
        for start in range(len(predictions), len(validation), CHECKPOINT_ROWS):
            tick = perf_counter()
            block = prompts[start:start + CHECKPOINT_ROWS]
            batch = batch_generate(model, tokenizer, [prompt[prefix_size:] for prompt in block],
                                   prompt_caches=[cache] * len(block), max_tokens=max_new,
                                   return_token_ids=True, sampler=sampler, **BATCH)
            for row, prompt, token_ids in zip(validation[start:start + len(block)], block, batch.token_ids, strict=True):
                response = tokenizer.decode(token_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
                predictions.append({'id': row['id'], 'truth': row['label'], 'writing': writing_system(row['text_raw']),
                                    'prompt_tokens': len(prompt), 'output_tokens': len(token_ids),
                                    'response': response, 'prediction': parse_label(response, task['labels'])})
            stage['seconds'] += perf_counter() - tick
            stage['peak_mlx_bytes'] = max(stage['peak_mlx_bytes'], mx.get_peak_memory())
            save_json(stage_file, stage)
            if len(predictions) % 256 == 0 or len(predictions) in (CHECKPOINT_ROWS, len(validation)):
                print(f'{corpus.upper()} | {len(predictions)}/{len(validation)} | {stage["seconds"]:.1f} s', flush=True)


def report(plan, output, target):
    from report_qwen_validation import score

    run_file = output / 'run.json'
    run = json.loads(run_file.read_text())
    if run['mode'] != 'few_shot' or run['split'] != 'validation' or run['training']:
        raise ValueError('Mauvais périmètre de génération.')
    results = {}
    for corpus, task, validation, demos, max_new, max_prompt in inputs(plan):
        stage_file = output / f'{corpus}.json'
        stage = json.loads(stage_file.read_text())
        if stage['run_sha256'] != digest(run_file):
            raise ValueError('Le fichier de prédictions diffère du plan de génération.')
        predictions = stage['predictions']
        validate_rows(predictions, validation, task['labels'], max_new, max_prompt)
        by_writing = defaultdict(list)
        for row in predictions:
            by_writing[row['writing']].append(row)
        results[corpus] = {**score(predictions, task['labels']), 'seconds': stage['seconds'],
                           'peak_mlx_bytes': stage['peak_mlx_bytes'], 'shared_prompt_tokens': stage['shared_prompt_tokens'],
                           'demonstration_count': len(demos), 'max_prompt_tokens': max(row['prompt_tokens'] for row in predictions),
                           'predictions_sha256': digest(stage_file), 'demonstrations_sha256': object_digest([row['id'] for row in demos]),
                           'by_writing': {key: score(rows, task['labels']) for key, rows in sorted(by_writing.items())}}
    target.parent.mkdir(parents=True, exist_ok=True)
    save_json(target, {'run': run, 'data_manifests_sha256': json.loads(AUDIT.read_text())['manifests_sha256'],
                       'metrics_code_sha256': digest(ROOT / 'scripts/baseline_metrics.py'),
                       'metrics_lock_sha256': digest(ROOT / 'uv.lock'), 'results': results})
    for corpus, metrics in results.items():
        print(f'{corpus.upper()} | {metrics["n"]} lignes | exactitude {metrics["accuracy"]:.2%} | '
              f'macro-F1 {metrics["macro_f1"]:.2%} | hors format {metrics["invalid_rows"]} | {metrics["seconds"]:.1f} s')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('convert', 'evaluate', 'report'))
    parser.add_argument('--run', type=Path, default=ROOT / 'outputs/qwen/qwen3_4b_few_shot_v1')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/qwen3_4b_few_shot_v1/summary.json')
    args = parser.parse_args()
    if ((ROOT / 'outputs/qwen').resolve() not in args.run.resolve().parents
            or (ROOT / 'reports').resolve() not in args.report.resolve().parents):
        parser.error('Conserver les prédictions dans outputs/qwen et les agrégats dans reports.')
    plan = load_plan()
    if args.action == 'convert':
        convert_model(plan)
    elif args.action == 'evaluate':
        evaluate(plan, args.run.resolve())
    else:
        report(plan, args.run.resolve(), args.report.resolve())
