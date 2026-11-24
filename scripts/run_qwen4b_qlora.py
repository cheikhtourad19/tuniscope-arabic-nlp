"""Train one shared Qwen3-4B adapter on Mac and compare it to the few-shot baseline."""

import argparse
from collections import Counter
import fcntl
from functools import partial
import json
from math import isfinite
from pathlib import Path
import random
from time import perf_counter

from build_qwen_sft import select_rows, tokenize_sft
from encoder_tokenizer_audit import object_digest
from load_working_data import load_records
from prepare_qwen4b import ROOT, digest, load_plan
from run_qwen4b_few_shot import AUDIT, LOCK, evaluate, hashes, model_paths, report, save_json


BASELINE = ROOT / 'reports/qwen3_4b_few_shot_v1/summary.json'
ADAPTER_FILES = ('adapter_config.json', 'adapters.safetensors')


def prepare_examples(plan):
    from transformers import AutoTokenizer

    comparison = plan['comparison']
    settings = json.loads((ROOT / plan['training']['sft_selection']).read_text())
    prompt_config = ROOT / comparison['prompt_config']
    prompts = json.loads(prompt_config.read_text())
    audit = json.loads(AUDIT.read_text())
    if digest(prompt_config) != audit['config_sha256'] or settings['selection_seed'] != comparison['seed']:
        raise ValueError('La sélection SFT ou les consignes diffèrent du protocole.')
    _, base = model_paths(plan)
    tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
    examples = []
    for corpus in comparison['corpora']:
        version = comparison['manifest_version']
        if digest(ROOT / f'data/manifests/{version}/{corpus}.csv') != audit['manifests_sha256'][corpus]:
            raise ValueError(f'Manifeste inattendu pour {corpus}.')
        train = load_records(corpus, splits=('train',), manifest_version=version)
        quotas = settings['label_counts'][corpus]
        selection = select_rows(corpus, train, quotas, settings['selection_seed'])
        encoded = list(tokenize_sft(tokenizer, prompts['tasks'][corpus], selection,
                                    settings['max_sequence_tokens'], audit['length_decision']['max_prompt_tokens']))
        if Counter(row['label'] for row in encoded) != quotas or len({row['id'] for row in encoded}) != len(encoded):
            raise ValueError(f'Sélection SFT inattendue pour {corpus}.')
        examples.extend({**row, 'corpus': corpus} for row in encoded)
    return examples, settings['max_sequence_tokens']


def checkpoint(path, model, optimizer, step, run_hash, seconds, peak):
    import mlx.core as mx
    from mlx.utils import tree_flatten

    tensors = {f'adapter.{key}': value for key, value in tree_flatten(model.trainable_parameters())}
    tensors.update({f'optimizer.{key}': value for key, value in tree_flatten(optimizer.state)})
    temp = path.with_suffix('.tmp.safetensors')
    mx.save_safetensors(str(temp), tensors, metadata={
        'run_sha256': run_hash, 'step': str(step), 'seconds': str(seconds), 'peak_mlx_bytes': str(peak)})
    temp.replace(path)


def restore(path, model, optimizer, run_hash):
    import mlx.core as mx
    from mlx.utils import tree_unflatten

    tensors, metadata = mx.load(str(path), return_metadata=True)
    if metadata['run_sha256'] != run_hash:
        raise ValueError('Ce point de reprise appartient à une autre expérience.')
    model.load_weights([(key.removeprefix('adapter.'), value) for key, value in tensors.items()
                        if key.startswith('adapter.')], strict=False)
    optimizer.state = tree_unflatten([(key.removeprefix('optimizer.'), value) for key, value in tensors.items()
                                      if key.startswith('optimizer.')])
    return int(metadata['step']), float(metadata['seconds']), int(metadata['peak_mlx_bytes'])


def fmt_time(seconds):
    minutes, seconds = divmod(round(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def verify_adapter(output, plan):
    _, base = model_paths(plan)
    completion = json.loads((output / 'completion.json').read_text())
    run_file = output / 'training.json'
    run = json.loads(run_file.read_text())
    if (completion['run_sha256'] != digest(run_file) or completion['steps'] != run['steps']
            or run['conversion_sha256'] != digest(base / 'conversion.json')
            or completion['adapter_sha256'] != {name: digest(output / name) for name in ADAPTER_FILES}):
        raise ValueError('L’adaptateur sauvegardé ou sa provenance diffère du reçu d’entraînement.')
    return run, completion


def train(plan, output, smoke):
    import mlx.core as mx
    import mlx.nn as nn
    from mlx.optimizers import AdamW, clip_grad_norm
    from mlx.utils import tree_flatten, tree_map
    from mlx_lm import load
    from mlx_lm.tuner.trainer import default_loss, grad_checkpoint
    from mlx_lm.tuner.utils import linear_to_lora_layers

    if (output / 'completion.json').exists():
        run, _ = verify_adapter(output, plan)
        if run['smoke'] != smoke:
            raise ValueError('Choisir un dossier distinct pour le contrôle court et pour le run complet.')
        print(f'Adaptateur déjà terminé et vérifié dans {output}', flush=True)
        return
    if not mx.metal.is_available():
        raise RuntimeError('Lancer ce notebook sur le Mac Apple Silicon avec accès à Metal.')
    mx.set_default_device(mx.gpu)
    mx.set_memory_limit(9 * 1024**3)
    mx.set_cache_limit(512 * 1024**2)
    source, base = model_paths(plan)
    conversion = base / 'conversion.json'
    receipt = json.loads(conversion.read_text())
    if (receipt['files'] != hashes(base) or receipt['environment_lock_sha256'] != digest(LOCK)
            or receipt['upstream_receipt_sha256'] != digest(source / 'qwen_source.json')):
        raise ValueError('La base MLX diffère du reçu de conversion.')

    settings = plan['training']
    seed = plan['comparison']['seed']
    accumulation = settings['accumulation_steps']
    print('Préparation des exemples du train TSAC et MADAR et vérification du masque de perte…', flush=True)
    examples, max_length = prepare_examples(plan)
    if smoke:
        long = max(examples, key=lambda row: len(row['input_ids']))
        examples = [row for row in examples if row['id'] != long['id']][:accumulation - 1] + [long]
    epochs = 1 if smoke else settings['epochs']
    rows = len(examples)
    total = epochs * rows
    if rows % accumulation or plan['execution']['initial_microbatch'] != 1:
        raise ValueError('Les époques doivent finir sur une mise à jour complète avec un texte par lot.')
    run = {'mode': 'qlora', 'split': 'train', 'smoke': smoke, 'steps': total, 'rows_per_epoch': rows,
           'epochs': epochs, 'seed': seed, 'settings': settings, 'max_sequence_tokens': max_length,
           'dropout_seed_policy': 'seed times 1000000000 plus one-based example step',
           'source_model_revision': plan['model']['revision'], 'conversion_sha256': digest(conversion),
           'lock_sha256': digest(LOCK), 'dataset_sha256': object_digest(examples),
           'code_sha256': {name: digest(ROOT / 'scripts' / name) for name in (
               'run_qwen4b_qlora.py', 'build_qwen_sft.py', 'load_working_data.py', 'qwen_prompting.py')},
           'manifest_hashes': json.loads(AUDIT.read_text())['manifests_sha256']}
    run_file = output / 'training.json'
    checkpoint_path = output / 'checkpoint.safetensors'
    if run_file.exists() and json.loads(run_file.read_text()) != run:
        raise ValueError('Impossible de reprendre ce dossier avec ce jeu de données ou ces réglages.')
    save_json(run_file, run)
    run_hash = digest(run_file)

    mx.random.seed(seed)
    model, _ = load(str(base))
    model.freeze()
    linear_to_lora_layers(model, len(model.layers), settings['lora'])
    grad_checkpoint(model.layers[0])
    model.train()
    adapter_config = {'fine_tune_type': 'lora', 'num_layers': len(model.layers), 'lora_parameters': settings['lora']}
    trainable = dict(tree_flatten(model.trainable_parameters()))
    if not trainable or not all(name.endswith(('.lora_a', '.lora_b')) for name in trainable):
        raise ValueError('Seules les matrices LoRA doivent être entraînables.')
    optimizer = AdamW(learning_rate=settings['learning_rate'], weight_decay=settings['weight_decay'])
    optimizer.init(model.trainable_parameters())
    completed, previous_seconds, previous_peak = 0, 0.0, 0
    if checkpoint_path.exists():
        completed, previous_seconds, previous_peak = restore(checkpoint_path, model, optimizer, run_hash)
    if not 0 <= completed <= total or completed % accumulation or optimizer.step.item() != completed // accumulation:
        raise ValueError('Le compteur du point de reprise ou celui d’AdamW est incohérent.')

    loss_and_gradient = nn.value_and_grad(model, default_loss)
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def update(batch, lengths, accumulated, apply_update):
        (loss, tokens), gradients = loss_and_gradient(model, batch, lengths)
        # Weight by the count of assistant tokens before averaging the accumulated gradients.
        gradients = tree_map(lambda grad: grad * tokens, gradients)
        if accumulated is not None:
            previous, previous_tokens = accumulated
            gradients = tree_map(lambda grad, old: grad + old, gradients, previous)
            tokens = tokens + previous_tokens
        if apply_update:
            averaged = tree_map(lambda grad: grad / tokens, gradients)
            clipped, norm = clip_grad_norm(averaged, settings['gradient_clip'])
            optimizer.update(model, clipped)
            return loss, None, norm
        return loss, (gradients, tokens), mx.array(0.0)

    print(f'QLoRA {"contrôle court" if smoke else "complet"} | {rows} textes par époque | '
          f'{epochs} époque(s) | {completed}/{total} exemples traités', flush=True)
    tick = perf_counter()
    losses = []
    grad_buffer = None
    peak = previous_peak
    step = completed
    for epoch in range(completed // rows, epochs):
        order = random.Random(seed + epoch).sample(range(rows), rows)
        for index in range(max(0, completed - epoch * rows), rows):
            row = examples[order[index]]
            ids = row['input_ids']
            padded = min(1 + 32 * ((len(ids) + 31) // 32), max_length)
            batch = mx.array([ids + [0] * (padded - len(ids))])
            lengths = mx.array([[row['prompt_tokens'], len(ids) - 1]])
            step = epoch * rows + index + 1
            # Give dropout the same random stream at this step after a restart.
            mx.random.seed(seed * 1_000_000_000 + step)
            apply_update = step % accumulation == 0
            loss, grad_buffer, norm = update(batch, lengths, grad_buffer, apply_update)
            mx.eval(state, loss, grad_buffer, norm)
            value = float(loss.item())
            if not isfinite(value) or apply_update and not isfinite(norm.item()):
                raise FloatingPointError(f'Perte ou norme des gradients non finie à l’étape {step}.')
            losses.append(value)
            peak = max(peak, mx.get_peak_memory())
            this_session = perf_counter() - tick
            seconds = previous_seconds + this_session
            if apply_update and (step % (accumulation * settings['checkpoint_interval_updates']) == 0 or step % rows == 0):
                checkpoint(checkpoint_path, model, optimizer, step, run_hash, seconds, peak)
            if step == completed + 1 or step % (2 * accumulation) == 0 or step % rows == 0:
                filled = 20 * step // total
                bar = '#' * filled + '-' * (20 - filled)
                remaining = (total - step) * this_session / (step - completed)
                print(f'[{bar}] {step / total:6.1%} | époque {epoch + 1}/{epochs} | '
                      f'{step}/{total} | loss {sum(losses) / len(losses):.4f} | '
                      f'écoulé {fmt_time(seconds)} | reste estimé {fmt_time(remaining)}', flush=True)
                losses.clear()

    temp = output / 'adapters.tmp.safetensors'
    mx.save_safetensors(str(temp), dict(tree_flatten(model.trainable_parameters())))
    temp.replace(output / 'adapters.safetensors')
    save_json(output / 'adapter_config.json', adapter_config)
    save_json(output / 'completion.json', {
        'run_sha256': run_hash, 'steps': step, 'optimizer_updates': step // accumulation,
        'trainable_parameters': sum(item.size for item in trainable.values()), 'peak_mlx_bytes': peak,
        'adapter_sha256': {name: digest(output / name) for name in ADAPTER_FILES},
    })
    verify_adapter(output, plan)
    print(f'Adaptateur sauvegardé et vérifié dans {output}', flush=True)


def summarize(plan, output):
    run, completion = verify_adapter(output, plan)
    if run['smoke']:
        raise ValueError('Les métriques complètes exigent l’adaptateur de la campagne complète.')
    report_path = output / 'validation_results.json'
    report(plan, output / 'validation', report_path)
    baseline = json.loads(BASELINE.read_text())
    adapted = json.loads(report_path.read_text())
    expected = {key: value for key, value in baseline['run'].items() if key != 'code_sha256'}
    observed = {key: value for key, value in adapted['run'].items() if key not in ('code_sha256', 'adapter_sha256')}
    if (expected != observed or baseline['data_manifests_sha256'] != adapted['data_manifests_sha256']
            or adapted['run']['adapter_sha256'] != completion['adapter_sha256']):
        raise ValueError('La validation avant et après adaptation diffère de la comparaison prévue.')
    scores = {}
    for corpus, before in baseline['results'].items():
        after = adapted['results'][corpus]
        if before['demonstrations_sha256'] != after['demonstrations_sha256'] or before['n'] != after['n']:
            raise ValueError(f'Exemples de démonstration ou de validation désalignés pour {corpus}.')
        keys = ('accuracy', 'macro_f1', 'invalid_rows')
        scores[corpus] = {'n': after['n'], 'base': {key: before[key] for key in keys},
                          'qlora': {key: after[key] for key in keys},
                          'macro_f1_gain_points': 100 * (after['macro_f1'] - before['macro_f1'])}
    save_json(output / 'comparison.json', {'validation': 'few_shot_v2', 'scores': scores,
                                          'baseline_sha256': digest(BASELINE), 'validation_sha256': digest(report_path),
                                          'training_sha256': digest(output / 'completion.json')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('train', 'evaluate', 'summarize'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true', help='Run one update and four validation cases per corpus.')
    args = parser.parse_args()
    output = args.output.resolve()
    if (ROOT / 'outputs/qwen').resolve() not in output.parents:
        parser.error('Conserver les artefacts et prédictions sous outputs/qwen/.')
    output.mkdir(parents=True, exist_ok=True)
    plan = load_plan()
    with (output / '.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error(f'Une exécution utilise déjà ce dossier : {output}')
        if args.action == 'train':
            train(plan, output, args.smoke)
        elif args.action == 'evaluate':
            run, _ = verify_adapter(output, plan)
            if run['smoke'] != args.smoke:
                parser.error('Le mode doit correspondre à celui de l’adaptateur sauvegardé.')
            evaluate(plan, output / 'validation', adapter=output, validation_limit=4 if args.smoke else None)
        else:
            summarize(plan, output)


if __name__ == '__main__':
    main()
