"""Run the fixed three-rate screening, or check an existing campaign without training."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from encoder_screening_plan import ROOT, PLAN, digest, effective_config, select_candidates, study_plan
from baseline_metrics import classification_metrics
from bigru_data import partitions as load_partitions
from character_lm_experiment import partition_summary
from encoder_source_provenance import verify_recorded_sources


def read(path):
    return json.loads(path.read_text())


def child(script, *args):
    if script not in {'run_encoder_finetuning.py', 'verify_encoder_finetuning.py', 'plot_encoder_finetuning.py'}:
        raise ValueError('Unexpected screening child')
    env = os.environ.copy(); env.pop('VIRTUAL_ENV', None); env['MPLBACKEND'] = 'Agg'
    subprocess.run([sys.executable, '-u', str(ROOT / 'scripts' / script), *map(str, args)],
                   cwd=ROOT, env=env, check=True)


def inside(directory, name):
    target = (directory / name).resolve()
    if directory.resolve() not in target.parents:
        raise ValueError('Campaign path escapes directory')
    return target


def checked_stage(campaign, stage, rate, mode):
    directory = inside(campaign, stage['directory'])
    if digest(directory / 'results.json') != stage['results_sha256']:
        raise ValueError('Stage results changed')
    if digest(directory / 'verification.json') != stage['verification_sha256']:
        raise ValueError('Stage verification changed')
    d, proof = read(directory / 'results.json'), read(directory / 'verification.json')
    if (d['mode'] != mode or d.get('experiment_scope') != 'screening'
            or d['config'] != effective_config('screening', rate) or d['test_loaded'] is not False
            or d['regime'] != ('paired' if mode == 'smoke' else 'finetuned')
            or d['execution_seeds'] != [42]):
        raise ValueError('Wrong stage scope/configuration')
    arms = ['frozen', 'finetuned'] if mode == 'smoke' else ['finetuned']
    expected = {(c, arm, 42) for c in ('tsac', 'madar') for arm in arms}
    if len(d['results']) != len(expected) or {(r['corpus'], r['regime'], r['seed']) for r in d['results']} != expected:
        raise ValueError('Incomplete stage model set')
    if (proof['results_sha256'] != stage['results_sha256'] or proof['models_replayed'] != len(expected)
            or proof['test_loaded'] is not False or proof['epoch_class_metrics_complete'] is not True
            or proof['device'] != d['device']):
        raise ValueError('Incomplete stage replay/epoch evidence')
    if d['study_plan_sha256'] != digest(PLAN) or d['protocol_sha256'] != digest(ROOT / study_plan()['protocol']):
        raise ValueError('Study plan or protocol changed')
    for name, h in d['artifacts'].items():
        if digest(inside(directory, name)) != h:
            raise ValueError('Stage artifact changed')
    verify_recorded_sources(d['code_sha256'])
    partitions = {corpus: load_partitions(corpus, mode, d['config']) for corpus in ('tsac', 'madar')}
    for r in d['results']:
        train, val = partitions[r['corpus']]
        if r['train'] != partition_summary(train) or r['validation'] != partition_summary(val):
            raise ValueError('Stage partitions differ from the v2 sources')
        preds = read(inside(directory, r['checkpoint'] + '/predictions.json'))
        if [(x['id'], x['group_id'], x['truth']) for x in preds] != [(x['id'], x['group_id'], x['label']) for x in val]:
            raise ValueError('Stage validation prediction alignment differs')
        metrics = classification_metrics([x['truth'] for x in preds], [x['predicted'] for x in preds], d['config']['labels'][r['corpus']])
        if metrics != {key: r['metrics'][key] for key in metrics}:
            raise ValueError('Candidate scores differ from saved predictions')
    return d


def verify_campaign(directory, expected_mode):
    directory = directory.resolve()
    d = read(directory / 'screening.json'); plan = study_plan()
    if expected_mode not in {'smoke', 'full'} or d['mode'] != expected_mode:
        raise ValueError('Explicit campaign replay mode mismatch')
    if d['plan_sha256'] != digest(PLAN) or d['protocol_sha256'] != digest(ROOT / plan['protocol']):
        raise ValueError('Campaign plan/protocol changed')
    verify_recorded_sources({'run_encoder_screening.py':d['workflow_sha256']})
    if [e['learning_rate'] for e in d['entries']] != plan['learning_rates']:
        raise ValueError('Missing, reordered or duplicate candidate rates')
    candidates, hardware, partitions, code, initializations = [], None, None, None, None
    stages_checked = 0
    for entry in d['entries']:
        rate = entry['learning_rate']
        smoke = checked_stage(directory, entry['smoke'], rate, 'smoke'); stages_checked += 1
        if hardware is None: hardware = smoke['hardware']
        if smoke['hardware'] != hardware:
            raise ValueError('Candidates must run on the same hardware')
        if expected_mode == 'smoke':
            if entry.get('full') is not None:
                raise ValueError('Smoke campaign includes undeclared FULL run')
            continue
        full = checked_stage(directory, entry['full'], rate, 'full'); stages_checked += 1
        if full['hardware'] != hardware:
            raise ValueError('FULL candidates use inconsistent hardware')
        if full['smoke_run'] != str(inside(directory, entry['smoke']['directory'])):
            raise ValueError('FULL run is not linked to this verified smoke')
        current = {r['corpus']: {'train': r['train'], 'validation': r['validation']} for r in full['results']}
        if partitions is None: partitions = current
        if current != partitions:
            raise ValueError('Candidates use different examples/groups/labels')
        if code is None: code = full['code_sha256']
        if full['code_sha256'] != code:
            raise ValueError('Candidate training code differs')
        current_initial = {r['corpus']: (r['initial_backbone_sha256'], r['initial_head_sha256']) for r in full['results']}
        if initializations is None: initializations = current_initial
        if current_initial != initializations:
            raise ValueError('Candidates did not start from identical weights')
        candidates.append({'learning_rate': rate, 'mode': 'full', 'experiment_scope': 'screening',
            'test_loaded': False, 'device': full['device'], 'seed': 42,
            'scores': {r['corpus']: r['metrics']['macro_f1'] for r in full['results']}})
    selected = select_candidates(candidates) if expected_mode == 'full' else None
    if selected != d['selected_learning_rates']:
        raise ValueError('Saved selection differs from the preregistered criterion')
    return d, {'campaign_sha256': digest(directory / 'screening.json'), 'stages_checked': stages_checked,
               'mode': expected_mode, 'selected_learning_rates': selected,
               'artifact_integrity_checked': True, 'prior_stage_replay_proofs_checked': True,
               'new_checkpoint_replay_performed_by_campaign_check': False}


def stage_reference(directory, campaign):
    return {'directory': str(directory.relative_to(campaign)),
            'results_sha256': digest(directory / 'results.json'),
            'verification_sha256': digest(directory / 'verification.json')}


def campaign(mode, output, device='auto'):
    plan = study_plan(); output = output.resolve()
    if mode not in {'smoke', 'full'} or device not in {'auto', 'cpu', 'mps', 'cuda'}:
        raise ValueError('Invalid campaign mode/device')
    if (ROOT / 'outputs/encoder_finetuning').resolve() not in output.parents or output.exists():
        raise ValueError('Choose a new private screening directory')
    output.mkdir(parents=True)
    d = {'mode': mode, 'plan_sha256': digest(PLAN), 'protocol_sha256': digest(ROOT / plan['protocol']),
         'workflow_sha256': digest(Path(__file__)), 'entries': [], 'selected_learning_rates': None,
         'test_loaded': False, 'final_confirmation_performed': False}
    candidates = []
    try:
        for rate in plan['learning_rates']:
            slug = f'lr_{rate:.0e}'
            smoke = output / f'{slug}_smoke'
            shared = ['--device', device, '--experiment-scope', 'screening', '--learning-rate', str(rate)]
            print(f'\n=== Taux {rate:g} — contrôle préalable ===', flush=True)
            child('run_encoder_finetuning.py', '--mode', 'smoke', '--regime', 'paired', '--output', smoke, *shared)
            child('verify_encoder_finetuning.py', smoke, '--device', device)
            entry = {'learning_rate': rate, 'smoke': stage_reference(smoke, output), 'full': None}
            if mode == 'full':
                full = output / f'{slug}_full'
                print(f'\n=== Taux {rate:g} — toutes les données, graine42 ===', flush=True)
                child('run_encoder_finetuning.py', '--mode', 'full', '--regime', 'finetuned',
                      '--output', full, '--smoke-run', smoke, *shared)
                child('verify_encoder_finetuning.py', full, '--device', device)
                child('plot_encoder_finetuning.py', full)
                entry['full'] = stage_reference(full, output)
                result = read(full / 'results.json')
                candidates.append({'learning_rate': rate, 'mode': 'full', 'experiment_scope': 'screening',
                    'test_loaded': False, 'device': result['device'], 'seed': 42,
                    'scores': {r['corpus']: r['metrics']['macro_f1'] for r in result['results']}})
            else:
                child('plot_encoder_finetuning.py', smoke)
            d['entries'].append(entry)
        if mode == 'full': d['selected_learning_rates'] = select_candidates(candidates)
        (output / 'screening.json').write_text(json.dumps(d, ensure_ascii=False, indent=2) + '\n')
        _, proof = verify_campaign(output, mode)
        (output / 'campaign_verification.json').write_text(json.dumps(proof, indent=2) + '\n')
        lines = ['# Criblage CAMeLBERT', '', f'Mode **{mode}**, graine42. Test officiel non évalué.', '']
        if mode == 'full':
            lines += ['| Taux | TSAC Macro-F1 | MADAR Macro-F1 |', '| ---: | ---: | ---: |']
            for c in candidates: lines.append(f"| {c['learning_rate']:g} | {c['scores']['tsac']:.6f} | {c['scores']['madar']:.6f} |")
            lines += ['', f"Taux proposés : TSAC **{d['selected_learning_rates']['tsac']:g}**, MADAR **{d['selected_learning_rates']['madar']:g}**.",
                      '', 'Résultat exploratoire sur une graine. Confirmation sur trois graines non exécutée.']
        else:
            lines += ['Trois contrôles techniques appariés réussis. **Aucun taux sélectionné sur les scores smoke.**']
        lines += ['', f'Dossier : `{output}`', '']
        (output / 'summary.md').write_text('\n'.join(lines))
        print('\n'.join(lines), flush=True)
        return output
    except Exception as error:
        (output / 'failure.json').write_text(json.dumps({'error': str(error), 'completed_entries': d['entries'],
            'selection_completed': False}, ensure_ascii=False, indent=2) + '\n')
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['smoke', 'full'])
    p.add_argument('--output', type=Path)
    p.add_argument('--device', choices=['auto', 'cpu', 'mps', 'cuda'], default='auto')
    p.add_argument('--verify', type=Path)
    p.add_argument('--expected-mode', choices=['smoke', 'full'])
    a = p.parse_args()
    if a.verify:
        if a.mode or a.output or not a.expected_mode: p.error('Verification requires --expected-mode only')
        _, proof = verify_campaign(a.verify, a.expected_mode)
        print(json.dumps(proof, ensure_ascii=False, indent=2))
    else:
        if not a.mode or not a.output: p.error('Training requires --mode and --output')
        campaign(a.mode, a.output, a.device)
