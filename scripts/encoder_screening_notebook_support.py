"""Lightweight notebook orchestration; neural imports stay in the encoder profile."""
import json
from pathlib import Path
import uuid
from encoder_finetuning_notebook_support import ROOT, run_tool, figures


def selected(directory, expected_mode):
    directory = Path(directory)
    directory = (ROOT / directory).resolve() if not directory.is_absolute() else directory.resolve()
    if (ROOT / 'outputs/encoder_finetuning').resolve() not in directory.parents:
        raise ValueError('Choose a private encoder campaign')
    d = json.loads((directory / 'screening.json').read_text())
    if expected_mode not in {'smoke', 'full'} or d['mode'] != expected_mode or d['test_loaded'] is not False:
        raise ValueError('Explicit smoke/full campaign mode mismatch')
    if len(d['entries']) != 3:
        raise ValueError('Incomplete three-rate campaign')
    if (d['selected_learning_rates'] is not None) != (expected_mode == 'full'):
        raise ValueError('Unexpected or missing FULL selection')
    return directory, d


def notebook_run(mode, device='auto', replay_directory=None, replay_mode='full'):
    if mode == 'off': return None
    if mode == 'replay':
        directory, _ = selected(replay_directory, replay_mode)
        run_tool('run_encoder_screening.py', '--verify', directory, '--expected-mode', replay_mode)
        return selected(directory, replay_mode)
    if mode not in {'smoke', 'full'}:
        raise ValueError('Choose full, smoke, replay or off')
    directory = ROOT / 'outputs/encoder_finetuning/screening/notebooks' / f'{uuid.uuid4().hex}_{mode}'
    run_tool('run_encoder_screening.py', '--mode', mode, '--device', device, '--output', directory)
    return selected(directory, mode)


def campaign_figures(directory, metadata):
    result = []
    for entry in metadata['entries']:
        stage = entry['full'] if metadata['mode'] == 'full' else entry['smoke']
        target = (directory / stage['directory']).resolve()
        if directory.resolve() not in target.parents:
            raise ValueError('Escaping campaign stage path')
        result.extend(figures(target))
    expected = 6 if metadata['mode'] == 'full' else 12
    if len(result) != expected:
        raise ValueError('Missing screening figures')
    return result
