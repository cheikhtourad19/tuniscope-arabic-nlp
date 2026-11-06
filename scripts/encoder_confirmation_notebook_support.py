"""Lightweight notebook08 bridge to fixed-rate confirmation in the encoder profile."""
import json
from pathlib import Path
import uuid
from encoder_finetuning_notebook_support import ROOT, run_tool, figures


def selected(directory, expected_mode):
    directory=Path(directory)
    directory=(ROOT/directory).resolve() if not directory.is_absolute() else directory.resolve()
    if (ROOT/'outputs/encoder_finetuning').resolve() not in directory.parents:
        raise ValueError('Choose a private confirmation campaign')
    d=json.loads((directory/'confirmation.json').read_text())
    if expected_mode not in {'full','smoke'} or d['mode']!=expected_mode or d['test_loaded'] is not False:
        raise ValueError('Explicit confirmation replay mode mismatch')
    if [e['corpus'] for e in d['entries']]!=['tsac','madar']:
        raise ValueError('Missing confirmation tasks')
    if (d['aggregates'] is not None)!=(expected_mode=='full'):
        raise ValueError('Missing or unexpected three-seed aggregate')
    return directory,d


def notebook_run(mode,device='auto',replay_directory=None,replay_mode='full'):
    if mode=='off':return None
    if mode=='replay':
        directory,_=selected(replay_directory,replay_mode)
        run_tool('run_encoder_confirmation.py','--verify',directory,'--expected-mode',replay_mode)
        return selected(directory,replay_mode)
    if mode not in {'full','smoke'}:raise ValueError('Choose full, smoke, replay or off')
    directory=ROOT/'outputs/encoder_finetuning/confirmation/notebooks'/f'{uuid.uuid4().hex}_{mode}'
    run_tool('run_encoder_confirmation.py','--mode',mode,'--device',device,'--output',directory)
    return selected(directory,mode)


def campaign_figures(directory,metadata):
    paths=[]
    for entry in metadata['entries']:
        stage=entry['full'] if metadata['mode']=='full' else entry['smoke']
        folder=(directory/stage['directory']).resolve()
        if directory.resolve() not in folder.parents:raise ValueError('Escaping stage directory')
        paths.extend(figures(folder))
    if len(paths)!=(6 if metadata['mode']=='full' else 4):raise ValueError('Missing confirmation figures')
    return paths
