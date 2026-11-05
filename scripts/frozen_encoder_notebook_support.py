"""Standard-library notebook bridge: no Torch import in the Jupyter kernel."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_tool(script, *arguments):
    if script not in {'run_frozen_encoder.py', 'verify_frozen_encoder.py', 'plot_frozen_encoder.py'}:
        raise ValueError('Expected a frozen encoder script')
    environment = os.environ.copy()
    environment.pop('VIRTUAL_ENV', None)
    environment['MPLBACKEND'] = 'Agg'
    command = ['sh', str(ROOT / 'scripts/encoder_uv.sh'), 'run', '--locked', '--offline', 'python',
               '-u', str(ROOT / 'scripts' / script), *map(str, arguments)]
    process = subprocess.Popen(command, cwd=ROOT, env=environment, text=True, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, bufsize=1, start_new_session=True)
    try:
        for line in process.stdout:
            print(line, end='', flush=True)
        status = process.wait()
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
        process.wait()
        raise
    finally:
        process.stdout.close()
    if status:
        raise subprocess.CalledProcessError(status, command)


def select_run(directory, expected_mode):
    directory = Path(directory)
    directory = (ROOT / directory).resolve() if not directory.is_absolute() else directory.resolve()
    if (ROOT / 'outputs/frozen_encoder').resolve() not in directory.parents:
        raise ValueError('Choose a frozen encoder run under outputs/frozen_encoder/')
    result = json.loads((directory / 'results.json').read_text())
    if result['mode'] != expected_mode or result['test_loaded'] is not False:
        raise ValueError('Run mode/test boundary mismatch; no smoke/full fallback')
    seeds = [42] if expected_mode == 'smoke' else [13, 42, 2026]
    if (len(result['results']) != 2 * len(seeds) or {(r['corpus'], r['seed']) for r in result['results']}
            != {(c, s) for c in ('tsac', 'madar') for s in seeds}):
        raise ValueError('Incomplete results')
    return directory, result


def notebook_run(mode, replay_directory=None):
    if mode not in {'full', 'smoke', 'replay'}:
        raise ValueError('Choose full, smoke or replay')
    if mode == 'replay':
        directory, _ = select_run(replay_directory, 'full')
        run_tool('verify_frozen_encoder.py', directory)
        return select_run(directory, 'full')
    parent = ROOT / 'outputs/frozen_encoder/notebooks'
    identifier = uuid.uuid4().hex
    smoke = parent / (identifier + '_smoke')
    run_tool('run_frozen_encoder.py', '--mode', 'smoke', '--output', smoke)
    run_tool('verify_frozen_encoder.py', smoke)
    if mode == 'smoke':
        return select_run(smoke, 'smoke')
    target = parent / (identifier + '_full')
    run_tool('run_frozen_encoder.py', '--mode', 'full', '--smoke-run', smoke, '--output', target)
    run_tool('verify_frozen_encoder.py', target)
    return select_run(target, 'full')


def figure_paths(directory):
    target = directory / 'figures'
    if not target.exists():
        run_tool('plot_frozen_encoder.py', directory)
    provenance = json.loads((target / 'provenance.json').read_text())
    if provenance['source_results_sha256'] != digest(directory / 'results.json'):
        raise ValueError('Figures refer to another results file')
    saved = json.loads((directory / 'results.json').read_text())
    expected = {f'{r["corpus"]}_seed_{r["seed"]}.png' for r in saved['results']}
    if set(provenance['files']) != expected:
        raise ValueError('Missing or extra figure records')
    paths = []
    for filename, checksum in provenance['files'].items():
        path = target / filename
        if path.name != filename or digest(path) != checksum:
            raise ValueError('Figure changed or invalid path')
        paths.append(path)
    return paths
