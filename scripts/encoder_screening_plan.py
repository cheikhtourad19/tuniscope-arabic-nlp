"""Bounded experiment scopes and validation-only learning-rate selection; no Torch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs/camelbert_finetuning_v1.json'
PLAN = ROOT / 'configs/camelbert_screening_v1.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def study_plan():
    p = json.loads(PLAN.read_text())
    if (p['version'] != 'camelbert-learning-rate-screening-v1'
            or p['base_config'] != str(CONFIG.relative_to(ROOT)) or p['base_config_sha256'] != digest(CONFIG)
            or p['learning_rates'] != [1e-5, 2e-5, 5e-5] or p['screening_seed'] != 42
            or p['corpora'] != ['tsac', 'madar'] or p['full_regime'] != 'finetuned'
            or p['smoke_regime'] != 'paired' or p['selection_metric'] != 'validation_macro_f1'
            or p['selection_per_corpus'] is not True or p['tie_break'] != 'smaller_learning_rate'
            or p['confirmation_seeds'] != [13, 42, 2026]
            or p['protocol'] != 'docs/52_protocole_selection_encodeur.md'):
        raise ValueError('Unsupported or changed screening plan')
    return p


def effective_config(scope='initial', learning_rate=None, corpus=None):
    c = json.loads(CONFIG.read_text())
    if scope != 'confirmation' and corpus is not None:
        raise ValueError('A task override is allowed only for fixed confirmation')
    if scope == 'initial':
        if learning_rate is not None:
            raise ValueError('Initial protocol does not accept learning-rate overrides')
    elif scope == 'screening':
        if learning_rate not in study_plan()['learning_rates']:
            raise ValueError('Choose one of the three preregistered learning rates')
        c['learning_rate'] = learning_rate
    elif scope == 'confirmation':
        if learning_rate is not None:
            raise ValueError('Confirmation rates are fixed by the selection report')
        from encoder_confirmation_plan import confirmation_config
        return confirmation_config(corpus)
    else:
        raise ValueError('Unknown experiment scope')
    return c


def validate_effective_config(c, scope='initial'):
    corpus = None
    if scope == 'confirmation':
        if len(c.get('corpora', [])) != 1:
            raise ValueError('Each confirmation stage must contain exactly one task')
        corpus = c['corpora'][0]
    expected = effective_config(scope, c.get('learning_rate') if scope == 'screening' else None, corpus)
    if c != expected or c['manifest_version'] != 'v2' or c['variant'] != 'raw-v0':
        raise ValueError('Expected fixed native CLS fine-tuning protocol')


def execution_seeds(mode, scope, config):
    validate_effective_config(config, scope)
    if mode not in {'smoke', 'full'}:
        raise ValueError('Invalid training mode')
    return [config['smoke_seed']] if mode == 'smoke' or scope == 'screening' else config['seeds']


def select_candidates(candidates):
    """Select per task, rejecting missing, repeated or smoke-scored candidates."""
    p = study_plan()
    if len(candidates) != len(p['learning_rates']):
        raise ValueError('Incomplete screening campaign')
    rates = [r['learning_rate'] for r in candidates]
    if sorted(rates) != p['learning_rates']:
        raise ValueError('Missing or duplicate learning-rate candidates')
    if len({r['device'] for r in candidates}) != 1:
        raise ValueError('Candidates must use the same device')
    for row in candidates:
        if row['mode'] != 'full' or row['experiment_scope'] != 'screening' or row['test_loaded'] is not False:
            raise ValueError('Selection requires FULL screening without test')
        if row['seed'] != p['screening_seed'] or set(row['scores']) != set(p['corpora']):
            raise ValueError('Wrong screening seed or tasks')
        if any(not isinstance(s, (float, int)) or not 0 <= s <= 1 for s in row['scores'].values()):
            raise ValueError('Invalid candidate score')
    return {corpus: min(candidates, key=lambda r: (-r['scores'][corpus], r['learning_rate']))['learning_rate']
            for corpus in p['corpora']}
