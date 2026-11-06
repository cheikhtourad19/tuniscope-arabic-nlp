"""Fixed selected-rate confirmation plan and portable aggregate provenance; no Torch."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / 'configs/camelbert_confirmation_v1.json'
BASE = ROOT / 'configs/camelbert_finetuning_v1.json'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def confirmation_plan():
    p = json.loads(PLAN.read_text())
    if (p['version'] != 'camelbert-selected-rates-confirmation-v1'
            or p['base_config_sha256'] != digest(BASE)
            or p['selected_learning_rates'] != {'tsac': 5e-5, 'madar': 2e-5}
            or p['corpora'] != ['tsac', 'madar'] or p['seeds'] != [13, 42, 2026]
            or p['smoke_seed'] != 42 or p['full_regime'] != 'finetuned' or p['smoke_regime'] != 'paired'
            or p['reuse_screening_checkpoints'] is not False
            or p['reporting'] != 'all-seeds-mean-and-sample-standard-deviation'
            or p['protocol'] != 'docs/55_protocole_confirmation_encodeur.md'):
        raise ValueError('Unsupported confirmation plan')
    source = (ROOT / p['selection_report']).resolve()
    if ROOT.resolve() not in source.parents or digest(source) != p['selection_report_sha256']:
        raise ValueError('Selection source report changed')
    report = json.loads(source.read_text())
    if (report['evidence_type'] != 'retained-local-full-artifacts-and-replay-proofs'
            or report['scope'] != 'screening' or report['seed'] != 42 or report['test_loaded'] is not False
            or report['selected_learning_rates'] != p['selected_learning_rates']):
        raise ValueError('Wrong source evidence for confirmation')
    for corpus in p['corpora']:
        candidates = [r for r in report['comparisons'] if r['corpus'] == corpus]
        if (len(candidates) != 3 or sorted(r['learning_rate'] for r in candidates) != [1e-5, 2e-5, 5e-5]
                or any(r['seed'] != 42 or not 0 <= r['macro_f1'] <= 1 for r in candidates)):
            raise ValueError('Incomplete or invalid source candidate set')
        chosen = min(candidates, key=lambda r: (-r['macro_f1'], r['learning_rate']))['learning_rate']
        if chosen != p['selected_learning_rates'][corpus]:
            raise ValueError('Confirmation rates differ from source selection')
    return p


def confirmation_config(corpus):
    p = confirmation_plan()
    if corpus not in p['corpora']:
        raise ValueError('Choose exactly one confirmation task')
    c = json.loads(BASE.read_text())
    c.update(corpora=[corpus], learning_rate=p['selected_learning_rates'][corpus])
    return c
