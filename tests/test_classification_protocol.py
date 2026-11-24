"""Regression checks for the frozen classification contract, without model loading."""
import copy
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from classification_protocol import check, validate_structure


@pytest.fixture
def protocol():
    return json.loads((ROOT / 'configs/classification_final_v1.json').read_text())


def test_repeated_lora_seed_cannot_replace_one_final_seed(protocol):
    row = next(r for r in protocol['evaluations'] if r['model'] == 'qwen_lora__42'
               and r['corpus'] == 'tsac' and r['prompt_mode'] == 'zero_shot')
    row['model'] = 'qwen_lora__13'
    with pytest.raises(ValueError, match='Trois graines'):
        validate_structure(protocol)


def test_cross_task_checkpoint_cannot_be_substituted(protocol):
    row = next(r for r in protocol['evaluations'] if r['model'] == 'tsac__bigru__13')
    row['model'] = 'madar__bigru__13'
    with pytest.raises(ValueError, match='mauvaise famille ou tâche'):
        validate_structure(protocol)


def test_changed_artifact_is_rejected_without_reading_test_texts(protocol, tmp_path):
    changed = copy.deepcopy(protocol)
    first_file = next(iter(changed['files_sha256']))
    changed['files_sha256'][first_file] = '0' * 64
    path = tmp_path / 'altered_protocol.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='Artefact différent du gel'):
        check(path)
