"""Check frozen-input enforcement and citation verification on invented data."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
from evaluate import sha
from final_evaluation import check_fingerprints, verify_rows
from search import Card, extract


def test_changed_input_is_rejected_before_model_loading(tmp_path):
    source = tmp_path / 'cards.json'
    source.write_text('version fictive initiale')
    fingerprints = {'cards.json': sha(source)}
    check_fingerprints(fingerprints, root=tmp_path)
    source.write_text('version fictive modifiée')
    with pytest.raises(ValueError, match='version gelée'):
        check_fingerprints(fingerprints, root=tmp_path)


def test_missing_frozen_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match='version gelée'):
        check_fingerprints({'missing.json': '0' * 64}, root=tmp_path)


def test_fabricated_answer_cannot_pass_as_cited_extract():
    card = Card('exemple.fr', 'fr', 1, 'Titre fictif', 'Passage fictif autorisé.', 'test', ['fictif.md'])
    question = {'id': 'invented', 'lang': 'fr', 'split': 'test', 'question': 'Question fictive',
                'relevant_ids': ['exemple.fr']}
    row = {**question, 'ranking': [{'id': card.id, 'score': 1.0}], 'recall': 1.0,
           'reciprocal_rank': 1.0, 'response': extract(question['question'], 'fr', [(card, 1.0)], 0)}
    verify_rows([row], [question], {card.id: card})
    row['response']['answer'] = 'Affirmation ajoutée artificiellement.'
    with pytest.raises(ValueError, match='passage cité'):
        verify_rows([row], [question], {card.id: card})
