"""Scientific invariants on tiny random BERT and invented data, no downloads."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch
from torch import nn
from transformers import BertConfig, BertModel, BertTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from frozen_encoder import masked_mean, extract_features, fit_head, head_output, state_digest
from run_frozen_encoder import run, validate_config


class FrozenEncoderTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        vocab = self.directory / 'vocab.txt'
        vocab.write_text('[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\na\nb\nc\n')
        self.tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=False)
        self.model = BertModel(BertConfig(vocab_size=8, hidden_size=8, intermediate_size=16,
                                          num_hidden_layers=1, num_attention_heads=2,
                                          max_position_embeddings=128), add_pooling_layer=False)
        self.model.requires_grad_(False).eval()
        self.config = json.loads((ROOT / 'configs/camelbert_frozen_v1.json').read_text())

    def test_masked_mean_excludes_padding_hand_calculation(self):
        hidden = torch.tensor([[[1., 3.], [3., 5.], [100., 100.]]])
        actual = masked_mean(hidden, torch.tensor([[1, 1, 0]]))
        self.assertTrue(torch.equal(actual, torch.tensor([[2., 4.]])))
        with self.assertRaises(ValueError):
            masked_mean(hidden, torch.tensor([[0, 0, 0]]))

    def test_features_invariant_to_other_rows_padding_and_order(self):
        rows = [{'text_raw': 'a'}, {'text_raw': 'a b c a b'}]
        alone, _ = extract_features(self.model, self.tokenizer, rows[:1], self.config, torch.device('cpu'))
        both, _ = extract_features(self.model, self.tokenizer, rows, self.config, torch.device('cpu'))
        self.assertTrue(np.allclose(alone[0], both[0], atol=1e-6))
        self.assertEqual(both.shape, (2, 8))
        self.assertEqual(both.dtype, np.float32)

    def test_encoder_eval_and_no_grad_required_and_unchanged(self):
        rows = [{'text_raw': 'a b'}]
        before = state_digest(self.model)
        extract_features(self.model, self.tokenizer, rows, self.config, torch.device('cpu'))
        self.assertEqual(before, state_digest(self.model))
        self.assertTrue(all(p.grad is None for p in self.model.parameters()))
        self.model.train()
        with self.assertRaises(ValueError):
            extract_features(self.model, self.tokenizer, rows, self.config, torch.device('cpu'))
        self.model.eval().requires_grad_(True)
        with self.assertRaises(ValueError):
            extract_features(self.model, self.tokenizer, rows, self.config, torch.device('cpu'))

    def test_truncation_really_bounds_encoder_input(self):
        vectors, timing = extract_features(self.model, self.tokenizer,
                    [{'text_raw': ' '.join(['a'] * 200)}], self.config, torch.device('cpu'))
        self.assertEqual(timing['maximum_padded_length'], 128)
        self.assertTrue(np.isfinite(vectors).all())

    def test_head_learning_repeatability_and_safe_reload(self):
        config = {**self.config, 'head_learning_rate': .05, 'head_patience': 8, 'head_max_epochs': 20}
        x = np.array([[-2., 0.], [2., 0.]] * 10, dtype=np.float32)
        y = [0, 1] * 10
        a = fit_head(x, x, y, y, ['NEG', 'POS'], config, 13, 'full')
        b = fit_head(x, x, y, y, ['NEG', 'POS'], config, 13, 'full')
        self.assertEqual(a[1]['metrics']['accuracy'], 1)
        self.assertEqual(a[1], b[1])
        self.assertEqual(a[2], b[2])
        self.assertTrue(a[3]['weights_changed'])
        self.assertEqual(a[3]['best_epoch'], max(a[2], key=lambda h: h['validation_macro_f1'])['epoch'])
        path = self.directory / 'head.pt'
        torch.save(a[0].state_dict(), path)
        head = nn.Linear(2, 2)
        head.load_state_dict(torch.load(path, weights_only=True, map_location='cpu'))
        self.assertEqual(head_output(head, x, torch.tensor(y), ['NEG', 'POS']), a[1])

    def test_full_requires_smoke_before_loading_weights_or_data(self):
        with self.assertRaisesRegex(ValueError, 'requires a verified smoke'):
            run('full', ROOT / 'outputs/frozen_encoder/test-not-created')
        self.assertFalse((ROOT / 'outputs/frozen_encoder/test-not-created').exists())

    def test_protocol_change_rejected(self):
        changed = copy.deepcopy(self.config)
        changed['manifest_version'] = 'v1'
        with self.assertRaises(ValueError):
            validate_config(changed)


if __name__ == '__main__':
    unittest.main()
