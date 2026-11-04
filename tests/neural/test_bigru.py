"""Run with scripts/bigru_uv.sh: model and pipeline checks on invented texts."""
import copy
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from bigru_data import build_vocabulary
from bigru_model import BiGRUClassifier, collate, resolve_device, set_seed
from bigru_training import evaluate, make_loader, train_epoch
from character_lm_experiment import ROOT
from run_bigru import run
from verify_bigru import verify


class BiGRUModelTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / "configs/bigru_v1.json").read_text())
        self.config.update(embedding_dim=8, hidden_dim=8, dropout=0.0)
        set_seed(42, 2)

    def test_padding_does_not_change_logits_including_reverse_direction(self):
        model = BiGRUClassifier(8, 2, self.config).eval()
        one = collate([([2, 3], 0)])
        batch = collate([([2, 3], 0), ([4, 5, 6, 7], 1)])
        with torch.inference_mode():
            expected = model(one[0], one[1])[0]
            actual = model(batch[0], batch[1])[0]
        torch.testing.assert_close(expected, actual, atol=1e-6, rtol=1e-6)
        self.assertTrue(model.gru.bidirectional)

    def test_gradients_update_embedding_gru_and_head_but_not_padding(self):
        model = BiGRUClassifier(8, 2, self.config)
        batch = collate([([2, 3], 0), ([4], 1)])
        initial = {k: p.detach().clone() for k, p in model.named_parameters()}
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        loss = torch.nn.functional.cross_entropy(model(batch[0], batch[1]), batch[2])
        loss.backward()
        for prefix in ("embedding", "gru", "classifier"):
            self.assertTrue(any(p.grad is not None and float(p.grad.abs().sum()) > 0 for n, p in model.named_parameters() if n.startswith(prefix)))
        self.assertEqual(float(model.embedding.weight.grad[0].abs().sum()), 0)
        optimizer.step()
        for prefix in ("embedding", "gru", "classifier"):
            self.assertTrue(any(not torch.equal(initial[n], p) for n, p in model.named_parameters() if n.startswith(prefix)))
        self.assertEqual(float(model.embedding.weight[0].detach().abs().sum()), 0)

    def test_eval_does_not_update_weights_and_checkpoint_reload(self):
        model = BiGRUClassifier(8, 2, self.config)
        loader = [(t, l, y) for t, l, y in [collate([([2, 3], 0), ([4], 1)])]]
        before = copy.deepcopy(model.state_dict())
        result = evaluate(model, loader, ["NEG", "POS"], torch.device("cpu"))
        self.assertFalse(model.training)
        self.assertTrue(all(torch.equal(before[k], v) for k, v in model.state_dict().items()))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.pt"
            torch.save(model.state_dict(), path)
            loaded = BiGRUClassifier(8, 2, self.config)
            loaded.load_state_dict(torch.load(path, weights_only=True))
            replay = evaluate(loaded, loader, ["NEG", "POS"], torch.device("cpu"))
        self.assertEqual(result["predicted"], replay["predicted"])
        self.assertEqual(result["probabilities"], replay["probabilities"])

    def test_seed_repeats_short_cpu_training(self):
        batch = collate([([2, 3], 0), ([4], 1)])
        final = []
        for _ in range(2):
            set_seed(13, 2)
            model = BiGRUClassifier(8, 2, self.config)
            optimizer = torch.optim.Adam(model.parameters(), lr=.01)
            train_epoch(model, [batch], optimizer, torch.device("cpu"), 1)
            final.append(copy.deepcopy(model.state_dict()))
        self.assertTrue(all(torch.equal(final[0][k], v) for k, v in final[1].items()))

    def test_small_learning_problem_reduces_loss(self):
        model = BiGRUClassifier(8, 2, self.config)
        batch = collate([([2, 2], 0), ([4, 4], 1), ([2], 0), ([4], 1)])
        optimizer = torch.optim.Adam(model.parameters(), lr=.03)
        before = evaluate(model, [batch], ["NEG", "POS"], torch.device("cpu"))["loss"]
        for _ in range(12):
            train_epoch(model, [batch], optimizer, torch.device("cpu"), 1)
        after = evaluate(model, [batch], ["NEG", "POS"], torch.device("cpu"))["loss"]
        self.assertLess(after, before * .5)

    def test_devices_unavailable_fail_explicitly(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")
        with patch("torch.cuda.is_available", return_value=False), self.assertRaises(RuntimeError):
            resolve_device("cuda")
        with patch("torch.backends.mps.is_available", return_value=False), self.assertRaises(RuntimeError):
            resolve_device("mps")
        with self.assertRaises(ValueError):
            resolve_device("invented")


class BiGRURunnerTests(unittest.TestCase):
    def setUp(self):
        base = ROOT / "outputs/neural_baselines"
        base.mkdir(exist_ok=True, parents=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="synthetic-test-", dir=base)
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name)
        self.config = ROOT / "configs/bigru_v1.json"

        def rows(corpus, mode, config):
            labels = config["labels"][corpus]
            train = [{"id": f"train-{label}-{i}", "group_id": f"tg-{label}-{i}", "text_raw": f"fictif {label} {label}", "label": label}
                     for label in labels for i in range(2)]
            validation = [{"id": f"val-{label}", "group_id": f"vg-{label}", "text_raw": f"fictif {label}", "label": label} for label in labels]
            return train, validation
        for target in ("run_bigru.partitions", "verify_bigru.partitions"):
            mocked = patch(target, side_effect=rows)
            mocked.start()
            self.addCleanup(mocked.stop)

    def test_end_to_end_smoke_and_reject_corruption(self):
        metadata = run(self.config, "smoke", self.path / "smoke")
        self.assertEqual(len(metadata["results"]), 2)
        verified = verify(self.path / "smoke")
        self.assertFalse(verified["test_loaded"])
        file = self.path / "smoke/results.json"
        data = json.loads(file.read_text())
        data["results"][0]["metrics"]["macro_f1"] = -1
        file.write_text(json.dumps(data))
        with self.assertRaises(ValueError):
            verify(self.path / "smoke")

    def test_full_requires_prior_smoke_and_overwrite_refused(self):
        with self.assertRaises(ValueError):
            run(self.config, "full", self.path / "full")
        (self.path / "exists").mkdir()
        with self.assertRaises(ValueError):
            run(self.config, "smoke", self.path / "exists")


if __name__ == "__main__":
    unittest.main()
