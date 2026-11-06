"""Small invented byte files exercise official SHA256 and Git-blob verification."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from fetch_qwen3 import load_plan, verify_file


class QwenAssetsTests(unittest.TestCase):
    def test_pinned_official_scope_without_training_or_quantization(self):
        p = load_plan()
        self.assertEqual(p['model_id'], 'Qwen/Qwen3-0.6B')
        self.assertEqual(len(p['files']), 9)
        self.assertIn('model.safetensors', p['files'])
        self.assertFalse(p['enable_thinking'])
        self.assertFalse(p['training_started'])
        self.assertFalse(p['corpus_read'])

    def test_both_digest_types_detect_same_size_corruption(self):
        contents = 'invented fixture é\n'.encode()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'example'
            for key, expected in [
                ('sha256', hashlib.sha256(contents).hexdigest()),
                ('git_blob_sha1', hashlib.sha1(f'blob {len(contents)}\0'.encode() + contents).hexdigest())]:
                info = {'bytes': len(contents), key: expected}; path.write_bytes(contents)
                self.assertEqual(verify_file(path, info)['sha256'], hashlib.sha256(contents).hexdigest())
                path.write_bytes(b'X' + contents[1:])
                with self.assertRaises(ValueError): verify_file(path, info)
                path.write_bytes(contents + b'X')
                with self.assertRaises(ValueError): verify_file(path, info)


if __name__ == '__main__':
    unittest.main()
