from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_kaggle_sources import link_directory, unique_parent


class SourceLinkTests(unittest.TestCase):
    def test_links_are_idempotent_and_never_replace_existing_directory(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); source = root / 'input'; source.mkdir()
            destination = root / 'project/original'
            link_directory(destination, source); link_directory(destination, source)
            self.assertTrue(destination.is_symlink())
            other = root / 'other'; other.mkdir()
            with self.assertRaises(FileExistsError): link_directory(destination, other)
            self.assertEqual(destination.resolve(), source.resolve())

    def test_discovery_rejects_missing_or_ambiguous_sources(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            with self.assertRaises(ValueError): unique_parent(root, 'train_pos.txt')
            a = root / 'a'; a.mkdir(); (a / 'train_pos.txt').write_text('invented')
            self.assertEqual(unique_parent(root, 'train_pos.txt'), a)
            b = root / 'b'; b.mkdir(); (b / 'train_pos.txt').write_text('invented')
            with self.assertRaises(ValueError): unique_parent(root, 'train_pos.txt')


if __name__ == '__main__': unittest.main()
