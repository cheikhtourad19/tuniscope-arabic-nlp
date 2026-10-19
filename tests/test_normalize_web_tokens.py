"""Synthetic-only tests for candidate URL/mention masking."""

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from normalize_web_tokens import mask_web_tokens, normalize_candidate  # noqa: E402


class WebTokenMaskingTests(unittest.TestCase):
    def test_url_and_terminal_punctuation(self):
        self.assertEqual(mask_web_tokens("voir https://site.example/a?x=1، bien!"),
                         ("voir <URL>، bien!", 1, 0))
        self.assertEqual(mask_web_tokens("(www.example.org)."), ("(<URL>).", 1, 0))

    def test_mentions_but_not_email(self):
        self.assertEqual(mask_web_tokens("merci @user_7 et @مستخدم! mail a@b.com"),
                         ("merci <USER> et <USER>! mail a@b.com", 0, 2))

    def test_url_path_mention_is_not_double_counted(self):
        self.assertEqual(mask_web_tokens("https://site.example/@someone @other"),
                         ("<URL> <USER>", 1, 1))

    def test_negation_emoji_arabizi_and_code_switch_are_preserved(self):
        self.assertEqual(normalize_candidate("  moch behi 😄 3andi @ali، très lent  "),
                         "moch behi 😄 3andi <USER>، très lent")


if __name__ == "__main__":
    unittest.main()
