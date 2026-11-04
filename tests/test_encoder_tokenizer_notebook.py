"""Notebook safety checks without loading tokenizer assets or protected corpora."""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import tokenizer_notebook_support as support
from check_bigru_notebook import validate_bigru
from check_notebooks import execute

ROOT = Path(__file__).resolve().parents[1]


class EncoderTokenizerNotebookTests(unittest.TestCase):
    def test_structure_and_off_mode_have_no_subprocess_or_tokenizer_load(self):
        document = json.loads((ROOT / "notebooks/04_tokenizer_encodeur.ipynb").read_text())
        validate_bigru(document, allow_saved_outputs=True)
        with patch.dict("os.environ", {"TUNISCOPE_TOKENIZER_MODE": "off"}), patch.object(support, "run_tool", side_effect=AssertionError("unexpected child")), patch("encoder_tokenizer_audit.load_tokenizer", side_effect=AssertionError("unexpected load")):
            namespace = execute(document, "04_tokenizer_encodeur.ipynb")
        self.assertIsNone(namespace["audit_result"])

    def test_child_uses_tokenizer_profile_and_removes_kernel_virtualenv(self):
        with patch.dict("os.environ", {"VIRTUAL_ENV": "invented-kernel"}), patch.object(support.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, "ok", "")) as child:
            support.run_tool("verify_encoder_tokenizer_audit.py", "invented")
        command = child.call_args.args[0]
        self.assertTrue(command[1].endswith("scripts/tokenizer_uv.sh"))
        self.assertEqual(command[2:6], ["run", "--locked", "--offline", "python"])
        self.assertNotIn("VIRTUAL_ENV", child.call_args.kwargs["env"])

    def test_unexpected_tool_and_mode_rejected(self):
        with self.assertRaises(ValueError):
            support.run_tool("run_bigru.py")
        with self.assertRaises(ValueError):
            support.notebook_audit("full")

    def test_failed_child_is_visible(self):
        output = StringIO()
        failed = subprocess.CompletedProcess([], 1, "invented output", "invented failure")
        with patch.object(support.subprocess, "run", return_value=failed), redirect_stdout(output), self.assertRaises(subprocess.CalledProcessError):
            support.run_tool("verify_encoder_tokenizer_audit.py")
        self.assertIn("invented failure", output.getvalue())


if __name__ == "__main__":
    unittest.main()
