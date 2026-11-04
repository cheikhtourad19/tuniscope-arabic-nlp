"""Notebook bridge to the tokenizer-only environment; no model runtime imports."""
import json
import os
from pathlib import Path
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]


def run_tool(script, *arguments):
    if script not in {"run_encoder_tokenizer_audit.py", "verify_encoder_tokenizer_audit.py"}:
        raise ValueError("Expected a tokenizer-audit script")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    command = ["sh", str(ROOT / "scripts/tokenizer_uv.sh"), "run", "--locked", "--offline", "python",
               str(ROOT / "scripts" / script), *map(str, arguments)]
    result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
    if result.returncode:
        print(result.stdout, end="")
        print(result.stderr, end="")
        raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
    return result.stdout


def notebook_audit(mode, replay_directory=None):
    if mode not in {"audit", "replay"}:
        raise ValueError("Choose audit or replay")
    if mode == "audit":
        directory = ROOT / "outputs/tokenizer_audits/notebooks" / uuid.uuid4().hex
        print("Audit réel de tout le train/validation v2 ; aucun entraînement.", flush=True)
        run_tool("run_encoder_tokenizer_audit.py", "--output", directory)
    else:
        directory = Path(replay_directory)
        if not directory.is_absolute():
            directory = ROOT / directory
        print("Relecture et vérification de l'audit explicitement choisi.", flush=True)
    directory = directory.resolve()
    if (ROOT / "outputs/tokenizer_audits").resolve() not in directory.parents:
        raise ValueError("Choose a tokenizer audit under outputs/tokenizer_audits/")
    run_tool("verify_encoder_tokenizer_audit.py", directory)
    result = json.loads((directory / "summary.json").read_text())
    if result["test_loaded"] is not False or result["weights_downloaded"] is not False:
        raise ValueError("Unexpected test/weight access")
    return directory, result
