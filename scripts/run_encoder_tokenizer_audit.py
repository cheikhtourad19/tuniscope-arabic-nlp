"""Run the frozen tokenizer-only audit and export aggregate provenance."""
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess

from encoder_tokenizer_audit import audit, readable_summary
from fetch_camelbert_tokenizer import ASSET_DIR, digest
from load_working_data import ROOT


def run(config_path, output):
    output = output.resolve()
    if (ROOT / "outputs/tokenizer_audits").resolve() not in output.parents or output.exists():
        raise ValueError("Choose a new directory under outputs/tokenizer_audits/")
    config = json.loads(config_path.read_text())
    result = audit(config)
    result.update({"config": config, "config_sha256": digest(config_path),
                   "created_at_utc": datetime.now(timezone.utc).isoformat(), "python": platform.python_version(),
                   "packages": {n: importlib.metadata.version(n) for n in ("transformers", "tokenizers", "huggingface-hub")},
                   "environment_lock_sha256": digest(ROOT / "environments/tokenizer/uv.lock"),
                   "classical_lock_sha256": digest(ROOT / "uv.lock"),
                   "bigru_lock_sha256": digest(ROOT / "environments/bigru/uv.lock"),
                   "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                   "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
                   "protocol_sha256": digest(ROOT / "docs/42_protocole_audit_tokenizer_encodeur.md"),
                   "code_sha256": {name: digest(ROOT / "scripts" / name) for name in (
                       "encoder_tokenizer_audit.py", "run_encoder_tokenizer_audit.py", "verify_encoder_tokenizer_audit.py")},
                   "manifests_sha256": {c: digest(ROOT / f"data/manifests/v2/{c}.csv") for c in config["corpora"]}})
    output.mkdir(parents=True)
    (output / "summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "summary.txt").write_text(readable_summary(result) + "\n", encoding="utf-8")
    print(readable_summary(result))
    print(f"Rapport agrégé : {output}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/camelbert_tokenizer_audit_v1.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
