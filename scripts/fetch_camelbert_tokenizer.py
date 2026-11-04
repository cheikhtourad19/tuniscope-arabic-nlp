"""Fetch five pinned public text files only; never fetch a model checkpoint."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-mix"
REVISION = "9be352797bdf28a9ae21e2ae582aaaca7abdb22d"
FILES = ("README.md", "config.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
ASSET_DIR = ROOT / ".tools/tokenizers" / f"camelbert_mix_{REVISION}"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_assets(directory=ASSET_DIR):
    manifest = json.loads((directory / "source.json").read_text())
    if manifest["model_id"] != MODEL_ID or manifest["revision"] != REVISION or set(manifest["files"]) != set(FILES):
        raise ValueError("Unexpected tokenizer source manifest")
    actual = {p.name for p in directory.iterdir() if p.is_file()}
    if actual != {*FILES, "source.json"}:
        raise ValueError("Unexpected files in tokenizer-only cache")
    for name, info in manifest["files"].items():
        if digest(directory / name) != info["sha256"]:
            raise ValueError(f"Tokenizer source checksum mismatch: {name}")
    return manifest


def fetch(directory=ASSET_DIR):
    if directory.resolve() != ASSET_DIR.resolve():
        raise ValueError("Use the pinned tokenizer-only cache directory")
    if (directory / "source.json").exists():
        return verify_assets(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest = {"model_id": MODEL_ID, "revision": REVISION,
                "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
                "weights_downloaded": False, "files": {}}
    for filename in FILES:
        url = f"https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{filename}"
        with urllib.request.urlopen(url, timeout=30) as response:
            contents = response.read(1024 * 1024 + 1)
        if len(contents) > 1024 * 1024:
            raise ValueError("Unexpectedly large tokenizer metadata file")
        contents.decode("utf-8")
        path = directory / filename
        path.write_bytes(contents)
        manifest["files"][filename] = {"url": url, "bytes": len(contents), "sha256": digest(path)}
    header = (directory / "README.md").read_text().split("---", 2)[1]
    license_match = re.search(r"^license:\s*(\S+)\s*$", header, flags=re.MULTILINE)
    if not license_match or license_match.group(1) != "apache-2.0":
        raise ValueError("Model-card licence declaration differs from expectation")
    manifest["model_license_declared"] = license_match.group(1)
    (directory / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return verify_assets(directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicitly fetch the five small public text files")
    args = parser.parse_args()
    result = fetch() if args.download else verify_assets()
    print(json.dumps({"directory": str(ASSET_DIR), "model_id": MODEL_ID, "revision": REVISION,
                      "files": len(result["files"]), "bytes": sum(f["bytes"] for f in result["files"].values()),
                      "model_license_declared": result["model_license_declared"], "weights_downloaded": False}, indent=2))
