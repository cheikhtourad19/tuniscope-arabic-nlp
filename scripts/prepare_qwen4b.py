"""Prepare or verify the separately pinned Qwen3-4B upstream snapshot."""

import argparse
from hashlib import file_digest
import json
from pathlib import Path
import re
from time import monotonic
from urllib.error import HTTPError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "configs/qwen3_4b_transition_v1.json"
METADATA_FILES = {
    "LICENSE", "README.md", "config.json", "generation_config.json", "merges.txt",
    "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
}


def digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def load_plan():
    plan = json.loads(PLAN.read_text())
    model = plan["model"]
    shards = {f"model-0000{i}-of-00003.safetensors" for i in range(1, 4)}
    if (model["id"] != "Qwen/Qwen3-4B" or model["license"] != "apache-2.0"
            or not re.fullmatch(r"[0-9a-f]{40}", model["revision"])
            or set(model["weights_sha256"]) != shards
            or any(not re.fullmatch(r"[0-9a-f]{64}", sha) for sha in model["weights_sha256"].values())):
        raise ValueError("Invalid Qwen3-4B snapshot plan")
    comparison = plan["comparison"]
    if (comparison["split"] != "validation" or comparison["manifest_version"] != "v2"
            or comparison["prompt_modes"] != ["few_shot"]
            or comparison["shared_adapter_count"] != 1 or comparison["hyperparameter_search"]
            or not comparison["same_quantized_base_before_after"]):
        raise ValueError("Invalid comparison plan")
    return plan


def check_files(directory, model):
    files = METADATA_FILES | model["weights_sha256"].keys()
    config = json.loads((directory / "config.json").read_text())
    if config.get("model_type") != "qwen3" or config.get("architectures") != ["Qwen3ForCausalLM"]:
        raise ValueError("Unexpected Qwen configuration")
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    if set(index["weight_map"].values()) != set(model["weights_sha256"]):
        raise ValueError("The weight index does not refer to the three pinned shards")
    if "Apache License" not in (directory / "LICENSE").read_text():
        raise ValueError("Missing Apache license")
    report = {}
    for name in sorted(files):
        file = directory / name
        sha = digest(file)
        expected = model["weights_sha256"].get(name)
        if expected is not None and sha != expected:
            raise ValueError(f"Upstream weight digest mismatch for {name}")
        report[name] = {"sha256": sha, "bytes": file.stat().st_size}
    return report


def transfer(url, destination, expected_digest):
    name = destination.name
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if expected_digest and partial.is_file() else 0
    headers = {"Range": f"bytes={offset}-"} if offset else {}
    action = f"Reprise à {offset / 1e6:,.1f} Mo" if offset else "Téléchargement"
    print(f"{action} de {name}", flush=True)
    try:
        source = urlopen(Request(url, headers=headers), timeout=90)
    except HTTPError as error:
        if error.code != 416 or not offset or digest(partial) != expected_digest:
            raise
        partial.replace(destination)
        print(f"  {name} déjà complet et vérifié ({offset / 1e6:,.1f} Mo)", flush=True)
        return

    with source:
        expected_bytes = int(source.headers.get("Content-Length") or 0)
        if offset:
            content_range = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", source.headers.get("Content-Range", ""))
            if source.status != 206 or content_range is None:
                raise ValueError(f"Reprise HTTP refusée pour {name}. Le fichier partiel est conservé.")
            start, end, total = map(int, content_range.groups())
            if start != offset or end != total - 1 or total <= offset:
                raise ValueError(f"Plage HTTP inattendue pour {name}. Le fichier partiel est conservé.")
            expected_bytes = total
            print(f"  Reprise HTTP acceptée. Reste {(total - offset) / 1e6:,.1f} Mo.", flush=True)

        received = offset
        last_update = monotonic()
        with partial.open("ab" if offset else "wb") as target:
            while chunk := source.read(4 * 1024 * 1024):
                target.write(chunk)
                received += len(chunk)
                now = monotonic()
                if now - last_update >= 5:
                    size = f"{received / 1e6:,.0f} Mo"
                    if expected_bytes:
                        size += f" / {expected_bytes / 1e6:,.0f} Mo ({received / expected_bytes:.1%})"
                    print(f"  {name} : {size}", flush=True)
                    last_update = now

    if expected_bytes and received != expected_bytes:
        raise ValueError(f"Téléchargement incomplet pour {name} ({received}/{expected_bytes} octets).")
    if expected_digest is not None and digest(partial) != expected_digest:
        raise ValueError(f"Upstream weight digest mismatch for {name}")
    partial.replace(destination)
    print(f"  {name} reçu ({received / 1e6:,.1f} Mo)", flush=True)


def download(directory, model):
    directory.mkdir(parents=True, exist_ok=True)
    url = f'https://huggingface.co/{model["id"]}/resolve/{model["revision"]}'
    for name in sorted(METADATA_FILES | model["weights_sha256"].keys()):
        destination = directory / name
        if not destination.exists():
            transfer(f"{url}/{name}", destination, model["weights_sha256"].get(name))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("action", choices=("plan", "download", "verify"), default="plan", nargs="?")
    args = cli.parse_args()
    plan = load_plan()
    model = plan["model"]
    directory = ROOT / ".tools/llms" / f'qwen3_4b_{model["revision"]}'
    receipt = directory / "qwen_source.json"
    info = {"model_id": model["id"], "revision": model["revision"], "cache": str(directory),
            "scope": "upstream-weights", "corpus_read": False, "training_started": False}
    if args.action == "plan":
        print(json.dumps({**info, "planned_adapters": plan["comparison"]["shared_adapter_count"],
                          "has_local_receipt": receipt.is_file()}, indent=2))
        return
    if args.action == "download" and not receipt.exists():
        download(directory, model)
    if args.action == "verify" and not receipt.is_file():
        cli.error("Aucun cache 4B vérifié. Lancez d’abord l’action download avec un accès réseau.")
    info["files"] = check_files(directory, model)
    if not receipt.exists() and args.action == "download":
        temporary = receipt.with_suffix(".json.part")
        temporary.write_text(json.dumps(info, indent=2) + "\n")
        temporary.replace(receipt)
    if info["files"] != json.loads(receipt.read_text())["files"]:
        raise ValueError("Local snapshot receipt mismatch")
    print("Cache Qwen3-4B vérifié.", flush=True)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
