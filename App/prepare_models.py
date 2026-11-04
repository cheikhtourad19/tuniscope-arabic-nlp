"""Copy an explicit full v2 TF-IDF run into the private Docker model bundle."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def prepare(run, destination, system="tfidf_char", replace=False):
    run, destination = Path(run).resolve(), Path(destination).resolve()
    # An accidental --output must never remove an archived run or arbitrary folder.
    allowed = (ROOT / "App/models").resolve()
    if destination.parent != allowed or destination.name != "current":
        raise ValueError("Le dossier de destination doit être App/models/current.")
    metadata = json.loads((run / "results.json").read_text())
    if (metadata.get("mode") != "full" or metadata.get("test_loaded") is not False
            or metadata.get("config", {}).get("manifest_version") != "v2"):
        raise ValueError("Un entraînement full v2, sans évaluation du test, est requis.")
    selected = [r for r in metadata["results"] if r["system"] == system]
    expected = {(c, v) for c in ("tsac", "madar") for v in ("raw-v0", "norm-nfc-space-v1")}
    if len(selected) != 4 or {(r["corpus"], r["variant"]) for r in selected} != expected:
        raise ValueError("Quatre modèles sont requis : deux tâches × brut/normalisé.")
    for result in selected:
        name = result["checkpoint"]
        if Path(name).name != name or (run / name).resolve().parent != run:
            raise ValueError("Chemin de modèle invalide.")
        if digest(run / name) != result["checkpoint_sha256"]:
            raise ValueError(f"Empreinte incorrecte : {name}")
    existing = [p for p in destination.iterdir() if p.name != ".gitkeep"] if destination.exists() else []
    if existing and not replace:
        raise ValueError("Le bundle existe. Utiliser --replace pour remplacer seulement cette copie.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".staging-", dir=destination.parent) as tmp:
        stage = Path(tmp)
        for result in selected:
            shutil.copy2(run / result["checkpoint"], stage / result["checkpoint"])
        # Aggregate metadata only: neither validation inputs nor prediction IDs.
        bundle = {**metadata, "results": selected, "source_run": run.name,
                  "source_results_sha256": digest(run / "results.json")}
        (stage / "results.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
        destination.mkdir(exist_ok=True)
        for path in existing:
            if path.is_dir():
                raise ValueError("Un sous-dossier inattendu empêche le remplacement.")
        for path in existing:
            path.unlink()
        for path in stage.iterdir():
            shutil.move(str(path), destination / path.name)
    return {"bundle": str(destination), "source_run": run.name, "models": len(selected),
            "system": system, "test_evaluated": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "App/models/current")
    parser.add_argument("--system", choices=["tfidf_char", "tfidf_word", "tfidf_word_char"], default="tfidf_char")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(args.run, args.output, args.system, args.replace), indent=2))
