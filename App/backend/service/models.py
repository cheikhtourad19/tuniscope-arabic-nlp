"""Load trusted local sklearn pipelines once; never import the training runners."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings

import joblib
from sklearn.exceptions import InconsistentVersionWarning

# The same normalizer is used by training and inference. Docker copies this file.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from normalize_text import normalize  # noqa: E402

TASKS = {"sentiment": ("tsac", {"NEG", "POS"}), "dialect": ("madar", {"MSA", "TN", "EG"})}
VARIANTS = {"raw-v0", "norm-nfc-space-v1"}
SYSTEMS = {"tfidf_char", "tfidf_word", "tfidf_word_char"}


class ModelStore:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.models = {}
        self.entries = []
        self.error = None
        self.source_run = None
        self.neural = {}
        self.neural_error = None

    def load(self):
        self.models.clear()
        self.entries.clear()
        self.error = None
        try:
            data = json.loads((self.directory / "results.json").read_text())
            if (data.get("mode") != "full" or data.get("test_loaded") is not False
                    or data.get("config", {}).get("manifest_version") != "v2"):
                raise ValueError("Le bundle doit provenir d'un run full v2 de validation.")
            for package in ("scikit-learn", "numpy", "scipy", "joblib", "threadpoolctl"):
                if data.get("packages", {}).get(package) != importlib.metadata.version(package):
                    raise ValueError(f"Version incompatible : {package}. Reconstruire l'environnement du modèle.")
            self.source_run = data.get("source_run", "run-local-v2")
            for result in data["results"]:
                if result["system"] not in SYSTEMS:
                    continue
                task = next((t for t, (c, _) in TASKS.items() if c == result["corpus"]), None)
                if task is None or result["variant"] not in VARIANTS:
                    raise ValueError("Tâche ou prétraitement non pris en charge.")
                model_id = f'{result["system"]}__{result["variant"]}'
                if (task, model_id) in {(e["task"], e["id"]) for e in self.entries}:
                    raise ValueError("Identifiant de modèle en double.")
                entry = {"id": model_id, "task": task, "system": result["system"],
                         "variant": result["variant"], "available": False,
                         "labels": data["config"]["labels"][result["corpus"]],
                         "checkpoint_sha256": result["checkpoint_sha256"],
                         "source_run": self.source_run, "split": "validation",
                         "train_rows": result["train_rows"], "validation_rows": result["validation_rows"],
                         "metrics": result["metrics"], "error": None}
                self.entries.append(entry)
                try:
                    path = self.directory / result["checkpoint"]
                    if Path(result["checkpoint"]).name != result["checkpoint"] or path.resolve().parent != self.directory.resolve():
                        raise ValueError("Chemin de checkpoint invalide.")
                    with path.open("rb") as stream:
                        if hashlib.file_digest(stream, "sha256").hexdigest() != result["checkpoint_sha256"]:
                            raise ValueError("Empreinte du checkpoint incorrecte.")
                    with warnings.catch_warnings():
                        warnings.simplefilter("error", InconsistentVersionWarning)
                        model = joblib.load(path)
                    if set(model.classes_) != TASKS[task][1] or set(entry["labels"]) != TASKS[task][1]:
                        raise ValueError("Labels du modèle incompatibles.")
                    if (not hasattr(model, "predict_proba") or list(model.named_steps) not in
                            (["tfidf", "classifier"], ["tfidf", "normalize", "classifier"])):
                        raise ValueError("Pipeline TF-IDF attendu.")
                    # Loading and a fictitious forward pass finish before startup.
                    model.predict(["exemple fictif اختبار"])
                    self.models[task, model_id] = model
                    entry["available"] = True
                except Exception as exc:
                    entry["error"] = f"Chargement impossible ({type(exc).__name__}). Vérifier le bundle et ses empreintes."
        except Exception as exc:
            self.error = str(exc) if isinstance(exc, ValueError) else "Bundle absent ou illisible. Préparer App/models/current puis reconstruire."
        return self

    @property
    def ready(self):
        return all(any(e["available"] and e["task"] == task for e in self.entries) for task in TASKS)

    def load_optional(self, directory):
        if not (directory / "manifest.json").is_file():
            self.neural_error = "Bundle neural absent."
            return False
        classical_count = len(self.entries)
        try:
            from .neural import load_neural
            load_neural(self, directory)
        except Exception as error:
            del self.entries[classical_count:]
            self.neural.clear()
            self.neural_error = f"Chargement neural impossible ({type(error).__name__})."
            raise
        return True

    def register_neural(self, task, system, predict, labels, metrics, run, checkpoint, train, validation,
                        seed, revision, aggregate, device):
        model_id = f"{system}__raw-v0"
        self.neural[task, model_id] = predict
        self.entries.append({"id": model_id, "task": task, "system": system, "variant": "raw-v0",
                             "available": True, "labels": labels, "checkpoint_sha256": checkpoint,
                             "source_run": run, "split": "validation", "train_rows": train,
                             "validation_rows": validation, "metrics": metrics, "error": None,
                             "seed": seed, "revision": revision, "aggregate": aggregate, "device": device,
                             "prompt_mode": "zero_shot" if system.startswith("qwen_") else None})

    def health(self):
        complete = self.ready and self.error is None and all(e["available"] for e in self.entries)
        return {"status": "ready" if complete else "degraded" if self.models or self.neural else "unavailable",
                "ready": self.ready, "version": "0.1.0", "source_run": self.source_run,
                "models": [{k: v for k, v in e.items() if k not in {"metrics", "aggregate"}} for e in self.entries],
                "error": self.error, "neural_error": self.neural_error}

    def predict(self, task, model_id, text):
        if (task, model_id) in self.neural:
            return self.neural[task, model_id](text)
        model = self.models.get((task, model_id))
        if model is None:
            raise KeyError("Modèle indisponible pour cette tâche.")
        entry = next(e for e in self.entries if e["task"] == task and e["id"] == model_id)
        text_norm = normalize(text)
        prepared = text if entry["variant"] == "raw-v0" else text_norm
        features = model[:-1].transform([prepared])
        if features.nnz == 0:
            return {"status": "abstained", "label": None, "scores": {}, "reason": "Aucun indice connu dans le vocabulaire du modèle."}
        scores = {str(label): float(value) for label, value in zip(model.classes_, model.predict_proba([prepared])[0])}
        return {"status": "predicted", "label": str(model.predict([prepared])[0]), "scores": scores, "reason": None}
