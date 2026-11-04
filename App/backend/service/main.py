"""FastAPI application: text/CSV inference and saved validation metrics."""
from contextlib import asynccontextmanager
import csv
import io
import os
from typing import Literal
import unicodedata
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, StrictStr

from .models import ModelStore, ROOT, normalize

MAX_CHARS = 2000
MAX_BYTES = 1024 * 1024


def writing_system(text):
    # Same Unicode-presence rule as audit_text_composition, without corpus imports.
    names = [unicodedata.name(c, "") for c in text if unicodedata.category(c).startswith("L")]
    ar, latin = any("ARABIC" in n for n in names), any("LATIN" in n for n in names)
    return "MIXED" if ar and latin else "AR" if ar else "LATIN" if latin else "UNKNOWN"


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: StrictStr
    task: Literal["sentiment", "dialect"]
    model: StrictStr = "tfidf_char__raw-v0"
    id: StrictStr | None = None


class AskRequest(BaseModel):
    question: StrictStr


def validate_text(text):
    if not text.strip():
        raise HTTPException(422, "Saisir un texte non vide.")
    if len(text) > MAX_CHARS:
        raise HTTPException(422, "Le texte dépasse 2 000 caractères. Aucune troncature automatique.")
    if "\x00" in text:
        raise HTTPException(422, "Le texte contient un caractère nul non pris en charge.")


def infer(store, request):
    validate_text(request.text)
    try:
        result = store.predict(request.task, request.model, request.text)
    except KeyError:
        raise HTTPException(503, "Modèle indisponible. Vérifier /health et reconstruire après préparation du bundle.")
    except Exception:
        raise HTTPException(503, "L'inférence a échoué. Vérifier le modèle chargé.")
    entry = next(e for e in store.entries if e["task"] == request.task and e["id"] == request.model)
    scope = ("Sentiment évalué sur des commentaires tunisiens POS/NEG ; neutre, ironie et hors domaine non validés."
             if request.task == "sentiment" else "MSA, Tunis/Sfax et arabe du Caire, domaine MADAR du voyage ; couverture limitée.")
    script = writing_system(request.text)
    return {"id": request.id or str(uuid.uuid4()), "task": request.task, "model": request.model,
            "text_raw": request.text, "text_norm": normalize(request.text), "script": script,
            **result, "scores_calibrated": False,
            "warnings": [scope, "Les scores ne sont pas des probabilités de correction calibrées."] +
                        (["L'écriture latine ne permet pas de distinguer français, anglais et Arabizi."] if script in {"LATIN", "MIXED"} else []),
            "versions": {"app": "0.1.0", "source_run": entry["source_run"], "normalization": entry["variant"],
                         "checkpoint_sha256": entry["checkpoint_sha256"], "manifest": "v2"}}


def create_app(model_dir=None):
    @asynccontextmanager
    async def lifespan(app):
        # Complete model loading before Uvicorn accepts requests.
        app.state.store = ModelStore(model_dir or os.environ.get("TUNISCOPE_MODEL_DIR", ROOT / "App/models/current")).load()
        if os.environ.get("TUNISCOPE_REQUIRE_MODELS", "false").lower() == "true" and (
                not app.state.store.ready or not all(e["available"] for e in app.state.store.entries)):
            raise RuntimeError("Les modèles requis sont absents ou incompatibles. Préparer le bundle avant le démarrage.")
        yield
        app.state.store.models.clear()

    app = FastAPI(title="TuniScope — Inférence", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        return app.state.store.health()

    @app.get("/ready")
    def ready():
        return JSONResponse({"ready": app.state.store.ready}, status_code=200 if app.state.store.ready else 503)

    @app.get("/models")
    def models():
        return {"models": app.state.store.health()["models"]}

    @app.post("/predict")
    def predict(request: PredictRequest):
        return infer(app.state.store, request)

    @app.post("/compare")
    def compare(request: PredictRequest):
        validate_text(request.text)
        identifier = request.id or str(uuid.uuid4())
        candidates = [e for e in app.state.store.entries if e["task"] == request.task and e["available"]]
        if not candidates:
            raise HTTPException(503, "Aucun modèle disponible pour cette tâche.")
        return {"id": identifier, "results": [infer(app.state.store, request.model_copy(update={"model": e["id"], "id": identifier})) for e in candidates]}

    @app.post("/batch")
    async def batch(file: UploadFile = File(...), task: Literal["sentiment", "dialect"] = Form(...),
                    model: str = Form("tfidf_char__raw-v0")):
        contents = await file.read(MAX_BYTES + 1)
        await file.close()
        if len(contents) > MAX_BYTES:
            raise HTTPException(413, "CSV limité à 1 Mo.")
        try:
            decoded = contents.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise HTTPException(422, "Encodage invalide : fournir un CSV UTF-8.")
        reader = csv.DictReader(io.StringIO(decoded, newline=""), strict=True)
        try:
            if not reader.fieldnames or not {"id", "text"} <= set(reader.fieldnames) or len(set(reader.fieldnames)) != len(reader.fieldnames):
                raise HTTPException(422, "Le CSV doit contenir les colonnes uniques id,text (séparateur virgule).")
            rows = []
            for row in reader:
                rows.append(row)
                if len(rows) > 100:
                    raise HTTPException(422, "Le CSV dépasse la limite de 100 lignes.")
        except csv.Error:
            raise HTTPException(422, "Structure CSV invalide.")
        if not rows:
            raise HTTPException(422, "Le CSV ne contient aucune entrée.")
        results, seen = [], set()
        for index, row in enumerate(rows, start=1):
            identifier = row.get("id") or ""
            error = None
            if not identifier.strip() or identifier in seen:
                error = "Identifiant vide ou en double."
            elif None in row or row.get("text") is None:
                error = "Nombre de colonnes incorrect."
            seen.add(identifier)
            if not error:
                try:
                    prediction = await run_in_threadpool(infer, app.state.store, PredictRequest(id=identifier, text=row["text"], task=task, model=model))
                    results.append({"row": index, **prediction})
                    continue
                except HTTPException as exc:
                    error = exc.detail
            results.append({"row": index, "id": identifier, "task": task, "model": model,
                            "text_raw": row.get("text"), "status": "error", "label": None, "error": error,
                            "versions": {"app": "0.1.0", "source_run": app.state.store.source_run}})
        return {"rows": len(results), "errors": sum(r["status"] == "error" for r in results), "results": results}

    @app.get("/metrics")
    def metrics():
        return {"split": "validation", "test_evaluated": False,
                "notice": "Résultats sauvegardés du run source ; ce ne sont pas les scores des textes saisis.",
                "results": [{k: v for k, v in entry.items() if k != "error"} for entry in app.state.store.entries]}

    @app.post("/ask")
    def ask(request: AskRequest):
        validate_text(request.question)
        return {"status": "not_implemented", "answer": None, "source_ids": [], "excerpts": [],
                "reason": "Le module documentaire RAG n'est pas encore implémenté."}

    return app


app = create_app()
