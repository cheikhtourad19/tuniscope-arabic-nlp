"""Functional inference contracts on invented examples, without protected data."""
import hashlib
import importlib.metadata
import json
from unittest.mock import patch

from fastapi.testclient import TestClient
import joblib
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from App.backend.service.main import create_app
from App.backend.service.models import ModelStore


@pytest.fixture
def bundle(tmp_path):
    metadata = {"mode": "full", "test_loaded": False, "source_run": "synthetic-fixture",
                "config": {"manifest_version": "v2", "labels": {"tsac": ["NEG", "POS"], "madar": ["MSA", "TN", "EG"]}},
                "packages": {p: importlib.metadata.version(p) for p in ("scikit-learn", "joblib", "numpy", "scipy", "threadpoolctl")}, "results": []}
    for corpus, task_labels in metadata["config"]["labels"].items():
        texts = [f"fictif classe {label} exemple {i}" for label in task_labels for i in range(3)]
        labels = [label for label in task_labels for _ in range(3)]
        model = Pipeline([("tfidf", TfidfVectorizer(analyzer="char", ngram_range=(2, 3))), ("classifier", LogisticRegression())]).fit(texts, labels)
        for variant in ("raw-v0", "norm-nfc-space-v1"):
            path = tmp_path / f"{corpus}__tfidf_char__{variant}.joblib"
            joblib.dump(model, path)
            metadata["results"].append({"corpus": corpus, "system": "tfidf_char", "variant": variant,
                                        "checkpoint": path.name, "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                        "train_rows": len(texts), "validation_rows": 3,
                                        "metrics": {"macro_f1": 0.5, "accuracy": 0.5, "n": 3}})
    (tmp_path / "results.json").write_text(json.dumps(metadata))
    return tmp_path


@pytest.fixture
def client(bundle):
    with TestClient(create_app(bundle)) as client:
        yield client


def test_load_once_and_no_training_during_requests(client):
    assert client.get("/health").json()["ready"] is True
    with patch("joblib.load", side_effect=AssertionError("Must not reload per request")), patch.object(Pipeline, "fit", side_effect=AssertionError("Must never train")):
        for task in ("sentiment", "dialect"):
            response = client.post("/predict", json={"text": "fictif classe POS exemple", "task": task})
            assert response.status_code == 200
            result = response.json()
            assert result["status"] == "predicted"
            assert result["scores_calibrated"] is False
            assert sum(result["scores"].values()) == pytest.approx(1)
    assert client.get("/ready").status_code == 200


def test_missing_bundle_explicit_unavailable(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/health").json()["status"] == "unavailable"
        assert client.get("/ready").status_code == 503
        assert client.post("/predict", json={"text": "texte fictif", "task": "sentiment"}).status_code == 503


def test_strict_startup_rejects_missing_models(tmp_path, monkeypatch):
    monkeypatch.setenv("TUNISCOPE_REQUIRE_MODELS", "true")
    with pytest.raises(RuntimeError), TestClient(create_app(tmp_path)):
        pass


def test_raw_and_normalization_preserved(client):
    text = "  fictif   classe\nPOS  "
    result = client.post("/predict", json={"text": text, "task": "sentiment", "model": "tfidf_char__norm-nfc-space-v1"}).json()
    assert result["text_raw"] == text
    assert result["text_norm"] == "fictif classe POS"
    assert result["versions"]["normalization"] == "norm-nfc-space-v1"


@pytest.mark.parametrize("text", ["", " \n ", "a" * 2001, "texte\x00"])
def test_invalid_text_rejected_without_truncation(client, text):
    assert client.post("/predict", json={"text": text, "task": "sentiment"}).status_code == 422


def test_bad_task_and_unavailable_model(client):
    assert client.post("/predict", json={"text": "test", "task": "invented"}).status_code == 422
    assert client.post("/predict", json={"text": "test", "task": "sentiment", "model": "encoder"}).status_code == 503


def test_unknown_vocabulary_abstains(client):
    result = client.post("/predict", json={"text": "🦊", "task": "sentiment"}).json()
    assert result["status"] == "abstained" and result["label"] is None and result["scores"] == {}


def test_comparison_keeps_same_input_and_id(client):
    results = client.post("/compare", json={"text": "fictif classe POS", "task": "sentiment", "id": "demo"}).json()["results"]
    assert len(results) == 2
    assert {r["id"] for r in results} == {"demo"}
    assert len({r["text_raw"] for r in results}) == 1


def test_csv_preserves_all_rows_and_row_errors(client):
    csv = '\ufeffid,text\r\na,"fictif classe POS"\r\nb,\r\na,doublon\r\nc,"fictif\nclasse NEG"\r\n'
    response = client.post("/batch", data={"task": "sentiment"}, files={"file": ("demo.csv", csv.encode(), "text/csv")})
    assert response.status_code == 200
    data = response.json()
    assert data["rows"] == 4 and data["errors"] == 2
    assert [r["id"] for r in data["results"]] == ["a", "b", "a", "c"]
    assert data["results"][-1]["text_raw"] == "fictif\nclasse NEG"


@pytest.mark.parametrize("contents", [b'id,text\na,\xff', b'other,columns\nx,y', b'id,text\n', b'id,text\n' + b'x,hello\n' * 101])
def test_invalid_csv_or_row_limit(client, contents):
    assert client.post("/batch", data={"task": "sentiment"}, files={"file": ("bad.csv", contents)}).status_code == 422


def test_oversized_csv(client):
    assert client.post("/batch", data={"task": "sentiment"}, files={"file": ("big.csv", b'x' * (1024 * 1024 + 1))}).status_code == 413


def test_metrics_keep_validation_boundary(client):
    metrics = client.get("/metrics").json()
    assert metrics["split"] == "validation" and metrics["test_evaluated"] is False
    assert len(metrics["results"]) == 4


def test_documentary_citation_and_abstention(client):
    known = client.post("/ask", json={"question": "Que signifie macro-F1 et comment calcule-t-on la moyenne des F1 des classes ?",
                                     "method": "bm25", "lang": "fr"}).json()
    assert known["status"] == "answered" and known["source_ids"] == ["macro-f1.fr"]
    source = client.get(known["sources"][0]["links"][0]["url"].removeprefix("/api"))
    assert source.status_code == 200 and source.headers["content-type"].startswith("text/plain")
    assert client.get("/documentation/AGENTS.md").status_code == 404
    unknown = client.post("/ask", json={"question": "Quel est le macro-F1 de TuniScope sur un corpus marocain indépendant ?"}).json()
    assert unknown["status"] == "insufficient_sources" and not unknown["sources"]
    assert client.post("/ask", json={"question": "test", "method": "e5"}).json()["status"] == "unavailable"


def test_compare_keeps_available_results_when_one_model_fails(client, monkeypatch):
    original = ModelStore.predict

    def predict(store, task, model, text):
        if model.endswith("raw-v0"):
            raise RuntimeError("fictitious failure")
        return original(store, task, model, text)

    monkeypatch.setattr(ModelStore, "predict", predict)
    response = client.post("/compare", json={"text": "fictif classe POS", "task": "sentiment", "id": "demo"})
    assert response.status_code == 200
    rows = response.json()["results"]
    assert {row["status"] for row in rows} == {"error", "predicted"}
    assert {row["id"] for row in rows} == {"demo"}


def test_corrupted_checkpoint_never_loaded(bundle):
    path = next(bundle.glob("tsac*raw-v0.joblib"))
    path.write_bytes(b"invalid")
    store = ModelStore(bundle).load()
    assert ("sentiment", "tfidf_char__raw-v0") not in store.models
    assert not next(e for e in store.entries if e["task"] == "sentiment" and e["variant"] == "raw-v0")["available"]


def test_incompatible_dependencies_rejected(bundle):
    path = bundle / "results.json"
    metadata = json.loads(path.read_text())
    metadata["packages"]["scikit-learn"] = "0.0.0"
    path.write_text(json.dumps(metadata))
    store = ModelStore(bundle).load()
    assert not store.models and "Version incompatible" in store.error


def test_checkpoint_path_cannot_escape_bundle(bundle):
    path = bundle / "results.json"
    metadata = json.loads(path.read_text())
    metadata["results"][0]["checkpoint"] = "../external.joblib"
    path.write_text(json.dumps(metadata))
    with patch("joblib.load", wraps=joblib.load) as loader:
        store = ModelStore(bundle).load()
        assert loader.call_count == 3
        assert not store.entries[0]["available"]


def test_arabic_and_latin_are_writing_presence_not_language(client):
    for text, script in [("مرحبا", "AR"), ("bonjour", "LATIN"), ("مرحبا bonjour", "MIXED"), ("🙂", "UNKNOWN")]:
        result = client.post("/predict", json={"text": text, "task": "sentiment"}).json()
        assert result["script"] == script


def test_prepare_models_copies_only_checkpoints_and_aggregates(bundle, tmp_path, monkeypatch):
    from App import prepare_models
    monkeypatch.setattr(prepare_models, "ROOT", tmp_path)
    (bundle / "protected-input.txt").write_text("must never be copied")
    destination = tmp_path / "App/models/current"
    prepare_models.prepare(bundle, destination)
    assert {p.suffix for p in destination.iterdir()} == {".joblib", ".json"}
    assert len(list(destination.iterdir())) == 5
    with pytest.raises(ValueError):
        prepare_models.prepare(bundle, destination)
    prepare_models.prepare(bundle, destination, replace=True)
    assert (bundle / "protected-input.txt").exists()
    with pytest.raises(ValueError):
        prepare_models.prepare(bundle, tmp_path, replace=True)


def test_failed_replacement_preserves_current_bundle(bundle, tmp_path, monkeypatch):
    from App import prepare_models
    monkeypatch.setattr(prepare_models, "ROOT", tmp_path)
    destination = tmp_path / "App/models/current"
    prepare_models.prepare(bundle, destination)
    original = (destination / "results.json").read_bytes()
    next(bundle.glob("*.joblib")).write_bytes(b"corrupted")
    with pytest.raises(ValueError):
        prepare_models.prepare(bundle, destination, replace=True)
    assert (destination / "results.json").read_bytes() == original
