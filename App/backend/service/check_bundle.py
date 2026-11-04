"""Build-time check; loads the bundle without starting an HTTP server."""
import os
from .models import ModelStore

if __name__ == "__main__":
    store = ModelStore(os.environ.get("TUNISCOPE_MODEL_DIR", "/srv/App/models/current")).load()
    strict = os.environ.get("TUNISCOPE_REQUIRE_MODELS", "true").lower() == "true"
    if strict and (not store.ready or not all(e["available"] for e in store.entries)):
        raise SystemExit(f"Bundle incomplet : {store.health()}")
    print(f"Bundle contrôlé : {len(store.models)} modèles chargés ; ready={store.ready}")
