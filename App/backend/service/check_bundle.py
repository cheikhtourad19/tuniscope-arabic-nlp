"""Build-time check of the model bundle and documentary engines."""
from .main import load_services

if __name__ == "__main__":
    store, documents = load_services()
    print(f"Bundle contrôlé : {len(store.models)} TF-IDF, {len(store.neural)} entrées neurales ; {documents.status()}")
