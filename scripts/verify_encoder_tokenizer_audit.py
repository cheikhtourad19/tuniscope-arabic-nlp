"""Replay the local audit and verify aggregate results without loading weights."""
import argparse
import importlib.metadata
import json
from pathlib import Path

from encoder_tokenizer_audit import audit, readable_summary
from fetch_camelbert_tokenizer import digest
from load_working_data import ROOT


def verify(directory, report=None):
    result = json.loads((directory / "summary.json").read_text())
    for package, version in result["packages"].items():
        if importlib.metadata.version(package) != version:
            raise ValueError("Tokenizer package version changed")
    for path, key in [("environments/tokenizer/uv.lock", "environment_lock_sha256"),
                      ("uv.lock", "classical_lock_sha256"), ("environments/bigru/uv.lock", "bigru_lock_sha256")]:
        if digest(ROOT / path) != result[key]:
            raise ValueError("Environment lock changed")
    for corpus, checksum in result["manifests_sha256"].items():
        if digest(ROOT / f"data/manifests/v2/{corpus}.csv") != checksum:
            raise ValueError("Manifest changed")
    recomputed = audit(result["config"])
    if any(result.get(key) != value for key, value in recomputed.items()):
        raise ValueError("Replayed tokenizer audit differs from saved results")
    if result["test_loaded"] is not False or result["weights_downloaded"] is not False:
        raise ValueError("Unexpected test/weight access")
    if report:
        if report.exists() or (ROOT / "reports").resolve() not in report.resolve().parents:
            raise ValueError("Choose a fresh directory under reports/")
        report.mkdir(parents=True)
        text = ("# Audit du tokenizer CAMeLBERT Mix\n\n"
                f"Révision : `{result['revision']}`. Texte raw-v0, manifestes v2.\n\n"
                "```text\n" + readable_summary(result) + "\n```\n\n"
                "L'audit complet a été rejoué depuis les fichiers locaux et manifestes vérifiés. "
                "Les petits groupes restent signalés ; aucune ligne protégée n'est publiée. "
                "Les seuls textes visibles dans summary.json sont des probes explicitement inventés. "
                "Les paramètres sont calculés depuis la configuration, sans chargement des poids.\n")
        (report / "README.md").write_text(text, encoding="utf-8")
        (report / "summary.json").write_text(json.dumps({**result, "source_audit_sha256": digest(directory / "summary.json")},
                                                        ensure_ascii=False, indent=2) + "\n")
    print("Audit rejoué et vérifié : mêmes statistiques, même longueur proposée ; aucun poids ni entraînement.")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    verify(args.directory, args.report)
