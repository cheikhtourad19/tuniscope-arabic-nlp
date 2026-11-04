"""Check the output-free BiGRU notebook; explicit modes for sequential replay."""
import argparse
import copy
import json
import os

from check_notebooks import ROOT, execute, validate


def validate_bigru(document, allow_saved_outputs=False):
    candidate = copy.deepcopy(document)
    if allow_saved_outputs:
        # Validate source in memory; never clear the user's notebook on disk.
        for cell in candidate.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []
                cell["execution_count"] = None
    validate(candidate)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--mode", choices=["off", "replay", "smoke", "full"], default="off")
    parser.add_argument("--allow-saved-outputs", action="store_true")
    args = parser.parse_args()
    if not args.execute and args.mode != "off":
        parser.error("Execution requires --execute")
    document = json.loads((ROOT / "notebooks/03_bigru.ipynb").read_text())
    validate_bigru(document, args.allow_saved_outputs)
    print("Structure vérifiée ; ce contrôle séquentiel n'est pas un test de noyau Jupyter.")
    if args.allow_saved_outputs:
        print("Sorties personnelles préservées ; seul le code est contrôlé sur une copie en mémoire.")
    if args.execute:
        os.environ["TUNISCOPE_BIGRU_MODE"] = args.mode
        os.chdir(ROOT)
        execute(document, "03_bigru.ipynb")
