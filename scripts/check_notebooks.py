"""Validate our two output-free notebooks; optionally execute cells in order.

This is a standard-library sequential check, NOT a native Jupyter/kernel test.
Only run trusted repository code. Real data and training require CLI opt-in.
"""

import argparse
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ("01_donnees_v2.ipynb", "02_baselines_classiques.ipynb")


def validate(document):
    if (document.get("nbformat"), document.get("nbformat_minor")) != (4, 4):
        raise ValueError("Expected notebook format 4.4")
    if document.get("metadata", {}).get("kernelspec", {}).get("language") != "python":
        raise ValueError("Expected a Python kernelspec")
    if not document.get("cells") or not any(cell.get("cell_type") == "code"
                                          for cell in document["cells"]):
        raise ValueError("Expected code and explanatory cells")
    if not any(cell.get("cell_type") == "markdown" for cell in document["cells"]):
        raise ValueError("Missing explanations")
    for cell in document["cells"]:
        if cell.get("cell_type") not in {"code", "markdown"}:
            raise ValueError("Unexpected cell type")
        if not isinstance(cell.get("source"), list) or not all(
                isinstance(line, str) for line in cell["source"]):
            raise ValueError("Expected source line strings")
        if cell["cell_type"] == "code":
            if cell.get("outputs") != [] or cell.get("execution_count") is not None:
                raise ValueError("Clear notebook outputs before sharing")
            compile("".join(cell["source"]), "<notebook-cell>", "exec")


def execute(document, filename):
    namespace = {"__name__": "__main__"}
    for number, cell in enumerate(document["cells"], 1):
        if cell["cell_type"] == "code":
            print(f"Executing {filename}, cell {number}", flush=True)
            exec(compile("".join(cell["source"]), f"{filename}:cell-{number}", "exec"), namespace)
    return namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--data-mode", choices=("synthetic", "real"), default="synthetic")
    parser.add_argument("--baseline-mode", choices=("off", "smoke", "full"), default="off")
    parser.add_argument("--family", choices=("characters", "words", "combined"), default="characters")
    args = parser.parse_args()
    if not args.execute and (args.data_mode != "synthetic" or args.baseline_mode != "off"):
        parser.error("Real modes require --execute")
    os.chdir(ROOT)
    os.environ.update(TUNISCOPE_DATA_MODE=args.data_mode,
                      TUNISCOPE_BASELINE_MODE=args.baseline_mode,
                      TUNISCOPE_BASELINE_FAMILY=args.family)
    print("Sequential Python check only; native Jupyter rendering/kernel remains unverified.", flush=True)
    for filename in NOTEBOOKS:
        document = json.loads((ROOT / "notebooks" / filename).read_text(encoding="utf-8"))
        validate(document)
        print(f"Validated output-free structure: {filename}", flush=True)
        if args.execute:
            execute(document, filename)


if __name__ == "__main__":
    main()
