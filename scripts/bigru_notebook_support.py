"""Standard-library notebook bridge to the isolated PyTorch environment."""
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import uuid

ROOT = Path(__file__).resolve().parents[1]
ALLOWED_SCRIPTS = {"run_bigru.py", "verify_bigru.py", "plot_bigru_run.py"}


def run_neural(script, *arguments, stream=False):
    if script not in ALLOWED_SCRIPTS:
        raise ValueError("Expected an allowed BiGRU script")
    environment = os.environ.copy()
    environment.pop("VIRTUAL_ENV", None)
    environment["MPLBACKEND"] = "Agg"
    command = ["sh", str(ROOT / "scripts/bigru_uv.sh"), "run", "--locked", "--offline",
               "python", str(ROOT / "scripts" / script), *map(str, arguments)]
    if not stream:
        result = subprocess.run(command, cwd=ROOT, env=environment, text=True, capture_output=True)
        if result.returncode:
            print(result.stdout, end="")
            print(result.stderr, end="")
            raise subprocess.CalledProcessError(result.returncode, command, result.stdout, result.stderr)
        return result.stdout
    process = subprocess.Popen(command, cwd=ROOT, env=environment, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=1,
                               start_new_session=True)
    output = []
    try:
        for line in process.stdout:
            output.append(line)
            print(line, end="", flush=True)
        returncode = process.wait()
    except BaseException:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait()
        raise
    finally:
        process.stdout.close()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command, "".join(output))
    return "".join(output)


def selected_run(directory, expected_mode=None):
    directory = Path(directory)
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory.resolve()
    if (ROOT / "outputs/neural_baselines").resolve() not in directory.parents:
        raise ValueError("Choose a run under outputs/neural_baselines/")
    saved = json.loads((directory / "results.json").read_text(encoding="utf-8"))
    mode = saved["mode"]
    if mode not in {"smoke", "full"} or expected_mode is not None and mode != expected_mode:
        raise ValueError("The requested run mode is missing; no smoke/full fallback")
    config = saved["config"]
    if saved["test_loaded"] is not False or config.get("manifest_version") != "v2":
        raise ValueError("Expected a v2 train/validation-only run")
    seeds = [config["smoke_seed"]] if mode == "smoke" else config["seeds"]
    if seeds != ([42] if mode == "smoke" else [13, 42, 2026]):
        raise ValueError("Unexpected seed protocol")
    expected = {(c, s) for c in ("tsac", "madar") for s in seeds}
    results = saved["results"]
    if (len(results) != len(expected) or {(r["corpus"], r["seed"]) for r in results} != expected
            or any(r["variant"] != "raw-v0" for r in results)):
        raise ValueError("Incomplete or unexpected BiGRU results")
    return {"directory": directory, "mode": mode, "results": results}


def train_bigru(mode):
    if mode not in {"smoke", "full"}:
        raise ValueError("Training requires smoke or full mode")
    parent = ROOT / "outputs/neural_baselines/notebooks"
    identifier = uuid.uuid4().hex
    smoke = parent / f"{identifier}_bigru_smoke"
    print("Contrôle préalable — 100 exemples par partition, deux tâches, graine 42.", flush=True)
    run_neural("run_bigru.py", "--mode", "smoke", "--output", smoke, stream=True)
    run_neural("verify_bigru.py", smoke)
    preliminary = selected_run(smoke, "smoke")
    if mode == "smoke":
        return preliminary
    target = parent / f"{identifier}_bigru_full"
    print("\nEntraînement FULL — tout le train, toute la validation, trois graines par tâche.", flush=True)
    run_neural("run_bigru.py", "--mode", "full", "--smoke-run", smoke, "--output", target, stream=True)
    run_neural("verify_bigru.py", target)
    return selected_run(target, "full")


def replay_bigru(directory):
    selected = selected_run(directory, "full")
    print("Relecture du run FULL choisi — aucun nouvel apprentissage.", flush=True)
    run_neural("verify_bigru.py", selected["directory"])
    return selected


def summary(run):
    lines = ["BiGRU — FULL — TROIS GRAINES PAR TÂCHE" if run["mode"] == "full" else "BiGRU — SMOKE — CONTRÔLE TECHNIQUE",
             "Scores de validation ; test officiel non évalué.",
             f"{'Corpus':<8}{'Graine':>8}{'Train':>9}{'Dev':>8}{'Époque':>9}{'Accuracy':>12}{'Macro-F1':>12}"]
    for r in run["results"]:
        lines.append(f"{r['corpus'].upper():<8}{r['seed']:>8}{r['train']['rows']:>9}{r['validation']['rows']:>8}"
                     f"{r['best_epoch']:>9}{r['metrics']['accuracy']:>12.6f}{r['metrics']['macro_f1']:>12.6f}")
    if run["mode"] == "full":
        lines += ["", "Macro-F1 moyen ± écart-type d'échantillon (trois graines) :"]
        for corpus in ("tsac", "madar"):
            values = [r["metrics"]["macro_f1"] for r in run["results"] if r["corpus"] == corpus]
            lines.append(f"{corpus.upper():<8} {statistics.mean(values):.6f} ± {statistics.stdev(values):.6f}")
    lines += ["", f"Modèles, historiques et résultats : {run['directory']}"]
    return "\n".join(lines)


def figures(run):
    source = run["directory"]
    selected_run(source, run["mode"])
    directory = source / "figures"
    if not directory.exists():
        run_neural("plot_bigru_run.py", source)
    provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    if provenance["source_results_sha256"] != digest(source / "results.json"):
        raise ValueError("Figures do not belong to the selected run")
    expected = {f"{r['corpus']}_seed_{r['seed']}.png" for r in run["results"]}
    if set(provenance["files"]) != expected:
        raise ValueError("Missing or unexpected figure files")
    for filename, checksum in provenance["files"].items():
        if Path(filename).name != filename or digest(directory / filename) != checksum:
            raise ValueError("Figure checksum mismatch")
    return {"directory": directory, "images": sorted(expected), "mode": run["mode"]}
