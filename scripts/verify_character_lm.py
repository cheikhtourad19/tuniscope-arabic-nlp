"""Reconstruct train counts, replay checkpoints and independently recalculate PPL."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path

from character_lm import CharacterLM, UNK, BOS, EOS
from character_lm_experiment import ROOT, digest, object_digest, partitions, partition_summary, read_artifact, validate_config


def close(actual, expected):
    if not math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError("Recomputed metric differs")


def independent_score(model, text):
    mapping = {c: i + 3 for i, c in enumerate(model.characters)}
    encoded = [mapping.get(c, UNK) for c in text] + [EOS]
    padded = [BOS] * (model.order - 1) + encoded
    terms = []
    for index, target in enumerate(encoded):
        history = tuple(padded[index:index + model.order - 1])
        counts = model.counts.get(history, {})
        numerator = counts.get(target, 0) + model.alpha
        denominator = sum(counts.values()) + model.alpha * (len(mapping) + 2)
        terms.append(math.log(denominator) - math.log(numerator))
    return {"nll": math.fsum(terms), "targets": len(encoded), "characters": len(text),
            "unknown_characters": sum(c not in mapping for c in text)}


def verify(directory, report=None):
    directory = directory.resolve()
    metadata = json.loads((directory / "results.json").read_text())
    config = metadata["config"]
    validate_config(config)
    if metadata["test_loaded"] is not False or metadata["mode"] not in {"smoke", "full"}:
        raise ValueError("Expected validation-only smoke/full run")
    if digest(ROOT / "uv.lock") != metadata["uv_lock_sha256"]:
        raise ValueError("Training environment lock changed")
    if object_digest(config) != object_digest(json.loads((ROOT / "configs/character_lm_v1.json").read_text())):
        raise ValueError("Configuration changed")
    if digest(ROOT / "configs/character_lm_v1.json") != metadata["config_sha256"]:
        raise ValueError("Configuration file checksum changed")
    if metadata["variant"] != "raw-v0" or metadata["seed"] is not None or metadata["deterministic"] is not True:
        raise ValueError("Unexpected representation or stochastic protocol")
    results = metadata["results"]
    if len(results) != 4 or {(r["corpus"], r["order"]) for r in results} != {(c, n) for c in config["corpora"] for n in config["orders"]}:
        raise ValueError("Expected four distinct character LM results")
    for corpus in config["corpora"]:
        if digest(ROOT / f"data/manifests/v2/{corpus}.csv") != metadata["manifests_sha256"][corpus]:
            raise ValueError("Manifest changed")
        train, validation = partitions(corpus, metadata["mode"], config)
        characters = sorted({c for row in train for c in row["text_raw"]})
        same_targets = None
        for result in [r for r in results if r["corpus"] == corpus]:
            if result["variant"] != "raw-v0":
                raise ValueError("Unexpected result representation")
            if result["train"] != partition_summary(train) or result["validation"] != partition_summary(validation):
                raise ValueError("Partition coverage changed")
            if result["vocabulary_sha256"] != object_digest(characters) or result["vocabulary_size"] != len(characters) + 2:
                raise ValueError("Vocabulary is not train-only")
            for key in ("checkpoint", "scores"):
                if digest(read_artifact(directory, result[key])) != result[f"{key}_sha256"]:
                    raise ValueError("Artifact checksum changed")
            model = CharacterLM.from_dict(json.loads(read_artifact(directory, result["checkpoint"]).read_text()))
            if model.order != result["order"] or model.alpha != config["alpha"] or model.characters != characters:
                raise ValueError("Model disagrees with protocol")
            # Different construction from fit/events: explicit padded-window counts.
            reference = {}
            mapping = {c: i + 3 for i, c in enumerate(characters)}
            for row in train:
                encoded = [mapping[c] for c in row["text_raw"]] + [EOS]
                padded = [BOS] * (model.order - 1) + encoded
                for index, target in enumerate(encoded):
                    history = tuple(padded[index:index + model.order - 1])
                    reference.setdefault(history, Counter())[target] += 1
            if reference != model.counts or result["train_targets"] != sum(sum(c.values()) for c in reference.values()):
                raise ValueError("Learned counts do not match training data")
            if (result["contexts"] != len(reference)
                    or result["observed_ngrams"] != sum(len(c) for c in reference.values())
                    or result["checkpoint_bytes"] != read_artifact(directory, result["checkpoint"]).stat().st_size):
                raise ValueError("Checkpoint dimensions or size do not match")
            scores = json.loads(read_artifact(directory, result["scores"]).read_text())
            if len(scores) != len(validation):
                raise ValueError("Missing validation documents")
            recomputed = []
            for row, saved in zip(validation, scores):
                if saved["id"] != row["id"] or saved["group_id"] != row["group_id"]:
                    raise ValueError("Validation IDs or groups misaligned")
                direct = independent_score(model, row["text_raw"])
                replay = model.score(row["text_raw"])
                for key in direct:
                    close(saved[key], direct[key])
                    close(replay[key], direct[key])
                recomputed.append(direct)
            nll = math.fsum(s["nll"] for s in recomputed)
            targets = sum(s["targets"] for s in recomputed)
            chars = sum(s["characters"] for s in recomputed)
            unknown = sum(s["unknown_characters"] for s in recomputed)
            expected = {"documents": len(validation), "nll": nll, "targets": targets,
                        "characters": chars, "unknown_characters": unknown,
                        "unknown_character_rate": unknown / chars if chars else 0,
                        "cross_entropy_nats": nll / targets, "bits_per_target": nll / targets / math.log(2),
                        "perplexity": math.exp(nll / targets)}
            for key, value in expected.items():
                close(result["metrics"][key], value)
            target_signature = (targets, chars, unknown, result["validation"]["ids_sha256"])
            if same_targets is not None and target_signature != same_targets:
                raise ValueError("Orders evaluated on different targets")
            same_targets = target_signature
    if report:
        if report.exists() or (ROOT / "reports").resolve() not in report.resolve().parents:
            raise ValueError("Choose a fresh report directory under reports/")
        report.mkdir(parents=True)
        table = ["| Corpus | Ordre | Train | Validation | Cibles | Vocabulaire | Inconnus | Perplexité ↓ | Bits/cible ↓ |",
                 "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for r in results:
            m = r["metrics"]
            table.append(f"| {r['corpus'].upper()} | {r['order']} | {r['train']['rows']} | {r['validation']['rows']} | "
                         f"{m['targets']} | {r['vocabulary_size']} | {m['unknown_character_rate']:.4%} | "
                         f"{m['perplexity']:.6f} | {m['bits_per_target']:.6f} |")
        content = ("# Modèles de langage caractères — validation v2\n\n"
                   f"Run : `{directory.name}` ; mode **{metadata['mode']}**, représentation **raw-v0**.\n\n" +
                   "\n".join(table) + "\n\nMême vocabulaire et mêmes cibles entre les deux ordres au sein de chaque corpus. "
                   "EOS inclus, BOS exclu ; lissage alpha=0,1. Les perplexités des deux corpus ne sont pas directement comparables. "
                   "Ce ne sont pas des scores de classification. Aucun test officiel évalué.\n\n"
                   "Comptes reconstruits depuis le train, checkpoints rechargés et validation intégralement rejouée ; "
                   "probabilités et agrégations recalculées indépendamment. L04/L07/L12 restent partiels.\n")
        (report / "README.md").write_text(content, encoding="utf-8")
        (report / "summary.json").write_text(json.dumps({
            "run": directory.name, "mode": metadata["mode"], "source_results_sha256": digest(directory / "results.json"),
            "manifests_sha256": metadata["manifests_sha256"], "test_loaded": False,
            "results": [{k: v for k, v in r.items() if k not in {"scores", "checkpoint"}} for r in results]
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {metadata['mode']}: four character LMs, train counts, checkpoints, all validation scores and perplexities", flush=True)
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    verify(args.run, args.report)
