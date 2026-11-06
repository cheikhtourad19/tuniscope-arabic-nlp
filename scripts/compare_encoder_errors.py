"""Compare archived FULL v2 validation predictions; no training or semantic review."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from baseline_metrics import classification_metrics
from load_working_data import ROOT, load_records


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare_predictions(reference, encoder):
    """Four disjoint outcomes; require identical ID/group/truth order."""
    if not reference or len(reference) != len(encoder):
        raise ValueError("Expected equally sized nonempty predictions")
    if len({r['id'] for r in reference}) != len(reference):
        raise ValueError("Repeated validation ID")
    counts = Counter(dict(both_correct=0, corrected=0, introduced=0, both_wrong=0))
    for a, b in zip(reference, encoder):
        if any(a[key] != b[key] for key in ('id', 'group_id', 'truth')):
            raise ValueError("Validation ID/group/truth mismatch")
        good_a, good_b = a['predicted'] == a['truth'], b['predicted'] == b['truth']
        key = ('both_correct' if good_b else 'introduced') if good_a else (
            'corrected' if good_b else 'both_wrong')
        counts[key] += 1
    result = dict(counts)
    result.update(rows=len(reference), baseline_errors=counts['corrected'] + counts['both_wrong'],
                  encoder_errors=counts['introduced'] + counts['both_wrong'])
    return result


def checked_predictions(directory, name, expected):
    path = (directory / name).resolve()
    if directory.resolve() not in path.parents or digest(path) != expected:
        raise ValueError("Prediction path or hash mismatch")
    return json.loads(path.read_text())


def run(baseline_run, encoder_run, output):
    if output.exists():
        raise FileExistsError("Preserve previous comparisons")
    baseline = json.loads((baseline_run / 'results.json').read_text())
    encoder = json.loads((encoder_run / 'results.json').read_text())
    proof = json.loads((encoder_run / 'verification.json').read_text())
    for meta in (baseline, encoder):
        if meta['mode'] != 'full' or meta['test_loaded'] is not False or meta['config']['manifest_version'] != 'v2':
            raise ValueError("Expected FULL v2 train/validation-only runs")
    if encoder['config']['variant'] != 'raw-v0' or proof['results_sha256'] != digest(encoder_run / 'results.json'):
        raise ValueError("Encoder variant or prior replay proof mismatch")
    if proof['models_replayed'] != len(encoder['results']):
        raise ValueError("Incomplete prior replay proof")
    rows, provenance = [], {}
    for corpus in ('tsac', 'madar'):
        manifest_hash = digest(ROOT / f'data/manifests/v2/{corpus}.csv')
        if any(meta['manifests_sha256'][corpus] != manifest_hash for meta in (baseline, encoder)):
            raise ValueError("Manifest hash mismatch")
        records = load_records(corpus, splits=('validation',), manifest_version='v2')
        source = [{**r, 'truth': r['label'], 'predicted': r['label']} for r in records]
        references = [r for r in baseline['results'] if
                      (r['corpus'], r['system'], r['variant']) == (corpus, 'tfidf_char', 'raw-v0')]
        if len(references) != 1:
            raise ValueError("Expected one raw character TF-IDF reference")
        ref = references[0]
        predicted = checked_predictions(baseline_run, ref['predictions'], ref['predictions_sha256'])
        compare_predictions(source, predicted)
        adapted = [r for r in encoder['results'] if r['corpus'] == corpus and r['regime'] == 'finetuned']
        if sorted(r['seed'] for r in adapted) != [13, 42, 2026]:
            raise ValueError("Expected all three seeds, without selection")
        provenance[corpus] = {'manifest_sha256': manifest_hash, 'baseline_predictions_sha256': ref['predictions_sha256']}
        labels = encoder['config']['labels'][corpus]
        for meta, preds in [(ref, predicted)]:
            actual = classification_metrics([p['truth'] for p in preds], [p['predicted'] for p in preds], labels)
            if actual != {k: meta['metrics'][k] for k in actual}:
                raise ValueError("Baseline metrics do not match predictions")
        for r in sorted(adapted, key=lambda r: r['seed']):
            name = f"{r['checkpoint']}/predictions.json"
            predictions = checked_predictions(encoder_run, name, encoder['artifacts'][name])
            counts = compare_predictions(predicted, predictions)
            actual = classification_metrics([p['truth'] for p in predictions], [p['predicted'] for p in predictions], labels)
            if actual != {k: r['metrics'][k] for k in actual}:
                raise ValueError("Encoder metrics do not match predictions")
            rows.append({'corpus': corpus, 'seed': r['seed'], **counts,
                         'baseline_macro_f1': ref['metrics']['macro_f1'], 'encoder_macro_f1': actual['macro_f1'],
                         'encoder_predictions_sha256': encoder['artifacts'][name]})
    result = {'analysis': 'quantitative-validation-error-overlap-v1', 'test_loaded': False,
              'new_training': False, 'semantic_error_review_complete': False,
              'baseline_run': str(baseline_run.relative_to(ROOT)), 'encoder_run': str(encoder_run.relative_to(ROOT)),
              'baseline_results_sha256': digest(baseline_run / 'results.json'),
              'encoder_results_sha256': digest(encoder_run / 'results.json'),
              'prior_encoder_verification_sha256': digest(encoder_run / 'verification.json'),
              'script_sha256': digest(Path(__file__)), 'provenance': provenance, 'comparisons': rows}
    lines = ['# Comparaison des erreurs CAMeLBERT / TF-IDF', '',
             'Validation FULL v2, texte brut, trois graines conservées. Aucun nouvel entraînement ni évaluation du test.', '',
             '| Corpus | Graine | Erreurs TF-IDF | Erreurs encodeur | Corrigées | Introduites | Communes | Correctes pour les deux |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for r in rows:
        lines.append(f"| {r['corpus'].upper()} | {r['seed']} | {r['baseline_errors']} | {r['encoder_errors']} | {r['corrected']} | {r['introduced']} | {r['both_wrong']} | {r['both_correct']} |")
    lines += ['', '« Corrigées » : TF-IDF se trompe et CAMeLBERT retrouve le label de référence. « Introduites » : TF-IDF retrouve le label de référence et CAMeLBERT se trompe. « Communes » : les deux se trompent, éventuellement avec des prédictions différentes.', '',
              'Les comptes suivent les annotations du corpus ; ils ne constituent pas une validation humaine des labels. Les quatre catégories couvrent tous les exemples de validation. Leurs effectifs ne sont pas des catégories linguistiques.', '',
              'Les identifiants, groupes, labels, empreintes des fichiers et métriques recalculées sont contrôlés. Le rapport réutilise la preuve antérieure de rechargement de l’encodeur, sans refaire l’inférence. La revue sémantique d’au moins 30 erreurs distinctes par tâche reste à effectuer. L07/L10 restent partiels.', '',
              'Reproduction depuis les runs locaux autorisés :', '',
              '```sh', 'sh scripts/project_uv.sh run --locked --offline python scripts/compare_encoder_errors.py --output reports/encoder_vs_tfidf_2026-11-05', '```', '']
    output.mkdir(parents=True)
    (output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (output / 'README.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:14]))
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-run', type=Path, default=ROOT / 'outputs/baselines/2026-10-17_grouped_v2_full')
    p.add_argument('--encoder-run', type=Path, default=ROOT / 'outputs/encoder_finetuning/2026-11-05_cls_paired_mps_full')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    run(a.baseline_run, a.encoder_run, a.output)
