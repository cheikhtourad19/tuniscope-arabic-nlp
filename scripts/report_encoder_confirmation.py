"""Export text-free FULL confirmation evidence and an aligned TF-IDF comparison."""
import argparse
import json
from pathlib import Path

from baseline_metrics import classification_metrics
from encoder_confirmation_plan import ROOT, digest
from encoder_confirmation_notebook_support import campaign_figures
from run_encoder_confirmation import inside, read, verify_campaign


def report(directory, target, classical_run):
    directory, target, classical_run = (p.resolve() for p in (directory, target, classical_run))
    if target.exists():
        raise FileExistsError('Preserve previous reports')
    campaign, proof = verify_campaign(directory, 'full')
    figures = campaign_figures(directory, campaign)
    classical = read(classical_run / 'results.json')
    if classical['mode'] != 'full' or classical['test_loaded'] is not False:
        raise ValueError('Require a full validation reference')
    payload = {'evidence_type': 'retained-full-three-seed-validation-confirmation',
               'test_loaded': False, 'source_campaign': str(directory.relative_to(ROOT)),
               'verification': proof, 'fixed_learning_rates': campaign['fixed_learning_rates'],
               'aggregates': campaign['aggregates'], 'results': [], 'stages': [], 'comparison': {},
               'figures_sha256': {str(p.relative_to(directory)): digest(p) for p in figures},
               'classical_results_sha256': digest(classical_run / 'results.json'),
               'report_generator_sha256': digest(Path(__file__)),
               'elapsed_seconds_before_final_campaign_checks': campaign['elapsed_seconds_before_final_campaign_checks']}
    for entry in campaign['entries']:
        corpus = entry['corpus']
        folder = inside(directory, entry['full']['directory'])
        stage = read(folder / 'results.json'); labels = stage['config']['labels'][corpus]
        refs = [r for r in classical['results'] if r['corpus'] == corpus
                and r['system'] == 'tfidf_char' and r['variant'] == 'raw-v0']
        if len(refs) != 1:
            raise ValueError('Need one raw character reference per task')
        ref = refs[0]; ref_path = inside(classical_run, ref['predictions'])
        if digest(ref_path) != ref['predictions_sha256']:
            raise ValueError('Classical predictions changed')
        old = read(ref_path)
        actual = classification_metrics([p['truth'] for p in old], [p['predicted'] for p in old], labels)
        if actual != {k: ref['metrics'][k] for k in actual}:
            raise ValueError('Classical metrics disagree with saved predictions')
        reference_keys = [(p['id'], p['group_id'], p['truth']) for p in old]
        for row in stage['results']:
            predictions = read(inside(folder, row['checkpoint'] + '/predictions.json'))
            if [(p['id'], p['group_id'], p['truth']) for p in predictions] != reference_keys:
                raise ValueError('Different encoder/TF-IDF validation examples')
            payload['results'].append({k: row[k] for k in (
                'corpus', 'seed', 'metrics', 'best_epoch', 'epochs_completed', 'parameters',
                'trainable_parameters', 'train', 'validation')})
        card_path = directory / f'model_card_{corpus}.json'; card = read(card_path)
        expected_checkpoints = [str(Path(entry['full']['directory']) / r['checkpoint']) for r in stage['results']]
        if (card['test_evaluated'] is not False or card['config'] != stage['config']
                or card['validation_aggregates'] != campaign['aggregates'][corpus]
                or card['checkpoint_references'] != expected_checkpoints
                or card['total_parameters'] != stage['results'][0]['parameters']
                or card['trainable_parameters'] != stage['results'][0]['trainable_parameters']):
            raise ValueError('Provisional model card differs from its run')
        value = campaign['aggregates'][corpus]['macro_f1']['mean']
        payload['comparison'][corpus] = {
            'validation_rows': len(old), 'same_validation_ids_groups_truth': True,
            'tfidf_char_raw_macro_f1': actual['macro_f1'], 'encoder_mean_macro_f1': value,
            'macro_f1_difference': value - actual['macro_f1'],
            'reference_predictions_sha256': ref['predictions_sha256'],
            'provisional_model_card_sha256': digest(card_path)}
        payload['stages'].append({'corpus': corpus, 'directory': entry['full']['directory'],
            'results_sha256': entry['full']['results_sha256'],
            'verification_sha256': entry['full']['verification_sha256'],
            'seconds': stage['seconds'], 'device': stage['device'],
            'config': stage['config'], 'manifests_sha256': stage['manifests_sha256'],
            'code_sha256': stage['code_sha256'], 'environment_lock_sha256': stage['environment_lock_sha256']})
    payload['full_stages_seconds'] = sum(s['seconds'] for s in payload['stages'])
    lines = ['# Confirmation de CAMeLBERT sur trois graines', '',
             'Résultats de **validation v2**, générés depuis une campagne FULL contrôlée. Test officiel non évalué.', '',
             '| Corpus | Taux | Graine | Macro-F1 | Accuracy | Époque retenue / exécutées |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in payload['results']:
        lines.append(f'| {r["corpus"].upper()} | {campaign["fixed_learning_rates"][r["corpus"]]:g} | '
                     f'{r["seed"]} | {r["metrics"]["macro_f1"]:.6f} | {r["metrics"]["accuracy"]:.6f} | '
                     f'{r["best_epoch"]} / {r["epochs_completed"]} |')
    lines += ['', '| Corpus | Macro-F1 moyen ± écart-type échantillonnal | TF-IDF caractères brut | Différence |',
              '| --- | ---: | ---: | ---: |']
    for corpus, values in payload['comparison'].items():
        a = campaign['aggregates'][corpus]['macro_f1']
        lines.append(f'| {corpus.upper()} | {a["mean"]:.6f} ± {a["sample_std"]:.6f} | '
                     f'{values["tfidf_char_raw_macro_f1"]:.6f} | {values["macro_f1_difference"]:+.6f} |')
    lines += ['', 'Les mêmes identifiants, groupes et labels de validation sont contrôlés pour toutes les graines et TF-IDF.',
              'Les trois scores sont conservés, sans sélection de meilleure graine. Aucune significativité ni généralisation au test établie.',
              'Les taux ont été sélectionnés sur cette validation ; ces moyennes ne sont pas une évaluation finale indépendante.',
              f'Stages FULL cumulés : {payload["full_stages_seconds"]:.6f} s. Campagne avant contrôles finaux : '
              f'{campaign["elapsed_seconds_before_final_campaign_checks"]:.6f} s (smokes, rejeux et figures inclus).',
              'Six checkpoints et six PNG conservés ; cartes provisoires vérifiées. Les données, prédictions individuelles et poids restent privés.',
              f'Campagne : `{directory.relative_to(ROOT)}`.',
              'L05 : confirmation de validation terminée ; L07/L08/L12 restent partiels (test, compléments d’évaluation, intégration et reproduction indépendante).', '']
    target.mkdir(parents=True)
    (target / 'summary.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    (target / 'README.md').write_text('\n'.join(lines))
    print(f'Rapport généré : {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--classical-run', type=Path, default=ROOT / 'outputs/baselines/2026-10-17_grouped_v2_full')
    args = parser.parse_args()
    report(args.run, args.output, args.classical_run)
