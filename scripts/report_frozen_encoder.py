"""Generate a public aggregate report from an explicitly verified full run."""
import argparse
import json
from pathlib import Path

from character_lm_experiment import ROOT
from fetch_camelbert_encoder import sha256
from run_frozen_encoder import save
from verify_frozen_encoder import verify


def report(directory, target, classical_run, bigru_run):
    if target.exists():
        raise ValueError('Preserve existing reports')
    result = verify(directory, reencode=False, reference_run=classical_run)
    proof = json.loads((directory / 'verification.json').read_text())
    if result['mode'] != 'full' or proof['reencoded'] is not True or proof['results_sha256'] != sha256(directory / 'results.json'):
        raise ValueError('Require a fully replayed and reencoded full run')
    classical = json.loads((classical_run / 'results.json').read_text())
    bigru = json.loads((bigru_run / 'results.json').read_text())
    if bigru['mode'] != 'full' or bigru['test_loaded'] is not False:
        raise ValueError('Expected full BiGRU reference')
    comparison = {}
    for corpus in result['config']['corpora']:
        selected = [r for r in classical['results'] if r['corpus'] == corpus and r['system'] == 'tfidf_char' and r['variant'] == 'raw-v0']
        neural = [r for r in bigru['results'] if r['corpus'] == corpus]
        partition = result['features'][corpus]['validation']['partition']
        if len(selected) != 1 or len(neural) != 3 or any(r['validation'] != partition for r in neural):
            raise ValueError('Reference validation partition mismatch')
        comparison[corpus] = {'tfidf_char_macro_f1': selected[0]['metrics']['macro_f1'],
                              'bigru_mean_macro_f1': sum(r['metrics']['macro_f1'] for r in neural) / len(neural),
                              'frozen_encoder': result['aggregates'][corpus]}
    aggregate = {k: result[k] for k in ('config', 'device', 'encoder', 'aggregates', 'memory', 'total_seconds',
                  'test_loaded', 'encoder_unchanged', 'packages', 'environment_lock_sha256', 'manifests_sha256')}
    aggregate['source_results_sha256'] = sha256(directory / 'results.json')
    aggregate['comparison'] = comparison
    aggregate['results'] = [{k: r[k] for k in ('corpus', 'seed', 'metrics', 'best_epoch', 'epochs_completed', 'trainable_parameters', 'seconds')} for r in result['results']]
    aggregate['features'] = result['features']
    aggregate['verification'] = proof
    aggregate['reference_results_sha256'] = {'classical': sha256(classical_run / 'results.json'), 'bigru': sha256(bigru_run / 'results.json')}
    lines = ['# CAMeLBERT gelé — résultats de validation', '',
             'Rapport généré depuis un run FULL explicite ; aucun score du test officiel. Contributions partielles à L05/L07/L12.', '',
             '| Corpus | Graine | Accuracy | Macro-F1 | Époque retenue | Époques exécutées |',
             '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in result['results']:
        lines.append(f'| {r["corpus"].upper()} | {r["seed"]} | {r["metrics"]["accuracy"]:.6f} | {r["metrics"]["macro_f1"]:.6f} | {r["best_epoch"]} | {r["epochs_completed"]} |')
    lines += ['', '| Corpus | TF-IDF caractères | BiGRU, moyenne 3 graines | CAMeLBERT gelé, moyenne ± écart-type |',
              '| --- | ---: | ---: | ---: |']
    for corpus, item in comparison.items():
        a = item['frozen_encoder']
        lines.append(f'| {corpus.upper()} | {item["tfidf_char_macro_f1"]:.6f} | {item["bigru_mean_macro_f1"]:.6f} | {a["mean_macro_f1"]:.6f} ± {a["sample_std_macro_f1"]:.6f} |')
    lines += ['', f'Durée réelle : **{result["total_seconds"]:.1f} s**, encodeur **{result["device"]}**, têtes CPU.',
              'Backbone chargé : **108 490 752** paramètres gelés ; têtes apprises : **1 538 / 2 307** paramètres.',
              'Même validation v2 ; TF-IDF et BiGRU sont des références déjà fixées, pas une sélection de la meilleure graine.',
              'Les têtes et métriques sont rejouées intégralement ; dix textes de validation par tâche sont réencodés depuis les poids originaux.',
              'Les empreintes du backbone avant/après sont identiques. Le cache de vecteurs et les poids restent privés.',
              '', 'Les poids généraux gelés ne dépassent pas TF-IDF caractères. Ces résultats ne mesurent pas le futur fine-tuning, ni une qualité universelle du modèle.',
              'Le run est local MPS ; aucun résultat CUDA ni budget mémoire de fine-tuning n’est établi.',
              '', f'Provenance du run : `{directory.relative_to(ROOT)}`.',
              'Les PNG du run privé montrent les pertes, le macro-F1 et les matrices, séparément pour chaque graine.', '']
    target.mkdir(parents=True)
    save(target / 'summary.json', aggregate)
    (target / 'README.md').write_text('\n'.join(lines))
    print(f'Rapport agrégé : {target}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--classical-run', type=Path, required=True)
    parser.add_argument('--bigru-run', type=Path, required=True)
    a = parser.parse_args()
    report(a.run.resolve(), a.output.resolve(), a.classical_run.resolve(), a.bigru_run.resolve())
