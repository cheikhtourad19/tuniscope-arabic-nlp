"""Consolidate Silver evidence from saved artifacts, without model inference."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'rag'))
from baseline_metrics import classification_metrics
from report_qwen_validation import aggregate as verify_qwen_base
from run_encoder_confirmation import verify_campaign

SEEDS = [13, 42, 2026]
CONFIRMATION = ROOT / 'outputs/encoder_finetuning/confirmation/notebooks/177d9f5ad78b4ef4907e97d97abe96f8_full'
LORA = ROOT / 'outputs/qwen/2026-11-07_cuda_lora_c10d02666e154607ac9d7faddbd47838_full'
REPORTS = {
    'classical': 'reports/baselines_grouped_v2_2026-10-17/metrics.json',
    'representations': 'reports/three_representations_v2_2026-10-17/comparison.json',
    'rules': 'reports/rule_baselines_v1_2026-10-17/summary.json',
    'language_model': 'reports/character_lm_v1_2026-11-04/summary.json',
    'bigru': 'reports/bigru_v1_2026-11-04/summary.json',
    'frozen': 'reports/camelbert_frozen_v1_2026-11-04/summary.json',
    'paired': 'reports/camelbert_finetuning_v1_2026-11-05/summary.json',
    'confirmation': 'reports/camelbert_confirmation_2026-11-06/summary.json',
    'base': 'reports/qwen_validation_v2_2026-11-07/summary.json',
    'lora': 'reports/qwen_lora_cuda_2026-11-07/summary.json',
    'generation': 'reports/rag_validation_v1_2026-11-09/generation.json',
    'retrieval_validation': 'reports/rag_application_validation_2026-11-11/results.json',
    'retrieval_test': 'reports/rag_documentary_test_v1_2026-11-12/results.json',
}


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def matrix_macro_f1(matrix):
    values = []
    for index, row in enumerate(matrix):
        denominator = sum(row) + sum(other[index] for other in matrix)
        values.append(2 * row[index] / denominator if denominator else 0.0)
    return statistics.mean(values)


def seed_statistics(rows):
    if sorted(seed for seed, _ in rows) != SEEDS:
        raise ValueError('Les trois graines distinctes 13, 42 et 2026 sont requises.')
    scores = {str(seed): value for seed, value in sorted(rows)}
    return {'seed_scores': scores, 'mean': statistics.mean(scores.values()),
            'sample_std': statistics.stdev(scores.values())}


def abstention_counts(rows, enabled):
    answered = [row for row in rows if (row['response']['status'] == 'answered' if enabled else bool(row['ranking']))]
    relevant = sum(row['ranking'][0]['id'] in row['relevant_ids'] for row in answered)
    unsupported = [row for row in rows if not row['relevant_ids']]
    return {'questions': len(rows), 'answers': len(answered), 'coverage': len(answered) / len(rows),
            'relevant_answers': relevant, 'irrelevant_answers': len(answered) - relevant,
            'relevant_rate_among_answers': relevant / len(answered) if answered else None,
            'unanswerable_rejected': sum(row not in answered for row in unsupported),
            'unanswerable': len(unsupported)}


def build():
    evidence = {}

    def checked(path, expected=None):
        path = path.resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Fichier absent ou hors projet — {path}')
        name = str(path.relative_to(ROOT))
        if name not in evidence:
            evidence[name] = sha(path)
        if expected is not None and evidence[name] != expected:
            raise ValueError(f'Empreinte différente — {name}')
        return path

    def read(path):
        return json.loads(checked(path).read_text())

    reports = {key: read(ROOT / name) for key, name in REPORTS.items()}
    for corpus in ('tsac', 'madar'):
        checked(ROOT / f'data/manifests/v2/{corpus}.csv', reports['classical']['manifests_sha256'][corpus])
    reference = {}
    scores = []
    provenance = {}

    def record_provenance(name, data):
        keys = ('git_commit', 'git_dirty', 'device', 'hardware', 'platform', 'python', 'packages',
                'config_sha256', 'protocol_sha256', 'code_sha256', 'manifests_sha256',
                'environment_lock_sha256', 'uv_lock_sha256', 'test_loaded')
        provenance[name] = {key: data.get(key) for key in keys}

    def add(family, corpus, mean, std=None, seeds=None):
        scores.append({'family': family, 'corpus': corpus, 'mean_macro_f1': mean,
                       'sample_std': std, 'seed_scores': seeds or {}})

    def predictions(path, row, corpus):
        values = read(path)
        expected_hash = row.get('predictions_sha256')
        if expected_hash is not None:
            checked(path, expected_hash)
        keys = [(value['id'], value['group_id'], value['truth']) for value in values]
        if keys != reference[corpus]:
            raise ValueError('Identifiants/groupes/labels de validation différents.')
        actual = classification_metrics([v['truth'] for v in values], [v['predicted'] for v in values],
                                        row['metrics']['labels'])
        if actual['confusion_matrix'] != row['metrics']['confusion_matrix'] or actual['macro_f1'] != row['metrics']['macro_f1']:
            raise ValueError('Métriques différentes des prédictions sauvegardées.')

    classical = ROOT / 'outputs/baselines/2026-10-17_grouped_v2_full'
    original = read(classical / 'results.json')
    if original['test_loaded'] is not False or original['mode'] != 'full':
        raise ValueError('Une référence complète de validation est requise.')
    for row in original['results']:
        if row['system'] == 'tfidf_char' and row['variant'] == 'raw-v0':
            values = read(classical / row['predictions'])
            reference[row['corpus']] = [(v['id'], v['group_id'], v['truth']) for v in values]
    for directory in ('2026-10-17_grouped_v2_full', '2026-10-17_words_v2_full', '2026-10-17_combined_v2_full'):
        folder = ROOT / 'outputs/baselines' / directory
        data = read(folder / 'results.json')
        record_provenance(directory, data)
        if data['test_loaded'] is not False or data['mode'] != 'full':
            raise ValueError('Référence classique hors périmètre.')
        for row in data['results']:
            checked(folder / row['checkpoint'], row['checkpoint_sha256'])
            predictions(folder / row['predictions'], row, row['corpus'])
            family = f'{row["system"]} / {row["variant"]}'
            if not any(s['family'] == family and s['corpus'] == row['corpus'] for s in scores):
                add(family, row['corpus'], row['metrics']['macro_f1'])

    for row in reports['rules']['results']:
        if row['system'] != 'majority':
            add(row['system'], row['corpus'], row['metrics']['macro_f1'])

    lm_folder = ROOT / 'outputs/language_models/2026-11-04_char_lm_v1_full'
    lm = read(lm_folder / 'results.json')
    checked(lm_folder / 'results.json', reports['language_model']['source_results_sha256'])
    if lm['test_loaded'] is not False or lm['mode'] != 'full':
        raise ValueError('Référence de modèle de langue hors périmètre.')
    record_provenance('character_language_model', lm)
    for row in lm['results']:
        checked(lm_folder / row['checkpoint'], row['checkpoint_sha256'])
        checked(lm_folder / row['scores'], row['scores_sha256'])
        values = read(lm_folder / row['scores'])
        if [(v['id'], v['group_id']) for v in values] != [(i, g) for i, g, _ in reference[row['corpus']]]:
            raise ValueError('Référence de modèle de langue désalignée.')
        perplexity = math.exp(sum(v['nll'] for v in values) / sum(v['targets'] for v in values))
        if not math.isclose(perplexity, row['metrics']['perplexity'], rel_tol=1e-12):
            raise ValueError('Perplexité différente des scores sauvegardés.')

    neural = [
        ('bigru', ROOT / 'outputs/neural_baselines/2026-11-04_bigru_v1_cpu_full'),
        ('frozen', ROOT / 'outputs/frozen_encoder/2026-11-04_camelbert_frozen_mps_full'),
        ('paired', ROOT / 'outputs/encoder_finetuning/2026-11-05_cls_paired_mps_full'),
    ]
    recomputed = {}
    for family, folder in neural:
        data = read(folder / 'results.json')
        record_provenance(family, data)
        if data['test_loaded'] is not False or data['mode'] != 'full':
            raise ValueError('Campagne neuronale hors périmètre.')
        checked(folder / 'results.json', reports[family]['source_results_sha256'])
        for name, digest in data.get('artifacts', {}).items():
            checked(folder / name, digest)
        groups = defaultdict(list)
        for row in data['results']:
            corpus = row['corpus']
            if family == 'paired':
                predictions(folder / row['checkpoint'] / 'predictions.json', row, corpus)
            else:
                checked(folder / row['checkpoint'], row.get('checkpoint_sha256'))
                predictions(folder / row['predictions'], row, corpus)
            group = (corpus, row['regime']) if family == 'paired' else (corpus, family)
            groups[group].append((row['seed'], row['metrics']['macro_f1']))
        for (corpus, regime), rows in groups.items():
            stats = seed_statistics(rows)
            key = f'{corpus}:{regime}'
            recomputed[f'{family}:{key}'] = stats
            source_aggregate = reports[family]['aggregate' if family == 'bigru' else 'aggregates']
            expected = source_aggregate[key if family == 'paired' else corpus]
            if not (math.isclose(stats['mean'], expected['mean_macro_f1'], abs_tol=1e-12)
                    and math.isclose(stats['sample_std'], expected['sample_std_macro_f1'], abs_tol=1e-12)):
                raise ValueError('Moyenne ou écart-type neuronal différent.')
            if family != 'paired':
                add(family, corpus, stats['mean'], stats['sample_std'], stats['seed_scores'])

    print('Baselines, BiGRU et références encodeur contrôlés.', flush=True)
    campaign, confirmation_proof = verify_campaign(CONFIRMATION, 'full')
    checked(CONFIRMATION / 'confirmation.json', reports['confirmation']['verification']['campaign_sha256'])
    for corpus in ('tsac', 'madar'):
        models = [r for r in reports['confirmation']['results'] if r['corpus'] == corpus]
        stats = seed_statistics([(r['seed'], r['metrics']['macro_f1']) for r in models])
        if campaign['aggregates'][corpus] != reports['confirmation']['aggregates'][corpus]:
            raise ValueError('Rapport de confirmation différent de sa campagne.')
        recomputed[f'confirmation:{corpus}'] = stats
        add('CAMeLBERT confirmé', corpus, stats['mean'], stats['sample_std'], stats['seed_scores'])
    for entry in campaign['entries']:
        record_provenance(f'confirmation_{entry["corpus"]}',
                          read(CONFIRMATION / entry['full']['directory'] / 'results.json'))
    print('Confirmation complète contrôlée, preuves de rejeu conservées vérifiées.', flush=True)

    base = verify_qwen_base(ROOT / reports['base']['run'])
    record_provenance('qwen_base', base['plan'])
    if base['stages'] != reports['base']['stages']:
        raise ValueError('Rapport Qwen différent de ses prédictions.')
    for name, stage in base['stages'].items():
        add(f'Qwen base / {stage["mode"]}', stage['corpus'], stage['macro_f1'])
    lora = reports['lora']
    if lora['test_loaded'] is not False or sorted(map(int, lora['runs'])) != SEEDS:
        raise ValueError('Campagne LoRA incomplète.')
    for seed, run in lora['runs'].items():
        folder = LORA / f'seed_{seed}'
        for filename, digest in run['training']['adapter_files'].items():
            checked(folder / filename, digest)
        if read(folder / 'training.json') != run['training']:
            raise ValueError('Preuve de formation LoRA différente.')
        for name, stage in run['validation'].items():
            if not math.isclose(matrix_macro_f1(stage['confusion_matrix']), stage['macro_f1'], abs_tol=1e-12):
                raise ValueError('Macro-F1 LoRA différent de sa matrice.')
    for name, aggregate in lora['aggregate'].items():
        corpus, mode = name.split('_', 1)
        stats = seed_statistics([(int(seed), run['validation'][name]['macro_f1']) for seed, run in lora['runs'].items()])
        if not (math.isclose(stats['mean'], aggregate['mean_macro_f1'], abs_tol=1e-12)
                and math.isclose(stats['sample_std'], aggregate['stdev_macro_f1'], abs_tol=1e-12)):
            raise ValueError('Moyenne ou écart-type LoRA différent.')
        recomputed[f'lora:{name}'] = stats
        add(f'Qwen LoRA / {mode}', corpus, stats['mean'], stats['sample_std'], stats['seed_scores'])
    if any(not c['all_parameter_hashes_match'] for c in lora['local_reload']['checks']):
        raise ValueError('Preuve de rechargement LoRA incohérente.')
    print('Qwen base et LoRA contrôlés, statistiques trois graines recalculées.', flush=True)

    configuration_inventory = {}
    for path in sorted((ROOT / 'configs').glob('*.json')):
        configuration_inventory[str(path.relative_to(ROOT))] = sha(checked(path))
    abstention = {method: {'with_abstention': abstention_counts(system['rows'], True),
                          'without_abstention': abstention_counts(system['rows'], False)}
                  for method, system in reports['retrieval_validation']['systems'].items()}
    generation = {condition: {key: value for key, value in values.items() if key != 'rows'}
                  for condition, values in reports['generation']['conditions'].items()}
    paired = {corpus: {regime: recomputed[f'paired:{corpus}:{regime}'] for regime in ('frozen', 'finetuned')}
              for corpus in ('tsac', 'madar')}
    return {
        'scope': 'Silver technical evidence consolidation', 'classification_split': 'validation',
        'new_training': False, 'new_model_inference': False, 'classification_test_evaluated': False,
        'classification_scores': scores, 'recalculated_seed_statistics': recomputed,
        'statistical_language_models': reports['language_model']['results'],
        'run_provenance': provenance,
        'confirmation_current_verification': confirmation_proof,
        'configuration_inventory': configuration_inventory,
        'ablations': {'A_raw_normalized': reports['classical']['paired_comparisons'],
                      'B_representations': reports['representations']['scores'],
                      'C_native_cls_frozen_adapted_initial_campaign': paired,
                      'D_qwen_with_without_context_validation': generation,
                      'E_documentary_abstention_validation_saved_rankings': abstention},
        'rag_documentary_test': {method: system['all'] for method, system in reports['retrieval_test']['systems'].items()},
        'limitations': {
            'rag_reference_review': 'AI proposals, independent human review and pedagogical waiver pending',
            'lora_original_cuda_predictions_received': lora['individual_predictions_received'],
            'lora_original_campaign_summary_received': lora['original_campaign_summary_received'],
            'lora_original_campaign_wall_seconds': lora['campaign_wall_seconds'],
            'paired_initial_all_epoch_class_metrics': reports['paired']['verification']['epoch_class_metrics_complete'],
            'independent_checkpoint_backup_confirmed': False,
            'independent_reproduction_confirmed': False,
            'qwen_base_git_commit_in_plan': base['plan'].get('git_commit'),
        },
        'source_files_sha256': evidence,
        'reporter_sha256': sha(Path(__file__)),
    }


def markdown(result):
    families = list(dict.fromkeys(row['family'] for row in result['classification_scores']))
    lines = ['# Bilan des preuves techniques Silver', '',
             'Scores de classification sur la validation v2. Les moyennes et écarts-types sont '
             'recalculés sur les trois graines pour les campagnes neuronales. '
             'Les références déterministes ou à génération fixée sont présentées sans écart-type.', '',
             '| Système | TSAC macro-F1 | MADAR macro-F1 |', '| --- | ---: | ---: |']
    for family in families:
        values = []
        for corpus in ('tsac', 'madar'):
            row = next((r for r in result['classification_scores'] if r['family'] == family and r['corpus'] == corpus), None)
            value = '—' if row is None else f'{row["mean_macro_f1"]:.6f}'
            if row is not None and row['sample_std'] is not None:
                value += f' ± {row["sample_std"]:.6f}'
            values.append(value)
        name = {'majority': 'Classe majoritaire', 'tfidf_char': 'TF-IDF caractères',
                'tfidf_word': 'TF-IDF mots', 'tfidf_word_char': 'TF-IDF mots + caractères',
                'lexicon': 'Vote lexical', 'lexicon_negation': 'Vote lexical avec négation',
                'bigru': 'BiGRU', 'frozen': 'CAMeLBERT gelé, moyenne des vecteurs'}.get(family.split(' / ')[0], family.split(' / ')[0])
        if ' / ' in family:
            variant = family.split(' / ', 1)[1]
            name += ' / ' + {'raw-v0': 'brut', 'norm-nfc-space-v1': 'NFC et espaces',
                             'zero_shot': 'zero-shot', 'few_shot': 'few-shot'}.get(variant, variant)
        lines.append(f'| {name} | {values[0]} | {values[1]} |')
    lines += ['', 'La référence statistique de modèle de langue est évaluée séparément par perplexité.', '',
              '| Corpus | Ordre du modèle caractères | Perplexité validation |', '| --- | ---: | ---: |']
    for row in result['statistical_language_models']:
        lines.append(f'| {row["corpus"].upper()} | {row["order"]} | {row["metrics"]["perplexity"]:.6f} |')
    lines.extend(['', 'La comparaison native CLS gelé/adapté de la campagne initiale est conservée '
                  'séparément de la confirmation finale à taux sélectionnés. La référence gelée '
                  'avec moyenne des vecteurs constitue une autre représentation.', '',
                  'Les matrices LoRA permettent de recalculer les scores et leur dispersion. '
                  'Les fichiers de prédictions individuelles CUDA et le résumé original complet '
                  'de campagne sont absents des archives reçues. Leur absence reste déclarée.', '',
                  'Le RAG dispose de résultats de validation et de test documentaire. Ses références '
                  'de pertinence restent proposées par IA. L’exigence de double revue humaine '
                  'et l’acceptation d’une adaptation au travail individuel sont ouvertes.', '',
                  'Les preuves d’entraînement, les artefacts locaux et les preuves de rechargement '
                  'enregistrées sont contrôlés. Cette consolidation n’exécute aucune nouvelle inférence. '
                  'Une sauvegarde sur support indépendant et une relance par un autre opérateur '
                  'restent à confirmer. Le test officiel de classification reste réservé.', '',
                  '[Scores, ablations et empreintes](summary.json)', ''])
    return '\n'.join(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Choisir un nouveau dossier pour préserver les bilans précédents.')
    result = build()
    args.output.mkdir(parents=True, exist_ok=False)
    path = args.output / 'summary.json'
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    if json.loads(path.read_text()) != result:
        raise ValueError('Le bilan sauvegardé diffère du résultat contrôlé.')
    (args.output / 'README.md').write_text(markdown(result))
    print(f'Bilan sauvegardé — {args.output.resolve()}', flush=True)
