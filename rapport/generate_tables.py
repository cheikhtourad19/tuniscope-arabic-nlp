"""Generate LaTeX tables and plot values from the checked aggregate JSON files."""

from collections import Counter
import csv
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / 'build' / 'generated'
CORPORA = ('tsac', 'madar')
TABLES = []


def load(name, file='summary.json'):
    return json.loads((ROOT / 'reports' / name / file).read_text())


def decimal(number, digits=2):
    return f'{number:.{digits}f}'.replace('.', '{,}')


def pct(number):
    return f'{decimal(100 * number)}\\,\\%'


def whole(number):
    return f'{number:,}'.replace(',', r'\,')


def table(name, rows):
    lines = [rf'\expandafter\def\csname tuniscoperows@{name}\endcsname{{%']
    lines.extend(' & '.join(map(str, row)) + r' \\' for row in rows)
    TABLES.extend([*lines, '}'])


def macro(name, text):
    return rf'\newcommand{{\{name}}}{{{text}}}'


def partitions():
    rows = []
    for corpus in CORPORA:
        labels = ('NEG', 'POS') if corpus == 'tsac' else ('MSA', 'TN', 'EG')
        with (ROOT / 'data' / 'manifests' / 'v2' / f'{corpus}.csv').open(encoding='utf-8', newline='') as stream:
            counts = Counter((row['split'], row['label']) for row in csv.DictReader(stream))
        for split, title in [('train', 'Train'), ('validation', 'Validation'), ('test', 'Test réservé')]:
            values = [counts[split, label] for label in labels]
            detail = ', '.join(f'{label} {whole(count)}' for label, count in zip(labels, values))
            rows.append([corpus.upper(), title, detail, whole(sum(values))])
    table('partitions', rows)


def classical():
    systems = load('baselines_grouped_v2_2026-10-17', 'metrics.json')['results']
    char = {(r['corpus'], r['system'], r['variant']): r['metrics']['macro_f1'] for r in systems}
    lexical = load('rule_baselines_v1_2026-10-17')['results']
    lex = {r['corpus']: r['metrics']['macro_f1'] for r in lexical if r['system'] == 'lexicon'}
    recurrent = load('bigru_v1_2026-11-04')['aggregate']
    frozen = load('camelbert_frozen_v1_2026-11-04')['aggregates']
    adapted = load('camelbert_confirmation_2026-11-06')['aggregates']
    rows = []
    for title, method, variant in [('Classe majoritaire', 'majority', 'raw-v0'),
                                   ('TF-IDF caractères, brut', 'tfidf_char', 'raw-v0'),
                                   ('TF-IDF caractères, NFC/espaces', 'tfidf_char', 'norm-nfc-space-v1')]:
        rows.append([title, *(pct(char[c, method, variant]) for c in CORPORA), 'Fixe'])
    rows.insert(1, ['Vote lexical', *(pct(lex[c]) for c in CORPORA), 'Fixe'])

    def spread(mean, std):
        return f'{decimal(100 * mean)} $\\pm$ {pct(std)}'

    for label, result in [('BiGRU', recurrent), ('CAMeLBERT gelé', frozen)]:
        rows.append([label, *(spread(result[c]['mean_macro_f1'], result[c]['sample_std_macro_f1'])
                              for c in CORPORA), '3 graines'])
    rows.append(['CAMeLBERT adapté', *(spread(adapted[c]['macro_f1']['mean'], adapted[c]['macro_f1']['sample_std'])
                                       for c in CORPORA), '3 graines'])
    table('classiques', rows)
    lm = load('character_lm_v1_2026-11-04')['results']
    pp = {(row['corpus'], row['order']): row['metrics']['perplexity'] for row in lm}
    table('langage', ([c.upper(), decimal(pp[c, 1]), decimal(pp[c, 3])] for c in CORPORA))


def qwen():
    old = load('qwen_lora_cuda_2026-11-07')['aggregate']
    table('qwen06', ([key.split('_')[0].upper(), 'zero-shot' if 'zero' in key else 'few-shot',
                      pct(data['base_macro_f1']), f"{decimal(data['mean_macro_f1'] * 100)} $\\pm$ {pct(data['stdev_macro_f1'])}"]
                     for key, data in old.items()))
    q = load('qwen3_4b_qlora_v1')
    results = q['results']
    rows = []
    statistics = []
    for corpus in CORPORA:
        before, after = (results[corpus][key] for key in ('base', 'qlora'))
        diff = after['macro_f1'] - before['macro_f1']
        rows.append([corpus.upper(), whole(after['n']), pct(before['macro_f1']), pct(after['macro_f1']),
                     f'{decimal(100 * diff)} pp', pct(after['accuracy']),
                     f"{before['invalid_rows']} $\\to$ {after['invalid_rows']}"])
        for label in before['labels']:
            statistics.append([corpus.upper(), label, whole(int(before['per_class'][label]['support'])),
                               pct(before['per_class'][label]['recall']), pct(after['per_class'][label]['recall'])])
    table('qwen4b', rows)
    table('qwen4b-classes', statistics)
    for corpus in CORPORA:
        for stage in ('base', 'qlora'):
            mat = results[corpus][stage]
            if len(mat['labels']) != len(mat['confusion_matrix']):
                raise ValueError(f'Matrice incomplète : {corpus}/{stage}')
            table(f'{corpus}-{stage}', ([label, *counts] for label, counts in zip(mat['labels'], mat['confusion_matrix'])))
    for corpus in CORPORA:
        before, after = (results[corpus][key]['per_class'] for key in ('base', 'qlora'))
        table(f'{corpus}-precision-f1', ([label, pct(before[label]['precision']), pct(after[label]['precision']),
                                        pct(before[label]['f1-score']), pct(after[label]['f1-score'])]
                                       for label in before))

    after_seconds = sum(results[c]['qlora']['seconds'] for c in CORPORA)
    hours, rest = divmod(q['training_seconds_from_log'], 3600)
    minutes, seconds = divmod(rest, 60)
    values = [macro('TrainingDuration', rf'{hours}\,h\,{minutes:02d}\,min\,{seconds:02d}\,s'),
              macro('AfterDuration', rf'{round(after_seconds) // 60}\,min\,{round(after_seconds) % 60:02d}\,s'),
              macro('AdapterParameters', whole(q['completion']['trainable_parameters'])),
              macro('TrainMemory', rf"{decimal(q['completion']['peak_mlx_bytes'] / 1e9)}\,Go"),
              macro('AdapterSize', rf"{decimal(q['adapter_bytes'] / 1e6)}\,Mo")]
    for corpus, name in [('tsac', 'Tsac'), ('madar', 'Madar')]:
        values += [macro(f'Qwen{name}{term}', f"{100 * results[corpus][stage]['macro_f1']:.2f}")
                   for term, stage in [('Base', 'base'), ('Adapted', 'qlora')]]
        values += [macro(f'Qwen{name}{term}Percent', pct(results[corpus][stage]['macro_f1']))
                   for term, stage in [('Base', 'base'), ('Adapted', 'qlora')]]
    (BUILD / 'values.tex').write_text('\n'.join(values) + '\n', encoding='utf-8')


def documentary_and_audit():
    source = load('rag_documentary_test_v1_2026-11-12', 'results.json')['systems']
    bm25, e5 = (source[name]['all'] for name in ('bm25', 'e5'))
    rows = [
        ['Rappel à 3, questions documentées', *(pct(r['recall_at_3']) for r in (bm25, e5))],
        ['MRR à 3, questions documentées', *(decimal(r['mrr_at_3'], 3) for r in (bm25, e5))],
        ['Réponses pertinentes', *(f"{r['answered_with_relevant_citation']}/{r['answerable']}" for r in (bm25, e5))],
        ['Pertinence parmi les réponses émises', *(f"{r['answered_with_relevant_citation']}/{r['answered']}" for r in (bm25, e5))],
        ['Réponses non pertinentes', *(r['answers_with_irrelevant_citation'] for r in (bm25, e5))],
        ['Abstentions sur les questions sans source', *(f"{r['unanswerable_rejected']}/{r['unanswerable']}" for r in (bm25, e5))],
    ]
    table('retrieval', rows)
    budget = load('gold_validation_2026-11-22')['budget']['summary']
    table('budget', ([f"{r['percent']}\\,\\%", f"{whole(r['min_train_rows'])}–{whole(r['max_train_rows'])}"
                     if r['min_train_rows'] != r['max_train_rows'] else whole(r['min_train_rows']),
                     pct(r['mean_macro_f1']), decimal(100 * r['sd_macro_f1'])] for r in budget))
    boot = load('gold_audit_2026-11-22', 'validation_audit.json')['bootstrap']
    table('bootstrap', ([c.upper(), whole(row['groups']), decimal(100 * row['difference']),
                         '[' + ' à '.join(decimal(100 * value) for value in row['percentile_interval_95']) + ']']
                        for c, row in boot.items()))


def supplementary_models():
    representations = load('three_representations_v2_2026-10-17', 'comparison.json')['scores']
    variants = {'raw-v0': 'Brut', 'norm-nfc-space-v1': 'NFC/espaces'}
    table('representations', ([r['corpus'].upper(), variants[r['variant']],
                              *(pct(r['macro_f1'][key]) for key in ('word', 'char', 'combined')), r['combined_errors']]
                             for r in representations))

    hybrids = load('character_rules_v1_2026-10-19')['results']
    systems = {'char_control': 'Caractères', 'char_lexicon': 'Caractères + lexique',
               'char_lexicon_negation': 'Caractères + lexique et négation'}
    table('hybrides', ([r['corpus'].upper(), systems[r['system']], pct(r['metrics']['macro_f1']),
                       pct(r['metrics']['accuracy'])] for r in hybrids))

    tokenizer = load('camelbert_tokenizer_v1_2026-11-04')['corpora']
    table('troncature', ([corpus.upper(), name, whole(m['rows']), m['length_with_special']['p95'],
                          m['length_with_special']['max'], whole(m['candidate_truncation']['128']['rows'])]
                         for corpus in CORPORA for split, name in [('train', 'Train'), ('validation', 'Validation')]
                         for m in (tokenizer[corpus][split]['all'],)))

    encoder = load('camelbert_finetuning_v1_2026-11-05')['aggregates']
    table('cls-paire', ([corpus.upper(), pct(encoder[f'{corpus}:frozen']['mean_macro_f1']),
                        pct(encoder[f'{corpus}:finetuned']['mean_macro_f1'])] for corpus in CORPORA))
    screen = load('camelbert_screening_mps_2026-11-06')['comparisons']
    table('criblage', ([r['corpus'].upper(), f"{r['learning_rate']:.0e}",
                        pct(r['macro_f1']), r['selected_epoch']] for r in sorted(screen, key=lambda r: (r['corpus'], r['learning_rate']), reverse=True)))

    for name, folder in [('bigru-graines', 'bigru_v1_2026-11-04'), ('encoder-graines', 'camelbert_confirmation_2026-11-06')]:
        runs = load(folder)['results']
        table(name, ([r['corpus'].upper(), r['seed'], r['best_epoch'], r['epochs_completed'],
                      pct(r['metrics']['macro_f1']), pct(r['metrics']['accuracy'])] for r in runs))

    old = load('qwen_lora_cuda_2026-11-07')['aggregate']
    table('qwen06-graines', ([key.split('_')[0].upper(), 'Zero-shot' if 'zero' in key else 'Few-shot',
                             *(pct(r['seeds'][str(seed)]) for seed in (13, 42, 2026))] for key, r in old.items()))


def supplementary_audits():
    robustness = load('gold_audit_2026-11-22', 'robustness.json')['models']
    rows = []
    for key, name in [('normalisation_nfc_espaces', 'NFC et espaces'), ('espaces_triplement', 'Espaces triplés'),
                      ('lettre_dupliquee', 'Duplication d’une lettre')]:
        base, alternate = (robustness[model][key] for model in ('tfidf_char', 'camelbert'))
        rows.append([name, whole(base['changed']['n']), base['all']['prediction_flips'],
                     decimal(100 * base['all']['macro_f1_difference']), alternate['all']['prediction_flips'],
                     decimal(100 * alternate['all']['macro_f1_difference'])])
    table('robustesse', rows)
    cal = load('gold_diagnostics_2026-11-22', 'calibration.json')['tasks']
    table('calibration', ([c.upper(), whole(item['raw']['n']), decimal(100 * item['raw']['ece_10']),
                           decimal(100 * item['cross_fit']['ece_10']), decimal(item['raw']['nll'], 3),
                           decimal(item['cross_fit']['nll'], 3), pct(item['cross_fit']['accuracy'])]
                          for c, item in cal.items()))
    diagnostic = load('gold_diagnostics_2026-11-22', 'paired_qwen.json')['tasks']
    table('qwen06-diagnostic', ([c.upper(), b['rows'], pct(b['reference_macro_f1']), pct(b['mean_alt_macro_f1']),
                                '[' + ' à '.join(decimal(100 * value) for value in b['percentile_interval_95']) + ']']
                               for c, item in diagnostic.items() for b in (item['paired_bootstrap'],)))

    for partition, folder in [('validation', 'rag_application_validation_2026-11-11'),
                              ('test', 'rag_documentary_test_v1_2026-11-12')]:
        systems = load(folder, 'results.json')['systems']
        table(f'rag-langues-{partition}', ([language_name, name.upper(), pct(r['recall_at_3']),
                                            decimal(r['mrr_at_3'], 3),
                                            f"{r['answered_with_relevant_citation']}/{r['answerable']}",
                                            f"{r['unanswerable_rejected']}/{r['unanswerable']}"]
                                           for language, language_name in [('fr', 'Français'), ('ar', 'Arabe standard')]
                                           for name in ('bm25', 'e5') for r in (systems[name]['by_language'][language],)))


if __name__ == '__main__':
    BUILD.mkdir(parents=True, exist_ok=True)
    partitions()
    classical()
    qwen()
    documentary_and_audit()
    supplementary_models()
    supplementary_audits()
    (BUILD / 'tables.tex').write_text('\n'.join(TABLES) + '\n', encoding='utf-8')
