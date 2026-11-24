"""Plot a verified documentary validation or test report."""
import argparse
import hashlib
import json
from pathlib import Path


def plot(run):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    source = run / 'results.json'
    proof = json.loads((run / 'verification.json').read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    verified = proof.get('rankings_and_metrics_recalculated',
                         proof.get('questions_rankings_and_metrics_verified', False))
    if digest != proof['results_sha256'] or not verified:
        raise ValueError('Le rapport ne correspond pas à la vérification enregistrée.')
    result = json.loads(source.read_text())
    if result['split'] not in ('validation', 'test') or result['top_k'] != 3:
        raise ValueError('Choisir un rapport documentaire vérifié à trois passages.')
    population = result['systems']['bm25']['all']
    supported, unsupported = population['answerable'], population['unanswerable']
    phase = 'Validation de développement' if result['split'] == 'validation' else 'Test documentaire'
    footer = f'{phase} · {population["questions"]} questions FR/AR · Pertinence proposée par IA'
    output = run / 'figures'
    output.mkdir(exist_ok=False)
    colours = ['#24689b', '#d27b32']
    figures = []
    for filename, title, labels, keys in [
        ('recherche.png', 'Classement des fiches avant abstention',
         ['Recall@3', 'MRR@3'], ['recall_at_3', 'mrr_at_3']),
        ('reponses.png', 'Réponses effectivement affichées',
         [f'Citation pertinente\nQuestions documentées\n(n = {supported})',
          f'Abstention attendue\nQuestions sans réponse\n(n = {unsupported})'],
         ['answered_with_relevant_citation', 'unanswerable_rejected']),
    ]:
        figure, axis = plt.subplots(figsize=(9, 5.6))
        for index, method in enumerate(('bm25', 'e5')):
            metrics = result['systems'][method]['all']
            values = [metrics[key] for key in keys]
            captions = [f'{value:.1%}' for value in values]
            if filename == 'reponses.png':
                denominators = [metrics['answerable'], metrics['unanswerable']]
                captions = [f'{value}/{total}' for value, total in zip(values, denominators, strict=True)]
                values = [value / total for value, total in zip(values, denominators, strict=True)]
            positions = [position + (index - 0.5) * 0.32 for position in range(2)]
            bars = axis.bar(positions, values, width=0.3, color=colours[index], label=method.upper())
            axis.bar_label(bars, labels=captions, padding=5)
        axis.set_xticks([0, 1], labels)
        axis.set_ylim(0, 1.13)
        axis.yaxis.set_major_formatter(PercentFormatter(1))
        axis.set_title(title, pad=28, fontweight='bold')
        axis.spines[['top', 'right']].set_visible(False)
        axis.set_axisbelow(True)
        axis.grid(axis='y', alpha=0.2)
        axis.legend(loc='upper center', bbox_to_anchor=(0.5, 1.08), ncols=2, frameon=False)
        if filename == 'recherche.png':
            axis.set_ylabel(f'Score moyen sur les {supported} questions documentées')
        else:
            axis.set_ylabel('Proportion de décisions attendues')
        figure.text(0.5, 0.025, footer,
                    ha='center', fontsize=9, color='#555555')
        figure.tight_layout(rect=(0, 0.055, 1, 1))
        path = output / filename
        figure.savefig(path, dpi=160, facecolor='white')
        plt.close(figure)
        figures.append(path)
    figure, axis = plt.subplots(figsize=(10, 4.8))
    categories = [
        ('Réponse pertinente', '#2c8559'), ('Réponse avec citation non pertinente', '#bd4c4c'),
        ('Abstention attendue', '#397aa3'), ('Abstention sur question documentée', '#adb4bd'),
    ]
    offsets = [0, 0]
    counts = []
    for method in ('bm25', 'e5'):
        row = result['systems'][method]['all']
        counts.append([row['answered_with_relevant_citation'],
                       row['answered'] - row['answered_with_relevant_citation'],
                       row['unanswerable_rejected'], row['answerable'] - row['answerable_accepted']])
    for index, (label, colour) in enumerate(categories):
        values = [row[index] for row in counts]
        bars = axis.barh(['BM25', 'E5'], values, left=offsets, color=colour, label=label)
        axis.bar_label(bars, labels=[str(value) if value else '' for value in values],
                       label_type='center', color='white' if index < 3 else '#1d2630', fontweight='bold')
        offsets = [start + value for start, value in zip(offsets, values, strict=True)]
    axis.set_xlim(0, population['questions'])
    axis.set_xlabel('Nombre de questions')
    axis.set_title(f'Décisions sur les {population["questions"]} questions', fontweight='bold', pad=16)
    axis.spines[['top', 'right']].set_visible(False)
    axis.invert_yaxis()
    axis.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18), ncols=2, frameon=False, fontsize=9)
    figure.text(0.5, 0.025, footer, ha='center', fontsize=9, color='#555555')
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    path = output / 'decisions.png'
    figure.savefig(path, dpi=160, facecolor='white')
    plt.close(figure)
    figures.append(path)
    manifest = {
        'results_sha256': digest,
        'plotter_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figures': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in figures},
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return figures


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    for path in plot(args.run):
        print(path.resolve())
