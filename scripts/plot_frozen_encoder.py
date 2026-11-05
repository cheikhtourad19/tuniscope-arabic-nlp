"""Six same-run head-learning and confusion PNGs, without retraining."""
import argparse
import json
import os
from pathlib import Path

from character_lm_experiment import ROOT, read_artifact
from fetch_camelbert_encoder import sha256
from run_frozen_encoder import save
from verify_frozen_encoder import verify


def export_figures(directory):
    os.environ['MPLBACKEND'] = 'Agg'
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.tools/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    saved = verify(directory, reencode=False)
    target = directory / 'figures'
    if target.exists():
        raise ValueError('Preserve existing figures; do not overwrite')
    target.mkdir()
    provenance = {'source_results_sha256': sha256(directory / 'results.json'), 'split': 'validation', 'files': {}}
    for r in saved['results']:
        history = json.loads(read_artifact(directory, r['history']).read_text())
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), layout='constrained')
        epochs = [h['epoch'] for h in history]
        axes[0].plot(epochs, [h['train_loss'] for h in history], label='Train', marker='.')
        axes[0].plot(epochs, [h['validation_loss'] for h in history], label='Validation', marker='.')
        axes[0].set(title='Perte de la tête linéaire', xlabel='Époque', ylabel='Entropie croisée')
        axes[0].legend()
        axes[1].plot(epochs, [h['validation_macro_f1'] for h in history], marker='.', color='#167a8a')
        axes[1].axvline(r['best_epoch'], linestyle='--', color='gray', label='Checkpoint retenu')
        axes[1].set(title='Macro-F1 de validation', xlabel='Époque de la tête', ylabel='Macro-F1', ylim=(0, 1))
        axes[1].legend()
        matrix, labels = r['metrics']['confusion_matrix'], r['metrics']['labels']
        axes[2].imshow(matrix, cmap='Blues', vmin=0)
        axes[2].set(xticks=range(len(labels)), yticks=range(len(labels)), xticklabels=labels, yticklabels=labels,
                    title='Checkpoint retenu', xlabel='Prédiction', ylabel='Label réel')
        maximum = max(map(max, matrix))
        for i in range(len(labels)):
            for j in range(len(labels)):
                axes[2].text(j, i, str(matrix[i][j]), ha='center', va='center',
                             color='white' if matrix[i][j] > maximum / 2 else 'black')
        fig.suptitle(f'CAMeLBERT gelé — {r["corpus"].upper()} — graine {r["seed"]} — {saved["mode"]}\n'
                     'Seule la tête apprend ; validation v2, test officiel non évalué', fontsize=12)
        filename = f'{r["corpus"]}_seed_{r["seed"]}.png'
        fig.savefig(target / filename, dpi=160)
        plt.close(fig)
        provenance['files'][filename] = sha256(target / filename)
    save(target / 'provenance.json', provenance)
    print(f'Figures enregistrées : {target}', flush=True)
    return target


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    export_figures(parser.parse_args().run.resolve())
