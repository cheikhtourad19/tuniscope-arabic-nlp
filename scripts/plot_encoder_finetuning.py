"""Export native CLS learning curves and matrices from one verified run."""
import argparse,json,os
from pathlib import Path
from character_lm_experiment import ROOT
from fetch_camelbert_encoder import sha256
from run_frozen_encoder import save


def export(directory):
    d=json.loads((directory/'results.json').read_text());proof=json.loads((directory/'verification.json').read_text())
    if proof['results_sha256']!=sha256(directory/'results.json'):raise ValueError('Verify this exact run first')
    for n,h in d['artifacts'].items():
        p=(directory/n).resolve()
        if directory.resolve() not in p.parents or sha256(p)!=h:raise ValueError('Artifact changed')
    os.environ['MPLBACKEND']='Agg';os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.tools/matplotlib-cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=directory/'figures'
    if target.exists():raise ValueError('Preserve prior figures')
    target.mkdir();provenance={'source_results_sha256':sha256(directory/'results.json'),'files':{},'mode':d['mode']}
    for r in d['results']:
        history=json.loads((directory/r['checkpoint']/'history.json').read_text());epochs=[h['epoch'] for h in history]
        fig,axes=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
        axes[0].plot(epochs,[h['train']['loss'] for h in history],marker='.',label='Train')
        axes[0].plot(epochs,[h['validation_loss'] for h in history],marker='.',label='Validation');axes[0].legend()
        axes[0].set(title='Perte de classification',xlabel='Époque',ylabel='Entropie croisée')
        axes[1].plot(epochs,[h['validation_macro_f1'] for h in history],marker='o',color='#167a8a')
        axes[1].axvline(r['best_epoch'],ls='--',color='gray',label='Checkpoint retenu');axes[1].legend()
        axes[1].set(title='Macro-F1 de validation',xlabel='Époque',ylabel='Macro-F1',ylim=(0,1))
        matrix=r['metrics']['confusion_matrix'];labels=r['metrics']['labels'];axes[2].imshow(matrix,cmap='Blues',vmin=0)
        axes[2].set(xticks=range(len(labels)),yticks=range(len(labels)),xticklabels=labels,yticklabels=labels,
                    title='Checkpoint retenu',xlabel='Prédiction',ylabel='Label réel')
        maximum=max(map(max,matrix))
        for i in range(len(labels)):
            for j in range(len(labels)):
                axes[2].text(j,i,str(matrix[i][j]),ha='center',va='center',color='white' if matrix[i][j]>maximum/2 else 'black')
        fig.suptitle(f'CAMeLBERT CLS — {r["corpus"].upper()} — {r["regime"]} — graine{r["seed"]} — {d["mode"]}\n'
                     +('Essai technique100 lignes/partition ; aucun résultat FULL' if d['mode']=='smoke' else 'Validation v2 ; test officiel non évalué'))
        name=f'{r["checkpoint"]}.png';fig.savefig(target/name,dpi=160);plt.close(fig);provenance['files'][name]=sha256(target/name)
    save(target/'provenance.json',provenance);print(f'Images enregistrées : {target}');return target


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);export(p.parse_args().run.resolve())
