"""Aggregate verified FULL paired CLS runs, with explicit historical references."""
import argparse,json
from pathlib import Path
from character_lm_experiment import ROOT
from fetch_camelbert_encoder import sha256
from run_frozen_encoder import save


def report(directory,target,classical_run,mean_run,bigru_run):
    if target.exists():raise ValueError('Preserve previous report')
    d=json.loads((directory/'results.json').read_text());proof=json.loads((directory/'verification.json').read_text())
    if d['mode']!='full' or proof['results_sha256']!=sha256(directory/'results.json') or d['test_loaded'] is not False:raise ValueError('Need verified full')
    classical=json.loads((classical_run/'results.json').read_text());mean=json.loads((mean_run/'results.json').read_text());bigru=json.loads((bigru_run/'results.json').read_text())
    if any(x['mode']!='full' or x['test_loaded'] is not False for x in (classical,mean,bigru)):raise ValueError('Full references only')
    comparison={}
    for corpus in ('tsac','madar'):
        current=[r for r in d['results'] if r['corpus']==corpus]
        expected=json.loads((directory/current[0]['checkpoint']/'predictions.json').read_text())
        keys=[(p['id'],p['group_id'],p['truth']) for p in expected]
        refs=[r for r in classical['results'] if r['corpus']==corpus and r['system']=='tfidf_char' and r['variant']=='raw-v0']
        if len(refs)!=1:raise ValueError('Unique raw character reference required')
        ref=refs[0];path=classical_run/ref['predictions']
        if sha256(path)!=ref['predictions_sha256']:raise ValueError('Reference checksum mismatch')
        old=json.loads(path.read_text())
        if [(p['id'],p['group_id'],p['truth']) for p in old]!=keys:raise ValueError('Different validation examples')
        for other in (mean,bigru):
            for r in [r for r in other['results'] if r['corpus']==corpus]:
                if r['validation']!=current[0]['validation']:raise ValueError('Reference partition mismatch')
        comparison[corpus]={'tfidf_char_macro_f1':ref['metrics']['macro_f1'],
            'bigru_mean_macro_f1':sum(r['metrics']['macro_f1'] for r in bigru['results'] if r['corpus']==corpus)/3,
            'mean_pooling_frozen':mean['aggregates'][corpus],
            'native_cls':{arm:d['aggregates'][f'{corpus}:{arm}'] for arm in ('frozen','finetuned') if f'{corpus}:{arm}' in d['aggregates']}}
    payload={k:d[k] for k in ('mode','regime','config','device','packages','environment_lock_sha256','other_locks_sha256',
              'manifests_sha256','seconds','peak_process_rss_bytes','aggregates')}
    payload.update(test_loaded=False,comparison=comparison,verification=proof,source_results_sha256=sha256(directory/'results.json'),
                   reference_results_sha256={str(p.relative_to(ROOT)):sha256(p/'results.json') for p in (classical_run,mean_run,bigru_run)})
    payload['results']=[{k:r[k] for k in ('corpus','seed','regime','metrics','best_epoch','epochs_completed','parameters','trainable_parameters',
                                         'epoch_train_seconds','epoch_validation_seconds','backbone_changed_during_training')} for r in d['results']]
    payload['report_generator_sha256']=sha256(Path(__file__))
    observed={}
    for r,item in zip(d['results'],payload['results']):
        folder=directory/r['checkpoint']
        history=json.loads((folder/'history.json').read_text())
        item['saved_checkpoint_directory_bytes']=sum(p.stat().st_size for p in folder.rglob('*') if p.is_file())
        item['observed_device_memory']={}
        for epoch in history:
            for name,value in epoch['train']['memory'].items():
                item['observed_device_memory'][name]=max(item['observed_device_memory'].get(name,0),value)
                observed[name]=max(observed.get(name,0),value)
    payload['maximum_observed_device_memory']=observed
    payload['memory_note']='MPS snapshots sampled after backward/step are not a guaranteed absolute peak. Apple unified CPU/GPU memory is not obtained by adding RSS and MPS allocations.'
    lines=['# CAMeLBERT CLS : référence gelée et fine-tuning','','Résultats **de validation v2** générés depuis un run FULL vérifié. Aucun test officiel évalué.',
           'L05/L07/L12 partiels : runs initiaux trois graines vérifiés ; configurations finales, analyses et test restent distincts.','',
           '| Corpus | Régime | Graine | Accuracy | Macro-F1 | Époque retenue / exécutées |','| --- | --- | ---: | ---: | ---: | ---: |']
    for r in d['results']:
        lines.append(f'| {r["corpus"].upper()} | {r["regime"]} | {r["seed"]} | {r["metrics"]["accuracy"]:.6f} | {r["metrics"]["macro_f1"]:.6f} | {r["best_epoch"]} / {r["epochs_completed"]} |')
    lines+=['','| Corpus | Méthode | Macro-F1 |','| --- | --- | ---: |']
    for corpus,c in comparison.items():
        lines.extend([f'| {corpus.upper()} | TF-IDF caractères brut | {c["tfidf_char_macro_f1"]:.6f} |',
                      f'| {corpus.upper()} | BiGRU, moyenne3graines | {c["bigru_mean_macro_f1"]:.6f} |'])
        a=c['mean_pooling_frozen'];lines.append(f'| {corpus.upper()} | Gelé mean-pooling, protocole antérieur | {a["mean_macro_f1"]:.6f} ± {a["sample_std_macro_f1"]:.6f} |')
        for arm,a in c['native_cls'].items():lines.append(f'| {corpus.upper()} | CLS {arm} | {a["mean_macro_f1"]:.6f} ± {a["sample_std_macro_f1"]:.6f} |')
    lines+=['',f'Durée réelle du run : **{d["seconds"]/60:.2f} min** ; appareil **{d["device"]}**. Hors installation, smoke et contrôles/export ultérieurs.',
            'Écart-type échantillonnal ; aucune meilleure graine sélectionnée. Les trois graines ne sont pas trois tests indépendants.',
            'Le témoin CLS partage initialisation, architecture, ordre de données et hyperparamètres avec l’adapté ; la référence mean-pooling suit une autre architecture et un budget différent.',
            'Le faible learning rate et le budget court du témoin CLS peuvent limiter sa tête ; ne pas présenter ce résultat comme le meilleur encodeur gelé possible.',
            'MPS utilise un avertissement pour les kernels sans déterminisme strict ; pas de répétabilité bit à bit promise.',
            'Aucun bootstrap/test final, essai CUDA, remplacement applicatif ou expérience LoRA n’est établi par ce run.',
            '',f'Run : `{directory.relative_to(ROOT)}`. Poids, exemples et prédictions individuelles restent privés.','']
    target.mkdir(parents=True);save(target/'summary.json',payload);(target/'README.md').write_text('\n'.join(lines))
    for corpus,c in comparison.items():
        for arm,a in c['native_cls'].items():
            card=[f'# Carte initiale — {corpus.upper()} / CAMeLBERT CLS {arm}','','État : expérience initiale de validation, non modèle final déployé.',
                  '- Base : CAMeL-Lab/bert-base-arabic-camelbert-mix, révision9be352797bdf28a9ae21e2ae582aaaca7abdb22d ; Apache-2.0 déclaré dans la fiche officielle.',
                  '- Tokenizer WordPiece officiel inchangé, texte brut, longueur128 ; tête CLS/pooler native.',
                  f'- Tâche et labels : {corpus}, {d["config"]["labels"][corpus]}.',
                  '- Sources : TSAC avis tunisiens de réseaux sociaux ; MADAR phrases de voyage MSA/Tunis/Sfax/Cairo, selon tâche. Aucune couverture universelle des dialectes.',
                  f'- Régime : {arm}. AdamW2e-5,3époquesmax,patience2,batch8accum2 ; graines13/42/2026.',
                  f'- Validation macro-F1 : {a["mean_macro_f1"]:.6f} ± {a["sample_std_macro_f1"]:.6f}. Résultats par classe/écriture dans summary.json.',
                  '- Rechargements et probabilités vérifiés ; scores softmax non calibrés, aucune garantie pour hors domaine/neutre/français seul.',
                  '- Absence de test final, étude exhaustive d’erreurs, calibration et relance indépendante. MPS warn_only ne garantit pas le déterminisme strict.',
                  '- Checkpoints privés référencés par résultats/empreintes ; aucun poids ou texte protégé redistribué. Application inchangée.','']
            (target/f'carte_{corpus}_{arm}.md').write_text('\n'.join(card))
    print(f'Rapport agrégé et cartes initiales : {target}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--classical-run',type=Path,default=ROOT/'outputs/baselines/2026-10-17_grouped_v2_full')
    p.add_argument('--mean-run',type=Path,default=ROOT/'outputs/frozen_encoder/2026-11-04_camelbert_frozen_mps_full')
    p.add_argument('--bigru-run',type=Path,default=ROOT/'outputs/neural_baselines/2026-11-04_bigru_v1_cpu_full')
    a=p.parse_args();report(a.run.resolve(),a.output.resolve(),a.classical_run.resolve(),a.mean_run.resolve(),a.bigru_run.resolve())
