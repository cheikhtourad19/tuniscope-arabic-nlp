"""Confirm fixed per-task rates on three seeds; no hyperparameter selection or test."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import time
from zipfile import ZipFile, ZIP_DEFLATED

from encoder_confirmation_plan import ROOT, PLAN, confirmation_plan, confirmation_config, digest
from encoder_source_provenance import verify_recorded_sources
from run_encoder_screening import child, inside, stage_reference
from baseline_metrics import classification_metrics
from bigru_data import partitions
from character_lm_experiment import partition_summary


def read(path):
    return json.loads(path.read_text())


def checked_stage(campaign, stage, corpus, mode):
    folder = inside(campaign, stage['directory'])
    if digest(folder/'results.json') != stage['results_sha256'] or digest(folder/'verification.json') != stage['verification_sha256']:
        raise ValueError('Confirmation stage results or proof changed')
    d, proof = read(folder/'results.json'), read(folder/'verification.json')
    expected_config = confirmation_config(corpus)
    seeds = [42] if mode == 'smoke' else [13,42,2026]
    arms = ['frozen','finetuned'] if mode == 'smoke' else ['finetuned']
    expected = {(corpus,s,a) for s in seeds for a in arms}
    if (d['mode'] != mode or d['experiment_scope'] != 'confirmation' or d['config'] != expected_config
            or d['test_loaded'] is not False or d['execution_seeds'] != seeds
            or d['regime'] != ('paired' if mode == 'smoke' else 'finetuned')
            or len(d['results']) != len(expected)
            or {(r['corpus'],r['seed'],r['regime']) for r in d['results']} != expected):
        raise ValueError('Wrong confirmation stage scope, tasks or seeds')
    p = confirmation_plan()
    if (d['confirmation_plan_sha256'] != digest(PLAN)
            or d['selection_report_sha256'] != p['selection_report_sha256']
            or d['protocol_sha256'] != digest(ROOT/p['protocol'])):
        raise ValueError('Confirmation provenance changed')
    if (proof['results_sha256'] != stage['results_sha256'] or proof['models_replayed'] != len(expected)
            or proof['epoch_class_metrics_complete'] is not True or proof['test_loaded'] is not False
            or proof['device'] != d['device']):
        raise ValueError('Incomplete confirmation stage replay evidence')
    for name,h in d['artifacts'].items():
        if digest(inside(folder,name)) != h: raise ValueError('Confirmation artifact changed')
    verify_recorded_sources(d['code_sha256'])
    if digest(ROOT/'environments/encoder/uv.lock') != d['environment_lock_sha256']:
        raise ValueError('Encoder lock changed')
    for name,h in d['other_locks_sha256'].items():
        if digest(ROOT/name) != h: raise ValueError('Other environment lock changed')
    if digest(ROOT/f'data/manifests/v2/{corpus}.csv') != d['manifests_sha256'][corpus]:
        raise ValueError('Confirmation manifest changed')
    train,val = partitions(corpus,mode,expected_config)
    for r in d['results']:
        if r['train'] != partition_summary(train) or r['validation'] != partition_summary(val):
            raise ValueError('Confirmation partitions changed')
        preds = read(inside(folder,r['checkpoint']+'/predictions.json'))
        if [(x['id'],x['group_id'],x['truth']) for x in preds] != [(x['id'],x['group_id'],x['label']) for x in val]:
            raise ValueError('Confirmation prediction alignment mismatch')
        actual = classification_metrics([x['truth'] for x in preds],[x['predicted'] for x in preds],expected_config['labels'][corpus])
        if actual != {k:r['metrics'][k] for k in actual}:
            raise ValueError('Confirmation metrics disagree with predictions')
    return d


def aggregate(models):
    if sorted(r['seed'] for r in models) != [13,42,2026] or len(models) != 3:
        raise ValueError('Report requires all three distinct seeds')
    return {metric:{'mean':statistics.mean(r['metrics'][metric] for r in models),
                    'sample_std':statistics.stdev(r['metrics'][metric] for r in models)}
            for metric in ('macro_f1','accuracy','balanced_accuracy')}


def verify_campaign(directory, expected_mode):
    directory = directory.resolve();d = read(directory/'confirmation.json');plan = confirmation_plan()
    if (expected_mode not in {'full','smoke'} or d['mode'] != expected_mode or d['test_loaded'] is not False
            or d['version'] != 'encoder-confirmation-v1'
            or d['fixed_learning_rates'] != plan['selected_learning_rates']):
        raise ValueError('Explicit confirmation mode mismatch')
    if d['plan_sha256'] != digest(PLAN) or d['selection_report_sha256'] != plan['selection_report_sha256']:
        raise ValueError('Campaign selection/plan changed')
    if d['protocol_sha256'] != digest(ROOT/plan['protocol']): raise ValueError('Protocol changed')
    verify_recorded_sources({'run_encoder_confirmation.py':d['workflow_sha256']})
    if [e['corpus'] for e in d['entries']] != plan['corpora']:
        raise ValueError('Missing or repeated confirmation task')
    hardware, packages, sources, stage_count, models = None, None, None, 0, 0
    for entry in d['entries']:
        corpus = entry['corpus'];smoke = checked_stage(directory,entry['smoke'],corpus,'smoke');stage_count += 1
        if hardware is None: hardware = smoke['hardware']
        if packages is None: packages = smoke['packages']
        if sources is None: sources = smoke['code_sha256']
        if smoke['hardware'] != hardware or smoke['packages'] != packages or smoke['code_sha256'] != sources:
            raise ValueError('Confirmation hardware/environment/code differs')
        if expected_mode == 'smoke':
            if entry.get('full') is not None or d['aggregates'] is not None:
                raise ValueError('Smoke cannot contain final three-seed results')
            models += 2;continue
        full = checked_stage(directory,entry['full'],corpus,'full');stage_count += 1
        if (full['hardware'] != hardware or full['packages'] != packages or full['code_sha256'] != sources
                or full['smoke_run'] != str(inside(directory,entry['smoke']['directory']))):
            raise ValueError('FULL confirmation differs from its preflight')
        if len({r['initial_backbone_sha256'] for r in full['results']}) != 1:
            raise ValueError('Seeds did not start from identical pretrained backbone')
        actual = aggregate(full['results'])
        if d['aggregates'][corpus] != actual:
            raise ValueError('Final means/stds differ from all three seed scores')
        models += 3
    return d, {'campaign_sha256':digest(directory/'confirmation.json'),'mode':expected_mode,
               'stages_checked':stage_count,'models_in_report':models,'test_loaded':False,
               'artifact_and_metric_checks':True,'prior_checkpoint_replay_proofs_checked':True,
               'new_checkpoint_inference_performed_by_campaign_check':False}


def export_evidence(directory):
    """A small transfer bundle of evidence; checkpoint weights remain in campaign folders."""
    target = directory/'evidence_without_weights.zip'
    if target.exists(): raise FileExistsError(target)
    with ZipFile(target,'w',compression=ZIP_DEFLATED,compresslevel=1) as archive:
        for path in sorted(directory.rglob('*')):
            if path.is_file() and path.suffix in {'.json','.md','.txt','.png'}:
                archive.write(path,str(path.relative_to(directory)))
        for source in [PLAN,ROOT/confirmation_plan()['protocol'],ROOT/confirmation_plan()['selection_report']]:
            archive.write(source,'project/'+str(source.relative_to(ROOT)))
    return target


def campaign(mode, output, device='auto'):
    plan = confirmation_plan();output = output.resolve()
    if mode not in {'full','smoke'} or device not in {'auto','cpu','mps','cuda'}: raise ValueError('Invalid mode/device')
    if (ROOT/'outputs/encoder_finetuning').resolve() not in output.parents or output.exists():
        raise ValueError('Choose a new private confirmation folder')
    output.mkdir(parents=True);start = time.perf_counter()
    d = {'version':'encoder-confirmation-v1','mode':mode,'test_loaded':False,
         'created_at_utc':datetime.now(timezone.utc).isoformat(),'plan_sha256':digest(PLAN),
         'selection_report_sha256':plan['selection_report_sha256'],'protocol_sha256':digest(ROOT/plan['protocol']),
         'workflow_sha256':digest(Path(__file__)),'fixed_learning_rates':plan['selected_learning_rates'],
         'entries':[],'aggregates':{} if mode == 'full' else None}
    try:
        for corpus in plan['corpora']:
            print(f'\n=== Confirmation {corpus.upper()} — taux fixé {plan["selected_learning_rates"][corpus]:g} ===',flush=True)
            smoke = output/f'{corpus}_smoke'
            shared = ['--device',device,'--experiment-scope','confirmation','--corpus',corpus]
            child('run_encoder_finetuning.py','--mode','smoke','--regime','paired','--output',smoke,*shared)
            child('verify_encoder_finetuning.py',smoke,'--device',device)
            entry = {'corpus':corpus,'smoke':stage_reference(smoke,output),'full':None}
            if mode == 'full':
                full = output/f'{corpus}_full'
                child('run_encoder_finetuning.py','--mode','full','--regime','finetuned',
                      '--output',full,'--smoke-run',smoke,*shared)
                child('verify_encoder_finetuning.py',full,'--device',device)
                child('plot_encoder_finetuning.py',full)
                entry['full'] = stage_reference(full,output)
                d['aggregates'][corpus] = aggregate(read(full/'results.json')['results'])
            else:
                child('plot_encoder_finetuning.py',smoke)
            d['entries'].append(entry)
        d['elapsed_seconds_before_final_campaign_checks'] = time.perf_counter()-start
        (output/'confirmation.json').write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
        _,proof = verify_campaign(output,mode)
        (output/'campaign_verification.json').write_text(json.dumps(proof,indent=2)+'\n')
        write_report(output,d)
        archive = export_evidence(output)
        print((output/'summary.md').read_text(),flush=True)
        print(f'Preuves sans poids : {archive}. Les poids restent dans les sous-dossiers des modèles.',flush=True)
        return output
    except Exception as error:
        (output/'failure.json').write_text(json.dumps({'error':str(error),'completed_entries':d['entries'],
            'confirmation_completed':False},ensure_ascii=False,indent=2)+'\n')
        raise


def write_report(directory,d):
    lines=['# Confirmation CAMeLBERT','',f'Mode **{d["mode"]}**. Taux fixés avant ces résultats. Test officiel non évalué.','']
    if d['mode'] == 'full':
        lines += ['| Corpus | Taux | Graine | Macro-F1 | Accuracy | Époque retenue |',
                  '| --- | ---: | ---: | ---: | ---: | ---: |']
        for entry in d['entries']:
            stage = read(directory/entry['full']['directory']/'results.json')
            for r in stage['results']:
                lines.append(f'| {r["corpus"].upper()} | {stage["config"]["learning_rate"]:g} | {r["seed"]} | {r["metrics"]["macro_f1"]:.6f} | {r["metrics"]["accuracy"]:.6f} | {r["best_epoch"]} |')
            corpus=entry['corpus']
            card={'corpus':corpus,'status':'three-seed-validation-confirmation-not-final-test',
                  'model_id':stage['config']['model_id'],'revision':stage['config']['revision'],'config':stage['config'],
                  'total_parameters':stage['results'][0]['parameters'],'trainable_parameters':stage['results'][0]['trainable_parameters'],
                  'validation_aggregates':d['aggregates'][corpus],'test_evaluated':False,
                  'checkpoint_references':[str(Path(entry['full']['directory'])/r['checkpoint']) for r in stage['results']],
                  'limitations':['selected on repeatedly used validation','MPS not bit-deterministic','uncalibrated scores','domain limited to source corpus']}
            (directory/f'model_card_{corpus}.json').write_text(json.dumps(card,ensure_ascii=False,indent=2)+'\n')
        lines += ['','| Corpus | Macro-F1 moyen | Écart-type échantillonnal |','| --- | ---: | ---: |']
        for corpus,values in d['aggregates'].items():
            score=values['macro_f1'];lines.append(f'| {corpus.upper()} | {score["mean"]:.6f} | {score["sample_std"]:.6f} |')
        lines += ['','Les trois graines sont toutes rapportées. Aucune meilleure graine sélectionnée. Validation utilisée pour sélectionner le taux ; aucun gain sur le test ni significativité établi.']
    else:
        lines += ['Contrôle technique seulement : quatre modèles appariés sur 100 exemples par partition. Aucune moyenne finale sur trois graines.']
    lines += ['',f'Dossier : `{directory}`','']
    (directory/'summary.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=['full','smoke']);p.add_argument('--output',type=Path)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    p.add_argument('--verify',type=Path);p.add_argument('--expected-mode',choices=['full','smoke'])
    a=p.parse_args()
    if a.verify:
        if a.mode or a.output or not a.expected_mode:p.error('Verification requires --expected-mode only')
        _,proof=verify_campaign(a.verify,a.expected_mode);print(json.dumps(proof,indent=2))
    else:
        if not a.mode or not a.output:p.error('Training requires --mode and --output')
        campaign(a.mode,a.output,a.device)
