"""Recompute all selected validation predictions from native CLS checkpoints."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import statistics
import numpy as np
import torch
from bigru_data import partitions
from bigru_model import resolve_device,set_seed
from bigru_training import add_writing_slices
from baseline_metrics import classification_metrics
from character_lm_experiment import ROOT,partition_summary
from encoder_finetuning import evaluate,tokens,transformer_probe
from encoder_finetuning_inference import load_checkpoint
from fetch_camelbert_encoder import sha256
from run_encoder_finetuning import validate_config,describe
from encoder_screening_plan import PLAN, execution_seeds, study_plan
from encoder_confirmation_plan import PLAN as CONFIRMATION_PLAN, confirmation_plan
from run_frozen_encoder import save
from verify_bigru import compare_nested


def artifact(directory,name):
    p=(directory/name).resolve()
    if directory.resolve() not in p.parents:raise ValueError('Escaping artifact path')
    return p


def verify_epoch_metrics(checkpoint,epoch,rows,labels):
    """Legacy runs lack these fields; never invent missing epoch-level evidence."""
    fields={'validation_metrics','validation_predictions'}
    present=fields & set(epoch)
    if not present:
        return False
    if present != fields:
        raise ValueError('Incomplete per-epoch metric evidence')
    predictions=json.loads(artifact(checkpoint,epoch['validation_predictions']).read_text())
    if [(p['id'],p['group_id'],p['truth']) for p in predictions]!=[(r['id'],r['group_id'],r['label']) for r in rows]:
        raise ValueError('Per-epoch prediction alignment mismatch')
    if [p['predicted'] for p in predictions]!=[labels[int(np.argmax(p['probabilities']))] for p in predictions]:
        raise ValueError('Per-epoch labels and probabilities disagree')
    metrics=classification_metrics([r['label'] for r in rows],[p['predicted'] for p in predictions],labels)
    compare_nested(metrics,epoch['validation_metrics'])
    if epoch['validation_macro_f1']!=metrics['macro_f1']:
        raise ValueError('Per-epoch macro-F1 differs')
    return True


def verify(directory,device_request=None):
    directory=directory.resolve();d=json.loads((directory/'results.json').read_text());config=d['config']
    scope=d.get('experiment_scope','initial');validate_config(config,scope)
    if d['test_loaded'] is not False or d['mode'] not in {'smoke','full'}:raise ValueError('Invalid mode/test boundary')
    if scope=='screening':
        if d['study_plan_sha256']!=sha256(PLAN) or d['protocol_sha256']!=sha256(ROOT/study_plan()['protocol']):
            raise ValueError('Screening plan/protocol changed')
        if d['regime']!=('paired' if d['mode']=='smoke' else 'finetuned'):
            raise ValueError('Wrong screening regime')
    if scope=='confirmation':
        plan=confirmation_plan()
        if (d['confirmation_plan_sha256']!=sha256(CONFIRMATION_PLAN)
                or d['selection_report_sha256']!=plan['selection_report_sha256']
                or d['protocol_sha256']!=sha256(ROOT/plan['protocol'])):
            raise ValueError('Confirmation plan/source/protocol changed')
        if d['regime']!=('paired' if d['mode']=='smoke' else 'finetuned'):
            raise ValueError('Wrong confirmation regime')
    if sha256(ROOT/'environments/encoder/uv.lock')!=d['environment_lock_sha256']:raise ValueError('Encoder lock changed')
    for name,expected in d['other_locks_sha256'].items():
        if sha256(ROOT/name)!=expected:raise ValueError('Existing environment lock changed')
    for p,v in d['packages'].items():
        if importlib.metadata.version(p)!=v:raise ValueError('Package changed')
    for n,h in d['artifacts'].items():
        if sha256(artifact(directory,n))!=h:raise ValueError(f'Artifact changed: {n}')
    seeds=execution_seeds(d['mode'],scope,config)
    if 'execution_seeds' in d and d['execution_seeds']!=seeds:raise ValueError('Wrong declared execution seeds')
    regimes=['frozen','finetuned'] if d['regime']=='paired' else [d['regime']]
    expected={(c,s,r) for c in config['corpora'] for s in seeds for r in regimes}
    if len(d['results'])!=len(expected) or {(r['corpus'],r['seed'],r['regime']) for r in d['results']}!=expected:raise ValueError('Incomplete/duplicate results')
    set_seed(42,config['cpu_threads']);device=resolve_device(device_request or d['device']);probes=[]
    initials={};epoch_evidence=[]
    for r in d['results']:
        corpus=r['corpus'];labels=config['labels'][corpus]
        if sha256(ROOT/f'data/manifests/v2/{corpus}.csv')!=d['manifests_sha256'][corpus]:raise ValueError('Manifest changed')
        train,val=partitions(corpus,d['mode'],config)
        if r['train']!=partition_summary(train) or r['validation']!=partition_summary(val):raise ValueError('Partition changed')
        key=(corpus,r['seed']);value=(r['initial_backbone_sha256'],r['initial_head_sha256'])
        if key in initials and initials[key]!=value:raise ValueError('Paired arms had different initialization')
        initials[key]=value
        checkpoint=artifact(directory,r['checkpoint'])
        model,tokenizer,meta=load_checkpoint(checkpoint,device)
        if (meta['config']!=config or meta['corpus']!=corpus or meta['seed']!=r['seed'] or meta['regime']!=r['regime']):raise ValueError('Checkpoint metadata mismatch')
        if (r['parameters']!=sum(p.numel() for p in model.parameters())
                or r['trainable_parameters']!=sum(p.numel() for p in model.parameters() if p.requires_grad)):raise ValueError('Parameter count mismatch')
        if r['parameters']>=1_000_000_000:raise ValueError('Parameter ceiling exceeded')
        changed=r['selected_backbone_sha256']!=r['initial_backbone_sha256']
        probe_changed=transformer_probe(model)!=r['initial_transformer_probe_sha256']
        if changed!=(r['regime']=='finetuned') or probe_changed!=changed:raise ValueError('Wrong frozen/adapted weights')
        if r['selected_head_sha256']==r['initial_head_sha256']:raise ValueError('Head did not learn')
        prediction=json.loads((checkpoint/'predictions.json').read_text())
        if [(p['id'],p['group_id'],p['truth']) for p in prediction]!=[(x['id'],x['group_id'],x['label']) for x in val]:raise ValueError('Prediction alignment mismatch')
        encoded=tokens(val,tokenizer,config);y=[labels.index(x['label']) for x in val]
        out=evaluate(model,encoded,y,tokenizer,labels,config,device)
        if out['predicted']!=[p['predicted'] for p in prediction]:raise ValueError('Reloaded labels differ')
        difference=float(np.max(np.abs(np.asarray(out['probabilities'])-np.asarray([p['probabilities'] for p in prediction]))))
        if difference>config['reload_probability_atol']:raise ValueError('Reloaded probabilities differ')
        compare_nested(add_writing_slices(out['metrics'],val,out['predicted'],labels),r['metrics'])
        history=json.loads((checkpoint/'history.json').read_text());best=max(history,key=lambda h:h['validation_macro_f1'])
        if best['epoch']!=r['best_epoch'] or len(history)!=r['epochs_completed'] or best['validation_macro_f1']!=r['metrics']['macro_f1']:raise ValueError('Wrong selected epoch')
        if any(h['train']['optimizer_steps']!=h['train']['expected_steps'] for h in history):raise ValueError('Accumulation step mismatch')
        for h in history:
            epoch_evidence.append(verify_epoch_metrics(checkpoint,h,val,labels))
            g=h['train']['gradients']
            if (g['head_gradient_norm']<=0 or g['backbone_any_gradient']!=(r['regime']=='finetuned')
                    or (r['regime']=='finetuned' and g['transformer_gradient_norm']<=0)):raise ValueError('Gradient audit differs')
        probes.append({'corpus':corpus,'seed':r['seed'],'regime':r['regime'],'validation_rows':len(val),
                       'first_ten_probability_max_abs_difference':float(np.max(np.abs(np.asarray(out['probabilities'][:10])-np.asarray([p['probabilities'] for p in prediction[:10]])))),
                       'all_probability_max_abs_difference':difference})
        del model
        if device.type=='mps':torch.mps.empty_cache()
    for c in config['corpora']:
        for arm in regimes:
            values=[r['metrics']['macro_f1'] for r in d['results'] if r['corpus']==c and r['regime']==arm]
            compare_nested(d['aggregates'][f'{c}:{arm}'],{'mean_macro_f1':statistics.mean(values),'sample_std_macro_f1':statistics.stdev(values) if len(values)>1 else None})
    if scope in {'screening','confirmation'} and not all(epoch_evidence):raise ValueError('This scope requires complete epoch metrics')
    save(directory/'verification.json',{'results_sha256':sha256(directory/'results.json'),'models_replayed':len(probes),
         'device':str(device),'test_loaded':False,'probes':probes,'verifier_sha256':sha256(Path(__file__)),
         'epoch_class_metrics_complete':all(epoch_evidence),
         'epochs_with_retained_class_metrics':sum(epoch_evidence),'epochs_checked':len(epoch_evidence)})
    print(f'Verified {d["mode"]}: {len(probes)} modèles, validation entière, gradients, poids et rechargement.',flush=True)
    return d


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--device',choices=['cpu','mps','cuda','auto'])
    a=p.parse_args();print(describe(verify(a.run,a.device)))
