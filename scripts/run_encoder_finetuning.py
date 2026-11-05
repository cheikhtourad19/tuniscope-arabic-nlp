"""Train native CLS frozen/fine-tuned models on v2 train/validation, never test."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import time
import gc
import torch
from bigru_data import partitions
from bigru_model import resolve_device,set_seed
from bigru_training import add_writing_slices
from character_lm_experiment import ROOT,partition_summary
from encoder_tokenizer_audit import load_tokenizer
from fetch_camelbert_encoder import sha256
from frozen_encoder import state_digest
from encoder_finetuning import build_model,set_regime,tokens,evaluate,train_epoch,transformer_probe,device_memory
from run_frozen_encoder import save
CONFIG=ROOT/'configs/camelbert_finetuning_v1.json'
CODE=('encoder_finetuning.py','run_encoder_finetuning.py','verify_encoder_finetuning.py','encoder_finetuning_inference.py')


def validate_config(c):
    if c!=json.loads(CONFIG.read_text()) or c['manifest_version']!='v2' or c['variant']!='raw-v0':
        raise ValueError('Expected fixed native CLS fine-tuning protocol')


def artifacts(directory):
    return {str(p.relative_to(directory)):sha256(p) for p in directory.rglob('*') if p.is_file() and p.name not in {'results.json','summary.txt','verification.json'}}


def describe(d):
    lines=[f'CAMeLBERT CLS — {d["mode"].upper()} — {d["regime"]} — {d["device"]}',
           'Train/validation v2 ; test officiel non évalué.',
           'Corpus  Régime      Graine  Macro-F1  Accuracy  Époque  Paramètres entraînables']
    for r in d['results']:
        lines.append(f'{r["corpus"].upper():<8}{r["regime"]:<12}{r["seed"]:<8}{r["metrics"]["macro_f1"]:<10.6f}'
                     f'{r["metrics"]["accuracy"]:<10.6f}{r["best_epoch"]:<8}{r["trainable_parameters"]}')
    lines.append(f'Durée : {d["seconds"]:.1f} s ; dossier : {d["directory"]}')
    if d['mode']=='smoke':lines.append('Essai technique sur100 lignes/partition/tâche ; aucun score FULL.')
    return '\n'.join(lines)


def run(mode,regime,output,device_request=None,smoke_run=None):
    config=json.loads(CONFIG.read_text());validate_config(config)
    if mode not in {'smoke','full'} or regime not in {'paired','frozen','finetuned'}:raise ValueError('Invalid mode/regime')
    output=output.resolve()
    if (ROOT/'outputs/encoder_finetuning').resolve() not in output.parents or output.exists():raise ValueError('Choose a new private run directory')
    set_seed(config['smoke_seed'],config['cpu_threads']);device=resolve_device(device_request or config['device'])
    if mode=='full':
        if smoke_run is None:raise ValueError('Full requires a verified smoke run')
        pre=json.loads((smoke_run/'results.json').read_text());proof=json.loads((smoke_run/'verification.json').read_text())
        if (pre['mode']!='smoke' or pre['config']!=config or pre['regime']!='paired' or pre['device']!=str(device)
                or proof['results_sha256']!=sha256(smoke_run/'results.json')):raise ValueError('Smoke configuration/device/verification mismatch')
    regimes=['frozen','finetuned'] if regime=='paired' else [regime]
    seeds=[config['smoke_seed']] if mode=='smoke' else config['seeds']
    output.mkdir(parents=True);tick=time.perf_counter()
    tokenizer,_,_=load_tokenizer()
    result={'mode':mode,'regime':regime,'device':str(device),'config':config,'test_loaded':False,'directory':str(output),
            'history_schema':'per-epoch-class-metrics-and-predictions-v2',
            'created_at_utc':datetime.now(timezone.utc).isoformat(),'platform':platform.platform(),'python':platform.python_version(),
            'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','tokenizers','numpy','scikit-learn')},
            'environment_lock_sha256':sha256(ROOT/'environments/encoder/uv.lock'),
            'other_locks_sha256':{n:sha256(ROOT/n) for n in ('uv.lock','environments/bigru/uv.lock','environments/tokenizer/uv.lock')},
            'code_sha256':{n:sha256(ROOT/'scripts'/n) for n in CODE},'config_sha256':sha256(CONFIG),
            'protocol_sha256':sha256(ROOT/'docs/46_protocole_fine_tuning_encodeur.md'),
            'manifests_sha256':{c:sha256(ROOT/f'data/manifests/v2/{c}.csv') for c in config['corpora']},
            'git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'git_dirty':bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True)),
            'results':[],'smoke_run':str(smoke_run) if smoke_run else None}
    for corpus in config['corpora']:
        train,validation=partitions(corpus,mode,config);labels=config['labels'][corpus]
        train_x=tokens(train,tokenizer,config);val_x=tokens(validation,tokenizer,config)
        train_y=[labels.index(r['label']) for r in train];val_y=[labels.index(r['label']) for r in validation]
        for seed in seeds:
            for arm in regimes:
                print(f'\n{corpus.upper()} {arm} graine{seed} — train{len(train)}, validation{len(validation)} — {device}',flush=True)
                model,loading=build_model(labels,seed,config,device);set_regime(model,arm)
                initial=state_digest(model.bert);initial_probe=transformer_probe(model);initial_head=state_digest(model.classifier)
                count=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
                generator=torch.Generator().manual_seed(seed)
                optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['learning_rate'],weight_decay=config['weight_decay'])
                stem=f'{corpus}_{arm}_seed_{seed}';target=output/stem;target.mkdir()
                best,best_epoch,stale,history=-1,None,0,[]
                epochs=config['smoke_epochs'] if mode=='smoke' else config['max_epochs']
                for epoch in range(1,epochs+1):
                    trained=train_epoch(model,train_x,train_y,tokenizer,optimizer,config,arm,device,generator)
                    val=evaluate(model,val_x,val_y,tokenizer,labels,config,device)
                    score=val['metrics']['macro_f1'];history.append({'epoch':epoch,'train':trained,'validation_loss':val['loss'],
                                                                 'validation_macro_f1':score,'validation_seconds':val['seconds']})
                    epoch_predictions=f'epoch_{epoch}_predictions.json'
                    save(target/epoch_predictions,[{'id':row['id'],'group_id':row['group_id'],'truth':row['label'],
                        'predicted':predicted,'probabilities':probabilities}
                        for row,predicted,probabilities in zip(validation,val['predicted'],val['probabilities'])])
                    history[-1]['validation_metrics']=val['metrics']
                    history[-1]['validation_predictions']=epoch_predictions
                    print(f'  époque{epoch} | perte train{trained["loss"]:.4f} | val{val["loss"]:.4f}'
                          f' | macro-F1{score:.6f} | train{trained["seconds"]:.1f}s | val{val["seconds"]:.1f}s',flush=True)
                    if score>best:
                        best,best_epoch,stale=score,epoch,0
                        if arm=='finetuned':model.save_pretrained(target,safe_serialization=True)
                        else:
                            model.config.save_pretrained(target)
                            torch.save(model.classifier.state_dict(),target/'head.pt')
                        tokenizer.save_pretrained(target)
                        selected_backbone=state_digest(model.bert);selected_probe=transformer_probe(model)
                        selected_head=state_digest(model.classifier)
                        selected_val=val
                    else:stale+=1
                    save(target/'history.json',history)
                    if stale>=config['patience']:break
                final_backbone=state_digest(model.bert)
                changed=final_backbone!=initial
                if (arm=='finetuned' and (not changed or transformer_probe(model)==initial_probe)
                        or arm=='frozen' and changed or selected_head==initial_head):raise ValueError('Weights did not obey training regime')
                metadata={'corpus':corpus,'seed':seed,'regime':arm,'config':config,'labels':labels,
                          'initial_backbone_sha256':initial,'selected_backbone_sha256':selected_backbone,
                          'initial_transformer_probe_sha256':initial_probe,'selected_transformer_probe_sha256':selected_probe,
                          'initial_head_sha256':initial_head,'selected_head_sha256':selected_head}
                save(target/'tuniscope.json',metadata)
                predictions=[{'id':r['id'],'group_id':r['group_id'],'truth':r['label'],'predicted':p,'probabilities':probs}
                             for r,p,probs in zip(validation,selected_val['predicted'],selected_val['probabilities'])]
                save(target/'predictions.json',predictions)
                metrics=add_writing_slices(selected_val['metrics'],validation,selected_val['predicted'],labels)
                result['results'].append({'corpus':corpus,'seed':seed,'regime':arm,'checkpoint':stem,'metrics':metrics,
                    'parameters':count,'trainable_parameters':trainable,'best_epoch':best_epoch,'epochs_completed':len(history),
                    'train':partition_summary(train),'validation':partition_summary(validation),'loading_info':loading,
                    'backbone_changed_during_training':changed,'initial_backbone_sha256':initial,
                    'initial_head_sha256':initial_head,'selected_backbone_sha256':selected_backbone,
                    'selected_head_sha256':selected_head,'selected_transformer_probe_sha256':selected_probe,
                    'initial_transformer_probe_sha256':initial_probe,
                    'epoch_train_seconds':[h['train']['seconds'] for h in history],
                    'epoch_validation_seconds':[h['validation_seconds'] for h in history]})
                del optimizer,model;gc.collect()
                if device.type=='mps':torch.mps.empty_cache()
                elif device.type=='cuda':torch.cuda.empty_cache()
    result['aggregates']={}
    for corpus in config['corpora']:
        for arm in regimes:
            values=[r['metrics']['macro_f1'] for r in result['results'] if r['corpus']==corpus and r['regime']==arm]
            result['aggregates'][f'{corpus}:{arm}']={'mean_macro_f1':statistics.mean(values),
                'sample_std_macro_f1':statistics.stdev(values) if len(values)>1 else None}
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result['peak_process_rss_bytes']=rss if platform.system()=='Darwin' else rss*1024
    result['seconds']=time.perf_counter()-tick;result['artifacts']=artifacts(output)
    save(output/'results.json',result);(output/'summary.txt').write_text(describe(result)+'\n')
    print(describe(result),flush=True);return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=['smoke','full'],required=True)
    p.add_argument('--regime',choices=['paired','frozen','finetuned'],default='paired');p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda']);p.add_argument('--smoke-run',type=Path)
    a=p.parse_args();run(a.mode,a.regime,a.output,a.device,a.smoke_run)
