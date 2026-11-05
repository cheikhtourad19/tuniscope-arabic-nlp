"""Bounded maximum-length gradient probe and clearly labelled full-run estimate."""
import argparse
import json
import time
from pathlib import Path
import torch
from bigru_model import resolve_device
from bigru_training import sync
from character_lm_experiment import ROOT
from encoder_finetuning import build_model,set_regime,device_memory
from run_encoder_finetuning import CONFIG
from run_frozen_encoder import save


def estimate(directory):
    d=json.loads((directory/'results.json').read_text());c=d['config']
    if d['mode']!='smoke' or d['regime']!='paired':raise ValueError('Need paired smoke')
    full={'tsac':(10053,1774),'madar':(6388,800)}
    rows=[]
    for r in d['results']:
        train_seconds=r['epoch_train_seconds'][-1];val_seconds=r['epoch_validation_seconds'][-1]
        n_train,n_val=full[r['corpus']]
        seconds=(train_seconds*n_train/r['train']['rows']+val_seconds*n_val/r['validation']['rows'])*c['max_epochs']*len(c['seeds'])
        rows.append({'corpus':r['corpus'],'regime':r['regime'],'estimated_full_seconds':seconds,
                     'last_measured_smoke_train_seconds':train_seconds,'last_measured_smoke_validation_seconds':val_seconds})
    return {'estimated_not_measured':True,'assumption':'Linear scaling from last smoke epoch; 3 epochs and 3 seeds, no early stopping; full text lengths and machine load may differ; excludes checkpoint I/O/replay.',
            'rows':rows,'estimated_paired_minutes':sum(r['estimated_full_seconds'] for r in rows)/60,
            'estimated_finetuned_only_minutes':sum(r['estimated_full_seconds'] for r in rows if r['regime']=='finetuned')/60}


def probe(device_name,output):
    config=json.loads(CONFIG.read_text());device=resolve_device(device_name)
    model,_=build_model(['NEG','POS'],42,config,device);set_regime(model,'finetuned')
    model.train();model.bert.eval();batch=config['batch_size'];length=config['max_length']
    # Invented valid IDs; no corpus or linguistic accuracy involved.
    ids=torch.full((batch,length),100,dtype=torch.long,device=device);ids[:,0]=2;ids[:,-1]=3
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=config['weight_decay'])
    sync(device);start=time.perf_counter()
    loss=model(input_ids=ids,attention_mask=torch.ones_like(ids),labels=torch.zeros(batch,dtype=torch.long,device=device)).loss
    loss.backward();before=device_memory(device);optimizer.step();sync(device)
    result={'device':str(device),'invented_inputs':True,'corpus_read':False,'batch':batch,'length':length,
            'forward_backward_adamw_step_succeeded':True,'seconds':time.perf_counter()-start,
            'memory_after_backward':before,'memory_after_step':device_memory(device),
            'memory_note':'Observed snapshots, not an absolute MPS peak; no claim about full-run duration or CUDA.'}
    if output.exists():raise ValueError('Preserve prior probe')
    save(output,result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--smoke-run',type=Path);p.add_argument('--probe-output',type=Path)
    p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto');a=p.parse_args()
    if a.probe_output:probe(a.device,a.probe_output)
    if a.smoke_run:
        value=estimate(a.smoke_run);save(a.smoke_run/'budget_estimate.json',value);print(json.dumps(value,indent=2))
