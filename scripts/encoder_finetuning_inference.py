"""Local inference of trusted saved CLS checkpoints, no corpus reads or training."""
import argparse
import json
from pathlib import Path
import torch
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from bigru_model import resolve_device
from encoder_finetuning import build_model,set_regime
from frozen_encoder import state_digest


def load_checkpoint(directory,device):
    metadata=json.loads((directory/'tuniscope.json').read_text());config=metadata['config'];labels=metadata['labels']
    if metadata['regime']=='finetuned':
        model=AutoModelForSequenceClassification.from_pretrained(str(directory),local_files_only=True,
                trust_remote_code=False,use_safetensors=True,attn_implementation='eager').to(device)
    elif metadata['regime']=='frozen':
        model,_=build_model(labels,metadata['seed'],config,device)
        model.classifier.load_state_dict(torch.load(directory/'head.pt',map_location='cpu',weights_only=True),strict=True)
    else:raise ValueError('Unexpected checkpoint regime')
    if (state_digest(model.bert)!=metadata['selected_backbone_sha256']
            or state_digest(model.classifier)!=metadata['selected_head_sha256']):raise ValueError('Checkpoint weight digest differs')
    if model.config.label2id!={label:i for i,label in enumerate(labels)}:raise ValueError('Wrong label mapping')
    tokenizer=AutoTokenizer.from_pretrained(str(directory),use_fast=True,local_files_only=True,trust_remote_code=False,
                                           model_max_length=512)
    set_regime(model,metadata['regime']);model.eval()
    return model,tokenizer,metadata


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--text',required=True);p.add_argument('--device',choices=['auto','cpu','mps','cuda'],default='auto')
    a=p.parse_args();device=resolve_device(a.device);model,tokenizer,meta=load_checkpoint(a.checkpoint.resolve(),device)
    full_length=len(tokenizer(a.text)['input_ids'])
    encoded=tokenizer(a.text,return_tensors='pt',truncation=True,max_length=meta['config']['max_length'])
    with torch.inference_mode():prob=model(**{k:v.to(device) for k,v in encoded.items()}).logits.softmax(-1)[0].cpu().tolist()
    print(json.dumps({'label':meta['labels'][max(range(len(prob)),key=prob.__getitem__)],
        'scores':dict(zip(meta['labels'],prob)),'scores_calibrated':False,'regime':meta['regime'],
        'truncated':full_length>meta['config']['max_length'],'task':meta['corpus'],'corpus_read':False},ensure_ascii=False,indent=2))
