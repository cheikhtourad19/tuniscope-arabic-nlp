"""Native BERT CLS classification, paired freeze policy and explicit training loop."""
import gc
import math
import time
import numpy as np
import torch
from torch import nn
from transformers import AutoModelForSequenceClassification
from baseline_metrics import classification_metrics
from bigru_model import set_seed
from bigru_training import sync
from encoder_tokenizer_audit import load_tokenizer
from fetch_camelbert_encoder import ENCODER_DIR, verify_encoder
from frozen_encoder import state_digest


def set_regime(model, regime):
    if regime not in {'frozen', 'finetuned'}:
        raise ValueError('Expected frozen or finetuned')
    model.requires_grad_(True)
    model.bert.requires_grad_(regime == 'finetuned')
    model.eval()


def build_model(labels, seed, config, device):
    set_seed(seed, config['cpu_threads'])
    # MPS embedding backward has no strict deterministic implementation in this runtime.
    # Seeds remain fixed; unsupported deterministic kernels emit a visible warning.
    if device.type == 'mps':
        torch.use_deterministic_algorithms(True, warn_only=True)
    verify_encoder()
    mapping = dict(enumerate(labels))
    model, info = AutoModelForSequenceClassification.from_pretrained(
        str(ENCODER_DIR), local_files_only=True, trust_remote_code=False, weights_only=True,
        use_safetensors=False, output_loading_info=True, attn_implementation='eager',
        num_labels=len(labels), id2label=mapping, label2id={v:k for k,v in mapping.items()},
        classifier_dropout=config['classifier_dropout'])
    if (set(info['missing_keys']) != {'classifier.weight', 'classifier.bias'}
            or info.get('mismatched_keys') or info.get('error_msgs')
            or any(not k.startswith('cls.') for k in info['unexpected_keys'])):
        raise ValueError(f'Unexpected model loading keys: {info}')
    return model.to(device), info


def tokens(rows, tokenizer, config):
    encoded = tokenizer([r['text_raw'] for r in rows], max_length=config['max_length'],
                        truncation=True, padding=False)
    return [{k: encoded[k][i] for k in encoded} for i in range(len(rows))]


def make_batch(encoded, indices, tokenizer, device):
    batch = tokenizer.pad([encoded[int(i)] for i in indices], padding=True, return_tensors='pt')
    return {k:v.to(device) for k,v in batch.items()}


def device_memory(device):
    if device.type == 'mps':
        return {'sampled_mps_tensor_bytes': torch.mps.current_allocated_memory(),
                'sampled_mps_driver_bytes': torch.mps.driver_allocated_memory()}
    if device.type == 'cuda':
        return {'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(device),
                'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(device)}
    return {}


def evaluate(model, encoded, targets, tokenizer, labels, config, device):
    model.eval()
    predicted, probabilities, loss_sum = [], [], 0.
    sync(device); tick = time.perf_counter()
    with torch.inference_mode():
        for start in range(0,len(encoded),config['batch_size']):
            indices = list(range(start,min(start+config['batch_size'],len(encoded))))
            batch = make_batch(encoded, indices, tokenizer, device)
            y = torch.tensor([targets[i] for i in indices],device=device)
            logits = model(**batch).logits
            loss = nn.functional.cross_entropy(logits,y,reduction='sum')
            if not torch.isfinite(loss):raise ValueError('Nonfinite validation loss')
            loss_sum += float(loss)
            probabilities.extend(logits.softmax(-1).cpu().tolist())
    sync(device)
    predicted = [labels[int(np.argmax(p))] for p in probabilities]
    return {'loss':loss_sum/len(encoded),'predicted':predicted,'probabilities':probabilities,
            'metrics':classification_metrics([labels[i] for i in targets],predicted,labels),
            'seconds':time.perf_counter()-tick}


def train_epoch(model, encoded, targets, tokenizer, optimizer, config, regime, device, generator):
    model.train()
    # eval() disables backbone dropout, not autograd: identical policy in both arms.
    model.bert.eval()
    if model.bert.training or not model.classifier.training:raise AssertionError('Wrong dropout policy')
    order = torch.randperm(len(encoded),generator=generator)
    effective = config['batch_size']*config['gradient_accumulation_steps']
    loss_sum, updates, sampled, gradients = 0.,0,{},None
    sync(device); tick=time.perf_counter()
    for window in order.split(effective):
        optimizer.zero_grad(set_to_none=True)
        for indices in window.split(config['batch_size']):
            batch=make_batch(encoded,indices,tokenizer,device)
            y=torch.tensor([targets[int(i)] for i in indices],device=device)
            loss=nn.functional.cross_entropy(model(**batch).logits,y,reduction='sum')
            if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
            loss_sum+=float(loss.detach())
            # Correct weighting also for final incomplete accumulation windows.
            (loss/len(window)).backward()
        if gradients is None:
            probe=model.bert.encoder.layer[0].attention.self.query.weight
            gradients={'head_gradient_norm':float(model.classifier.weight.grad.norm()),
                       'transformer_gradient_norm':float(probe.grad.norm()) if probe.grad is not None else 0.,
                       'backbone_any_gradient':any(p.grad is not None for p in model.bert.parameters())}
            if gradients['head_gradient_norm']<=0:raise ValueError('Missing head gradient')
            if regime=='finetuned' and gradients['transformer_gradient_norm']<=0:raise ValueError('Missing Transformer gradient')
            if regime=='frozen' and gradients['backbone_any_gradient']:raise ValueError('Frozen backbone got gradients')
        for k,v in device_memory(device).items():sampled[k]=max(sampled.get(k,0),v)
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],config['gradient_clip'],error_if_nonfinite=True)
        optimizer.step();updates+=1
        for k,v in device_memory(device).items():sampled[k]=max(sampled.get(k,0),v)
    sync(device)
    seconds=time.perf_counter()-tick
    return {'loss':loss_sum/len(encoded),'rows':len(encoded),'optimizer_steps':updates,
            'expected_steps':math.ceil(len(encoded)/effective),'seconds':seconds,
            'rows_per_second':len(encoded)/seconds,'gradients':gradients,'memory':sampled}


def transformer_probe(model):
    p=model.bert.encoder.layer[0].attention.self.query.weight
    import hashlib
    return hashlib.sha256(p.detach().cpu().numpy().tobytes()).hexdigest()


def release(model, device):
    del model
    gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    elif device.type=='cuda':torch.cuda.empty_cache()
