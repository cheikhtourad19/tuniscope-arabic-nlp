"""Local pinned BERT, masked mean features, and trainable CPU linear heads."""
import hashlib
import time

import numpy as np
import torch
from torch import nn
from transformers import AutoModel

from baseline_metrics import classification_metrics
from bigru_model import set_seed
from bigru_training import sync
from encoder_tokenizer_audit import load_tokenizer
from fetch_camelbert_encoder import ENCODER_DIR, verify_encoder


def state_digest(model):
    result = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        result.update(name.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def load_encoder(device):
    source = verify_encoder()
    tokenizer, _, _ = load_tokenizer()
    model, info = AutoModel.from_pretrained(
        str(ENCODER_DIR), local_files_only=True, trust_remote_code=False,
        use_safetensors=False, weights_only=True, add_pooling_layer=False,
        output_loading_info=True, attn_implementation='eager')
    if (info['missing_keys'] or info.get('mismatched_keys') or info.get('error_msgs')
            or any(not key.startswith('cls.') and key not in {'pooler.dense.weight', 'pooler.dense.bias'}
                   for key in info['unexpected_keys'])):
        raise ValueError(f'Unexpected backbone loading mismatch: {info}')
    model.requires_grad_(False)
    model.eval()
    count = sum(p.numel() for p in model.parameters())
    if count != 108490752 or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Unexpected actual backbone parameter count/freeze status')
    return tokenizer, model.to(device), {'source': source, 'loading_info': info,
                                        'backbone_parameters': count, 'backbone_trainable_parameters': 0}


def masked_mean(hidden, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
    denominator = mask.sum(dim=1)
    if bool((denominator <= 0).any()):
        raise ValueError('Empty attention mask')
    return (hidden * mask).sum(dim=1) / denominator


def extract_features(model, tokenizer, rows, config, device, progress=False):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('Encoder must be frozen and in eval mode')
    batch_size, blocks, durations, lengths = config['encoder_batch_size'], [], [], []
    sync(device)
    started = time.perf_counter()
    with torch.inference_mode():
        for offset in range(0, len(rows), batch_size):
            batch = tokenizer([r['text_raw'] for r in rows[offset:offset + batch_size]],
                              padding=True, truncation=True, max_length=config['max_length'],
                              return_tensors='pt')
            lengths.append(batch['input_ids'].shape[1])
            batch = {k: v.to(device) for k, v in batch.items()}
            sync(device)
            tick = time.perf_counter()
            hidden = model(**batch).last_hidden_state
            vectors = masked_mean(hidden, batch['attention_mask']).cpu().numpy().copy()
            sync(device)
            durations.append(time.perf_counter() - tick)
            blocks.append(vectors)
            if progress and (offset == 0 or (offset // batch_size + 1) % 50 == 0 or offset + batch_size >= len(rows)):
                print(f'  Encodage {min(offset + batch_size, len(rows))}/{len(rows)} textes', flush=True)
    result = np.concatenate(blocks).astype(np.float32, copy=False)
    if result.shape != (len(rows), model.config.hidden_size) or not np.isfinite(result).all():
        raise ValueError('Invalid cached features')
    elapsed = time.perf_counter() - started
    return result, {'rows': len(rows), 'seconds': elapsed, 'rows_per_second': len(rows) / elapsed,
                    'batch_size': batch_size, 'maximum_padded_length': max(lengths),
                    'first_batch_seconds': durations[0], 'forward_seconds': sum(durations)}


def head_output(head, features, targets, labels):
    head.eval()
    with torch.inference_mode():
        logits = head(torch.from_numpy(features))
        loss = float(nn.functional.cross_entropy(logits, targets))
        probabilities = logits.softmax(-1).numpy().copy()
    predicted = [labels[i] for i in probabilities.argmax(axis=1)]
    return {'loss': loss, 'probabilities': probabilities.tolist(), 'predicted': predicted,
            'metrics': classification_metrics([labels[i] for i in targets.tolist()], predicted, labels)}


def fit_head(train_x, validation_x, train_y, validation_y, labels, config, seed, mode, progress=False):
    set_seed(seed, config['cpu_threads'])
    head = nn.Linear(train_x.shape[1], len(labels))
    before = state_digest(head)
    optimizer = torch.optim.AdamW(head.parameters(), lr=config['head_learning_rate'],
                                 weight_decay=config['head_weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(train_x)
    train_y = torch.tensor(train_y, dtype=torch.long)
    validation_y = torch.tensor(validation_y, dtype=torch.long)
    max_epochs = config['smoke_epochs'] if mode == 'smoke' else config['head_max_epochs']
    best, best_epoch, best_state, history, stale = -1, None, None, [], 0
    started = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        head.train()
        order = torch.randperm(len(x), generator=generator)
        total_loss = 0.
        for indices in order.split(config['head_batch_size']):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(head(x[indices]), train_y[indices])
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite head loss')
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        evaluation = head_output(head, validation_x, validation_y, labels)
        score = evaluation['metrics']['macro_f1']
        history.append({'epoch': epoch, 'train_loss': total_loss / len(x),
                        'validation_loss': evaluation['loss'], 'validation_macro_f1': score})
        if progress:
            print(f'  Graine {seed} époque {epoch:02d} | perte train {total_loss / len(x):.4f}'
                  f' | perte val {evaluation["loss"]:.4f} | macro-F1 {score:.6f}', flush=True)
        if score > best:
            best, best_epoch, stale = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        else:
            stale += 1
        if stale >= config['head_patience']:
            break
    head.load_state_dict(best_state)
    if before == state_digest(head):
        raise ValueError('Head weights did not change')
    evaluation = head_output(head, validation_x, validation_y, labels)
    return head, evaluation, history, {'best_epoch': best_epoch, 'epochs_completed': len(history),
                                      'seconds': time.perf_counter() - started, 'weights_changed': True,
                                      'trainable_parameters': sum(p.numel() for p in head.parameters()),
                                      'initial_head_sha256': before, 'selected_head_sha256': state_digest(head)}
