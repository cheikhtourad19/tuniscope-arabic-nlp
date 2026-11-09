"""Load verified neural inference weights without any training or dataset access."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
from threading import Lock

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, GenerationConfig

from qwen_prompting import parse_label, render_prompts

from .models import ROOT, TASKS


def read(path):
    return json.loads(path.read_text())


def verify_files(directory, files):
    for name, expected in files.items():
        if Path(name).name != name:
            raise ValueError('Nom de fichier de modèle invalide.')
        path = directory / name
        if path.resolve().parent != directory.resolve():
            raise ValueError('Fichier de modèle hors bundle.')
        with path.open('rb') as stream:
            if hashlib.file_digest(stream, 'sha256').hexdigest() != expected:
                raise ValueError(f'Empreinte incorrecte : {name}')


def device_for_inference():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


class Encoder:
    def __init__(self, directory, labels, device):
        self.model = AutoModelForSequenceClassification.from_pretrained(
            directory, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, attn_implementation='eager').eval().to(device)
        if self.model.config.label2id != {label: index for index, label in enumerate(labels)}:
            raise ValueError('Labels CAMeLBERT incompatibles.')
        self.tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
        self.device, self.labels = device, labels
        self.max_length = read(directory / 'tuniscope.json')['config']['max_length']
        self.lock = Lock()

    @torch.inference_mode()
    def predict(self, text):
        with self.lock:
            batch = self.tokenizer(text, return_tensors='pt', verbose=False)
            truncated = batch.input_ids.shape[-1] > self.max_length
            if truncated:
                batch = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=self.max_length)
            values = self.model(**batch.to(self.device)).logits.softmax(-1)[0].cpu().tolist()
        scores = dict(zip(self.labels, values, strict=True))
        return {'status': 'predicted', 'label': max(scores, key=scores.get), 'scores': scores,
                'reason': f'Entrée tronquée à {self.max_length} tokens selon le protocole CAMeLBERT.' if truncated else None}


class Qwen:
    def __init__(self, directory, revision, device):
        base, adapter = directory / 'base', directory / 'adapter'
        config = read(adapter / 'adapter_config.json')
        if config['revision'] != revision or config['base_model_name_or_path'] != 'Qwen/Qwen3-0.6B':
            raise ValueError('Révision de l’adaptateur incompatible.')
        self.tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=False)
        model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            dtype=torch.float32, attn_implementation='eager')
        self.model = PeftModel.from_pretrained(model, adapter, is_trainable=False, local_files_only=True).eval().to(device)
        self.prompt = read(ROOT / 'configs/qwen_prompting_v2.json')
        self.generation = GenerationConfig.from_pretrained(base, local_files_only=True)
        eos = self.generation.eos_token_id
        self.stops = set(eos if isinstance(eos, list) else [eos])
        self.device = device
        self.lock = Lock()

    @torch.inference_mode()
    def predict(self, text, corpus, adapted):
        task = self.prompt['tasks'][corpus]
        with self.lock:
            prompt, = render_prompts(self.tokenizer, task, [text])
            batch = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt', verbose=False)
            width = batch.input_ids.shape[1]
            if width > 480:
                return {'status': 'abstained', 'label': None, 'scores': {},
                        'reason': 'Le prompt dépasse le budget Qwen de 480 tokens.'}
            with nullcontext() if adapted else self.model.disable_adapter():
                tokens = self.model.generate(
                    **batch.to(self.device), generation_config=self.generation,
                    **self.prompt['initial_generation'], max_new_tokens=self.prompt['max_new_tokens'],
                    use_cache=True)[0, width:].cpu().tolist()
            end = next((i for i, value in enumerate(tokens) if value in self.stops), len(tokens))
            raw = self.tokenizer.decode(tokens[:end], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        label = parse_label(raw, task['labels'])
        return {'status': 'predicted' if label else 'abstained', 'label': label, 'scores': {},
                'reason': None if label else 'Qwen n’a produit aucun label valide pour cette tâche.'}


def load_neural(store, directory):
    bundle = read(directory / 'manifest.json')
    if bundle['version'] != 1 or bundle['split'] != 'validation' or bundle['test_loaded'] is not False:
        raise ValueError('Manifest neural de validation requis.')
    torch.set_num_threads(2)
    device = device_for_inference()
    seed = bundle['seed']
    for task, (corpus, expected) in TASKS.items():
        item = bundle['encoder'][corpus]
        checkpoint = directory / 'encoder' / corpus
        labels = item['metrics']['labels']
        if set(labels) != expected:
            raise ValueError('Labels de tâche incohérents.')
        verify_files(checkpoint, item['files'])
        model = Encoder(checkpoint, labels, device)
        if model.predict('exemple fictif اختبار')['label'] not in expected:
            raise ValueError('Sonde CAMeLBERT invalide.')
        store.register_neural(task, 'camelbert', model.predict, labels, item['metrics'], item['source_run'],
                              item['files']['model.safetensors'], item['train_rows'], item['validation_rows'],
                              seed, item['revision'], item['aggregate'], str(device))
    item = bundle['qwen']
    verify_files(directory / 'qwen/base', item['base_files'])
    verify_files(directory / 'qwen/adapter', item['adapter_files'])
    model = Qwen(directory / 'qwen', item['revision'], device)
    for task, (corpus, expected) in TASKS.items():
        stage = f'{corpus}_zero_shot'
        for adapted in (False, True):
            metrics = item['metrics' if adapted else 'base_metrics'][stage]
            if set(metrics['labels']) != expected:
                raise ValueError('Labels Qwen incompatibles.')
            weights = (item['adapter_files']['adapter_model.safetensors'] if adapted
                       else item['base_files']['model.safetensors'])
            store.register_neural(
                task, 'qwen_lora' if adapted else 'qwen_base',
                lambda text, corpus=corpus, adapted=adapted: model.predict(text, corpus, adapted),
                metrics['labels'], metrics, item['source_run' if adapted else 'base_source_run'], weights,
                2000 if adapted else None, metrics['n'], seed if adapted else None, item['revision'],
                item['aggregates'][stage] if adapted else None, str(device))
    for corpus in ('tsac', 'madar'):
        model.predict('exemple fictif اختبار', corpus, True)
