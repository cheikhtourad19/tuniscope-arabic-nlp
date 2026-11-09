"""Local multilingual E5 retrieval with the official pooling and prefixes."""
import torch
from transformers import AutoModel, AutoTokenizer

from fetch_e5 import CACHE, assets
from search import ordered


class DenseE5:
    def __init__(self, cards, device='cpu', directory=CACHE):
        cache = assets(directory=directory)
        self.cards = cards
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(cache, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(cache, local_files_only=True, trust_remote_code=False,
                                              use_safetensors=True).eval().to(self.device)
        self.vectors = self.encode([f'passage: {card.passage}' for card in cards])

    @torch.inference_mode()
    def encode(self, texts):
        vectors = []
        for start in range(0, len(texts), 16):
            batch = self.tokenizer(texts[start:start+16], padding=True, return_tensors='pt').to(self.device)
            if batch.input_ids.shape[1] > 512:
                raise ValueError('E5 input exceeds 512 tokens')
            hidden = self.model(**batch).last_hidden_state
            mask = batch.attention_mask[..., None]
            pooled = (hidden * mask).sum(1) / mask.sum(1)
            vectors.append(torch.nn.functional.normalize(pooled, p=2, dim=1))
        return torch.cat(vectors)

    def rank_many(self, questions, k=3):
        vectors = self.encode([f'query: {q["question"]}' for q in questions])
        scores = (vectors @ self.vectors.T).cpu().tolist()
        return [ordered(((card, score) for card, score in zip(self.cards, row, strict=True)
                         if card.lang == question['lang']), k)
                for question, row in zip(questions, scores, strict=True)]
