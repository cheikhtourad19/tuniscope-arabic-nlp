"""Small randomly initialized embedding + bidirectional GRU + masked pooling."""
import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from bigru_data import PAD


def resolve_device(requested):
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be cpu, cuda, mps or auto")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(requested)


def set_seed(seed, threads):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class BiGRUClassifier(nn.Module):
    def __init__(self, vocabulary_size, num_classes, config):
        super().__init__()
        self.embedding = nn.Embedding(vocabulary_size, config["embedding_dim"], padding_idx=PAD)
        self.gru = nn.GRU(config["embedding_dim"], config["hidden_dim"], num_layers=1,
                          batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(config["dropout"])
        self.classifier = nn.Linear(2 * config["hidden_dim"], num_classes)

    def forward(self, tokens, lengths):
        embedded = self.embedding(tokens)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        outputs, _ = self.gru(packed)
        outputs, _ = pad_packed_sequence(outputs, batch_first=True, total_length=tokens.shape[1])
        mask = torch.arange(tokens.shape[1], device=tokens.device)[None, :] < lengths.to(tokens.device)[:, None]
        pooled = (outputs * mask.unsqueeze(-1)).sum(dim=1) / lengths.to(tokens.device).unsqueeze(-1)
        return self.classifier(self.dropout(pooled))


def collate(examples):
    sequences, labels = zip(*examples)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.long)
    tokens = torch.zeros((len(sequences), int(lengths.max())), dtype=torch.long)
    for index, sequence in enumerate(sequences):
        tokens[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long)
    return tokens, lengths, torch.tensor(labels, dtype=torch.long)
