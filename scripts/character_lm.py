"""Deterministic add-alpha Unicode character LM; no corpus access at import."""
from collections import Counter
import math

UNK, BOS, EOS = 0, 1, 2


class CharacterLM:
    def __init__(self, order, alpha, characters):
        if type(order) is not int or order not in (1, 3):
            raise ValueError("Supported orders are 1 and 3")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Smoothing alpha must be finite and positive")
        if (not characters or list(characters) != sorted(set(characters))
                or any(not isinstance(c, str) or len(c) != 1 for c in characters)):
            raise ValueError("Expected sorted distinct training characters")
        self.order, self.alpha = order, float(alpha)
        self.characters = list(characters)
        self.ids = {c: i + 3 for i, c in enumerate(characters)}
        self.counts = {}
        self.totals = {}

    @property
    def vocabulary_size(self):
        return len(self.characters) + 2  # UNK and EOS; never BOS.

    def events(self, text):
        history = [BOS] * (self.order - 1)
        for target in [self.ids.get(c, UNK) for c in text] + [EOS]:
            yield tuple(history), target
            if self.order > 1:
                history = (history + [target])[-(self.order - 1):]

    def fit(self, texts):
        self.counts = {}
        for text in texts:
            if any(c not in self.ids for c in text):
                raise ValueError("Training character missing from declared vocabulary")
            for history, target in self.events(text):
                self.counts.setdefault(history, Counter())[target] += 1
        if not self.counts:
            raise ValueError("No training documents")
        self.totals = {history: sum(counts.values()) for history, counts in self.counts.items()}
        return self

    def probability(self, history, target):
        if not self.counts:
            raise ValueError("Unfitted model")
        if len(history) != self.order - 1 or target == BOS or not 0 <= target < len(self.characters) + 3:
            raise ValueError("Invalid history or target")
        return ((self.counts.get(tuple(history), {}).get(target, 0) + self.alpha) /
                (self.totals.get(tuple(history), 0) + self.alpha * self.vocabulary_size))

    def score(self, text):
        events = list(self.events(text))
        nll = math.fsum(-math.log(self.probability(h, c)) for h, c in events)
        return {"nll": nll, "targets": len(events), "characters": len(text),
                "unknown_characters": sum(c not in self.ids for c in text)}

    def to_dict(self):
        return {"version": "character-lm-v1", "order": self.order, "alpha": self.alpha,
                "characters": self.characters, "counts": [
                    [list(h), [[c, n] for c, n in sorted(counts.items())]]
                    for h, counts in sorted(self.counts.items())]}

    @classmethod
    def from_dict(cls, data):
        if data["version"] != "character-lm-v1":
            raise ValueError("Unknown checkpoint version")
        model = cls(data["order"], data["alpha"], data["characters"])
        for history, pairs in data["counts"]:
            history = tuple(history)
            if history in model.counts or len(history) != model.order - 1:
                raise ValueError("Duplicate or invalid context")
            if any(type(x) is not int or not 0 <= x < len(model.characters) + 3 for x in history):
                raise ValueError("Invalid context token")
            counts = Counter()
            for target, count in pairs:
                if (type(target) is not int or target == BOS or not 0 <= target < len(model.characters) + 3
                        or type(count) is not int or count <= 0 or target in counts):
                    raise ValueError("Invalid count or target")
                counts[target] = count
            if not counts:
                raise ValueError("Empty context")
            model.counts[history] = counts
        if not model.counts:
            raise ValueError("Empty checkpoint")
        model.totals = {h: sum(c.values()) for h, c in model.counts.items()}
        return model


def aggregate(scores):
    if not scores:
        raise ValueError("No evaluation documents")
    targets = sum(s["targets"] for s in scores)
    characters = sum(s["characters"] for s in scores)
    unknown = sum(s["unknown_characters"] for s in scores)
    nll = math.fsum(s["nll"] for s in scores)
    entropy = nll / targets
    return {"documents": len(scores), "targets": targets, "characters": characters,
            "unknown_characters": unknown, "unknown_character_rate": unknown / characters if characters else 0,
            "nll": nll, "cross_entropy_nats": entropy, "bits_per_target": entropy / math.log(2),
            "perplexity": math.exp(entropy)}
