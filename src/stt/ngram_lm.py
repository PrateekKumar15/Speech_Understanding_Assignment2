from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


class NGramLM:
    def __init__(self, order: int = 3, alpha: float = 0.1):
        self.order = order
        self.alpha = alpha
        self.ngram_counts: Dict[int, Counter[Tuple[str, ...]]] = defaultdict(Counter)
        self.vocab: set[str] = set()

    def fit_text(self, text: str) -> None:
        tokens = [x.strip().lower() for x in text.split() if x.strip()]
        self.fit_tokens(tokens)

    def fit_tokens(self, tokens: List[str]) -> None:
        self.vocab.update(tokens)
        for n in range(1, self.order + 1):
            for i in range(len(tokens) - n + 1):
                gram = tuple(tokens[i : i + n])
                self.ngram_counts[n][gram] += 1

    def score_next(self, history: List[str], token: str) -> float:
        token = token.lower()
        hist = [h.lower() for h in history]
        self.vocab.add(token)
        v = max(len(self.vocab), 1)

        max_context = min(self.order - 1, len(hist))
        for context_size in range(max_context, -1, -1):
            context = tuple(hist[-context_size:]) if context_size > 0 else tuple()
            n = context_size + 1
            gram = context + (token,)

            if n == 1:
                num = self.ngram_counts[1][(token,)] + self.alpha
                den = sum(self.ngram_counts[1].values()) + self.alpha * v
                return math.log(num / den)

            context_count = self.ngram_counts[context_size][context]
            if context_count > 0:
                num = self.ngram_counts[n][gram] + self.alpha
                den = context_count + self.alpha * v
                return math.log(num / den)

        return -12.0

    def save(self, path: str | Path) -> None:
        data = {
            "order": self.order,
            "alpha": self.alpha,
            "vocab": sorted(self.vocab),
            "counts": {
                str(n): {"|||".join(k): v for k, v in self.ngram_counts[n].items()}
                for n in range(1, self.order + 1)
            },
        }
        with Path(path).open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "NGramLM":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        obj = cls(order=data["order"], alpha=data["alpha"])
        obj.vocab = set(data["vocab"])
        for n in range(1, obj.order + 1):
            key = str(n)
            for gram, count in data["counts"][key].items():
                obj.ngram_counts[n][tuple(gram.split("|||"))] = count
        return obj
