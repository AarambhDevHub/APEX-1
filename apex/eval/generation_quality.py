"""Lightweight generation quality checks.

These metrics are not a replacement for human evaluation or benchmark suites.
They are quick educational signals that help learners inspect generated text.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationQualityReport:
    count: int
    average_length: float
    distinct_1: float
    distinct_2: float
    repetition_rate: float

    def as_dict(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "average_length": self.average_length,
            "distinct_1": self.distinct_1,
            "distinct_2": self.distinct_2,
            "repetition_rate": self.repetition_rate,
        }


def _tokenize(text: str) -> list[str]:
    return [tok for tok in text.strip().split() if tok]


def average_length(texts: list[str]) -> float:
    if not texts:
        return 0.0
    return sum(len(_tokenize(text)) for text in texts) / len(texts)


def distinct_n(texts: list[str], n: int = 1) -> float:
    """Return ratio of unique n-grams to total n-grams."""
    if n <= 0:
        raise ValueError("n must be positive")
    total = 0
    unique: set[tuple[str, ...]] = set()
    for text in texts:
        tokens = _tokenize(text)
        if len(tokens) < n:
            continue
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i : i + n])
            unique.add(ngram)
            total += 1
    return len(unique) / total if total else 0.0


def repetition_rate(texts: list[str]) -> float:
    """Return token repetition rate across generated texts.

    0.0 means no token repeats. Higher values mean more repeated tokens.
    """
    repeated = 0
    total = 0
    for text in texts:
        counts = Counter(_tokenize(text))
        total += sum(counts.values())
        repeated += sum(max(0, count - 1) for count in counts.values())
    return repeated / total if total else 0.0


def evaluate_generated_texts(texts: list[str]) -> GenerationQualityReport:
    return GenerationQualityReport(
        count=len(texts),
        average_length=average_length(texts),
        distinct_1=distinct_n(texts, n=1),
        distinct_2=distinct_n(texts, n=2),
        repetition_rate=repetition_rate(texts),
    )
