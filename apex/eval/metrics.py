"""Small evaluation metrics for APEX-1.

These helpers are deliberately simple so students can read and modify them.
They work with raw logits returned by ``APEX1Model`` or ``APEX1VisionModel``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ClassificationMetrics:
    """Token-level classification metrics."""

    total: int
    correct: int
    accuracy: float


def _validate_logits_and_labels(logits: torch.Tensor, labels: torch.Tensor) -> None:
    if logits.ndim != 3:
        raise ValueError(f"logits must have shape [batch, seq, vocab], got {tuple(logits.shape)}")
    if labels.ndim != 2:
        raise ValueError(f"labels must have shape [batch, seq], got {tuple(labels.shape)}")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            "logits and labels sequence shapes must match: "
            f"logits[:2]={tuple(logits.shape[:2])}, labels={tuple(labels.shape)}"
        )


def next_token_accuracy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> ClassificationMetrics:
    """Return token accuracy for logits already aligned with labels.

    Args:
        logits: Prediction scores ``[batch, seq, vocab]``.
        labels: Target token IDs ``[batch, seq]``. Positions equal to
            ``ignore_index`` are skipped.
        ignore_index: Label value to ignore.

    Note:
        This function does not shift inputs. If you want standard language-model
        next-token accuracy, pass ``logits[:, :-1]`` and ``input_ids[:, 1:]``.
    """
    _validate_logits_and_labels(logits, labels)
    predictions = logits.argmax(dim=-1)
    mask = labels != ignore_index
    total = int(mask.sum().item())
    if total == 0:
        return ClassificationMetrics(total=0, correct=0, accuracy=0.0)
    correct = int(((predictions == labels) & mask).sum().item())
    return ClassificationMetrics(total=total, correct=correct, accuracy=correct / total)


def token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
    reduction: str = "mean",
) -> torch.Tensor:
    """Cross-entropy for aligned token logits and labels.

    This is useful for demos and tests where logits and labels are already the
    same sequence length. For standard LM evaluation, shift first.
    """
    _validate_logits_and_labels(logits, labels)
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]),
        labels.reshape(-1),
        ignore_index=ignore_index,
        reduction=reduction,
    )


def shift_for_language_modeling(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Shift logits/labels for next-token language-model evaluation."""
    if logits.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected logits [B,S,V] and labels [B,S]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("logits and labels must share batch/sequence dimensions before shift")
    if logits.shape[1] < 2:
        raise ValueError("Need sequence length >= 2 for language-model shifting")
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()
