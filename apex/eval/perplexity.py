"""Perplexity evaluation for APEX-1 language models."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class PerplexityResult:
    """Aggregated perplexity result."""

    loss: float
    perplexity: float
    token_count: int
    batch_count: int

    def as_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "perplexity": self.perplexity,
            "token_count": self.token_count,
            "batch_count": self.batch_count,
        }


def _extract_tensor_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(input_ids, labels)`` from common batch formats."""
    if isinstance(batch, torch.Tensor):
        return batch, batch
    if isinstance(batch, dict):
        if "input_ids" not in batch:
            raise KeyError("Batch dict must contain 'input_ids'")
        input_ids = batch["input_ids"]
        labels = batch.get("labels", input_ids)
        return input_ids, labels
    if isinstance(batch, (tuple, list)) and len(batch) >= 1:
        input_ids = batch[0]
        labels = batch[1] if len(batch) > 1 else input_ids
        return input_ids, labels
    raise TypeError(
        "Unsupported batch type. Expected Tensor, dict with input_ids, or tuple/list."
    )


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    dataloader: Iterable[Any],
    device: torch.device | str | None = None,
    ignore_index: int = -100,
    max_batches: int | None = None,
) -> PerplexityResult:
    """Compute next-token cross-entropy and perplexity.

    Args:
        model: APEX model returning a dict with ``logits``.
        dataloader: Iterable of batches. Supported batch formats:
            ``Tensor``; ``dict(input_ids=..., labels=...)``; or tuple/list.
        device: Optional device. If omitted, uses the model parameter device.
        ignore_index: Label value to skip.
        max_batches: Optional cap for quick smoke tests.
    """
    was_training = model.training
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)

    total_loss = 0.0
    total_tokens = 0
    batch_count = 0

    for batch_idx, batch in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        input_ids, labels = _extract_tensor_batch(batch)
        input_ids = input_ids.to(device)
        labels = labels.to(device)

        if input_ids.ndim != 2:
            raise ValueError(f"input_ids must be [batch, seq], got {tuple(input_ids.shape)}")
        if labels.shape != input_ids.shape:
            raise ValueError("labels must have the same shape as input_ids")
        if input_ids.shape[1] < 2:
            continue

        output = model(input_ids)
        logits = output["logits"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        flat_labels = shift_labels.reshape(-1)
        valid_tokens = int((flat_labels != ignore_index).sum().item())
        if valid_tokens == 0:
            continue

        loss_sum = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            flat_labels,
            ignore_index=ignore_index,
            reduction="sum",
        )
        total_loss += float(loss_sum.item())
        total_tokens += valid_tokens
        batch_count += 1

    if was_training:
        model.train()

    if total_tokens == 0:
        return PerplexityResult(loss=0.0, perplexity=float("inf"), token_count=0, batch_count=batch_count)

    mean_loss = total_loss / total_tokens
    perplexity = math.exp(min(mean_loss, 100.0))
    return PerplexityResult(
        loss=mean_loss,
        perplexity=perplexity,
        token_count=total_tokens,
        batch_count=batch_count,
    )
