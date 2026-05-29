"""Training utilities for APEX-1 vision-language SFT."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def expand_labels_for_visual_tokens(
    token_ids: torch.Tensor,
    labels: torch.Tensor,
    image_token_id: int,
    n_visual_tokens: int,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Expand labels so they match logits after visual-token insertion.

    The vision model replaces one ``<|img|>`` text token with ``n_visual_tokens``
    continuous embeddings. Those positions are context only, so labels are set
    to ``ignore_index``.

    Args:
        token_ids: Original text tokens ``[B, S]``.
        labels: Original text labels ``[B, S]``.
        image_token_id: Token ID for ``<|img|>``.
        n_visual_tokens: Number of inserted visual tokens.
        ignore_index: Loss ignore label.

    Returns:
        Expanded labels ``[B, S - 1 + n_visual_tokens]`` if image placeholder
        exists, otherwise ``[B, S + n_visual_tokens]`` with visual tokens after
        the first token.
    """
    if token_ids.shape != labels.shape:
        raise ValueError("token_ids and labels must have the same shape")
    if token_ids.ndim != 2:
        raise ValueError("token_ids and labels must be [B,S]")
    if n_visual_tokens <= 0:
        raise ValueError("n_visual_tokens must be positive")

    rows: list[torch.Tensor] = []
    expected_len: int | None = None
    ignore = torch.full((n_visual_tokens,), ignore_index, dtype=labels.dtype, device=labels.device)

    for b in range(token_ids.shape[0]):
        matches = (token_ids[b] == image_token_id).nonzero(as_tuple=False).flatten()
        if matches.numel() > 0:
            idx = int(matches[0].item())
            row = torch.cat([labels[b, :idx], ignore, labels[b, idx + 1 :]], dim=0)
        else:
            idx = 1 if labels.shape[1] > 0 else 0
            row = torch.cat([labels[b, :idx], ignore, labels[b, idx:]], dim=0)
        if expected_len is None:
            expected_len = row.shape[0]
        elif row.shape[0] != expected_len:
            raise ValueError("Expanded labels must have equal sequence length across the batch")
        rows.append(row)

    return torch.stack(rows, dim=0)


def compute_vision_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute autoregressive SFT loss for text outputs in multimodal batches.

    Args:
        logits: Model logits ``[B, S, vocab]``.
        labels: Expanded labels ``[B, S]``. Non-answer and visual positions
            should be ``ignore_index``.
        ignore_index: Label ID ignored by cross entropy.
    """
    if logits.ndim != 3:
        raise ValueError("logits must be [B,S,V]")
    if labels.ndim != 2:
        raise ValueError("labels must be [B,S]")
    if logits.shape[:2] != labels.shape:
        raise ValueError(
            f"logits sequence shape {tuple(logits.shape[:2])} must match labels {tuple(labels.shape)}"
        )

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )
    with torch.no_grad():
        valid = (shift_labels != ignore_index).sum().item()
    return loss, {"loss_vision_sft": float(loss.detach().cpu()), "valid_tokens": float(valid)}
