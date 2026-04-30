"""
Training Loss Functions for APEX-1.

Implements:
1. Pretraining loss: next-token prediction + multi-token auxiliary (Section 12a)
2. SFT loss: cross-entropy on assistant tokens only (Section 12b)

Fix BUG-12: Speculative head losses now guard against short sequences
where the offset ``k`` leaves fewer than 1 overlapping token between
the logits slice ``[:, :-k]`` and the target slice ``[:, k:]``.
Previously the guard ``if k >= token_ids.shape[1]: break`` was
off-by-one — it allowed ``k == seq_len - 1`` which gives only a single
token, but missed the case where ``seq_len - k < 1`` after the shift,
causing an empty cross-entropy that returned ``nan``.

Pretraining formula:
    L = L_main + λ × mean(L_offset_k for k in 1..n_predict)
    λ = 0.1

SFT formula:
    L = CrossEntropy(logits, labels) where labels[non-assistant] = -100
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def compute_pretrain_loss(
    logits_main: torch.Tensor,
    logits_speculative: Optional[list[torch.Tensor]],
    token_ids: torch.Tensor,
    vocab_size: int,
    lambda_spec: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute pretraining loss with multi-token auxiliary heads.

    Main loss: predict next token at every position (shift by 1).
    Speculative losses: predict token at offset k (shift by k).

    Args:
        logits_main: Main LM head logits ``[batch, seq_len, vocab_size]``.
        logits_speculative: List of speculative logits, each
                           ``[batch, seq_len, vocab_size]``. Can be None.
        token_ids: Ground truth token IDs ``[batch, seq_len]``.
        vocab_size: Vocabulary size for reshaping.
        lambda_spec: Weight of speculative heads in total loss (default: 0.1).

    Returns:
        Tuple of (total_loss, metrics_dict) where metrics_dict contains
        individual loss components for logging.
    """
    # Main loss: predict next token at every position
    # Shift: input positions 0..T-1 predict targets 1..T
    main_logits = logits_main[:, :-1, :].contiguous().view(-1, vocab_size)
    main_targets = token_ids[:, 1:].contiguous().view(-1)
    l_main = F.cross_entropy(main_logits, main_targets)

    metrics = {"loss_main": l_main.item()}

    # Speculative head losses: predict token at offset k
    if logits_speculative is not None and len(logits_speculative) > 0:
        l_spec_total = torch.tensor(0.0, device=logits_main.device)
        n_spec = 0

        for k, spec_logits in enumerate(logits_speculative, start=1):
            # BUG-12 FIX: skip heads where the offset k leaves fewer
            # than 1 overlapping position between logits and targets.
            # seq_len - k is the number of positions remaining after
            # slicing; need at least 1 for a valid cross-entropy.
            if token_ids.shape[1] - k < 1:
                break
            spec_l = spec_logits[:, :-k, :].contiguous().view(-1, vocab_size)
            spec_t = token_ids[:, k:].contiguous().view(-1)
            l_k = F.cross_entropy(spec_l, spec_t)
            l_spec_total = l_spec_total + l_k
            n_spec += 1
            metrics[f"loss_spec_{k}"] = l_k.item()

        if n_spec > 0:
            l_spec_avg = l_spec_total / n_spec
            metrics["loss_spec_avg"] = l_spec_avg.item()
            total_loss = l_main + lambda_spec * l_spec_avg
        else:
            total_loss = l_main
    else:
        total_loss = l_main

    metrics["loss_total"] = total_loss.item()
    return total_loss, metrics


def compute_sft_loss(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    token_types: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute SFT loss on assistant tokens only.

    Only computes loss where token_type == 2 (assistant). System and user
    tokens are masked with ignore_index=-100 so they contribute zero
    to the loss.

    Args:
        logits: Model logits ``[batch, seq_len, vocab_size]``.
        token_ids: Ground truth token IDs ``[batch, seq_len]``.
        token_types: Token type labels ``[batch, seq_len]``.
                    0=system, 1=user, 2=assistant.
        vocab_size: Vocabulary size for reshaping.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    labels = token_ids.clone()

    # Ignore system and user tokens in loss
    labels[token_types != 2] = -100

    # Shift by 1 (predict next token)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100,
    )

    # Count how many tokens contributed to loss
    n_assistant = (shift_labels != -100).sum().item()
    n_total = shift_labels.numel()

    metrics = {
        "loss_sft": loss.item(),
        "assistant_tokens": n_assistant,
        "total_tokens": n_total,
        "assistant_ratio": n_assistant / max(n_total, 1),
    }

    return loss, metrics
