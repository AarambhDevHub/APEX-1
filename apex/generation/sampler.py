"""
Sampling Strategies for APEX-1.

Implements temperature scaling, top-p (nucleus) sampling, top-k sampling,
and repetition penalty. These can be combined in the generation loop.

Recommended defaults:
    Factual Q&A:       temperature=0.3, top_p=0.9,  top_k=off
    Creative writing:  temperature=0.9, top_p=0.95, top_k=50
    Code generation:   temperature=0.1, top_p=1.0,  top_k=off
    Reasoning (final): temperature=0.3, top_p=0.9
    Reasoning (think): temperature=0.6, top_p=0.95
"""

from __future__ import annotations

from typing import Optional

import torch


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Args:
        logits: Raw logits ``[..., vocab_size]``.
        temperature: Scaling factor.
            - 1.0: raw distribution
            - < 1.0: sharper (more deterministic)
            - > 1.0: flatter (more random)
            - → 0: greedy (always argmax)

    Returns:
        Scaled logits.
    """
    if temperature <= 0:
        return logits  # Will use argmax
    return logits / temperature


def apply_top_p(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """Apply top-p (nucleus) sampling filter.

    Removes tokens once cumulative probability exceeds top_p,
    keeping only the smallest set of tokens whose cumulative
    probability is at least top_p.

    Args:
        logits: Logits ``[..., vocab_size]``.
        top_p: Cumulative probability threshold (0.0 to 1.0).
              1.0 disables filtering.

    Returns:
        Filtered logits with removed tokens set to -inf.
    """
    if top_p >= 1.0:
        return logits

    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(
        torch.softmax(sorted_logits, dim=-1), dim=-1
    )

    # Remove tokens once cumulative probability exceeds top_p
    remove_mask = cumulative_probs > top_p
    # Shift right: keep the first token that crosses the threshold
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False  # always keep the best token

    sorted_logits[remove_mask] = float("-inf")

    # Scatter back to original order
    result = torch.zeros_like(logits)
    result.scatter_(-1, sorted_idx, sorted_logits)

    return result


def apply_top_k(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Apply top-k sampling filter.

    Keeps only the top-k highest probability tokens and sets
    all others to -inf.

    Args:
        logits: Logits ``[..., vocab_size]``.
        top_k: Number of top tokens to keep. 0 disables filtering.

    Returns:
        Filtered logits.
    """
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits

    top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, top_k_indices, top_k_logits)

    return filtered


def apply_repetition_penalty(
    logits: torch.Tensor,
    generated_ids: list[int],
    penalty: float = 1.1,
) -> torch.Tensor:
    """Apply repetition penalty to discourage repeated tokens.

    For tokens that have already been generated:
    - Positive logits are divided by the penalty
    - Negative logits are multiplied by the penalty

    Args:
        logits: Logits ``[vocab_size]`` (single position).
        generated_ids: List of previously generated token IDs.
        penalty: Penalty factor (1.0 = no penalty, 1.1-1.3 typical).

    Returns:
        Penalized logits.
    """
    if penalty == 1.0 or not generated_ids:
        return logits

    logits = logits.clone()
    unique_ids = set(generated_ids)

    for token_id in unique_ids:
        if 0 <= token_id < logits.shape[-1]:
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty

    return logits


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_p: float = 1.0,
    top_k: int = 0,
    generated_ids: Optional[list[int]] = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    """Sample the next token from logits with all strategies combined.

    Applies strategies in order: repetition penalty → temperature →
    top-k → top-p → sampling.

    Args:
        logits: Raw logits ``[vocab_size]`` for a single position.
        temperature: Temperature scaling factor.
        top_p: Nucleus sampling threshold.
        top_k: Top-k filtering (0 = disabled).
        generated_ids: Previously generated token IDs (for repetition penalty).
        repetition_penalty: Repetition penalty factor.

    Returns:
        Sampled token ID tensor ``[1]``.
    """
    # Apply repetition penalty
    if generated_ids and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # Apply temperature
    if temperature <= 0:
        # Greedy decoding
        return logits.argmax(dim=-1, keepdim=True)

    logits = apply_temperature(logits, temperature)

    # Apply top-k
    logits = apply_top_k(logits, top_k)

    # Apply top-p
    logits = apply_top_p(logits, top_p)

    # Sample
    probs = torch.softmax(logits, dim=-1)

    # Handle case where all probs are 0 (all filtered out)
    if probs.sum() == 0:
        probs = torch.ones_like(probs) / probs.shape[-1]

    next_token = torch.multinomial(probs, num_samples=1)
    return next_token
