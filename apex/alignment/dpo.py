"""
DPO — Direct Preference Optimization for APEX-1.

DPO skips the reward model entirely and trains directly on preference pairs.
It uses the implicit reward from log-probability ratios between the policy
and a frozen reference model.

Loss (Section 13b):
    reward_chosen   = β × (log π(chosen|prompt) - log π_ref(chosen|prompt))
    reward_rejected = β × (log π(rejected|prompt) - log π_ref(rejected|prompt))
    loss = -log(sigmoid(reward_chosen - reward_rejected))
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sequence_logprob(
    logits: torch.Tensor,
    token_ids: torch.Tensor,
    response_start_idx: int,
) -> torch.Tensor:
    """Compute log-probability of a response given logits.

    Sums log-probabilities of response tokens only (not the prompt).

    Args:
        logits: Model logits ``[1, seq_len, vocab_size]``.
        token_ids: Full token IDs ``[1, seq_len]``.
        response_start_idx: Index where the response starts.

    Returns:
        Scalar log-probability.
    """
    # Shift logits and targets
    shift_logits = logits[:, :-1, :]  # predict next token
    shift_targets = token_ids[:, 1:]  # ground truth next token

    # Compute per-token log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        2, shift_targets.unsqueeze(-1)
    ).squeeze(-1)

    # Sum only response tokens (from response_start_idx - 1 due to shift)
    start = max(0, response_start_idx - 1)
    response_log_prob = token_log_probs[:, start:].sum(dim=-1)

    return response_log_prob


def dpo_loss(
    model: nn.Module,
    reference_model: nn.Module,
    prompt_ids: torch.Tensor,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    prompt_len: int,
    beta: float = 0.1,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DPO loss for a preference pair.

    Args:
        model: Policy model being trained.
        reference_model: Frozen reference (SFT) model.
        prompt_ids: Prompt token IDs ``[1, prompt_len]``.
        chosen_ids: Full (prompt + chosen response) IDs ``[1, total_len]``.
        rejected_ids: Full (prompt + rejected response) IDs ``[1, total_len]``.
        prompt_len: Length of the prompt.
        beta: KL penalty coefficient. Higher = more conservative.

    Returns:
        Tuple of (loss, metrics_dict).
    """
    # Log-probabilities from policy (model being trained)
    chosen_logits = model(chosen_ids)["logits"]
    rejected_logits = model(rejected_ids)["logits"]

    log_pi_chosen = compute_sequence_logprob(chosen_logits, chosen_ids, prompt_len)
    log_pi_rejected = compute_sequence_logprob(rejected_logits, rejected_ids, prompt_len)

    # Log-probabilities from reference (frozen SFT model)
    with torch.no_grad():
        ref_chosen_logits = reference_model(chosen_ids)["logits"]
        ref_rejected_logits = reference_model(rejected_ids)["logits"]

        log_ref_chosen = compute_sequence_logprob(ref_chosen_logits, chosen_ids, prompt_len)
        log_ref_rejected = compute_sequence_logprob(ref_rejected_logits, rejected_ids, prompt_len)

    # Implicit reward = β × (log π(y|x) - log π_ref(y|x))
    reward_chosen = beta * (log_pi_chosen - log_ref_chosen)
    reward_rejected = beta * (log_pi_rejected - log_ref_rejected)

    # Loss: maximize margin between chosen and rejected implicit rewards
    loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

    metrics = {
        "dpo_loss": loss.item(),
        "reward_chosen": reward_chosen.mean().item(),
        "reward_rejected": reward_rejected.mean().item(),
        "reward_margin": (reward_chosen - reward_rejected).mean().item(),
        "log_pi_chosen": log_pi_chosen.mean().item(),
        "log_pi_rejected": log_pi_rejected.mean().item(),
    }

    return loss, metrics
