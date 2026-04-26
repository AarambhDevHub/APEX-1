"""
GRPO — Group Relative Policy Optimization for APEX-1.

DeepSeek-R1's key innovation for reasoning training. Eliminates the
critic/value network by using within-group relative reward as the
training signal.

For each prompt, sample G responses, rank them using combined reward,
and use group-normalized advantages with clipped surrogate objective.

Full algorithm from Section 13c of the architecture document.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def extract_thinking_steps(response_ids: list[int], thinking_start_id: int, thinking_end_id: int) -> list[list[int]]:
    """Extract reasoning steps from a response's thinking section.

    Splits the thinking content into steps based on newline tokens.

    Args:
        response_ids: Full response token IDs.
        thinking_start_id: Token ID for <|thinking|>.
        thinking_end_id: Token ID for <|/thinking|>.

    Returns:
        List of token ID lists, one per reasoning step.
    """
    steps: list[list[int]] = []
    in_thinking = False
    current_step: list[int] = []

    for tid in response_ids:
        if tid == thinking_start_id:
            in_thinking = True
            current_step = []
            continue
        if tid == thinking_end_id:
            if current_step:
                steps.append(current_step)
            in_thinking = False
            continue
        if in_thinking:
            current_step.append(tid)

    if current_step and in_thinking:
        steps.append(current_step)

    return steps if steps else [response_ids]


def compute_sequence_log_prob(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_start: int,
) -> torch.Tensor:
    """Compute log-probability of response tokens under a model.

    Args:
        model: Model to compute log-probs with.
        input_ids: Full token IDs ``[1, seq_len]``.
        response_start: Index where response begins.

    Returns:
        Scalar log-probability sum over response tokens.
    """
    output = model(input_ids)
    logits = output["logits"]

    shift_logits = logits[:, :-1, :]
    shift_targets = input_ids[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(
        2, shift_targets.unsqueeze(-1)
    ).squeeze(-1)

    start = max(0, response_start - 1)
    response_log_prob = token_log_probs[:, start:].sum(dim=-1)

    return response_log_prob.squeeze()


def grpo_training_step(
    model: nn.Module,
    reference_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    prompt_ids: torch.Tensor,
    response_ids_list: list[torch.Tensor],
    rewards: torch.Tensor,
    prompt_len: int,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    max_grad_norm: float = 1.0,
) -> tuple[float, dict[str, float]]:
    """Execute one GRPO training step.

    Takes pre-generated responses and their rewards, computes
    group-normalized advantages, and updates the policy with
    the clipped surrogate objective.

    Args:
        model: Policy model being trained.
        reference_model: Frozen reference (SFT) model.
        optimizer: Optimizer for the policy model.
        prompt_ids: Prompt token IDs ``[1, prompt_len]``.
        response_ids_list: List of G response token tensors.
        rewards: Combined rewards for each response ``[G]``.
        prompt_len: Length of the prompt.
        beta: KL penalty coefficient.
        clip_eps: PPO clipping epsilon.
        max_grad_norm: Maximum gradient norm for clipping.

    Returns:
        Tuple of (loss_value, metrics_dict).
    """
    device = next(model.parameters()).device
    G = len(response_ids_list)

    # Step 1: Compute group-normalized advantages
    group_mean = rewards.mean()
    group_std = rewards.std().clamp(min=1e-6)
    advantages = (rewards - group_mean) / group_std

    # Step 2: Compute GRPO loss for each response
    all_losses: list[torch.Tensor] = []
    all_kl: list[float] = []
    all_ratios: list[float] = []

    for i, response_ids in enumerate(response_ids_list):
        advantage = advantages[i]

        # Build full input (prompt + response)
        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)
        full_ids = torch.cat([prompt_ids, response_ids], dim=1).to(device)

        # Log-probability under current policy
        log_pi = compute_sequence_log_prob(model, full_ids, prompt_len)

        # Log-probability under reference policy (frozen)
        with torch.no_grad():
            log_ref = compute_sequence_log_prob(reference_model, full_ids, prompt_len)

        # KL divergence
        kl_div = log_pi - log_ref
        all_kl.append(kl_div.item())

        # Policy ratio
        ratio = torch.exp(log_pi - log_ref.detach())
        all_ratios.append(ratio.item())

        # Clipped surrogate objective
        l_clip = torch.min(
            ratio * advantage,
            torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage,
        )

        # Full loss: maximize reward, penalize KL drift
        loss = -(l_clip - beta * kl_div)
        all_losses.append(loss)

    # Step 3: Backprop and update
    total_loss = torch.stack(all_losses).mean()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    metrics = {
        "grpo_loss": total_loss.item(),
        "mean_reward": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "mean_kl": sum(all_kl) / len(all_kl),
        "mean_ratio": sum(all_ratios) / len(all_ratios),
        "advantage_max": advantages.max().item(),
        "advantage_min": advantages.min().item(),
    }

    return total_loss.item(), metrics


def grpo_full_loop(
    model: nn.Module,
    reference_model: nn.Module,
    optimizer: torch.optim.Optimizer,
    prompts: list[torch.Tensor],
    reward_fn: Callable,
    G: int = 8,
    beta: float = 0.04,
    clip_eps: float = 0.2,
    generation_config: Optional[Any] = None,
) -> dict[str, float]:
    """Full GRPO training loop over a batch of prompts.

    For each prompt:
    1. Sample G responses from current policy
    2. Score with reward function
    3. Run GRPO training step

    Args:
        model: Policy model.
        reference_model: Frozen reference model.
        optimizer: Policy optimizer.
        prompts: List of prompt token ID tensors.
        reward_fn: Function(prompt_ids, response_ids) -> float reward.
        G: Group size (rollouts per prompt).
        beta: KL penalty coefficient.
        clip_eps: Clipping epsilon.
        generation_config: Config for generation.

    Returns:
        Aggregated metrics across all prompts.
    """
    all_metrics: list[dict[str, float]] = []

    for prompt_ids in prompts:
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_len = prompt_ids.shape[1]

        # Sample G responses
        response_ids_list: list[torch.Tensor] = []
        for _ in range(G):
            # Generate a response (simplified — in production use APEX1Generator)
            with torch.no_grad():
                model.eval()
                output = model(prompt_ids)
                logits = output["logits"][:, -1, :]
                # Simple sampling for training rollouts
                probs = torch.softmax(logits / 0.7, dim=-1)
                tokens: list[int] = []
                for _ in range(128):
                    token = torch.multinomial(probs, 1)
                    tokens.append(token.item())
                    out = model(token)
                    logits = out["logits"][:, -1, :]
                    probs = torch.softmax(logits / 0.7, dim=-1)
                model.train()

            response_ids_list.append(
                torch.tensor([tokens], device=prompt_ids.device)
            )

        # Score responses
        rewards: list[float] = []
        for resp_ids in response_ids_list:
            r = reward_fn(prompt_ids, resp_ids)
            rewards.append(float(r))
        rewards_tensor = torch.tensor(rewards, device=prompt_ids.device)

        # GRPO step
        loss, metrics = grpo_training_step(
            model, reference_model, optimizer,
            prompt_ids, response_ids_list, rewards_tensor,
            prompt_len, beta, clip_eps,
        )
        all_metrics.append(metrics)

    # Aggregate metrics
    if all_metrics:
        agg = {}
        for key in all_metrics[0]:
            vals = [m[key] for m in all_metrics]
            agg[key] = sum(vals) / len(vals)
        return agg

    return {"grpo_loss": 0.0}
