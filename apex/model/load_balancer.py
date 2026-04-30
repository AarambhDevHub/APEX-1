"""
Auxiliary-Loss-Free Load Balancer for MoE.

Maintains per-expert bias terms that nudge routing toward underused experts.
Updated every step via a simple rule — NOT via gradient / backprop.
The LM objective sees zero interference from load balancing.

Algorithm (per step):
    1. Count how many tokens each expert received
    2. Compute observed load fraction per expert
    3. delta = target_rate - observed_rate
    4. bias += alpha * sign(delta)
    5. Clamp bias to [-1.0, 1.0]

This is called AFTER optimizer.step() in the training loop.

Source: DeepSeek-V3 — achieves near-uniform expert utilization with
zero perplexity degradation vs auxiliary-loss balancing.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class LoadBalancer:
    """Auxiliary-loss-free expert load balancer.

    Maintains a per-expert bias that is injected into router logits
    to encourage balanced expert utilization. The bias is updated
    outside the computation graph — no gradient flows through it.

    Args:
        n_experts: Total number of routed experts.
        target_rate: Ideal fraction of tokens per expert.
                    Defaults to ``1 / n_experts`` (uniform).
        alpha: Bias update step size. Small values (0.001) give
              slow but stable convergence.
    """

    def __init__(
        self,
        n_experts: int,
        target_rate: Optional[float] = None,
        alpha: float = 0.001,
    ) -> None:
        self.n_experts = n_experts
        self.target_rate = target_rate if target_rate is not None else (1.0 / n_experts)
        self.alpha = alpha
        self.bias = torch.zeros(n_experts)

        # Tracking statistics
        self._total_updates: int = 0
        self._cumulative_counts = torch.zeros(n_experts)

    def update(self, top_k_idx: torch.Tensor) -> dict[str, float]:
        """Update expert biases based on observed routing decisions.

        Must be called AFTER optimizer.step() in the training loop.
        The bias update is outside the computation graph.

        Args:
            top_k_idx: Routing decisions of shape ``[n_tokens, n_active]``
                      containing expert indices chosen per token.

        Returns:
            Dict with load balancing statistics:
            - ``max_load``: highest observed load fraction
            - ``min_load``: lowest observed load fraction
            - ``load_std``: standard deviation of load fractions
            - ``bias_range``: (min_bias, max_bias) tuple
        """
        n_tokens = top_k_idx.shape[0]
        n_active = top_k_idx.shape[1]

        # Count how many tokens each expert received
        counts = (
            torch.bincount(
                top_k_idx.flatten().to(torch.long),
                minlength=self.n_experts,
            )
            .float()
            .to(top_k_idx.device)
        )

        # Normalize to get observed load fraction per expert
        total_assignments = n_tokens * n_active
        observed_rate = counts / total_assignments

        # Adjust bias:
        #   overloaded expert  (rate > target) → decrease bias
        #   underloaded expert (rate < target) → increase bias
        delta = self.target_rate - observed_rate.cpu()
        self.bias += self.alpha * delta.sign()

        # Bias is bounded to prevent extreme values
        self.bias = self.bias.clamp(-1.0, 1.0)

        # Update statistics
        self._total_updates += 1
        self._cumulative_counts += counts.cpu()

        stats = {
            "max_load": observed_rate.max().item(),
            "min_load": observed_rate.min().item(),
            "load_std": observed_rate.std().item(),
            "bias_min": self.bias.min().item(),
            "bias_max": self.bias.max().item(),
        }

        if self._total_updates % 100 == 0:
            logger.debug(
                "LoadBalancer step %d: load_std=%.4f, bias_range=[%.3f, %.3f]",
                self._total_updates,
                stats["load_std"],
                stats["bias_min"],
                stats["bias_max"],
            )

        return stats

    def get_bias(self) -> torch.Tensor:
        """Get current expert biases for injection into router logits.

        Returns:
            Bias tensor of shape ``[n_experts]``.
        """
        return self.bias.clone()

    def get_cumulative_distribution(self) -> torch.Tensor:
        """Get the cumulative token distribution across experts.

        Returns:
            Normalized distribution tensor of shape ``[n_experts]``.
        """
        total = self._cumulative_counts.sum()
        if total == 0:
            return torch.ones(self.n_experts) / self.n_experts
        return self._cumulative_counts / total

    def reset_statistics(self) -> None:
        """Reset cumulative tracking statistics."""
        self._total_updates = 0
        self._cumulative_counts.zero_()

    def state_dict(self) -> dict:
        """Serialize balancer state for checkpointing.

        Returns:
            Dict containing bias and statistics.
        """
        return {
            "bias": self.bias.clone(),
            "total_updates": self._total_updates,
            "cumulative_counts": self._cumulative_counts.clone(),
            "n_experts": self.n_experts,
            "target_rate": self.target_rate,
            "alpha": self.alpha,
        }

    def load_state_dict(self, state: dict) -> None:
        """Load balancer state from a checkpoint.

        Args:
            state: State dict from ``state_dict()``.
        """
        self.bias = state["bias"]
        self._total_updates = state["total_updates"]
        self._cumulative_counts = state["cumulative_counts"]
        self.n_experts = state["n_experts"]
        self.target_rate = state["target_rate"]
        self.alpha = state["alpha"]
        logger.info(
            "LoadBalancer state loaded: %d updates, alpha=%.4f",
            self._total_updates,
            self.alpha,
        )
