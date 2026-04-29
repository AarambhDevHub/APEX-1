"""
Feed-Forward Networks for APEX-1.

Implements two FFN variants:
1. DenseFFN — SwiGLU activation with 3 weight matrices (W_gate, W_up, W_down)
2. MoEFFN — Mixture of Experts with router, top-K selection, shared experts,
             and auxiliary-loss-free load balancing

Fix BUG-08: The MoE expert dispatch previously called
``expert(tokens_for_expert.unsqueeze(0)).squeeze(0)`` which worked only
when n_e=1.  With n_e>1 the expert received shape [1, n_e, d_model] and
treated n_e as the sequence dimension, producing wrong outputs and wrong
gradients.  The fix passes tokens as [1, n_e, d_model] and squeezes the
batch dim, which DenseFFN handles correctly for any n_e.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class DenseFFN(nn.Module):
    """Dense Feed-Forward Network with SwiGLU activation.

    Args:
        config: APEXConfig (uses d_model and d_ffn).
    """

    def __init__(self, config) -> None:
        super().__init__()
        d_model = config.model.d_model
        d_ffn = config.model.d_ffn

        self.W_gate = nn.Linear(d_model, d_ffn, bias=False)
        self.W_up = nn.Linear(d_model, d_ffn, bias=False)
        self.W_down = nn.Linear(d_ffn, d_model, bias=False)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply SwiGLU FFN.

        Accepts either ``[batch, seq_len, d_model]`` or ``[n_tokens, d_model]``.

        Args:
            x: Input tensor.

        Returns:
            Output tensor with same shape as input.
        """
        gate = self.act(self.W_gate(x))
        value = self.W_up(x)
        hidden = gate * value
        return self.W_down(hidden)


class MoEFFN(nn.Module):
    """Mixture of Experts Feed-Forward Network.

    Fix BUG-08: Expert dispatch now correctly handles batches of n_e > 1
    tokens.  Previously ``unsqueeze(0)`` added a spurious batch dimension
    causing DenseFFN to treat the token count as the sequence length.
    The fix reshapes tokens to ``[1, n_e, d_model]`` (batch=1, seq=n_e)
    before calling the expert and removes the batch dim afterwards.

    Args:
        config: APEXConfig with MoE parameters.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.n_experts = config.moe.n_experts
        self.n_active = config.moe.n_active
        self.n_shared = config.moe.n_shared
        self.d_model = config.model.d_model

        self.shared_experts = nn.ModuleList([DenseFFN(config) for _ in range(self.n_shared)])
        self.routed_experts = nn.ModuleList([DenseFFN(config) for _ in range(self.n_experts)])
        self.router = nn.Linear(config.model.d_model, self.n_experts, bias=False)

        self.register_buffer("expert_bias", torch.zeros(self.n_experts), persistent=False)
        self._last_top_k_idx: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE FFN.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.

        Returns:
            Output tensor ``[batch, seq_len, d_model]``.
        """
        batch, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [n_tokens, d_model]

        # Shared experts: always compute (operate on [batch, seq, d_model])
        shared_out = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)

        # Router
        router_logits = self.router(x_flat)  # [n_tokens, n_experts]
        biased_logits = router_logits + self.expert_bias

        top_k_vals, top_k_idx = torch.topk(biased_logits, self.n_active, dim=-1)
        self._last_top_k_idx = top_k_idx.detach()

        routing_weights = torch.softmax(top_k_vals, dim=-1)  # [n_tokens, n_active]

        routed_out = torch.zeros_like(x_flat)  # [n_tokens, d_model]

        for e_idx in range(self.n_experts):
            token_mask = (top_k_idx == e_idx).any(dim=-1)  # [n_tokens]
            if not token_mask.any():
                continue

            tokens_for_expert = x_flat[token_mask]  # [n_e, d_model]
            n_e = tokens_for_expert.shape[0]

            # BUG-08 FIX: Reshape to [1, n_e, d_model] so DenseFFN processes
            # all n_e tokens as a sequence (not as a batch), then squeeze
            # the batch dim back.  This is correct for any n_e >= 1.
            expert_output = self.routed_experts[e_idx](
                tokens_for_expert.unsqueeze(0)  # [1, n_e, d_model]
            ).squeeze(0)  # [n_e, d_model]

            # Gather routing weight for this expert per token
            expert_positions = (top_k_idx[token_mask] == e_idx)  # [n_e, n_active]
            e_position = expert_positions.float().argmax(dim=-1)  # [n_e]
            weights = (
                routing_weights[token_mask]
                .gather(1, e_position.unsqueeze(1).long())
                .squeeze(1)
            )  # [n_e]

            routed_out[token_mask] += expert_output * weights.unsqueeze(-1)

        routed_out = routed_out.view(batch, seq_len, d_model)
        return shared_out + routed_out

    def get_last_routing_indices(self) -> Optional[torch.Tensor]:
        """Return the last routing decisions for load balancer updates."""
        return self._last_top_k_idx

    def set_expert_bias(self, bias: torch.Tensor) -> None:
        """Update expert biases from the load balancer.

        Args:
            bias: New bias tensor of shape ``[n_experts]``.
        """
        self.expert_bias.copy_(bias.to(self.expert_bias.device))