"""
Feed-Forward Networks for APEX-1.

Implements two FFN variants:
1. DenseFFN — SwiGLU activation with 3 weight matrices (W_gate, W_up, W_down)
2. MoEFFN — Mixture of Experts with router, top-K selection, shared experts,
             and auxiliary-loss-free load balancing

SwiGLU formula:
    FFN(x) = W_down(SiLU(W_gate(x)) ⊙ W_up(x))

MoE architecture:
    output = shared_experts(x) + Σ(routing_weight_k × expert_k(x))
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

    Uses three linear projections and elementwise gating:
    - W_gate: projects to intermediate dim and applies SiLU
    - W_up: projects to intermediate dim (value branch)
    - W_down: projects gated output back to model dim

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

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.

        Returns:
            Output tensor ``[batch, seq_len, d_model]``.
        """
        gate = self.act(self.W_gate(x))   # [batch, seq, d_ffn]
        value = self.W_up(x)              # [batch, seq, d_ffn]
        hidden = gate * value             # elementwise gating
        return self.W_down(hidden)        # [batch, seq, d_model]


class MoEFFN(nn.Module):
    """Mixture of Experts Feed-Forward Network.

    Replaces a single large FFN with multiple smaller expert FFNs.
    Only top-K experts process each token, scaling capacity without
    scaling compute proportionally.

    Features:
    - Shared experts: always active, bypass router
    - Routed experts: top-K selected per token via learned router
    - Load balancing via external bias (auxiliary-loss-free)

    Args:
        config: APEXConfig with MoE parameters.
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.n_experts = config.moe.n_experts
        self.n_active = config.moe.n_active
        self.n_shared = config.moe.n_shared
        self.d_model = config.model.d_model

        # Shared experts — always computed, bypass router
        self.shared_experts = nn.ModuleList([
            DenseFFN(config) for _ in range(self.n_shared)
        ])

        # Routed experts — only top-K activated per token
        self.routed_experts = nn.ModuleList([
            DenseFFN(config) for _ in range(self.n_experts)
        ])

        # Router: scores each expert for each token
        self.router = nn.Linear(config.model.d_model, self.n_experts, bias=False)

        # Load-balancing biases — updated each step, NOT via backprop
        self.register_buffer(
            "expert_bias", torch.zeros(self.n_experts), persistent=False
        )

        # Store last routing decisions for load balancer
        self._last_top_k_idx: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through MoE FFN.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.

        Returns:
            Output tensor ``[batch, seq_len, d_model]``.
        """
        batch, seq_len, d_model = x.shape
        # Flatten tokens for routing
        x_flat = x.view(-1, d_model)   # [batch * seq, d_model]
        n_tokens = x_flat.shape[0]

        # Shared experts: always compute
        shared_out = torch.zeros_like(x)
        for expert in self.shared_experts:
            shared_out = shared_out + expert(x)

        # Router
        router_logits = self.router(x_flat)                # [n_tokens, n_experts]
        biased_logits = router_logits + self.expert_bias   # add load-balance bias

        top_k_vals, top_k_idx = torch.topk(
            biased_logits, self.n_active, dim=-1
        )
        # top_k_vals: [n_tokens, n_active]
        # top_k_idx:  [n_tokens, n_active]

        # Store for load balancer
        self._last_top_k_idx = top_k_idx.detach()

        # Routing weights: softmax over selected experts only
        routing_weights = torch.softmax(top_k_vals, dim=-1)   # [n_tokens, n_active]

        # Dispatch tokens to experts
        routed_out = torch.zeros_like(x_flat)   # [n_tokens, d_model]

        # Group tokens by expert for efficient batched computation
        for e_idx in range(self.n_experts):
            # Find all tokens routed to expert e_idx
            token_mask = (top_k_idx == e_idx).any(dim=-1)   # [n_tokens]
            if not token_mask.any():
                continue   # no tokens for this expert this step

            tokens_for_expert = x_flat[token_mask]           # [n_e, d_model]
            expert_output = self.routed_experts[e_idx](
                tokens_for_expert.unsqueeze(0)
            ).squeeze(0)

            # Handle single-token case
            if expert_output.dim() == 1:
                expert_output = expert_output.unsqueeze(0)

            # Find routing weight for this expert for each relevant token
            expert_positions = (top_k_idx[token_mask] == e_idx)  # [n_e, n_active]
            # Get the position index where this expert appears in the top-k
            e_position = expert_positions.float().argmax(dim=-1)  # [n_e]
            weights = routing_weights[token_mask].gather(
                1, e_position.unsqueeze(1).long()
            ).squeeze(1)   # [n_e]

            routed_out[token_mask] += expert_output * weights.unsqueeze(-1)

        routed_out = routed_out.view(batch, seq_len, d_model)
        return shared_out + routed_out

    def get_last_routing_indices(self) -> Optional[torch.Tensor]:
        """Return the last routing decisions for load balancer updates.

        Returns:
            Tensor of shape ``[n_tokens, n_active]`` with expert indices,
            or None if no forward pass has been run yet.
        """
        return self._last_top_k_idx

    def set_expert_bias(self, bias: torch.Tensor) -> None:
        """Update expert biases from the load balancer.

        Args:
            bias: New bias tensor of shape ``[n_experts]``.
        """
        self.expert_bias.copy_(bias.to(self.expert_bias.device))
