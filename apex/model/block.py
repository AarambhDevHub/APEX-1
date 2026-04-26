"""
APEX-1 Transformer Block.

Implements a single transformer block with the full APEX-1 architecture:
1. Pre-norm with RMSNorm
2. Attention (MLA for global layers, GQA+SW for local layers)
3. Residual connection
4. Dynamic skip gate
5. FFN (Dense or MoE on alternating layers)
6. Residual connection (gated by skip gate)

The block structure:
    Input x [batch, seq_len, d_model]
      │
      ├─ RMSNorm(x)
      │       └─► Attention → + x  (residual)
      │                         x'
      ├─ Dynamic Skip Gate — evaluates x' per token
      │       ├─ gate < 0.15 → SKIP FFN
      │       └─ gate ≥ 0.15 → FFN:
      │                   RMSNorm(x')
      │                       └─► FFN → + x'  (residual)
    Output x'' [batch, seq_len, d_model]
"""

from __future__ import annotations

from typing import Any, Optional, Union

import torch
import torch.nn as nn

from apex.model.attention import GQASlidingWindowAttention, MLAAttention
from apex.model.ffn import DenseFFN, MoEFFN
from apex.model.mask import is_global_layer
from apex.model.norm import RMSNorm
from apex.model.skip_gate import SkipGate


class APEXTransformerBlock(nn.Module):
    """Single APEX-1 Transformer Block.

    Dispatches between MLA (global) and GQA+SW (local) attention based
    on layer index, and between Dense FFN and MoE FFN on alternating layers.
    Includes a dynamic skip gate that can bypass FFN for simple tokens.

    Args:
        layer_idx: Zero-based index of this layer in the stack.
        config: APEXConfig with all model parameters.
    """

    def __init__(self, layer_idx: int, config: Any) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.is_global = is_global_layer(layer_idx, config.attention.global_layer_freq)

        # Pre-norm layers
        self.norm1 = RMSNorm(config.model.d_model)
        self.norm2 = RMSNorm(config.model.d_model)

        # Attention: MLA for global layers, GQA+SW for local layers
        if self.is_global:
            self.attn = MLAAttention(config)
        else:
            self.attn = GQASlidingWindowAttention(config)

        # FFN: alternate dense and MoE
        if config.moe.enabled and layer_idx % config.moe.moe_layer_freq != 0:
            self.ffn = MoEFFN(config)
            self.is_moe = True
        else:
            self.ffn = DenseFFN(config)
            self.is_moe = False

        # Dynamic skip gate
        if config.skip_gate.enabled:
            self.skip_gate = SkipGate(
                config.model.d_model,
                hidden_dim=config.skip_gate.hidden_dim,
                threshold=config.skip_gate.threshold,
            )
            self.use_skip_gate = True
        else:
            self.use_skip_gate = False

        self.skip_threshold = config.skip_gate.threshold

    def forward(
        self,
        x: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> tuple[torch.Tensor, Any]:
        """Forward pass through one transformer block.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            cos_cache: RoPE cosine cache.
            sin_cache: RoPE sine cache.
            positions: Position indices ``[seq_len]``.
            attn_mask: Attention mask (bool, True = attend).
            kv_cache: Cached KV from previous steps.
                     For MLA: ``torch.Tensor`` (c_kv latent).
                     For GQA: ``tuple[Tensor, Tensor]`` (K, V caches).

        Returns:
            Tuple of (output ``[batch, seq_len, d_model]``, new_kv_cache).
        """
        # ── Attention sub-layer (always runs) ────────────────────────────
        h, new_kv = self.attn(
            self.norm1(x),
            cos_cache,
            sin_cache,
            positions,
            attn_mask,
            kv_cache,
        )
        x = x + h   # residual

        # ── Dynamic skip gate — may bypass FFN ───────────────────────────
        if self.use_skip_gate:
            gate = self.skip_gate(x)  # [batch, seq, 1] values in (0, 1)
            skip_mask = gate < self.skip_threshold  # True = skip FFN

            if skip_mask.all():
                # Every token skips — avoid FFN entirely (fast path)
                return x, new_kv

            # ── FFN sub-layer (skipped for low-complexity tokens) ─────────
            ffn_out = self.ffn(self.norm2(x))

            # Apply gate: skipped tokens contribute 0, active tokens contribute ffn_out
            x = x + ffn_out * (~skip_mask).float()  # residual with selective gate
        else:
            # No skip gate — always run FFN
            ffn_out = self.ffn(self.norm2(x))
            x = x + ffn_out

        return x, new_kv

    def extra_repr(self) -> str:
        """Return string representation of block configuration."""
        attn_type = "MLA (global)" if self.is_global else "GQA+SW (local)"
        ffn_type = "MoE" if self.is_moe else "Dense"
        skip = "enabled" if self.use_skip_gate else "disabled"
        return (
            f"layer_idx={self.layer_idx}, "
            f"attn={attn_type}, ffn={ffn_type}, skip_gate={skip}"
        )
