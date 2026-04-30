"""
APEX-1 Transformer Block.

Fix BUG-19: The ``is_moe`` flag now correctly checks ``config.moe.enabled``
before evaluating the layer-frequency condition.  Previously, blocks in a
non-MoE model could be labelled as MoE in ``extra_repr()`` output.
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

    Args:
        layer_idx: Zero-based index of this layer in the stack.
        config: APEXConfig with all model parameters.
    """

    def __init__(self, layer_idx: int, config: Any) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.is_global = is_global_layer(layer_idx, config.attention.global_layer_freq)

        self.norm1 = RMSNorm(config.model.d_model)
        self.norm2 = RMSNorm(config.model.d_model)

        if self.is_global:
            self.attn = MLAAttention(config)
        else:
            self.attn = GQASlidingWindowAttention(config)

        # BUG-19 FIX: check config.moe.enabled FIRST.  Without this check,
        # when MoE is disabled the flag could still be True if the
        # layer-frequency condition happened to be satisfied.
        if config.moe.enabled and layer_idx % config.moe.moe_layer_freq != 0:
            self.ffn = MoEFFN(config)
            self.is_moe = True
        else:
            self.ffn = DenseFFN(config)
            self.is_moe = False

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
        kv_cache: Optional[Union[torch.Tensor, tuple]] = None,
    ) -> tuple[torch.Tensor, Any]:
        """Forward pass through one transformer block.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            cos_cache: RoPE cosine cache.
            sin_cache: RoPE sine cache.
            positions: Position indices ``[seq_len]``.
            attn_mask: Attention mask (bool, True = attend).
            kv_cache: Cached KV from previous steps.

        Returns:
            Tuple of (output ``[batch, seq_len, d_model]``, new_kv_cache).
        """
        h, new_kv = self.attn(
            self.norm1(x),
            cos_cache,
            sin_cache,
            positions,
            attn_mask,
            kv_cache,
        )
        x = x + h

        if self.use_skip_gate:
            gate = self.skip_gate(x)
            skip_mask = gate < self.skip_threshold

            if skip_mask.all():
                return x, new_kv

            ffn_out = self.ffn(self.norm2(x))
            x = x + ffn_out * (~skip_mask).float()
        else:
            ffn_out = self.ffn(self.norm2(x))
            x = x + ffn_out

        return x, new_kv

    def extra_repr(self) -> str:
        """Return string representation of block configuration."""
        attn_type = "MLA (global)" if self.is_global else "GQA+SW (local)"
        ffn_type = "MoE" if self.is_moe else "Dense"
        skip = "enabled" if self.use_skip_gate else "disabled"
        return f"layer_idx={self.layer_idx}, " f"attn={attn_type}, ffn={ffn_type}, skip_gate={skip}"
