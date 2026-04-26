"""
Complete APEX-1 Model.

Assembles all components into the full APEX-1 decoder-only transformer:
- Input embedding with sqrt(d_model) scaling
- Weight tying between embedding and LM head
- Stack of APEXTransformerBlock layers
- Final RMSNorm
- LM head (weight-tied to embedding)
- Multi-token prediction heads

Full forward pass follows Section 16 of the architecture document.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn

from apex.config import APEXConfig
from apex.model.block import APEXTransformerBlock
from apex.model.mask import build_apex_attention_mask, is_global_layer
from apex.model.multi_token_head import MultiTokenHead
from apex.model.norm import RMSNorm
from apex.model.rope import precompute_rope_cache_with_yarn

logger = logging.getLogger(__name__)


class APEX1Model(nn.Module):
    """Complete APEX-1 Language Model.

    A decoder-only transformer combining MLA, GQA+SW, MoE, skip gates,
    multi-token prediction, and built-in thinking mode support.

    Args:
        config: Complete APEXConfig instance.
    """

    def __init__(self, config: APEXConfig) -> None:
        super().__init__()
        self.config = config
        m = config.model

        # ── Embedding ────────────────────────────────────────────────────
        self.embedding = nn.Embedding(m.vocab_size, m.d_model)
        self.embed_scale = math.sqrt(m.d_model)

        # ── Transformer blocks ───────────────────────────────────────────
        self.blocks = nn.ModuleList([APEXTransformerBlock(i, config) for i in range(m.n_layers)])

        # ── Final norm ───────────────────────────────────────────────────
        self.final_norm = RMSNorm(m.d_model)

        # ── LM head (weight-tied to embedding) ──────────────────────────
        # We don't create a separate Linear — we use embedding.weight.T
        # This saves vocab_size × d_model parameters

        # ── Multi-token prediction heads ─────────────────────────────────
        if config.multi_token_head.enabled:
            self.multi_token_head = MultiTokenHead(
                d_model=m.d_model,
                vocab_size=m.vocab_size,
                n_predict=config.multi_token_head.n_predict,
            )
        else:
            self.multi_token_head = None

        # ── Precompute RoPE cache ────────────────────────────────────────
        cos_cache, sin_cache, self.attn_factor = precompute_rope_cache_with_yarn(
            d_head=m.d_head,
            max_seq_len=m.max_seq_len,
            rope_base=m.rope_base,
            scale_factor=m.rope_scaling,
        )
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

        # Also precompute for d_head_rope (used by MLA decoupled RoPE)
        cos_rope, sin_rope, _ = precompute_rope_cache_with_yarn(
            d_head=m.d_head_rope,
            max_seq_len=m.max_seq_len,
            rope_base=m.rope_base,
            scale_factor=m.rope_scaling,
        )
        self.register_buffer("cos_cache_rope", cos_rope, persistent=False)
        self.register_buffer("sin_cache_rope", sin_rope, persistent=False)

        # Initialize weights
        self._init_weights()

        logger.info(
            "APEX-1 Model initialized: %d layers, d_model=%d, vocab=%d, "
            "total_params=%s, active_params=%s",
            m.n_layers,
            m.d_model,
            m.vocab_size,
            self._format_params(self.total_parameters()),
            self._format_params(self.active_parameters()),
        )

    def _init_weights(self) -> None:
        """Initialize weights with standard normal scaled by 1/sqrt(d_model)."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        prefix_len: int = 0,
        kv_caches: Optional[list[Any]] = None,
        return_hidden: bool = False,
    ) -> dict[str, Any]:
        """Full forward pass through APEX-1.

        Args:
            token_ids: Input token IDs ``[batch, seq_len]``.
            positions: Position indices ``[seq_len]``. If None, uses 0..seq_len-1.
            prefix_len: Number of prefix tokens for bidirectional attention.
            kv_caches: List of per-layer KV caches from previous steps.
            return_hidden: If True, also return final hidden states.

        Returns:
            Dict containing:
            - ``logits``: Main LM head logits ``[batch, seq_len, vocab_size]``
            - ``spec_logits``: List of speculative logits (if multi-token head enabled)
            - ``kv_caches``: Updated KV caches for each layer
            - ``hidden_states``: Final hidden states (if return_hidden=True)
        """
        batch, seq_len = token_ids.shape
        device = token_ids.device

        # ── Step 1: Embedding ────────────────────────────────────────────
        x = self.embedding(token_ids) * self.embed_scale
        # x: [batch, seq_len, d_model]

        # ── Step 2: Positions ────────────────────────────────────────────
        if positions is None:
            if kv_caches is not None and kv_caches[0] is not None:
                # Autoregressive: continue from previous length
                if isinstance(kv_caches[0], torch.Tensor):
                    prev_len = kv_caches[0].shape[1]
                elif isinstance(kv_caches[0], tuple):
                    prev_len = kv_caches[0][0].shape[2]
                else:
                    prev_len = 0
                positions = torch.arange(prev_len, prev_len + seq_len, device=device)
            else:
                positions = torch.arange(seq_len, device=device)

        # ── Step 3: Build attention masks ────────────────────────────────
        total_len = positions[-1].item() + 1 if positions.numel() > 0 else seq_len

        # ── Step 4: Pass through transformer blocks ──────────────────────
        new_kv_caches: list[Any] = []

        for i, block in enumerate(self.blocks):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            layer_is_global = is_global_layer(i, self.config.attention.global_layer_freq)

            # Build per-layer mask
            attn_mask = build_apex_attention_mask(
                prefix_len=prefix_len if kv_caches is None else 0,
                total_len=seq_len,
                local_window=self.config.attention.local_window,
                is_global_layer=layer_is_global,
                device=device,
            )

            # Select appropriate RoPE cache
            if layer_is_global:
                # MLA uses d_head_rope for decoupled RoPE
                cos = self.cos_cache_rope
                sin = self.sin_cache_rope
            else:
                # GQA uses d_head for standard RoPE
                cos = self.cos_cache
                sin = self.sin_cache

            x, new_kv = block(x, cos, sin, positions, attn_mask, layer_kv)
            new_kv_caches.append(new_kv)

        # ── Step 5: Final norm ───────────────────────────────────────────
        x = self.final_norm(x)
        # x: [batch, seq_len, d_model]

        # ── Step 6: LM head (weight-tied to embedding) ──────────────────
        logits = torch.matmul(x, self.embedding.weight.T)
        # logits: [batch, seq_len, vocab_size]

        # ── Step 7: Multi-token prediction ───────────────────────────────
        spec_logits = None
        if self.multi_token_head is not None:
            spec_logits = self.multi_token_head(x)

        result: dict[str, Any] = {
            "logits": logits,
            "spec_logits": spec_logits,
            "kv_caches": new_kv_caches,
        }

        if return_hidden:
            result["hidden_states"] = x

        return result

    def total_parameters(self) -> int:
        """Count total parameters in the model.

        Returns:
            Total number of parameters.
        """
        return sum(p.numel() for p in self.parameters())

    def active_parameters(self) -> int:
        """Estimate active parameters per token (excluding inactive MoE experts).

        Returns:
            Estimated active parameters per forward pass per token.
        """
        total = 0

        # Embedding + final norm
        total += self.embedding.weight.numel()
        total += sum(p.numel() for p in self.final_norm.parameters())

        for block in self.blocks:
            # Attention (always active)
            total += sum(p.numel() for p in block.attn.parameters())
            # Norms
            total += sum(p.numel() for p in block.norm1.parameters())
            total += sum(p.numel() for p in block.norm2.parameters())

            # FFN
            if block.is_moe:
                moe = block.ffn
                # Shared experts always active
                for expert in moe.shared_experts:
                    total += sum(p.numel() for p in expert.parameters())
                # Only n_active routed experts active per token
                if len(moe.routed_experts) > 0:
                    expert_params = sum(p.numel() for p in moe.routed_experts[0].parameters())
                    total += expert_params * moe.n_active
                # Router
                total += sum(p.numel() for p in moe.router.parameters())
            else:
                total += sum(p.numel() for p in block.ffn.parameters())

            # Skip gate
            if block.use_skip_gate:
                total += sum(p.numel() for p in block.skip_gate.parameters())

        # Multi-token head
        if self.multi_token_head is not None:
            total += sum(p.numel() for p in self.multi_token_head.parameters())

        return total

    def get_moe_layers(self) -> list[tuple[int, Any]]:
        """Get all MoE FFN layers and their indices.

        Returns:
            List of (layer_idx, MoEFFN) tuples.
        """
        moe_layers = []
        for i, block in enumerate(self.blocks):
            if block.is_moe:
                moe_layers.append((i, block.ffn))
        return moe_layers

    @staticmethod
    def _format_params(n: int) -> str:
        """Format parameter count to human-readable string."""
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        elif n >= 1e6:
            return f"{n / 1e6:.1f}M"
        elif n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)
