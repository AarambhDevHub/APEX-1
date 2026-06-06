"""
Complete APEX-1 Model.

v2.5.0 adds optional PEFT/LoRA adapter injection; v2.7.0 adds QLoRA-style 4-bit adapter injection. When ``config.peft.enabled``
is true, selected linear projections are wrapped with ``LoRALinear`` or ``QLoRALinear`` after base
weight initialization. The base model can be frozen while only adapter weights
remain trainable.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional

import torch
import torch.nn as nn

from apex.config import APEXConfig
from apex.model.block import APEXTransformerBlock
from apex.model.lora import apply_lora_adapters, peft_parameter_summary
from apex.model.mask import build_apex_attention_mask, is_global_layer
from apex.model.multi_token_head import MultiTokenHead
from apex.model.norm import RMSNorm
from apex.model.rope import precompute_rope_cache_with_yarn

logger = logging.getLogger(__name__)


class APEX1Model(nn.Module):
    """Complete APEX-1 Language Model."""

    def __init__(self, config: APEXConfig) -> None:
        super().__init__()
        self.config = config
        m = config.model

        self.embedding = nn.Embedding(m.vocab_size, m.d_model)
        self.embed_scale = math.sqrt(m.d_model)

        self.blocks = nn.ModuleList([APEXTransformerBlock(i, config) for i in range(m.n_layers)])

        self.final_norm = RMSNorm(m.d_model)

        if config.multi_token_head.enabled:
            self.multi_token_head = MultiTokenHead(
                d_model=m.d_model,
                vocab_size=m.vocab_size,
                n_predict=config.multi_token_head.n_predict,
            )
        else:
            self.multi_token_head = None

        cos_cache, sin_cache, self.attn_factor = precompute_rope_cache_with_yarn(
            d_head=m.d_head,
            max_seq_len=m.max_seq_len,
            rope_base=m.rope_base,
            scale_factor=m.rope_scaling,
        )
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

        cos_rope, sin_rope, _ = precompute_rope_cache_with_yarn(
            d_head=m.d_head_rope,
            max_seq_len=m.max_seq_len,
            rope_base=m.rope_base,
            scale_factor=m.rope_scaling,
        )
        self.register_buffer("cos_cache_rope", cos_rope, persistent=False)
        self.register_buffer("sin_cache_rope", sin_rope, persistent=False)

        self._init_weights()

        if config.peft.enabled:
            apply_lora_adapters(self, config.peft)
            peft_stats = peft_parameter_summary(self)
            logger.info(
                "LoRA/PEFT enabled: trainable=%s / total=%s (%.4f%%)",
                self._format_params(int(peft_stats["trainable"])),
                self._format_params(int(peft_stats["total"])),
                float(peft_stats["trainable_percent"]),
            )

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
        """Full forward pass through APEX-1."""
        batch, seq_len = token_ids.shape
        device = token_ids.device

        x = self.embedding(token_ids) * self.embed_scale

        if positions is None:
            if kv_caches is not None and kv_caches[0] is not None:
                layer_0_is_global = is_global_layer(0, self.config.attention.global_layer_freq)
                cache_0 = kv_caches[0]
                if layer_0_is_global:
                    prev_len = cache_0[0].shape[1]
                else:
                    prev_len = cache_0[0].shape[2]
                positions = torch.arange(prev_len, prev_len + seq_len, device=device)
            else:
                positions = torch.arange(seq_len, device=device)

        new_kv_caches: list[Any] = []

        for i, block in enumerate(self.blocks):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            layer_is_global = is_global_layer(i, self.config.attention.global_layer_freq)

            attn_mask = build_apex_attention_mask(
                prefix_len=prefix_len if kv_caches is None else 0,
                total_len=seq_len,
                local_window=self.config.attention.local_window,
                is_global_layer=layer_is_global,
                device=device,
            )

            if layer_is_global:
                cos = self.cos_cache_rope
                sin = self.sin_cache_rope
            else:
                cos = self.cos_cache
                sin = self.sin_cache

            x, new_kv = block(x, cos, sin, positions, attn_mask, layer_kv)
            new_kv_caches.append(new_kv)

        x = self.final_norm(x)
        logits = torch.matmul(x, self.embedding.weight.T)

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
        return sum(p.numel() for p in self.parameters())

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def active_parameters(self) -> int:
        total = 0
        total += self.embedding.weight.numel()
        total += sum(p.numel() for p in self.final_norm.parameters())

        for block in self.blocks:
            total += sum(p.numel() for p in block.attn.parameters())
            total += sum(p.numel() for p in block.norm1.parameters())
            total += sum(p.numel() for p in block.norm2.parameters())

            if block.is_moe:
                moe = block.ffn
                for expert in moe.shared_experts:
                    total += sum(p.numel() for p in expert.parameters())
                if len(moe.routed_experts) > 0:
                    expert_params = sum(p.numel() for p in moe.routed_experts[0].parameters())
                    total += expert_params * moe.n_active
                total += sum(p.numel() for p in moe.router.parameters())
            else:
                total += sum(p.numel() for p in block.ffn.parameters())

            if block.use_skip_gate:
                total += sum(p.numel() for p in block.skip_gate.parameters())

        if self.multi_token_head is not None:
            total += sum(p.numel() for p in self.multi_token_head.parameters())

        return total

    def get_moe_layers(self) -> list[tuple[int, Any]]:
        return [(i, block.ffn) for i, block in enumerate(self.blocks) if block.is_moe]

    @staticmethod
    def _format_params(n: int) -> str:
        if n >= 1e9:
            return f"{n / 1e9:.1f}B"
        if n >= 1e6:
            return f"{n / 1e6:.1f}M"
        if n >= 1e3:
            return f"{n / 1e3:.1f}K"
        return str(n)
