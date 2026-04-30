# 14 — The Transformer Block: One Complete Layer

> **Difficulty:** ⭐⭐☆☆☆ Intermediate  
> **Source file:** `apex/model/block.py`  
> **You will learn:** How all components assemble into a single transformer layer, what residual connections do, and the BUG-19 fix.

---

## 1. The Building Block

The entire APEX-1 model is built by stacking $n_{layers}$ identical blocks (one on top of the other). Each block is one "step of thinking" — it reads the current representation of all tokens and produces an improved representation.

---

## 2. Inside One Block

```
Input x  [batch, seq_len, d_model]
  │
  ├── RMSNorm (norm1)
  │       │
  │       └── Attention (MLA or GQA+SW depending on layer)
  │               │
  └───────────────┤ ← Residual connection (add attention output to x)
  x = x + attn_out
  │
  ├── Skip Gate decides: should we skip the FFN?
  │
  ├── (if not skip) RMSNorm (norm2)
  │                   │
  │                   └── FFN (Dense or MoE depending on layer)
  │                           │
  └───────────────────────────┤ ← Residual + skip mask
  x = x + ffn_out * (1 - skip_mask)
  │
  Output x  [batch, seq_len, d_model]
```

---

## 3. The Residual Connection

The residual connection (`x = x + layer_output`) is the single most important design choice in the transformer architecture (He et al., ResNet, 2015).

**Why is it essential?**

Without residual connections, in deep networks, gradients vanish as they propagate backward through many layers (the chain rule multiplies fractions repeatedly, approaching zero).

With residual connections:

$$x_{out} = x_{in} + \text{Layer}(x_{in})$$

The gradient can flow directly back through the $+$ operation without passing through the layer at all:

$$\frac{\partial L}{\partial x_{in}} = \frac{\partial L}{\partial x_{out}} + \frac{\partial L}{\partial x_{out}} \cdot \frac{\partial \text{Layer}}{\partial x_{in}}$$

The first term bypasses the layer entirely — gradients never vanish, even in 72-layer models.

---

## 4. Pre-Norm Architecture

APEX-1 uses **pre-norm** (normalise before each sub-layer):

```python
x = x + Attention(RMSNorm(x))   # normalise BEFORE attention
x = x + FFN(RMSNorm(x))          # normalise BEFORE FFN
```

The original "Attention Is All You Need" paper used **post-norm** (normalise after):

```python
x = RMSNorm(x + Attention(x))
```

Post-norm is harder to train for deep models. Pre-norm stabilises training and is standard in all modern LLMs (GPT-3, Llama, DeepSeek, etc.).

---

## 5. The BUG-19 Story: is_moe Always True

The original code computed `is_moe`:

```python
# ORIGINAL (broken):
is_moe = layer_idx % config.moe.moe_layer_freq != 0
```

This always evaluated to `True` or `False` regardless of `config.moe.enabled`. So even with `enabled=False` (dense-only mode), the code would try to instantiate a `MoEFFN` — and crash.

**Fix:**

```python
# FIXED: check enabled first
is_moe = config.moe.enabled and (layer_idx % config.moe.moe_layer_freq != 0)
```

---

## 6. Full Annotated Source: `apex/model/block.py`

```python
"""
APEX-1 Transformer Block.

Each block contains:
  1. Pre-norm + Attention (MLA or GQA+SW)
  2. Residual connection
  3. Skip gate decision
  4. Pre-norm + FFN (Dense or MoE)
  5. Residual connection with skip mask

BUG-19 FIX: is_moe now checks config.moe.enabled first.
"""

import torch
import torch.nn as nn

from apex.model.attention import MLAAttention, GQASlidingWindowAttention
from apex.model.ffn import DenseFFN, MoEFFN
from apex.model.norm import RMSNorm
from apex.model.skip_gate import SkipGate
from apex.model.mask import is_global_layer


class APEXTransformerBlock(nn.Module):
    """One complete APEX-1 transformer layer.
    
    Args:
        layer_idx: Zero-based index of this layer (determines type).
        config:    Full APEXConfig.
    """

    def __init__(self, layer_idx: int, config) -> None:
        super().__init__()
        m = config.model

        # ── Attention type: global (MLA) or local (GQA+SW)? ─────────────
        # BUG-07 context: layer type must be consistent between block and model
        self.is_global = is_global_layer(layer_idx, config.attention.global_layer_freq)

        if self.is_global:
            # Every 6th layer: Multi-Head Latent Attention (full causal)
            self.attention = MLAAttention(config)
        else:
            # Other 5 of 6 layers: GQA + Sliding Window (local)
            self.attention = GQASlidingWindowAttention(config)

        # ── FFN type: Dense or MoE? ──────────────────────────────────────
        # BUG-19 FIX: check enabled BEFORE computing moe_layer_freq condition
        is_moe_layer = config.moe.enabled and (layer_idx % config.moe.moe_layer_freq != 0)

        if is_moe_layer:
            # Mixture of Experts FFN
            self.ffn = MoEFFN(
                d_model=m.d_model,
                d_ffn=m.d_ffn,
                n_experts=config.moe.n_experts,
                n_active=config.moe.n_active,
                n_shared=config.moe.n_shared,
                dropout=m.dropout,
            )
        else:
            # Standard Dense SwiGLU FFN
            self.ffn = DenseFFN(
                d_model=m.d_model,
                d_ffn=m.d_ffn,
                dropout=m.dropout,
            )

        # ── Normalisations ───────────────────────────────────────────────
        self.norm1 = RMSNorm(m.d_model)   # Applied before attention
        self.norm2 = RMSNorm(m.d_model)   # Applied before FFN

        # ── Skip Gate (optional) ─────────────────────────────────────────
        if config.skip_gate.enabled:
            self.skip_gate = SkipGate(
                d_model=m.d_model,
                hidden_dim=config.skip_gate.hidden_dim,
                threshold=config.skip_gate.threshold,
            )
        else:
            self.skip_gate = None

    def forward(
        self,
        x: torch.Tensor,               # [batch, seq_len, d_model]
        cos_cache: torch.Tensor,        # RoPE cos table
        sin_cache: torch.Tensor,        # RoPE sin table
        positions: torch.Tensor,        # [seq_len] absolute positions
        attn_mask=None,                 # Boolean attention mask
        kv_cache=None,                  # Cached KV from previous steps
    ) -> tuple[torch.Tensor, object]:
        """Forward pass through one transformer block.
        
        Returns:
            (output_x, new_kv_cache)
        """
        # ── Sub-layer 1: Attention ───────────────────────────────────────
        # Pre-norm: normalise input before passing to attention
        normed_x = self.norm1(x)

        # Run attention (MLA or GQA+SW based on layer type)
        attn_out, new_kv_cache = self.attention(
            normed_x,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            positions=positions,
            attn_mask=attn_mask,
            kv_cache=kv_cache,
        )

        # Residual connection: add attention output to original x
        x = x + attn_out

        # ── Sub-layer 2: FFN (with optional skip gate) ───────────────────
        if self.skip_gate is not None:
            # Compute gate value and skip mask
            _, skip_mask = self.skip_gate(x)   # [B, S, 1]

            # Optimisation: if ALL tokens want to skip, avoid FFN entirely
            if not skip_mask.all():
                # Pre-norm before FFN
                normed_x2 = self.norm2(x)
                ffn_out = self.ffn(normed_x2)

                # Apply skip mask: skip_mask=1 → zero out FFN contribution
                # The residual (x) still passes through unchanged
                x = x + ffn_out * (1.0 - skip_mask)
        else:
            # No skip gate: always run FFN
            normed_x2 = self.norm2(x)
            ffn_out = self.ffn(normed_x2)
            x = x + ffn_out

        return x, new_kv_cache
```

---

## 7. Layer Assignment Summary

For a 12-layer model with `global_layer_freq=6`, `moe_layer_freq=2`:

| Layer | Attention Type | FFN Type |
|---|---|---|
| 0 | LOCAL (GQA+SW) | Dense |
| 1 | LOCAL (GQA+SW) | MoE |
| 2 | LOCAL (GQA+SW) | Dense |
| 3 | LOCAL (GQA+SW) | MoE |
| 4 | LOCAL (GQA+SW) | Dense |
| 5 | **GLOBAL (MLA)** | MoE |
| 6 | LOCAL (GQA+SW) | Dense |
| ... | ... | ... |
| 11 | **GLOBAL (MLA)** | MoE |

---

**Next:** [15 — APEX1Model — The Complete Model →](15-full-model.md)
