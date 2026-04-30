# 15 — APEX1Model: The Complete Model Assembly

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/model/apex_model.py`  
> **You will learn:** How all blocks are stacked, the two RoPE cache problem (BUG-07), KV cache position detection (BUG-09), and the full forward pass.

---

## 1. The Complete Architecture

```
Input token IDs  [batch, seq_len]
        │
        ▼
  Embedding Lookup × √d_model   → [batch, seq_len, d_model]
        │
        ▼
  Block 0 (LOCAL  / Dense)  ─┐
  Block 1 (LOCAL  / MoE  )   │
  Block 2 (LOCAL  / Dense)   │  × n_layers blocks
  Block 3 (LOCAL  / MoE  )   │
  Block 4 (LOCAL  / Dense)   │
  Block 5 (GLOBAL / MoE  )  ─┘
        │  ... (repeats)
        ▼
  Final RMSNorm
        │
        ▼
  LM Head (= Embedding.weight.T)   → [batch, seq_len, vocab_size]
        │
        ▼
  Output: {logits, spec_logits, kv_caches, [hidden_states]}
```

---

## 2. The Two RoPE Cache Problem (BUG-07)

APEX-1 has two types of attention layers that use RoPE with **different head dimensions**:

| Layer Type | RoPE applied to | Dimension |
|---|---|---|
| GQA (local) | Full Q and K | `d_head` (e.g., 64) |
| MLA (global) | Decoupled rope projection | `d_head_rope` (e.g., 32) |

If you precompute only one cache, you get a shape mismatch when the wrong cache is passed to a layer.

**BUG-07 Fix:** Precompute **two** separate caches in `__init__`:

```python
# For GQA layers: full head dimension
self.cos_cache, self.sin_cache = precompute_rope_cache_with_yarn(
    d_head=m.d_head, ...)

# For MLA layers: smaller rope-only dimension
self.cos_cache_rope, self.sin_cache_rope = precompute_rope_cache_with_yarn(
    d_head=m.d_head_rope, ...)
```

And pass the correct cache to each block based on `is_global_layer()`.

---

## 3. KV Cache Position Detection (BUG-09)

During autoregressive generation, the model needs to know **how many tokens have already been processed** (to compute the correct positions for RoPE).

The original code used `isinstance(cache, torch.Tensor)` to detect whether layer 0's cache was MLA or GQA:

```python
# ORIGINAL (fragile): relies on cache type — wrong if layer ordering changes
if isinstance(kv_caches[0], torch.Tensor):
    prev_len = kv_caches[0].shape[1]   # assumed MLA
else:
    prev_len = kv_caches[0][0].shape[2]  # assumed GQA
```

**Problem:** Both MLA and GQA caches are tuples — `isinstance` always returned the same thing.

**BUG-09 Fix:** Use `is_global_layer()` to determine layer 0's type, then access the correct field:

```python
# FIXED: use is_global_layer to know which cache format
if is_global_layer(0, global_layer_freq):
    # MLA cache: (c_kv, K_rope), c_kv is [B, prev_len, d_kv_compressed]
    prev_len = kv_caches[0][0].shape[1]
else:
    # GQA cache: (K, V), K is [B, n_kv, prev_len, d_head]
    prev_len = kv_caches[0][0].shape[2]
```

---

## 4. Full Annotated Source: `apex/model/apex_model.py`

```python
"""
APEX-1: Complete Language Model.

Assembles all components into the full model:
  Embedding → n_layers blocks → final norm → LM head

BUG-07: Two separate RoPE caches (d_head and d_head_rope).
BUG-09: KV cache position detection via is_global_layer().
"""

import math
import torch
import torch.nn as nn

from apex.config import APEXConfig
from apex.model.block import APEXTransformerBlock
from apex.model.multi_token_head import MultiTokenHead
from apex.model.norm import RMSNorm
from apex.model.rope import precompute_rope_cache_with_yarn
from apex.model.mask import build_apex_attention_mask, is_global_layer


class APEX1Model(nn.Module):
    def __init__(self, config: APEXConfig) -> None:
        super().__init__()
        self.config = config
        m = config.model

        # ── Embedding ────────────────────────────────────────────────────
        # Input: token IDs [batch, seq]
        # Output: dense vectors [batch, seq, d_model]
        self.embedding = nn.Embedding(m.vocab_size, m.d_model)
        self.embed_scale = math.sqrt(m.d_model)   # scale for stable magnitudes

        # ── Transformer Blocks ───────────────────────────────────────────
        # Build n_layers blocks — each decides its own type from layer_idx
        self.blocks = nn.ModuleList([
            APEXTransformerBlock(layer_idx=i, config=config)
            for i in range(m.n_layers)
        ])

        # ── Final Normalisation ───────────────────────────────────────────
        self.final_norm = RMSNorm(m.d_model)

        # ── Multi-Token Prediction Head (optional) ───────────────────────
        if config.multi_token_head.enabled:
            self.multi_token_head = MultiTokenHead(
                d_model=m.d_model,
                vocab_size=m.vocab_size,
                n_predict=config.multi_token_head.n_predict,
            )
        else:
            self.multi_token_head = None

        # ── RoPE Caches (BUG-07 FIX: two separate caches) ────────────────
        # Cache 1: for GQA local layers (full d_head)
        cos_cache, sin_cache, _ = precompute_rope_cache_with_yarn(
            d_head=m.d_head,
            max_seq_len=m.max_seq_len,
            rope_base=m.rope_base,
            scale_factor=m.rope_scaling,
        )
        # Register as buffers: they move to GPU with .to(device)
        # but are NOT trained (no gradient)
        self.register_buffer("cos_cache", cos_cache)
        self.register_buffer("sin_cache", sin_cache)

        # Cache 2: for MLA global layers (smaller d_head_rope)
        cos_cache_rope, sin_cache_rope, _ = precompute_rope_cache_with_yarn(
            d_head=m.d_head_rope,
            max_seq_len=m.max_seq_len,
            rope_base=m.rope_base,
            scale_factor=m.rope_scaling,
        )
        self.register_buffer("cos_cache_rope", cos_cache_rope)
        self.register_buffer("sin_cache_rope", sin_cache_rope)

        # ── Weight Initialisation ────────────────────────────────────────
        self._init_weights()

    def _init_weights(self) -> None:
        """Initialise weights with Gaussian distribution.
        
        Standard deviation = 0.02 (following GPT-2 and most modern LLMs).
        Linear layers and embeddings get this treatment.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        token_ids: torch.Tensor,       # [batch, seq_len]
        prefix_len: int = 0,           # Number of prefix (bidirectional) tokens
        kv_caches=None,                # List of per-layer KV caches (or None)
        return_hidden: bool = False,   # Whether to return hidden states
    ) -> dict:
        """Full forward pass.
        
        Args:
            token_ids:    Input token IDs [batch, seq_len].
            prefix_len:   Tokens before this index get full bidirectional attention.
            kv_caches:    List of KV caches from previous decoding steps.
            return_hidden:If True, include final hidden states in output.
        
        Returns:
            dict with keys:
              'logits':      [batch, seq_len, vocab_size]
              'spec_logits': List of [batch, seq_len, vocab_size] or None
              'kv_caches':   List of updated per-layer KV caches
              'hidden_states': [batch, seq_len, d_model] if return_hidden
        """
        batch, seq_len = token_ids.shape
        m = self.config.model

        # ── Step 1: Compute absolute positions ───────────────────────────
        # If using KV cache, new tokens start at position = prev_len
        if kv_caches is not None:
            # BUG-09 FIX: use is_global_layer() to determine cache format
            global_layer_freq = self.config.attention.global_layer_freq
            layer_0_is_global = is_global_layer(0, global_layer_freq)

            if layer_0_is_global:
                # MLA cache format: (c_kv, K_rope); c_kv.shape[1] = prev_seq_len
                prev_len = kv_caches[0][0].shape[1]
            else:
                # GQA cache format: (K, V); K.shape[2] = prev_seq_len
                prev_len = kv_caches[0][0].shape[2]
        else:
            prev_len = 0

        # Positions for the new tokens: [prev_len, prev_len+1, ..., prev_len+seq_len-1]
        positions = torch.arange(
            prev_len, prev_len + seq_len,
            device=token_ids.device
        )

        # ── Step 2: Token embedding + scale ──────────────────────────────
        x = self.embedding(token_ids) * self.embed_scale   # [B, S, d_model]

        # ── Step 3: Build attention mask ──────────────────────────────────
        # Compute total sequence length including cached context
        total_len = prev_len + seq_len

        # Build masks for each layer type (global vs local)
        # We build them here once and pass to each block
        attn_mask_global = build_apex_attention_mask(
            prefix_len=prefix_len,
            total_len=total_len,
            local_window=self.config.attention.local_window,
            is_global_layer=True,
            device=token_ids.device,
        )
        attn_mask_local = build_apex_attention_mask(
            prefix_len=prefix_len,
            total_len=total_len,
            local_window=self.config.attention.local_window,
            is_global_layer=False,
            device=token_ids.device,
        )

        # ── Step 4: Pass through all transformer blocks ──────────────────
        new_kv_caches = []
        for layer_idx, block in enumerate(self.blocks):
            # Select correct RoPE cache for this layer's type
            if block.is_global:
                # MLA uses the smaller d_head_rope cache
                cos = self.cos_cache_rope
                sin = self.sin_cache_rope
                mask = attn_mask_global
            else:
                # GQA uses the full d_head cache
                cos = self.cos_cache
                sin = self.sin_cache
                mask = attn_mask_local

            # Get this layer's existing KV cache (or None for first pass)
            layer_kv_cache = kv_caches[layer_idx] if kv_caches is not None else None

            # Run the block
            x, new_layer_kv = block(
                x,
                cos_cache=cos,
                sin_cache=sin,
                positions=positions,
                attn_mask=mask,
                kv_cache=layer_kv_cache,
            )
            new_kv_caches.append(new_layer_kv)

        # ── Step 5: Final normalisation ───────────────────────────────────
        x = self.final_norm(x)   # [B, S, d_model]

        # ── Step 6: LM head (weight-tied with embedding) ─────────────────
        # logits[i,j,k] = probability that token k follows tokens 0..j in example i
        logits = torch.matmul(x, self.embedding.weight.T)   # [B, S, vocab]

        # ── Step 7: Speculative heads (optional) ─────────────────────────
        spec_logits = None
        if self.multi_token_head is not None:
            spec_logits = self.multi_token_head(x)

        # ── Build output dict ─────────────────────────────────────────────
        output = {
            "logits": logits,           # Main LM logits
            "spec_logits": spec_logits, # Speculative head logits (or None)
            "kv_caches": new_kv_caches, # Updated KV caches for next step
        }

        if return_hidden:
            output["hidden_states"] = x  # Used by reward model and PRM

        return output

    def count_parameters(self) -> dict[str, int]:
        """Count total and active parameters."""
        total = sum(p.numel() for p in self.parameters())

        # Active = total minus idle MoE expert weights
        # Idle experts = total routed - active routed
        moe_idle = 0
        for block in self.blocks:
            if hasattr(block.ffn, "n_experts"):
                n_idle = block.ffn.n_experts - block.ffn.n_active
                expert_params = sum(p.numel() for p in block.ffn.experts[0].parameters())
                moe_idle += n_idle * expert_params

        return {"total": total, "active": total - moe_idle, "moe_idle": moe_idle}
```

---

## 5. Forward Pass in One Sentence

> Tokens → embedding → 12–72 blocks of (norm, attention, norm, FFN) with residuals → final norm → dot with embedding weights → vocabulary scores.

---

**Next:** [16 — Training Losses →](16-training-losses.md)
