# 30 — Utilities: Shape Checker, FLOPs, and Param Counter

> **Difficulty:** ⭐⭐☆☆☆ Intermediate  
> **Source files:** `apex/utils/shape_checker.py`, `apex/utils/flops.py`, `apex/utils/param_counter.py`  
> **You will learn:** How to debug tensor shape errors, estimate compute, and count model parameters — with BUG-17 and BUG-23.

---

## 1. Why Utilities Matter

When building neural networks, the most common errors are **shape mismatches** — passing a tensor of the wrong size to a layer. Shape checkers catch these instantly.

**FLOPs** (Floating-Point Operations) measure the computational cost of one forward pass — used to compare model efficiency and estimate hardware requirements.

**Param counters** show the "size" of the model and how much memory it needs.

---

## 2. Shape Checker — `apex/utils/shape_checker.py`

### BUG-23: Initialising a Random Model for Shape Checks

The original `verify_shapes()`:

```python
def verify_shapes(config: APEXConfig) -> dict:
    # BUG-23: creates a RANDOM model just to run a forward pass!
    # For a 900B model, this takes minutes and 500GB of RAM.
    model = APEX1Model(config)   # ← bug: always creates new model
    test_input = torch.zeros(1, 16, dtype=torch.long)
    output = model(test_input)
    return output
```

For verification during training, you already **have** the model. Creating a fresh random one wastes gigabytes of memory and minutes of time.

**Fix:** Accept an optional pre-built model:

```python
def verify_shapes(config: APEXConfig, model: Optional[APEX1Model] = None) -> dict:
    # BUG-23 FIX: use the existing model if provided
    if model is None:
        # Only create a new model if none was given (e.g., standalone check)
        model = APEX1Model(config)
    # ...
```

### Full Annotated Source

```python
"""
Tensor Shape Verification for APEX-1.

Runs a minimal forward pass and checks every output tensor's shape
against the expected shape given the configuration.

BUG-23 FIX: verify_shapes() accepts an optional pre-built model
instead of always creating a new random one.
"""

import logging
from typing import Optional
import torch
from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model

logger = logging.getLogger(__name__)


def verify_shapes(
    config: APEXConfig,
    model: Optional[APEX1Model] = None,    # BUG-23 FIX: optional model
    seq_len: int = 16,
    batch_size: int = 1,
) -> dict:
    """Verify all output tensor shapes are correct.
    
    Args:
        config:     Model config to verify.
        model:      Pre-built model to test (create new if None).
        seq_len:    Test sequence length.
        batch_size: Test batch size.
    
    Returns:
        dict of {shape_name: actual_shape} for logging.
    
    Raises:
        AssertionError: If any shape is wrong.
    """
    # BUG-23 FIX: use provided model; only create if none given
    if model is None:
        logger.warning(
            "No model provided to verify_shapes — creating a new one. "
            "This may use significant memory for large configs."
        )
        model = APEX1Model(config)

    model.eval()
    device = next(model.parameters()).device

    # Create dummy input
    test_input = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

    with torch.no_grad():
        output = model(test_input, return_hidden=True)

    m = config.model
    shapes = {}

    # ── Check logits shape ────────────────────────────────────────────
    logits = output["logits"]
    expected = (batch_size, seq_len, m.vocab_size)
    assert logits.shape == expected, (
        f"logits shape {logits.shape} != expected {expected}"
    )
    shapes["logits"] = logits.shape

    # ── Check hidden states shape ─────────────────────────────────────
    hidden = output.get("hidden_states")
    if hidden is not None:
        expected_h = (batch_size, seq_len, m.d_model)
        assert hidden.shape == expected_h, (
            f"hidden_states shape {hidden.shape} != expected {expected_h}"
        )
        shapes["hidden_states"] = hidden.shape

    # ── Check speculative head shapes ─────────────────────────────────
    spec_logits = output.get("spec_logits")
    if spec_logits is not None:
        for k, sl in enumerate(spec_logits):
            expected_s = (batch_size, seq_len, m.vocab_size)
            assert sl.shape == expected_s, (
                f"spec_logits[{k}] shape {sl.shape} != {expected_s}"
            )
        shapes["spec_logits_count"] = len(spec_logits)

    # ── Check KV cache shapes ─────────────────────────────────────────
    kv_caches = output.get("kv_caches")
    if kv_caches is not None:
        shapes["n_kv_caches"] = len(kv_caches)
        # (individual KV cache shapes vary by layer type — skip detailed check)

    logger.info("Shape verification passed: %s", shapes)
    return shapes
```

---

## 3. FLOPs Estimator — `apex/utils/flops.py`

### BUG-17: Missing SwiGLU Elementwise FLOPs

The original FLOPs counter:

```python
def estimate_ffn_flops(d_model, d_ffn, seq_len):
    # gate projection:  2 × d_model × d_ffn
    gate_flops = 2 * d_model * d_ffn
    # up projection:    2 × d_model × d_ffn
    up_flops = 2 * d_model * d_ffn
    # down projection:  2 × d_ffn × d_model
    down_flops = 2 * d_ffn * d_model
    return (gate_flops + up_flops + down_flops) * seq_len
    # BUG-17: MISSING the SiLU activation and elementwise multiply!
    # SiLU is approximately 4 FLOPs per element
    # Element-wise multiply: d_ffn FLOPs
```

**Fix:** Add the missing elementwise operations:

```python
def estimate_ffn_flops(d_model, d_ffn, seq_len):
    gate_flops = 2 * d_model * d_ffn    # W_gate linear
    up_flops   = 2 * d_model * d_ffn    # W_up linear
    down_flops = 2 * d_ffn * d_model    # W_down linear
    # BUG-17 FIX: include SiLU (~4 FLOPs/element) + elementwise multiply (1 FLOP/element)
    activation_flops = d_ffn * 5        # SiLU + multiply
    return (gate_flops + up_flops + down_flops + activation_flops) * seq_len
```

### Full Annotated Source

```python
"""
FLOPs estimation for APEX-1.

BUG-17 FIX: FFN FLOPs now include the SwiGLU activation + multiply.
"""

import math
from apex.config import APEXConfig


def estimate_model_flops(
    config: APEXConfig,
    seq_len: int = 2048,
    batch_size: int = 1,
) -> dict:
    """Estimate total FLOPs for one forward pass.
    
    Returns dict with detailed breakdown by component.
    """
    m = config.model
    a = config.attention
    moe = config.moe

    total_flops = 0
    flops_by_component = {}

    # ── 1. Embedding ──────────────────────────────────────────────────
    # Embedding is a lookup — no arithmetic FLOPs
    flops_by_component["embedding"] = 0

    # ── 2. Transformer Blocks ─────────────────────────────────────────
    for layer_idx in range(m.n_layers):
        # Determine layer type
        is_global = (layer_idx % a.global_layer_freq) == (a.global_layer_freq - 1)
        is_moe_layer = moe.enabled and (layer_idx % moe.moe_layer_freq != 0)

        layer_flops = 0

        # ── 2a. Attention ─────────────────────────────────────────────
        if is_global:
            # MLA: compression + decompression + attention
            kv_compress_flops = 2 * m.d_model * m.d_kv_compressed * seq_len
            kv_decomp_flops = 2 * m.d_kv_compressed * m.n_heads_kv * m.d_head * seq_len
            q_compress_flops = 2 * m.d_model * m.d_q_compressed * seq_len
            q_decomp_flops = 2 * m.d_q_compressed * m.n_heads_q * m.d_head * seq_len
            rope_proj_flops = 2 * m.d_model * (m.n_heads_q + m.n_heads_kv) * m.d_head_rope * seq_len
            # Attention scores: Q × K^T + weights × V
            attn_flops = 2 * m.n_heads_q * seq_len * seq_len * (m.d_head + m.d_head_rope)
            out_proj_flops = 2 * m.n_heads_q * m.d_head * m.d_model * seq_len
            layer_attn_flops = (kv_compress_flops + kv_decomp_flops + q_compress_flops +
                                q_decomp_flops + rope_proj_flops + attn_flops + out_proj_flops)
        else:
            # GQA+SW: standard linear projections + local attention
            qkv_flops = 2 * m.d_model * (m.n_heads_q + 2 * m.n_heads_kv) * m.d_head * seq_len
            window = min(a.local_window, seq_len)
            attn_flops = 2 * m.n_heads_q * seq_len * window * m.d_head
            out_proj_flops = 2 * m.n_heads_q * m.d_head * m.d_model * seq_len
            layer_attn_flops = qkv_flops + attn_flops + out_proj_flops

        layer_flops += layer_attn_flops

        # ── 2b. FFN ───────────────────────────────────────────────────
        if is_moe_layer:
            # MoE: only n_active + n_shared experts active per token
            n_active_total = moe.n_active + moe.n_shared
            effective_d_ffn = m.d_ffn
            # Router: d_model → n_experts (tiny)
            router_flops = 2 * m.d_model * moe.n_experts * seq_len
            # Expert computation for active experts
            gate_flops = 2 * m.d_model * effective_d_ffn * n_active_total * seq_len
            up_flops = 2 * m.d_model * effective_d_ffn * n_active_total * seq_len
            down_flops = 2 * effective_d_ffn * m.d_model * n_active_total * seq_len
            # BUG-17 FIX: include SwiGLU activation flops
            act_flops = effective_d_ffn * 5 * n_active_total * seq_len
            layer_ffn_flops = router_flops + gate_flops + up_flops + down_flops + act_flops
        else:
            # Dense FFN
            gate_flops = 2 * m.d_model * m.d_ffn * seq_len
            up_flops = 2 * m.d_model * m.d_ffn * seq_len
            down_flops = 2 * m.d_ffn * m.d_model * seq_len
            # BUG-17 FIX: include SwiGLU activation
            act_flops = m.d_ffn * 5 * seq_len
            layer_ffn_flops = gate_flops + up_flops + down_flops + act_flops

        layer_flops += layer_ffn_flops

        # RMSNorm: ~2 × d_model per position (negligible but included)
        norm_flops = 2 * 2 * m.d_model * seq_len   # 2 norms per block
        layer_flops += norm_flops

        total_flops += layer_flops

    # ── 3. LM Head ────────────────────────────────────────────────────
    lm_head_flops = 2 * m.d_model * m.vocab_size * seq_len
    total_flops += lm_head_flops
    flops_by_component["lm_head"] = lm_head_flops

    # Scale by batch size
    total_flops *= batch_size

    return {
        "total_flops": total_flops,
        "total_flops_per_token": total_flops / (seq_len * batch_size),
        "total_gflops": total_flops / 1e9,
        "components": flops_by_component,
    }
```

---

## 4. Param Counter — `apex/utils/param_counter.py`

```python
def count_parameters(model: torch.nn.Module) -> dict:
    """Count total, trainable, and active parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Count idle (non-active) MoE expert params
    idle_expert_params = 0
    for module in model.modules():
        if hasattr(module, "n_experts") and hasattr(module, "n_active"):
            n_idle = module.n_experts - module.n_active
            expert_size = sum(p.numel() for p in module.experts[0].parameters())
            idle_expert_params += n_idle * expert_size

    return {
        "total_params": total,
        "trainable_params": trainable,
        "active_params": total - idle_expert_params,
        "idle_moe_params": idle_expert_params,
        "total_params_M": total / 1e6,
        "active_params_M": (total - idle_expert_params) / 1e6,
    }
```

---

## 5. Quick Usage

```python
from apex.utils.shape_checker import verify_shapes
from apex.utils.flops import estimate_model_flops
from apex.utils.param_counter import count_parameters

# After building model:
shapes = verify_shapes(config, model=model)       # BUG-23 fix
flops = estimate_model_flops(config, seq_len=2048)
params = count_parameters(model)

print(f"Total params:    {params['total_params_M']:.1f}M")
print(f"Active params:   {params['active_params_M']:.1f}M")
print(f"FLOPs per token: {flops['total_flops_per_token']:.1f}")
```

---

**Next:** [31 — End-to-End Walkthrough →](31-end-to-end-walkthrough.md)
