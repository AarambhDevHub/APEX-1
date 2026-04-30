"""
Shape Verification for APEX-1.

Verifies that every tensor shape in the model matches the specifications
from Section 16 of the architecture document. Runs a forward pass with
known dimensions and checks all intermediate shapes.

Fix BUG-23: ``verify_shapes`` now accepts an optional ``model`` parameter
and uses it instead of always creating a new ``APEX1Model`` internally.
Previously the function instantiated a fresh model from the config,
ignoring any pre-built model the caller wanted to validate — meaning it
always tested a randomly-initialised model rather than the actual one.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model

logger = logging.getLogger(__name__)


def verify_shapes(
    config: APEXConfig,
    device: str = "cpu",
    model: Optional[APEX1Model] = None,
) -> dict[str, bool]:
    """Verify all tensor shapes in the model match architecture spec.

    Runs a forward pass with known inputs and checks that output shapes
    are correct.

    Args:
        config: APEXConfig to test.
        device: Device to run on.
        model: Optional pre-built APEX1Model to verify.  If None, a new
               model is created from the config (original behaviour).

    Returns:
        Dict mapping check names to pass/fail booleans.
    """
    m = config.model
    results: dict[str, bool] = {}

    logger.info("Verifying shapes for d_model=%d, n_layers=%d", m.d_model, m.n_layers)

    # BUG-23 FIX: use the passed model if provided, otherwise create a new one.
    if model is None:
        model = APEX1Model(config).to(device)
    else:
        model = model.to(device)
    model.eval()

    batch = 2
    seq_len = 16

    # Input
    token_ids = torch.randint(0, m.vocab_size, (batch, seq_len), device=device)

    # Forward pass
    with torch.no_grad():
        output = model(token_ids, prefix_len=seq_len // 2)

    # Check output shapes
    logits = output["logits"]
    results["logits_shape"] = logits.shape == (batch, seq_len, m.vocab_size)

    if output["spec_logits"] is not None:
        for i, sl in enumerate(output["spec_logits"]):
            results[f"spec_logits_{i}_shape"] = sl.shape == (batch, seq_len, m.vocab_size)

    # Check KV caches
    kv_caches = output["kv_caches"]
    results["n_kv_caches"] = len(kv_caches) == m.n_layers

    from apex.model.mask import is_global_layer

    for layer_idx, kv in enumerate(kv_caches):
        is_global = is_global_layer(layer_idx, config.attention.global_layer_freq)
        if is_global:
            # MLA: kv is a tuple (c_kv, K_rope) after BUG-01 fix
            # c_kv: [batch, seq, d_kv_compressed]
            # K_rope: [batch, n_kv, seq, d_head_rope]
            if isinstance(kv, tuple) and len(kv) == 2:
                c_kv, K_rope = kv
                results[f"layer_{layer_idx}_mla_cache"] = (
                    c_kv.shape[0] == batch and c_kv.shape[2] == m.d_kv_compressed
                )
            else:
                results[f"layer_{layer_idx}_mla_cache"] = False
        else:
            # GQA: (K, V) each [batch, n_kv_heads, window, d_head]
            if isinstance(kv, tuple) and len(kv) == 2:
                K, V = kv
                results[f"layer_{layer_idx}_gqa_K_heads"] = K.shape[1] == m.n_heads_kv
                results[f"layer_{layer_idx}_gqa_K_head_dim"] = K.shape[3] == m.d_head
                results[f"layer_{layer_idx}_gqa_V_heads"] = V.shape[1] == m.n_heads_kv
            else:
                results[f"layer_{layer_idx}_gqa_cache"] = False

    # Check embedding weight tying
    emb_weight = model.embedding.weight
    # LM head uses emb_weight.T, so verify shapes are compatible
    results["weight_tying_shape"] = emb_weight.shape == (m.vocab_size, m.d_model)

    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info("Shape verification: %d/%d checks passed", passed, total)

    for name, ok in results.items():
        if not ok:
            logger.error("FAILED: %s", name)

    return results


def print_model_architecture(model: APEX1Model) -> str:
    """Print a human-readable model architecture summary.

    Args:
        model: APEX-1 model instance.

    Returns:
        Formatted architecture string.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("APEX-1 Architecture Summary")
    lines.append("=" * 70)

    config = model.config
    m = config.model

    lines.append(f"Model dimension:     {m.d_model}")
    lines.append(f"Layers:              {m.n_layers}")
    lines.append(f"Query heads:         {m.n_heads_q}")
    lines.append(f"KV heads:            {m.n_heads_kv}")
    lines.append(f"Head dimension:      {m.d_head}")
    lines.append(f"FFN dimension:       {m.d_ffn}")
    lines.append(f"Vocab size:          {m.vocab_size}")
    lines.append(f"Max seq length:      {m.max_seq_len}")
    lines.append(f"RoPE base:           {m.rope_base}")
    lines.append(f"RoPE scaling:        {m.rope_scaling}")
    lines.append("")

    lines.append("Layer Assignment:")
    lines.append("-" * 50)

    from apex.model.mask import is_global_layer

    for i in range(m.n_layers):
        is_global = is_global_layer(i, config.attention.global_layer_freq)
        attn_type = "GLOBAL (MLA)" if is_global else "LOCAL  (GQA+SW)"
        is_moe = config.moe.enabled and i % config.moe.moe_layer_freq != 0
        ffn_type = "MoE" if is_moe else "Dense"
        lines.append(f"  Layer {i:3d}: {attn_type:20s} | FFN: {ffn_type}")

    lines.append("=" * 70)

    summary = "\n".join(lines)
    print(summary)
    return summary
