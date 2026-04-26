"""
FLOPs Estimation for APEX-1.

Estimates the floating-point operations per forward pass for each
model size configuration.
"""

from __future__ import annotations

from apex.config import APEXConfig
from apex.model.mask import is_global_layer


def estimate_flops(config: APEXConfig, seq_len: int | None = None) -> dict[str, float]:
    """Estimate FLOPs for one forward pass.

    Computes approximate FLOPs for each component: embedding, attention,
    FFN, MoE routing, etc.

    Args:
        config: APEXConfig instance.
        seq_len: Sequence length (defaults to training seq_len).

    Returns:
        Dict mapping component names to FLOPs counts.
    """
    m = config.model
    S = seq_len or config.training.seq_len

    flops: dict[str, float] = {}

    # Embedding lookup: negligible FLOPs (just memory)
    flops["embedding"] = 0.0

    # Per-layer FLOPs
    n_global = 0
    n_local = 0
    attn_flops_total = 0.0
    ffn_flops_total = 0.0

    for layer_idx in range(m.n_layers):
        is_global = is_global_layer(layer_idx, config.attention.global_layer_freq)

        if is_global:
            n_global += 1
            # MLA attention FLOPs
            # Q compression: S * d_model * d_q_compressed
            q_comp = 2 * S * m.d_model * m.d_q_compressed
            q_decomp = 2 * S * m.d_q_compressed * (m.n_heads_q * m.d_head)
            # KV compression: S * d_model * d_kv_compressed
            kv_comp = 2 * S * m.d_model * m.d_kv_compressed
            kv_decomp = 2 * S * m.d_kv_compressed * (m.n_heads_kv * m.d_head) * 2
            # RoPE projections
            rope = 2 * S * m.d_model * (m.n_heads_q + m.n_heads_kv) * m.d_head_rope
            # Attention scores: S * S * d_head * n_heads
            d_total = m.d_head + m.d_head_rope
            attn_score = 2 * m.n_heads_q * S * S * d_total
            # Attention value: S * S * d_head * n_heads
            attn_val = 2 * m.n_heads_q * S * S * m.d_head
            # Output projection
            out_proj = 2 * S * (m.n_heads_q * m.d_head) * m.d_model

            layer_attn = (
                q_comp + q_decomp + kv_comp + kv_decomp + rope + attn_score + attn_val + out_proj
            )
        else:
            n_local += 1
            # GQA attention FLOPs
            W = min(S, config.attention.local_window)
            # Q/K/V projections
            qkv = 2 * S * m.d_model * (m.n_heads_q + 2 * m.n_heads_kv) * m.d_head
            # Attention scores: S * W * d_head * n_heads
            attn_score = 2 * m.n_heads_q * S * W * m.d_head
            attn_val = 2 * m.n_heads_q * S * W * m.d_head
            out_proj = 2 * S * (m.n_heads_q * m.d_head) * m.d_model

            layer_attn = qkv + attn_score + attn_val + out_proj

        attn_flops_total += layer_attn

        # FFN FLOPs
        is_moe = config.moe.enabled and layer_idx % config.moe.moe_layer_freq != 0

        if is_moe:
            # Shared experts
            shared_ffn = config.moe.n_shared * 2 * S * m.d_model * m.d_ffn * 3
            # Routed experts (only n_active per token)
            routed_ffn = config.moe.n_active * 2 * S * m.d_model * m.d_ffn * 3
            # Router
            router = 2 * S * m.d_model * config.moe.n_experts
            layer_ffn = shared_ffn + routed_ffn + router
        else:
            # Dense SwiGLU: 3 matrices
            layer_ffn = 2 * S * m.d_model * m.d_ffn * 3

        ffn_flops_total += layer_ffn

    flops["attention_total"] = attn_flops_total
    flops["ffn_total"] = ffn_flops_total
    flops["global_layers"] = n_global
    flops["local_layers"] = n_local

    # RMSNorm: 2 * n_layers * 2 * S * d_model (pre-attn + pre-ffn)
    flops["rmsnorm"] = 2.0 * m.n_layers * 2 * S * m.d_model

    # LM head: S * d_model * vocab_size
    flops["lm_head"] = 2.0 * S * m.d_model * m.vocab_size

    # Total
    flops["total"] = sum(v for k, v in flops.items() if k not in ("global_layers", "local_layers"))

    return flops


def format_flops(flops: float) -> str:
    """Format FLOPs to human-readable string.

    Args:
        flops: Number of FLOPs.

    Returns:
        Formatted string (e.g., '1.2 TFLOPs').
    """
    if flops >= 1e15:
        return f"{flops / 1e15:.1f} PFLOPs"
    elif flops >= 1e12:
        return f"{flops / 1e12:.1f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops / 1e9:.1f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops / 1e6:.1f} MFLOPs"
    return f"{flops:.0f} FLOPs"


def print_flops_summary(config: APEXConfig, seq_len: int | None = None) -> str:
    """Print FLOPs breakdown for a config.

    Args:
        config: APEXConfig instance.
        seq_len: Override sequence length.

    Returns:
        Formatted summary string.
    """
    flops = estimate_flops(config, seq_len)

    lines = [
        "=" * 50,
        "APEX-1 FLOPs Estimate",
        "=" * 50,
        f"Sequence length:  {seq_len or config.training.seq_len}",
        f"Global layers:    {int(flops['global_layers'])}",
        f"Local layers:     {int(flops['local_layers'])}",
        "",
        f"Attention:        {format_flops(flops['attention_total'])}",
        f"FFN:              {format_flops(flops['ffn_total'])}",
        f"RMSNorm:          {format_flops(flops['rmsnorm'])}",
        f"LM Head:          {format_flops(flops['lm_head'])}",
        "-" * 50,
        f"Total:            {format_flops(flops['total'])}",
        "=" * 50,
    ]

    summary = "\n".join(lines)
    print(summary)
    return summary
