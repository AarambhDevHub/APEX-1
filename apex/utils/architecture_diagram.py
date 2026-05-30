"""Text architecture diagram generator for APEX-1."""

from __future__ import annotations

from typing import Any


def _is_global_layer(layer_idx: int, global_layer_freq: int) -> bool:
    return (layer_idx % global_layer_freq) == (global_layer_freq - 1)


def _is_moe_layer(layer_idx: int, moe_enabled: bool, moe_layer_freq: int) -> bool:
    return bool(moe_enabled) and layer_idx % moe_layer_freq != 0


def build_architecture_diagram(config: Any, title: str = "APEX-1") -> str:
    """Return a readable ASCII architecture diagram from ``APEXConfig``."""
    m = config.model
    a = config.attention
    moe = config.moe
    vision = getattr(config, "vision", None)
    vision_enabled = bool(getattr(vision, "enabled", False))

    lines: list[str] = []
    lines.append(f"{title}")
    lines.append("=" * len(title))
    lines.append("")
    lines.append(f"d_model={m.d_model}, layers={m.n_layers}, vocab={m.vocab_size}, max_seq_len={m.max_seq_len}")
    lines.append(f"global_layer_freq={a.global_layer_freq}, local_window={a.local_window}")
    lines.append("")

    if vision_enabled:
        lines.append("Image Input")
        lines.append(f"  └─ Vision Encoder: {vision.encoder_type}, image={vision.image_size}px, patch={vision.patch_size}px")
        lines.append(f"      └─ Vision Projector: {vision.projector_type}, visual_tokens={vision.n_visual_tokens}")
        lines.append("          └─ Insert at <|img|> inside token embedding stream")
        lines.append("")

    lines.append("Text Input")
    lines.append("  └─ Token Embedding × √d")
    lines.append("      └─ Transformer Blocks")

    for i in range(m.n_layers):
        is_global = _is_global_layer(i, a.global_layer_freq)
        attn = "Global MLA" if is_global else "Local GQA+SW"
        ffn = "MoE FFN" if _is_moe_layer(i, moe.enabled, moe.moe_layer_freq) else "Dense FFN"
        skip = "SkipGate" if config.skip_gate.enabled else "NoSkipGate"
        connector = "          ├─" if i < m.n_layers - 1 else "          └─"
        lines.append(f"{connector} Layer {i:02d}: {attn} + {ffn} + {skip}")

    lines.append("              └─ Final RMSNorm")
    lines.append("                  └─ Tied LM Head → logits")
    if config.multi_token_head.enabled:
        lines.append("                  └─ Multi-token speculative heads")
    return "\n".join(lines)


def build_layer_table(config: Any) -> str:
    """Return a Markdown table describing every transformer layer."""
    lines = ["| Layer | Attention | FFN | Skip Gate |", "|---:|---|---|---|"]
    for i in range(config.model.n_layers):
        attn = "Global MLA" if _is_global_layer(i, config.attention.global_layer_freq) else "Local GQA+SW"
        ffn = "MoE" if _is_moe_layer(i, config.moe.enabled, config.moe.moe_layer_freq) else "Dense"
        skip = "Enabled" if config.skip_gate.enabled else "Disabled"
        lines.append(f"| {i} | {attn} | {ffn} | {skip} |")
    return "\n".join(lines)
