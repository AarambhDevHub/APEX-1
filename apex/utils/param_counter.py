"""
Parameter Counting Utilities for APEX-1.

Counts total and active parameters, breaking down by component type
(embedding, attention, FFN, MoE, skip gate, etc.).
"""

from __future__ import annotations

import logging

import torch.nn as nn

logger = logging.getLogger(__name__)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count total parameters in a model.

    Args:
        model: PyTorch model.
        trainable_only: If True, count only parameters with requires_grad.

    Returns:
        Total parameter count.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def count_parameters_by_module(model: nn.Module) -> dict[str, int]:
    """Count parameters broken down by top-level module.

    Args:
        model: PyTorch model.

    Returns:
        Dict mapping module name to parameter count.
    """
    counts: dict[str, int] = {}
    for name, module in model.named_children():
        n = sum(p.numel() for p in module.parameters())
        counts[name] = n
    return counts


def format_params(n: int) -> str:
    """Format parameter count to human-readable string.

    Args:
        n: Number of parameters.

    Returns:
        Formatted string (e.g., '100.5M', '7.2B').
    """
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    elif n >= 1e6:
        return f"{n / 1e6:.1f}M"
    elif n >= 1e3:
        return f"{n / 1e3:.1f}K"
    return str(n)


def print_parameter_summary(model: nn.Module) -> str:
    """Print a detailed parameter summary.

    Args:
        model: APEX-1 model instance.

    Returns:
        Formatted summary string.
    """
    lines: list[str] = []
    lines.append("=" * 70)
    lines.append("APEX-1 Parameter Summary")
    lines.append("=" * 70)

    total = count_parameters(model)
    trainable = count_parameters(model, trainable_only=True)

    lines.append(f"Total parameters:     {format_params(total)} ({total:,})")
    lines.append(f"Trainable parameters: {format_params(trainable)} ({trainable:,})")
    lines.append("")

    # Break down by component
    lines.append("Component breakdown:")
    lines.append("-" * 50)

    by_module = count_parameters_by_module(model)
    for name, count in sorted(by_module.items(), key=lambda x: -x[1]):
        pct = 100.0 * count / max(total, 1)
        lines.append(f"  {name:30s} {format_params(count):>10s} ({pct:5.1f}%)")

    lines.append("-" * 50)

    # Active parameters estimate
    if hasattr(model, "active_parameters"):
        active = model.active_parameters()
        lines.append(f"\nActive params per token: {format_params(active)} ({active:,})")
        lines.append(f"Active/Total ratio:      {100.0 * active / max(total, 1):.1f}%")

    # MoE breakdown
    if hasattr(model, "get_moe_layers"):
        moe_layers = model.get_moe_layers()
        if moe_layers:
            n_moe = len(moe_layers)
            _, first_moe = moe_layers[0]
            expert_params = sum(p.numel() for p in first_moe.routed_experts[0].parameters())
            lines.append(f"\nMoE layers:             {n_moe}")
            lines.append(f"Experts per MoE:        {first_moe.n_experts}")
            lines.append(f"Active experts/token:   {first_moe.n_active}")
            lines.append(f"Params per expert:      {format_params(expert_params)}")

    lines.append("=" * 70)

    summary = "\n".join(lines)
    print(summary)
    return summary
