"""Model inspection utilities for APEX-1.

The inspector prints what learners usually want to know before training or
benchmarking:

- how many parameters the model has
- which layers are global MLA vs local GQA
- which layers are Dense FFN vs MoE
- whether vision modules are present
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LayerInfo:
    index: int
    attention: str
    ffn: str
    skip_gate: bool
    parameters: int


@dataclass(frozen=True)
class ModelInspection:
    model_type: str
    total_parameters: int
    trainable_parameters: int
    active_parameters: int | None
    vocab_size: int | None
    d_model: int | None
    n_layers: int
    global_layers: int
    local_layers: int
    moe_layers: int
    dense_layers: int
    skip_gate_layers: int
    vision_enabled: bool
    vision_parameters: int
    layers: list[LayerInfo]

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_type": self.model_type,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "active_parameters": self.active_parameters,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "global_layers": self.global_layers,
            "local_layers": self.local_layers,
            "moe_layers": self.moe_layers,
            "dense_layers": self.dense_layers,
            "skip_gate_layers": self.skip_gate_layers,
            "vision_enabled": self.vision_enabled,
            "vision_parameters": self.vision_parameters,
            "layers": [layer.__dict__ for layer in self.layers],
        }


def count_parameters(module: torch.nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    return sum(p.numel() for p in module.parameters())


def format_parameter_count(n: int | None) -> str:
    if n is None:
        return "n/a"
    if n >= 1_000_000_000:
        return f"{n / 1_000_000_000:.2f}B"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.2f}M"
    if n >= 1_000:
        return f"{n / 1_000:.2f}K"
    return str(n)


def _language_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "language_model", model)


def inspect_model(model: torch.nn.Module) -> ModelInspection:
    """Inspect an ``APEX1Model`` or ``APEX1VisionModel`` instance."""
    lm = _language_model(model)
    config = getattr(model, "config", getattr(lm, "config", None))
    blocks = list(getattr(lm, "blocks", []))

    layers: list[LayerInfo] = []
    for i, block in enumerate(blocks):
        attention = "Global MLA" if getattr(block, "is_global", False) else "Local GQA+SW"
        ffn = "MoE" if getattr(block, "is_moe", False) else "Dense"
        skip_gate = bool(getattr(block, "use_skip_gate", False))
        layers.append(
            LayerInfo(
                index=i,
                attention=attention,
                ffn=ffn,
                skip_gate=skip_gate,
                parameters=count_parameters(block),
            )
        )

    active_params = None
    if hasattr(lm, "active_parameters"):
        active_params = int(lm.active_parameters())

    vision_params = 0
    if hasattr(model, "vision_encoder"):
        vision_params += count_parameters(model.vision_encoder)
    if hasattr(model, "vision_projector"):
        vision_params += count_parameters(model.vision_projector)

    vision_enabled = bool(getattr(getattr(config, "vision", None), "enabled", False)) or vision_params > 0
    model_cfg = getattr(config, "model", None)

    return ModelInspection(
        model_type=model.__class__.__name__,
        total_parameters=count_parameters(model),
        trainable_parameters=count_parameters(model, trainable_only=True),
        active_parameters=active_params,
        vocab_size=getattr(model_cfg, "vocab_size", None),
        d_model=getattr(model_cfg, "d_model", None),
        n_layers=len(layers),
        global_layers=sum(1 for layer in layers if layer.attention.startswith("Global")),
        local_layers=sum(1 for layer in layers if layer.attention.startswith("Local")),
        moe_layers=sum(1 for layer in layers if layer.ffn == "MoE"),
        dense_layers=sum(1 for layer in layers if layer.ffn == "Dense"),
        skip_gate_layers=sum(1 for layer in layers if layer.skip_gate),
        vision_enabled=vision_enabled,
        vision_parameters=vision_params,
        layers=layers,
    )


def format_inspection_report(report: ModelInspection, show_layers: bool = True) -> str:
    lines = [
        f"# {report.model_type} Inspection",
        "",
        f"Total parameters:      {format_parameter_count(report.total_parameters)}",
        f"Trainable parameters:  {format_parameter_count(report.trainable_parameters)}",
        f"Active parameters:     {format_parameter_count(report.active_parameters)}",
        f"Vocabulary size:       {report.vocab_size}",
        f"Hidden size d_model:   {report.d_model}",
        f"Layers:                {report.n_layers}",
        f"Global MLA layers:     {report.global_layers}",
        f"Local GQA+SW layers:   {report.local_layers}",
        f"MoE layers:            {report.moe_layers}",
        f"Dense FFN layers:      {report.dense_layers}",
        f"Skip-gate layers:      {report.skip_gate_layers}",
        f"Vision enabled:        {report.vision_enabled}",
        f"Vision parameters:     {format_parameter_count(report.vision_parameters)}",
    ]
    if show_layers:
        lines.extend(["", "## Layer Map", "", "| Layer | Attention | FFN | Skip Gate | Params |", "|---:|---|---|---|---:|"])
        for layer in report.layers:
            lines.append(
                f"| {layer.index} | {layer.attention} | {layer.ffn} | "
                f"{str(layer.skip_gate)} | {format_parameter_count(layer.parameters)} |"
            )
    return "\n".join(lines)
