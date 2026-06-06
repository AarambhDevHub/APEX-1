"""
APEX-1 LoRA / QLoRA / DoRA inference loading helpers.

v2.5.0 added LoRA training and adapter checkpoints.
v2.6.0 completes the workflow by making adapters usable for generation and
merge/export flows:

    base checkpoint + PEFT adapter -> generation model
    base checkpoint + PEFT adapter -> merged plain APEX checkpoint

The key implementation detail is load order. A normal base checkpoint uses
plain linear keys such as:

    blocks.0.attn.W_Q.weight

A LoRA-wrapped model uses keys such as:

    blocks.0.attn.W_Q.base_layer.weight
    blocks.0.attn.W_Q.lora_A.weight
    blocks.0.attn.W_Q.lora_B.weight

So for safe loading we:
1. temporarily disable PEFT
2. build the plain base model
3. load the base checkpoint if provided
4. re-enable PEFT and inject LoRA, QLoRA, DoRA, or QDoRA wrappers
5. load the adapter-only checkpoint
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model
from apex.model.lora import (
    apply_lora_adapters,
    count_lora_modules,
    load_lora_adapters,
    merge_and_unload_lora_weights,
    merge_lora_weights,
    save_merged_lora_checkpoint,
)
from apex.training.checkpoint import load_checkpoint

logger = logging.getLogger(__name__)


@dataclass
class LoRAInferenceLoadResult:
    """Return object for LoRA inference loading."""

    model: APEX1Model
    adapter_info: dict[str, Any]
    device: torch.device
    lora_modules: int
    merged_for_runtime: bool = False
    unloaded: bool = False


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Resolve a requested device or pick CUDA when available."""
    if device is not None:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_base_model_then_apply_lora(
    config: APEXConfig,
    checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
    strict_base: bool = True,
) -> APEX1Model:
    """Build APEX-1 safely for LoRA inference.

    Args:
        config: APEX config. ``config.peft`` controls adapter injection.
        checkpoint: Optional full base model checkpoint.
        device: Optional device string or torch device.
        strict_base: Whether to strictly load the base checkpoint.

    Returns:
        A model with PEFT wrappers attached when ``config.peft.enabled`` is true.
    """
    target_device = resolve_device(device)
    peft_enabled = bool(config.peft.enabled)

    # Build/load plain base model first so a normal checkpoint matches.
    config.peft.enabled = False
    model = APEX1Model(config)

    if checkpoint is not None:
        info = load_checkpoint(
            checkpoint,
            model,
            map_location="cpu",
            strict=strict_base,
        )
        logger.info("Loaded base checkpoint from step %s", info.get("step", "unknown"))

    # Re-enable LoRA wrappers after base weights are loaded.
    config.peft.enabled = peft_enabled
    if peft_enabled:
        apply_lora_adapters(model, config.peft)

    model.to(target_device)
    model.eval()
    return model


def load_lora_model_for_inference(
    config: APEXConfig,
    adapter: str | Path,
    checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
    strict_base: bool = True,
    strict_adapter: bool = False,
    merge_for_runtime: bool = False,
) -> LoRAInferenceLoadResult:
    """Build an inference-ready model and load a PEFT adapter.

    Args:
        config: APEX config. LoRA will be enabled automatically.
        adapter: Adapter checkpoint path from ``save_lora_adapters``.
        checkpoint: Optional base model checkpoint.
        device: Optional runtime device.
        strict_base: Strict loading for the base checkpoint.
        strict_adapter: Strict loading for the adapter checkpoint.
        merge_for_runtime: Merge adapter weights into the wrapped base layers
            before generation. Wrappers remain attached, but their forward path
            skips the adapter branch because weights are already merged.

    Returns:
        ``LoRAInferenceLoadResult`` with the model and adapter metadata.
    """
    target_device = resolve_device(device)
    config.peft.enabled = True

    model = build_base_model_then_apply_lora(
        config=config,
        checkpoint=checkpoint,
        device=target_device,
        strict_base=strict_base,
    )
    adapter_info = load_lora_adapters(
        model,
        adapter,
        strict=strict_adapter,
        map_location="cpu",
    )

    if merge_for_runtime:
        merge_lora_weights(model)

    model.eval()
    return LoRAInferenceLoadResult(
        model=model,
        adapter_info=adapter_info,
        device=target_device,
        lora_modules=count_lora_modules(model),
        merged_for_runtime=merge_for_runtime,
        unloaded=False,
    )


def load_merge_and_unload_lora_model(
    config: APEXConfig,
    adapter: str | Path,
    checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
    strict_base: bool = True,
    strict_adapter: bool = False,
) -> LoRAInferenceLoadResult:
    """Load adapter, merge it into base weights, and remove LoRA wrappers.

    The returned model is a plain APEX-1 model in memory. Its state dict can be
    loaded into another plain APEX-1 model with ``config.peft.enabled=False``.
    """
    result = load_lora_model_for_inference(
        config=config,
        adapter=adapter,
        checkpoint=checkpoint,
        device=device,
        strict_base=strict_base,
        strict_adapter=strict_adapter,
        merge_for_runtime=False,
    )
    merge_and_unload_lora_weights(result.model)
    config.peft.enabled = False
    result.lora_modules = count_lora_modules(result.model)
    result.merged_for_runtime = True
    result.unloaded = True
    return result


def export_merged_lora_checkpoint(
    config: APEXConfig,
    adapter: str | Path,
    output: str | Path,
    checkpoint: str | Path | None = None,
    device: str | torch.device | None = None,
    strict_base: bool = True,
    strict_adapter: bool = False,
) -> Path:
    """Create a plain merged checkpoint from a base checkpoint plus adapter."""
    result = load_merge_and_unload_lora_model(
        config=config,
        adapter=adapter,
        checkpoint=checkpoint,
        device=device,
        strict_base=strict_base,
        strict_adapter=strict_adapter,
    )
    output_path = Path(output)
    save_merged_lora_checkpoint(
        result.model,
        output_path,
        config=config,
        extra={
            "source_adapter": str(adapter),
            "source_checkpoint": str(checkpoint) if checkpoint else None,
            "adapter_info": result.adapter_info,
        },
    )
    return output_path
