"""
LoRA and PEFT utilities for APEX-1.

This module implements LoRA from scratch without depending on the external
``peft`` package. The goal is educational clarity:

    frozen Linear(x) + trainable low-rank update(x)

For a frozen weight W with shape [out_features, in_features], LoRA learns:

    ΔW = B @ A

where:
    A has shape [r, in_features]
    B has shape [out_features, r]
    r is small, for example 4, 8, or 16.

Forward pass:

    y = x W^T + (alpha / r) * x A^T B^T

Only A and B are trained when ``freeze_base_model=True``.

v2.6.0 adds production-style adapter inference helpers:
- count/check whether adapters are attached
- merge adapters into base weights
- unload LoRA wrappers after merge so the checkpoint becomes plain APEX-1
- save merged checkpoints that can be loaded with normal generation scripts
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` layer with trainable LoRA adapters.

    Args:
        base_layer: Existing linear projection to wrap.
        r: LoRA rank.
        alpha: LoRA scaling alpha.
        dropout: Dropout probability before the LoRA A projection.
        freeze_base: Whether to freeze the original linear parameters.
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        freeze_base: bool = True,
    ) -> None:
        super().__init__()

        if r <= 0:
            raise ValueError("LoRA rank r must be positive")

        self.base_layer = base_layer
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.merged = False

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)

        self.reset_lora_parameters()

        if freeze_base:
            for param in self.base_layer.parameters():
                param.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        """Expose base weight for compatibility with code that reads ``.weight``."""
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        """Expose base bias for compatibility with code that reads ``.bias``."""
        return self.base_layer.bias

    def reset_lora_parameters(self) -> None:
        """Initialize A randomly and B to zero so initial output equals base layer."""
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply base projection plus low-rank adapter update."""
        result = self.base_layer(x)
        if self.merged:
            return result
        update = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return result + update

    @torch.no_grad()
    def merge(self) -> None:
        """Merge LoRA weights into the frozen base weight for inference/export."""
        if self.merged:
            return
        delta = self.lora_B.weight @ self.lora_A.weight
        self.base_layer.weight += delta * self.scaling
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo a previous merge."""
        if not self.merged:
            return
        delta = self.lora_B.weight @ self.lora_A.weight
        self.base_layer.weight -= delta * self.scaling
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}, "
            f"merged={self.merged}"
        )


def apply_lora_adapters(model: nn.Module, peft_config: Any) -> nn.Module:
    """Inject LoRA adapters into matching ``nn.Linear`` modules.

    The function modifies ``model`` in place and returns it.

    Matching rule:
    - exact child module name match, for example ``W_Q``
    - full dotted module path suffix match, for example ``attn.W_Q``
    - path fragment match for names in ``target_modules``

    Args:
        model: APEX model or any PyTorch module.
        peft_config: ``PEFTConfig`` or compatible object.

    Returns:
        The same model with LoRA wrappers inserted.
    """
    if getattr(peft_config, "method", "lora") != "lora":
        raise ValueError("Only LoRA PEFT is implemented")

    if getattr(peft_config, "freeze_base_model", True):
        for param in model.parameters():
            param.requires_grad = False

    target_modules = list(getattr(peft_config, "target_modules", []))
    replaced: list[str] = []

    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                continue
            if not isinstance(child, nn.Linear):
                continue

            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not _matches_target(full_name, child_name, target_modules):
                continue

            wrapped = LoRALinear(
                child,
                r=int(getattr(peft_config, "r", 8)),
                alpha=int(getattr(peft_config, "alpha", 16)),
                dropout=float(getattr(peft_config, "dropout", 0.0)),
                freeze_base=bool(getattr(peft_config, "freeze_base_model", True)),
            )
            setattr(parent, child_name, wrapped)
            replaced.append(full_name)

    mark_only_lora_as_trainable(
        model,
        bias=getattr(peft_config, "bias", "none"),
        modules_to_save=getattr(peft_config, "modules_to_save", []),
    )

    if not replaced:
        logger.warning("LoRA enabled but no target Linear modules were matched")
    else:
        logger.info("Inserted LoRA adapters into %d modules", len(replaced))
        logger.debug("LoRA target modules: %s", replaced)

    return model


def mark_only_lora_as_trainable(
    model: nn.Module,
    bias: str = "none",
    modules_to_save: Iterable[str] | None = None,
) -> None:
    """Set ``requires_grad`` for LoRA fine-tuning.

    Args:
        model: Model containing LoRA modules.
        bias: ``none``, ``all``, or ``lora_only``.
        modules_to_save: Optional module-name fragments that remain trainable.
    """
    modules_to_save = list(modules_to_save or [])

    for name, param in model.named_parameters():
        trainable = "lora_A" in name or "lora_B" in name

        if bias == "all" and name.endswith(".bias"):
            trainable = True
        elif bias == "lora_only" and "base_layer.bias" in name:
            trainable = True

        if any(fragment in name for fragment in modules_to_save):
            trainable = True

        param.requires_grad = trainable


def iter_lora_modules(model: nn.Module):
    """Yield ``(name, module)`` pairs for all LoRA-wrapped linear modules."""
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def count_lora_modules(model: nn.Module) -> int:
    """Return the number of LoRA-wrapped linear modules in ``model``."""
    return sum(1 for _name, _module in iter_lora_modules(model))


def has_lora_adapters(model: nn.Module) -> bool:
    """Return True when at least one LoRA adapter is attached."""
    return count_lora_modules(model) > 0


def require_lora_adapters(model: nn.Module) -> None:
    """Raise a clear error when adapter utilities are called on a plain model."""
    if not has_lora_adapters(model):
        raise ValueError(
            "No LoRA adapters found in model. Build the model with config.peft.enabled=True "
            "or call apply_lora_adapters(model, config.peft) before loading adapters."
        )


def get_lora_state_dict(
    model: nn.Module,
    modules_to_save: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Return an adapter-only state dict.

    This keeps checkpoints small: only LoRA A/B matrices and optional
    ``modules_to_save`` parameters are saved.
    """
    modules_to_save = list(modules_to_save or [])
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            state[name] = tensor.detach().cpu()
        elif any(fragment in name for fragment in modules_to_save):
            state[name] = tensor.detach().cpu()
    return state


def peft_config_to_dict(peft_config: Any | None) -> dict[str, Any]:
    """Convert a PEFT config object/dataclass to a JSON-friendly dict."""
    if peft_config is None:
        return {}
    if dataclasses.is_dataclass(peft_config):
        return dataclasses.asdict(peft_config)
    if isinstance(peft_config, dict):
        return dict(peft_config)
    return dict(vars(peft_config))


def save_lora_adapters(
    model: nn.Module,
    path: str | Path,
    peft_config: Any | None = None,
) -> None:
    """Save only LoRA adapter weights and PEFT metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    modules_to_save = getattr(peft_config, "modules_to_save", []) if peft_config else []
    payload: dict[str, Any] = {
        "format": "apex_lora_adapter",
        "version": "2.6.0",
        "adapter_state_dict": get_lora_state_dict(model, modules_to_save=modules_to_save),
        "num_lora_modules": count_lora_modules(model),
    }

    if peft_config is not None:
        payload["peft_config"] = peft_config_to_dict(peft_config)

    torch.save(payload, path)
    logger.info("Saved LoRA adapters to %s", path)


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
    """Load a torch file while supporting older and newer PyTorch versions."""
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_lora_adapters(
    model: nn.Module,
    path: str | Path,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load LoRA adapter weights into a model that already has adapters.

    Args:
        model: Model with LoRA wrappers already inserted.
        path: Adapter checkpoint path produced by ``save_lora_adapters``.
        strict: Whether to enforce an exact adapter state match.
        map_location: Torch map location.

    Returns:
        Metadata with missing/unexpected keys and saved PEFT config.
    """
    require_lora_adapters(model)

    path = Path(path)
    payload = _torch_load(path, map_location=map_location)

    if "adapter_state_dict" not in payload:
        raise ValueError(f"{path} is not an APEX LoRA adapter checkpoint")

    incompatible = model.load_state_dict(payload["adapter_state_dict"], strict=strict)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    logger.info(
        "Loaded LoRA adapters from %s | missing=%d unexpected=%d",
        path,
        len(missing),
        len(unexpected),
    )
    return {
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "peft_config": payload.get("peft_config", {}),
        "format": payload.get("format", "unknown"),
        "version": payload.get("version", "unknown"),
        "num_lora_modules": payload.get("num_lora_modules"),
    }


@torch.no_grad()
def merge_lora_weights(model: nn.Module) -> None:
    """Merge all LoRA adapters into base weights but keep LoRA wrappers attached."""
    require_lora_adapters(model)
    merged = 0
    for _name, module in iter_lora_modules(model):
        module.merge()
        merged += 1
    logger.info("Merged %d LoRA modules into base weights", merged)


@torch.no_grad()
def unmerge_lora_weights(model: nn.Module) -> None:
    """Undo merge for all LoRA adapters."""
    require_lora_adapters(model)
    unmerged = 0
    for _name, module in iter_lora_modules(model):
        module.unmerge()
        unmerged += 1
    logger.info("Unmerged %d LoRA modules from base weights", unmerged)


@torch.no_grad()
def merge_and_unload_lora_weights(model: nn.Module) -> nn.Module:
    """Merge LoRA weights and replace every ``LoRALinear`` with plain ``nn.Linear``.

    This is the important v2.6.0 deployment path. After this function runs, the
    model no longer contains LoRA wrappers and its ``state_dict`` uses normal
    base model keys like ``blocks.0.attn.W_Q.weight`` instead of
    ``blocks.0.attn.W_Q.base_layer.weight``.

    Returns:
        The same model, modified in place.
    """
    require_lora_adapters(model)
    unloaded = 0

    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRALinear):
                child.merge()
                setattr(parent, child_name, child.base_layer)
                unloaded += 1

    logger.info("Merged and unloaded %d LoRA modules", unloaded)
    return model


def save_merged_lora_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a merged APEX checkpoint after ``merge_and_unload_lora_weights``.

    The saved file follows the same basic shape as normal APEX checkpoints so it
    can be loaded by ``apex.training.checkpoint.load_checkpoint``:

    ``{"model_state_dict": model.state_dict(), ...}``
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format": "apex_merged_lora_checkpoint",
        "version": "2.6.0",
        "step": 0,
        "epoch": 0,
        "loss": 0.0,
        "model_state_dict": model.state_dict(),
        "extra": {
            "merged_from_lora": True,
            "contains_lora_wrappers": has_lora_adapters(model),
        },
    }

    if config is not None:
        payload["config"] = dataclasses.asdict(config) if dataclasses.is_dataclass(config) else config
    if extra:
        payload["extra"].update(extra)

    torch.save(payload, path)
    logger.info("Saved merged LoRA checkpoint to %s", path)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    """Return total number of parameters."""
    return sum(p.numel() for p in model.parameters())


def peft_parameter_summary(model: nn.Module) -> dict[str, float | int]:
    """Return total/trainable/frozen parameter counts and trainable percentage."""
    total = count_total_parameters(model)
    trainable = count_trainable_parameters(model)
    frozen = total - trainable
    percent = 100.0 * trainable / max(total, 1)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": percent,
        "lora_modules": count_lora_modules(model),
    }


def print_peft_parameter_summary(model: nn.Module) -> None:
    """Pretty-print PEFT parameter counts."""
    s = peft_parameter_summary(model)
    print("\n" + "=" * 70)
    print("APEX-1 PEFT / LoRA Parameter Summary")
    print("=" * 70)
    print(f"Total parameters:     {s['total']:,}")
    print(f"Trainable parameters: {s['trainable']:,}")
    print(f"Frozen parameters:    {s['frozen']:,}")
    print(f"Trainable percent:    {s['trainable_percent']:.4f}%")
    print(f"LoRA modules:         {s['lora_modules']:,}")
    print("=" * 70 + "\n")


def _matches_target(full_name: str, child_name: str, target_modules: list[str]) -> bool:
    for target in target_modules:
        if child_name == target:
            return True
        if full_name == target:
            return True
        if full_name.endswith(f".{target}"):
            return True
        # This allows targeted fragments like "attn.W_Q" if desired.
        if target in full_name:
            return True
    return False
