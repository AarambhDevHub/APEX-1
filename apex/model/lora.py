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
        "adapter_state_dict": get_lora_state_dict(model, modules_to_save=modules_to_save),
    }

    if peft_config is not None:
        if dataclasses.is_dataclass(peft_config):
            payload["peft_config"] = dataclasses.asdict(peft_config)
        else:
            payload["peft_config"] = dict(vars(peft_config))

    torch.save(payload, path)
    logger.info("Saved LoRA adapters to %s", path)


def load_lora_adapters(
    model: nn.Module,
    path: str | Path,
    strict: bool = False,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Load LoRA adapter weights into a model that already has adapters."""
    path = Path(path)
    payload = torch.load(path, map_location=map_location)

    if "adapter_state_dict" not in payload:
        raise ValueError(f"{path} is not an APEX LoRA adapter checkpoint")

    missing, unexpected = model.load_state_dict(payload["adapter_state_dict"], strict=strict)
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
    }


@torch.no_grad()
def merge_lora_weights(model: nn.Module) -> None:
    """Merge all LoRA adapters into base weights."""
    for _name, module in iter_lora_modules(model):
        module.merge()


@torch.no_grad()
def unmerge_lora_weights(model: nn.Module) -> None:
    """Undo merge for all LoRA adapters."""
    for _name, module in iter_lora_modules(model):
        module.unmerge()


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
