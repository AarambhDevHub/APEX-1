"""
LoRA, QLoRA, and PEFT utilities for APEX-1.

This module implements adapter fine-tuning from scratch without depending on
external PEFT libraries. It intentionally favors clarity over kernel-level
performance so learners can inspect every step.

LoRA idea:

    frozen Linear(x) + trainable low-rank update(x)

For a frozen weight W with shape [out_features, in_features], LoRA learns:

    ΔW = B @ A

Forward pass:

    y = x W^T + (alpha / r) * x A^T B^T

v2.7.0 adds an educational QLoRA-style path:

    frozen 4-bit quantized Linear(x) + trainable LoRA update(x)

The QLoRA implementation in this repository includes:
- NF4-style 4-bit codebook quantization
- optional double quantization of row scales
- trainable LoRA adapters on top of frozen quantized base weights
- merge-and-unload export into a plain nn.Linear checkpoint

This is an educational CPU/PyTorch implementation. It does not include custom
CUDA kernels or paged optimizers.
"""

from __future__ import annotations

import dataclasses
import logging
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# NF4-like codebook used by QLoRA. These values are commonly used in educational
# reproductions of NormalFloat4: more resolution near zero, less near extremes.
NF4_CODEBOOK_VALUES = [
    -1.0000000,
    -0.6961928,
    -0.5250731,
    -0.3949175,
    -0.2844414,
    -0.1847734,
    -0.0910500,
    0.0000000,
    0.0795803,
    0.1609302,
    0.2461123,
    0.3379152,
    0.4407098,
    0.5626170,
    0.7229568,
    1.0000000,
]


def get_4bit_codebook(
    quant_type: str = "nf4",
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a 16-value codebook for 4-bit quantization.

    Args:
        quant_type: ``nf4`` or ``fp4``. ``nf4`` is the recommended default.
        device: Optional device.
        dtype: Output dtype.

    Returns:
        Tensor with shape ``[16]``.
    """
    quant_type = quant_type.lower()
    if quant_type == "nf4":
        values = NF4_CODEBOOK_VALUES
    elif quant_type == "fp4":
        # Symmetric linear 4-bit codebook. Useful for comparison tests.
        values = torch.linspace(-1.0, 1.0, steps=16).tolist()
    else:
        raise ValueError("quant_type must be one of: nf4, fp4")
    return torch.tensor(values, device=device, dtype=dtype)


def pack_4bit_indices(indices: torch.Tensor) -> torch.Tensor:
    """Pack 4-bit indices into uint8 bytes.

    Two 4-bit values are stored in one byte:
    - low nibble: first value
    - high nibble: second value
    """
    flat = indices.reshape(-1).to(torch.uint8)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.uint8, device=flat.device)])
    low = flat[0::2] & 0x0F
    high = (flat[1::2] & 0x0F) << 4
    return low | high


def unpack_4bit_indices(packed: torch.Tensor, num_values: int) -> torch.Tensor:
    """Unpack uint8 bytes back into 4-bit indices."""
    packed = packed.reshape(-1).to(torch.uint8)
    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    out = torch.empty(packed.numel() * 2, dtype=torch.long, device=packed.device)
    out[0::2] = low.long()
    out[1::2] = high.long()
    return out[:num_values]


def quantize_4bit_weight(
    weight: torch.Tensor,
    quant_type: str = "nf4",
    double_quant: bool = True,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor | tuple[int, ...] | str | bool]:
    """Quantize a 2D linear weight tensor row-wise into packed 4-bit values.

    This is intentionally simple and readable:
    1. compute one absmax scale per output row
    2. normalize weights into [-1, 1]
    3. map each value to the nearest codebook entry
    4. pack two 4-bit indices per uint8
    5. optionally quantize the row scales to uint8 (double quantization)
    """
    if weight.dim() != 2:
        raise ValueError("quantize_4bit_weight expects a 2D [out, in] weight tensor")

    w = weight.detach().to(torch.float32)
    codebook = get_4bit_codebook(quant_type, device=w.device, dtype=torch.float32)
    scales = w.abs().amax(dim=1, keepdim=True).clamp_min(eps)
    normalized = (w / scales).clamp(-1.0, 1.0)

    # nearest-codebook assignment, shape [out, in]
    distances = (normalized.unsqueeze(-1) - codebook.view(1, 1, 16)).abs()
    indices = distances.argmin(dim=-1).to(torch.uint8)
    packed = pack_4bit_indices(indices)

    payload: dict[str, torch.Tensor | tuple[int, ...] | str | bool] = {
        "qweight": packed.cpu(),
        "shape": tuple(w.shape),
        "quant_type": quant_type,
        "double_quant": bool(double_quant),
    }

    if double_quant:
        scale_max = scales.max().clamp_min(eps)
        scale_q = torch.round(scales / scale_max * 255.0).clamp(0, 255).to(torch.uint8)
        payload["scale_q"] = scale_q.cpu()
        payload["scale_scale"] = scale_max.detach().cpu()
    else:
        payload["scales"] = scales.cpu()

    return payload


def dequantize_4bit_weight(
    qweight: torch.Tensor,
    shape: tuple[int, int] | torch.Size,
    quant_type: str = "nf4",
    scales: torch.Tensor | None = None,
    scale_q: torch.Tensor | None = None,
    scale_scale: torch.Tensor | None = None,
    double_quant: bool = True,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Dequantize packed 4-bit weights back to a floating point tensor."""
    out_features, in_features = int(shape[0]), int(shape[1])
    num_values = out_features * in_features
    device = qweight.device
    codebook = get_4bit_codebook(quant_type, device=device, dtype=torch.float32)
    indices = unpack_4bit_indices(qweight, num_values)
    normalized = codebook[indices].reshape(out_features, in_features)

    if double_quant:
        if scale_q is None or scale_scale is None:
            raise ValueError("double_quant=True requires scale_q and scale_scale")
        row_scales = scale_q.to(torch.float32) / 255.0 * scale_scale.to(torch.float32)
    else:
        if scales is None:
            raise ValueError("double_quant=False requires scales")
        row_scales = scales.to(torch.float32)

    return (normalized * row_scales).to(dtype)


class QuantizedLinear4bit(nn.Module):
    """Frozen 4-bit quantized replacement for ``nn.Linear``.

    The quantized weight is stored as buffers, not trainable parameters. During
    forward pass it is dequantized to the input dtype and used with
    ``torch.nn.functional.linear``. This is slower than CUDA QLoRA kernels, but
    much easier to understand and works on CPU for small educational models.
    """

    def __init__(
        self,
        linear: nn.Linear,
        quant_type: str = "nf4",
        double_quant: bool = True,
        compute_dtype: str | torch.dtype = "float32",
    ) -> None:
        super().__init__()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.quant_type = quant_type
        self.double_quant = bool(double_quant)
        self.compute_dtype = _resolve_compute_dtype(compute_dtype)

        payload = quantize_4bit_weight(
            linear.weight.detach(),
            quant_type=quant_type,
            double_quant=double_quant,
        )
        self.weight_shape = tuple(payload["shape"])  # type: ignore[arg-type]
        self.register_buffer("qweight", payload["qweight"].clone().detach())  # type: ignore[index]

        if self.double_quant:
            self.register_buffer("scale_q", payload["scale_q"].clone().detach())  # type: ignore[index]
            self.register_buffer("scale_scale", payload["scale_scale"].clone().detach())  # type: ignore[index]
            self.register_buffer("scales", None)
        else:
            self.register_buffer("scales", payload["scales"].clone().detach())  # type: ignore[index]
            self.register_buffer("scale_q", None)
            self.register_buffer("scale_scale", None)

        if linear.bias is not None:
            self.bias = nn.Parameter(linear.bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Return dequantized weight for compatibility with inspection code."""
        return self.dequantize_weight(dtype=torch.float32)

    def dequantize_weight(self, dtype: torch.dtype | None = None) -> torch.Tensor:
        """Return the dequantized weight matrix."""
        return dequantize_4bit_weight(
            qweight=self.qweight,
            shape=self.weight_shape,
            quant_type=self.quant_type,
            scales=self.scales,
            scale_q=self.scale_q,
            scale_scale=self.scale_scale,
            double_quant=self.double_quant,
            dtype=dtype or self.compute_dtype,
        )

    def to_float_linear(self) -> nn.Linear:
        """Convert the quantized layer into a plain frozen ``nn.Linear``."""
        linear = nn.Linear(
            self.in_features,
            self.out_features,
            bias=self.bias is not None,
            device=self.qweight.device,
        )
        linear.weight.data.copy_(self.dequantize_weight(dtype=linear.weight.dtype))
        if self.bias is not None:
            linear.bias.data.copy_(self.bias.detach().to(linear.bias.dtype))
        for param in linear.parameters():
            param.requires_grad = False
        return linear

    def storage_summary(self) -> dict[str, int | float]:
        """Return a small memory summary for the quantized base weight."""
        float32_bytes = self.out_features * self.in_features * 4
        quant_bytes = int(self.qweight.numel())
        if self.double_quant:
            quant_bytes += int(self.scale_q.numel()) + int(self.scale_scale.numel() * 4)
        else:
            quant_bytes += int(self.scales.numel() * 4)
        if self.bias is not None:
            quant_bytes += int(self.bias.numel() * self.bias.element_size())
        return {
            "float32_weight_bytes": float32_bytes,
            "quantized_storage_bytes": quant_bytes,
            "compression_ratio": float32_bytes / max(quant_bytes, 1),
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.dequantize_weight(dtype=x.dtype).to(x.device)
        bias = self.bias.to(x.dtype).to(x.device) if self.bias is not None else None
        return F.linear(x, weight, bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"quant_type={self.quant_type}, double_quant={self.double_quant}"
        )


class LoRALinear(nn.Module):
    """Wrap an ``nn.Linear`` layer with trainable LoRA adapters."""

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
        return self.base_layer.weight

    @property
    def bias(self) -> torch.Tensor | None:
        return self.base_layer.bias

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def lora_delta_weight(self) -> torch.Tensor:
        """Return the dense LoRA delta weight with shape [out, in]."""
        return self.lora_B.weight @ self.lora_A.weight * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.merged:
            return result
        update = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return result + update

    @torch.no_grad()
    def merge(self) -> None:
        if self.merged:
            return
        self.base_layer.weight += self.lora_delta_weight().to(self.base_layer.weight.dtype)
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        if not self.merged:
            return
        self.base_layer.weight -= self.lora_delta_weight().to(self.base_layer.weight.dtype)
        self.merged = False

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, scaling={self.scaling:.3f}, "
            f"merged={self.merged}"
        )


class QLoRALinear(nn.Module):
    """QLoRA-style layer: frozen 4-bit base projection + trainable LoRA update."""

    def __init__(
        self,
        base_layer: nn.Linear,
        r: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
        freeze_base: bool = True,
        quant_type: str = "nf4",
        double_quant: bool = True,
        compute_dtype: str | torch.dtype = "float32",
    ) -> None:
        super().__init__()

        if r <= 0:
            raise ValueError("QLoRA rank r must be positive")

        self.base_layer: nn.Module = QuantizedLinear4bit(
            base_layer,
            quant_type=quant_type,
            double_quant=double_quant,
            compute_dtype=compute_dtype,
        )
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.quant_type = quant_type
        self.double_quant = bool(double_quant)
        self.compute_dtype = _resolve_compute_dtype(compute_dtype)
        self.merged = False
        self._quantized_base_before_merge: QuantizedLinear4bit | None = None

        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        self.reset_lora_parameters()

        if freeze_base:
            for param in self.base_layer.parameters():
                param.requires_grad = False

    @property
    def weight(self) -> torch.Tensor:
        if isinstance(self.base_layer, QuantizedLinear4bit):
            return self.base_layer.dequantize_weight(dtype=torch.float32)
        return self.base_layer.weight  # type: ignore[return-value]

    @property
    def bias(self) -> torch.Tensor | None:
        return getattr(self.base_layer, "bias", None)

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=5**0.5)
        nn.init.zeros_(self.lora_B.weight)

    def lora_delta_weight(self) -> torch.Tensor:
        return self.lora_B.weight @ self.lora_A.weight * self.scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        if self.merged:
            return result
        update = self.lora_B(self.lora_A(self.lora_dropout(x))) * self.scaling
        return result + update

    @torch.no_grad()
    def merge(self) -> None:
        """Merge QLoRA into a plain floating ``nn.Linear`` inside this wrapper.

        Unlike classic LoRA, the base weight is quantized. For export we first
        dequantize the base weight, add the LoRA delta, and replace the internal
        base layer with a frozen plain ``nn.Linear``.
        """
        if self.merged:
            return
        if isinstance(self.base_layer, QuantizedLinear4bit):
            self._quantized_base_before_merge = self.base_layer
            linear = self.base_layer.to_float_linear()
            linear.weight += self.lora_delta_weight().to(linear.weight.dtype)
            for param in linear.parameters():
                param.requires_grad = False
            self.base_layer = linear
        else:
            self.base_layer.weight += self.lora_delta_weight().to(self.base_layer.weight.dtype)  # type: ignore[attr-defined]
        self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Undo a runtime merge when the original quantized base is still stored."""
        if not self.merged:
            return
        if self._quantized_base_before_merge is not None:
            self.base_layer = self._quantized_base_before_merge
            self._quantized_base_before_merge = None
        else:
            self.base_layer.weight -= self.lora_delta_weight().to(self.base_layer.weight.dtype)  # type: ignore[attr-defined]
        self.merged = False

    def storage_summary(self) -> dict[str, int | float]:
        if isinstance(self.base_layer, QuantizedLinear4bit):
            return self.base_layer.storage_summary()
        float32_bytes = self.out_features * self.in_features * 4
        return {
            "float32_weight_bytes": float32_bytes,
            "quantized_storage_bytes": float32_bytes,
            "compression_ratio": 1.0,
        }

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"r={self.r}, alpha={self.alpha}, quant_type={self.quant_type}, "
            f"double_quant={self.double_quant}, merged={self.merged}"
        )


LoRA_MODULE_TYPES = (LoRALinear, QLoRALinear)


def apply_lora_adapters(model: nn.Module, peft_config: Any) -> nn.Module:
    """Inject LoRA or QLoRA adapters into matching ``nn.Linear`` modules."""
    method = str(getattr(peft_config, "method", "lora")).lower()
    if method not in {"lora", "qlora"}:
        raise ValueError("Only peft.method='lora' or peft.method='qlora' is implemented")

    if getattr(peft_config, "freeze_base_model", True):
        for param in model.parameters():
            param.requires_grad = False

    target_modules = list(getattr(peft_config, "target_modules", []))
    replaced: list[str] = []

    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRA_MODULE_TYPES):
                continue
            if not isinstance(child, nn.Linear):
                continue

            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            if not _matches_target(full_name, child_name, target_modules):
                continue

            if method == "qlora":
                wrapped: nn.Module = QLoRALinear(
                    child,
                    r=int(getattr(peft_config, "r", 8)),
                    alpha=int(getattr(peft_config, "alpha", 16)),
                    dropout=float(getattr(peft_config, "dropout", 0.0)),
                    freeze_base=bool(getattr(peft_config, "freeze_base_model", True)),
                    quant_type=str(getattr(peft_config, "quant_type", "nf4")),
                    double_quant=bool(getattr(peft_config, "double_quant", True)),
                    compute_dtype=getattr(peft_config, "compute_dtype", "float32"),
                )
            else:
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

    label = "QLoRA" if method == "qlora" else "LoRA"
    if not replaced:
        logger.warning("%s enabled but no target Linear modules were matched", label)
    else:
        logger.info("Inserted %s adapters into %d modules", label, len(replaced))
        logger.debug("%s target modules: %s", label, replaced)

    return model


def mark_only_lora_as_trainable(
    model: nn.Module,
    bias: str = "none",
    modules_to_save: Iterable[str] | None = None,
) -> None:
    """Set ``requires_grad`` for LoRA/QLoRA fine-tuning."""
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
    """Yield ``(name, module)`` pairs for all LoRA/QLoRA wrapped modules."""
    for name, module in model.named_modules():
        if isinstance(module, LoRA_MODULE_TYPES):
            yield name, module


def iter_qlora_modules(model: nn.Module):
    """Yield ``(name, module)`` pairs for all QLoRA wrapped modules."""
    for name, module in model.named_modules():
        if isinstance(module, QLoRALinear):
            yield name, module


def count_lora_modules(model: nn.Module) -> int:
    return sum(1 for _name, _module in iter_lora_modules(model))


def count_qlora_modules(model: nn.Module) -> int:
    return sum(1 for _name, _module in iter_qlora_modules(model))


def has_lora_adapters(model: nn.Module) -> bool:
    return count_lora_modules(model) > 0


def require_lora_adapters(model: nn.Module) -> None:
    if not has_lora_adapters(model):
        raise ValueError(
            "No LoRA/QLoRA adapters found in model. Build the model with "
            "config.peft.enabled=True or call apply_lora_adapters(model, config.peft)."
        )


def get_lora_state_dict(
    model: nn.Module,
    modules_to_save: Iterable[str] | None = None,
) -> dict[str, torch.Tensor]:
    """Return an adapter-only state dict."""
    modules_to_save = list(modules_to_save or [])
    state: dict[str, torch.Tensor] = {}
    for name, tensor in model.state_dict().items():
        if "lora_A" in name or "lora_B" in name:
            state[name] = tensor.detach().cpu()
        elif any(fragment in name for fragment in modules_to_save):
            state[name] = tensor.detach().cpu()
    return state


def peft_config_to_dict(peft_config: Any | None) -> dict[str, Any]:
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
    """Save only adapter weights and PEFT metadata."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    modules_to_save = getattr(peft_config, "modules_to_save", []) if peft_config else []
    method = str(getattr(peft_config, "method", "lora")).lower() if peft_config else "lora"
    payload: dict[str, Any] = {
        "format": "apex_lora_adapter",
        "version": "2.7.0",
        "method": method,
        "adapter_state_dict": get_lora_state_dict(model, modules_to_save=modules_to_save),
        "num_lora_modules": count_lora_modules(model),
        "num_qlora_modules": count_qlora_modules(model),
    }

    if peft_config is not None:
        payload["peft_config"] = peft_config_to_dict(peft_config)

    torch.save(payload, path)
    logger.info("Saved %s adapters to %s", method.upper(), path)


def _torch_load(path: Path, map_location: str | torch.device = "cpu") -> Any:
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
    """Load adapter weights into a model that already has LoRA/QLoRA wrappers."""
    require_lora_adapters(model)

    path = Path(path)
    payload = _torch_load(path, map_location=map_location)

    if "adapter_state_dict" not in payload:
        raise ValueError(f"{path} is not an APEX LoRA/QLoRA adapter checkpoint")

    incompatible = model.load_state_dict(payload["adapter_state_dict"], strict=strict)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)

    logger.info(
        "Loaded adapters from %s | missing=%d unexpected=%d",
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
        "method": payload.get("method", payload.get("peft_config", {}).get("method", "unknown")),
        "num_lora_modules": payload.get("num_lora_modules"),
        "num_qlora_modules": payload.get("num_qlora_modules"),
    }


@torch.no_grad()
def merge_lora_weights(model: nn.Module) -> None:
    """Merge all LoRA/QLoRA adapters into their base weights but keep wrappers."""
    require_lora_adapters(model)
    merged = 0
    for _name, module in iter_lora_modules(model):
        module.merge()
        merged += 1
    logger.info("Merged %d LoRA/QLoRA modules into base weights", merged)


@torch.no_grad()
def unmerge_lora_weights(model: nn.Module) -> None:
    """Undo runtime merge for all adapters when possible."""
    require_lora_adapters(model)
    unmerged = 0
    for _name, module in iter_lora_modules(model):
        module.unmerge()
        unmerged += 1
    logger.info("Unmerged %d LoRA/QLoRA modules from base weights", unmerged)


@torch.no_grad()
def merge_and_unload_lora_weights(model: nn.Module) -> nn.Module:
    """Merge adapters and replace wrappers with plain ``nn.Linear`` modules.

    Works for both classic LoRA and QLoRA. For QLoRA, the quantized base weight
    is dequantized, the adapter delta is added, and a normal ``nn.Linear`` is
    placed back into the model so the resulting checkpoint can be loaded without
    PEFT enabled.
    """
    require_lora_adapters(model)
    unloaded = 0

    for parent in list(model.modules()):
        for child_name, child in list(parent.named_children()):
            if isinstance(child, LoRA_MODULE_TYPES):
                child.merge()
                setattr(parent, child_name, child.base_layer)
                unloaded += 1

    logger.info("Merged and unloaded %d LoRA/QLoRA modules", unloaded)
    return model


def save_merged_lora_checkpoint(
    model: nn.Module,
    path: str | Path,
    config: Any | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Save a merged APEX checkpoint after adapter merge/unload."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "format": "apex_merged_lora_checkpoint",
        "version": "2.7.0",
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
    logger.info("Saved merged adapter checkpoint to %s", path)


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def qlora_storage_summary(model: nn.Module) -> dict[str, int | float]:
    """Return estimated quantized-base storage across all QLoRA modules."""
    float32_bytes = 0
    quantized_bytes = 0
    modules = 0
    for _name, module in iter_qlora_modules(model):
        summary = module.storage_summary()
        float32_bytes += int(summary["float32_weight_bytes"])
        quantized_bytes += int(summary["quantized_storage_bytes"])
        modules += 1
    return {
        "qlora_modules": modules,
        "float32_weight_bytes": float32_bytes,
        "quantized_storage_bytes": quantized_bytes,
        "compression_ratio": float32_bytes / max(quantized_bytes, 1),
    }


def peft_parameter_summary(model: nn.Module) -> dict[str, float | int]:
    total = count_total_parameters(model)
    trainable = count_trainable_parameters(model)
    frozen = total - trainable
    percent = 100.0 * trainable / max(total, 1)
    qsum = qlora_storage_summary(model)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "trainable_percent": percent,
        "lora_modules": count_lora_modules(model),
        "qlora_modules": qsum["qlora_modules"],
        "qlora_float32_weight_bytes": qsum["float32_weight_bytes"],
        "qlora_quantized_storage_bytes": qsum["quantized_storage_bytes"],
        "qlora_compression_ratio": qsum["compression_ratio"],
    }


def print_peft_parameter_summary(model: nn.Module) -> None:
    s = peft_parameter_summary(model)
    print("\n" + "=" * 70)
    print("APEX-1 PEFT / LoRA / QLoRA Parameter Summary")
    print("=" * 70)
    print(f"Total parameters:     {s['total']:,}")
    print(f"Trainable parameters: {s['trainable']:,}")
    print(f"Frozen parameters:    {s['frozen']:,}")
    print(f"Trainable percent:    {s['trainable_percent']:.4f}%")
    print(f"LoRA modules:         {s['lora_modules']:,}")
    print(f"QLoRA modules:        {s['qlora_modules']:,}")
    if s["qlora_modules"]:
        print(f"QLoRA base fp32 bytes: {s['qlora_float32_weight_bytes']:,}")
        print(f"QLoRA quant bytes:     {s['qlora_quantized_storage_bytes']:,}")
        print(f"QLoRA compression:     {s['qlora_compression_ratio']:.2f}x")
    print("=" * 70 + "\n")


def _resolve_compute_dtype(dtype: str | torch.dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower()
    if name in {"float32", "fp32", "torch.float32"}:
        return torch.float32
    if name in {"float16", "fp16", "torch.float16"}:
        return torch.float16
    if name in {"bfloat16", "bf16", "torch.bfloat16"}:
        return torch.bfloat16
    raise ValueError("compute_dtype must be float32, float16, or bfloat16")


def _matches_target(full_name: str, child_name: str, target_modules: list[str]) -> bool:
    for target in target_modules:
        if child_name == target:
            return True
        if full_name == target:
            return True
        if full_name.endswith(f".{target}"):
            return True
        if target in full_name:
            return True
    return False
