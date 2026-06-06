"""Tests for APEX-1 v2.7.0 QLoRA / 4-bit PEFT support."""

from __future__ import annotations

import torch
import torch.nn as nn

from apex.config import get_tiny_config, get_tiny_qlora_config
from apex.model.apex_model import APEX1Model
from apex.model.lora import (
    QLoRALinear,
    QuantizedLinear4bit,
    count_lora_modules,
    count_qlora_modules,
    dequantize_4bit_weight,
    iter_qlora_modules,
    load_lora_adapters,
    merge_and_unload_lora_weights,
    pack_4bit_indices,
    qlora_storage_summary,
    quantize_4bit_weight,
    save_lora_adapters,
    unpack_4bit_indices,
)


def test_pack_unpack_4bit_indices_roundtrip():
    indices = torch.arange(31) % 16
    packed = pack_4bit_indices(indices)
    unpacked = unpack_4bit_indices(packed, indices.numel())
    assert torch.equal(indices.long(), unpacked.long())
    assert packed.dtype == torch.uint8


def test_quantize_dequantize_4bit_weight_shape():
    weight = torch.randn(7, 5)
    payload = quantize_4bit_weight(weight, quant_type="nf4", double_quant=True)
    restored = dequantize_4bit_weight(
        payload["qweight"],
        payload["shape"],
        quant_type=payload["quant_type"],
        scale_q=payload["scale_q"],
        scale_scale=payload["scale_scale"],
        double_quant=True,
    )
    assert restored.shape == weight.shape
    assert torch.isfinite(restored).all()


def test_quantized_linear_forward_shape():
    base = nn.Linear(8, 4)
    quant = QuantizedLinear4bit(base, quant_type="nf4", double_quant=True)
    x = torch.randn(2, 3, 8)
    y = quant(x)
    assert y.shape == (2, 3, 4)
    assert quant.qweight.dtype == torch.uint8


def test_qlora_modules_are_inserted():
    config = get_tiny_qlora_config()
    model = APEX1Model(config)
    assert count_qlora_modules(model) > 0
    assert count_lora_modules(model) == count_qlora_modules(model)
    for _name, module in iter_qlora_modules(model):
        assert isinstance(module, QLoRALinear)
        assert isinstance(module.base_layer, QuantizedLinear4bit)


def test_qlora_freezes_quantized_base_and_trains_only_adapters():
    config = get_tiny_qlora_config()
    model = APEX1Model(config)

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all("lora_A" in name or "lora_B" in name for name in trainable_names)

    for name, param in model.named_parameters():
        if "base_layer" in name:
            assert not param.requires_grad, name


def test_qlora_forward_backward_trains_adapters():
    config = get_tiny_qlora_config()
    model = APEX1Model(config)
    token_ids = torch.randint(0, config.model.vocab_size, (2, 8))
    out = model(token_ids)
    assert out["logits"].shape == (2, 8, config.model.vocab_size)

    loss = out["logits"].mean()
    loss.backward()

    lora_grad_count = 0
    base_grad_count = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            if param.grad is not None:
                lora_grad_count += 1
        elif "base_layer" in name and param.grad is not None:
            base_grad_count += 1

    assert lora_grad_count > 0
    assert base_grad_count == 0


def test_qlora_adapter_save_and_load(tmp_path):
    config = get_tiny_qlora_config()
    model = APEX1Model(config)
    adapter_path = tmp_path / "qlora_adapter.pt"
    save_lora_adapters(model, adapter_path, config.peft)

    fresh = APEX1Model(config)
    info = load_lora_adapters(fresh, adapter_path, strict=False)
    assert info["method"] == "qlora"
    assert info["format"] == "apex_lora_adapter"
    assert info["num_qlora_modules"] == count_qlora_modules(model)


def test_qlora_merge_and_unload_loads_into_plain_model():
    qconfig = get_tiny_qlora_config()
    qmodel = APEX1Model(qconfig)
    merge_and_unload_lora_weights(qmodel)
    assert count_lora_modules(qmodel) == 0

    plain_config = get_tiny_config()
    plain = APEX1Model(plain_config)
    incompatible = plain.load_state_dict(qmodel.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_qlora_storage_summary_reports_compression():
    config = get_tiny_qlora_config()
    model = APEX1Model(config)
    summary = qlora_storage_summary(model)
    assert summary["qlora_modules"] > 0
    assert summary["float32_weight_bytes"] > summary["quantized_storage_bytes"]
    assert summary["compression_ratio"] > 1.0
