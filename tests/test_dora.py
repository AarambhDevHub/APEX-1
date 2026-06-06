"""Tests for APEX-1 v2.8.0 DoRA / Weight-Decomposed LoRA support."""

from __future__ import annotations

import torch
import torch.nn as nn

from apex.config import get_tiny_config, get_tiny_dora_config, get_tiny_qdora_config
from apex.model.apex_model import APEX1Model
from apex.model.lora import (
    DoRALinear,
    QDoRALinear,
    QuantizedLinear4bit,
    count_dora_modules,
    count_lora_modules,
    iter_dora_modules,
    load_lora_adapters,
    merge_and_unload_lora_weights,
    save_lora_adapters,
)


def test_dora_linear_initial_output_matches_base():
    torch.manual_seed(7)
    base = nn.Linear(8, 4)
    reference = nn.Linear(8, 4)
    reference.load_state_dict(base.state_dict())

    dora = DoRALinear(base, r=2, alpha=4, dropout=0.0)
    x = torch.randn(3, 5, 8)

    torch.testing.assert_close(dora(x), reference(x), atol=1e-5, rtol=1e-5)


def test_dora_modules_are_inserted():
    config = get_tiny_dora_config()
    model = APEX1Model(config)
    assert count_dora_modules(model) > 0
    assert count_lora_modules(model) == count_dora_modules(model)
    for _name, module in iter_dora_modules(model):
        assert isinstance(module, DoRALinear)


def test_dora_trains_magnitude_and_adapters_only():
    config = get_tiny_dora_config()
    model = APEX1Model(config)

    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    assert trainable_names
    assert any("dora_magnitude" in name for name in trainable_names)
    assert any("lora_A" in name or "lora_B" in name for name in trainable_names)
    assert all(
        "lora_A" in name or "lora_B" in name or "dora_magnitude" in name
        for name in trainable_names
    )

    for name, param in model.named_parameters():
        if "base_layer" in name:
            assert not param.requires_grad, name


def test_dora_forward_backward_trains_adapters():
    config = get_tiny_dora_config()
    model = APEX1Model(config)
    token_ids = torch.randint(0, config.model.vocab_size, (2, 8))
    out = model(token_ids)
    assert out["logits"].shape == (2, 8, config.model.vocab_size)

    loss = out["logits"].mean()
    loss.backward()

    adapter_grad_count = 0
    base_grad_count = 0
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "dora_magnitude" in name:
            if param.grad is not None:
                adapter_grad_count += 1
        elif "base_layer" in name and param.grad is not None:
            base_grad_count += 1

    assert adapter_grad_count > 0
    assert base_grad_count == 0


def test_dora_adapter_save_and_load(tmp_path):
    config = get_tiny_dora_config()
    model = APEX1Model(config)
    adapter_path = tmp_path / "dora_adapter.pt"
    save_lora_adapters(model, adapter_path, config.peft)

    fresh = APEX1Model(config)
    info = load_lora_adapters(fresh, adapter_path, strict=False)
    assert info["method"] == "dora"
    assert info["format"] == "apex_lora_adapter"
    assert info["num_dora_modules"] == count_dora_modules(model)


def test_dora_merge_and_unload_loads_into_plain_model():
    dconfig = get_tiny_dora_config()
    dmodel = APEX1Model(dconfig)
    merge_and_unload_lora_weights(dmodel)
    assert count_lora_modules(dmodel) == 0

    plain_config = get_tiny_config()
    plain = APEX1Model(plain_config)
    incompatible = plain.load_state_dict(dmodel.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


def test_qdora_modules_are_inserted_and_merge_to_plain_model():
    config = get_tiny_qdora_config()
    model = APEX1Model(config)
    assert count_dora_modules(model) > 0
    for _name, module in iter_dora_modules(model):
        assert isinstance(module, QDoRALinear)
        assert isinstance(module.base_layer, QuantizedLinear4bit)

    merge_and_unload_lora_weights(model)
    assert count_lora_modules(model) == 0

    plain = APEX1Model(get_tiny_config())
    incompatible = plain.load_state_dict(model.state_dict(), strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
