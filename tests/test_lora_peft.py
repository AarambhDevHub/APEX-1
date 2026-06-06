"""
Tests for APEX-1 LoRA / PEFT support.
"""

from __future__ import annotations

import torch

from apex.config import get_tiny_lora_config
from apex.model.apex_model import APEX1Model
from apex.model.lora import (
    LoRALinear,
    get_lora_state_dict,
    iter_lora_modules,
    load_lora_adapters,
    peft_parameter_summary,
    save_lora_adapters,
)


def test_lora_modules_are_inserted():
    config = get_tiny_lora_config()
    model = APEX1Model(config)

    lora_modules = list(iter_lora_modules(model))
    assert len(lora_modules) > 0
    assert all(isinstance(module, LoRALinear) for _, module in lora_modules)


def test_peft_freezes_base_and_keeps_lora_trainable():
    config = get_tiny_lora_config()
    model = APEX1Model(config)

    stats = peft_parameter_summary(model)
    assert stats["trainable"] > 0
    assert stats["trainable"] < stats["total"]
    assert stats["trainable_percent"] < 20.0

    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            assert param.requires_grad
        elif "bias" not in name:
            assert not param.requires_grad


def test_lora_forward_pass_shape():
    config = get_tiny_lora_config()
    model = APEX1Model(config)

    token_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    output = model(token_ids)

    assert output["logits"].shape == (2, 16, config.model.vocab_size)
    assert output["kv_caches"] is not None


def test_lora_backward_only_trains_adapters():
    config = get_tiny_lora_config()
    model = APEX1Model(config)

    token_ids = torch.randint(0, config.model.vocab_size, (2, 16))
    output = model(token_ids)
    loss = output["logits"].mean()
    loss.backward()

    saw_lora_grad = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"missing grad for trainable param {name}"
            if "lora_" in name:
                saw_lora_grad = True
        else:
            assert param.grad is None, f"frozen param should not have grad: {name}"

    assert saw_lora_grad


def test_adapter_save_and_load(tmp_path):
    config = get_tiny_lora_config()
    model = APEX1Model(config)

    path = tmp_path / "adapter.pt"
    save_lora_adapters(model, path, config.peft)

    state = get_lora_state_dict(model)
    assert len(state) > 0
    assert path.exists()

    model2 = APEX1Model(config)
    info = load_lora_adapters(model2, path, strict=False)
    assert "missing_keys" in info
    assert "unexpected_keys" in info
