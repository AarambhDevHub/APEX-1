"""Tests for APEX-1 v2.6.0 LoRA inference and merge workflow."""

from __future__ import annotations

import torch

from apex.config import get_tiny_config, get_tiny_lora_config
from apex.model.apex_model import APEX1Model
from apex.model.lora import (
    count_lora_modules,
    has_lora_adapters,
    merge_and_unload_lora_weights,
    save_lora_adapters,
)
from apex.model.lora_inference import load_lora_model_for_inference, load_merge_and_unload_lora_model


def test_lora_adapter_load_for_inference(tmp_path):
    config = get_tiny_lora_config()
    model = APEX1Model(config)
    adapter_path = tmp_path / "adapter.pt"
    save_lora_adapters(model, adapter_path, peft_config=config.peft)

    infer_config = get_tiny_lora_config()
    result = load_lora_model_for_inference(
        config=infer_config,
        adapter=adapter_path,
        checkpoint=None,
        device="cpu",
    )

    assert has_lora_adapters(result.model)
    assert result.lora_modules > 0
    assert result.adapter_info["format"] == "apex_lora_adapter"

    input_ids = torch.randint(0, infer_config.model.vocab_size, (1, 8))
    output = result.model(input_ids)
    assert output["logits"].shape == (1, 8, infer_config.model.vocab_size)


def test_lora_merge_for_runtime_keeps_wrappers(tmp_path):
    config = get_tiny_lora_config()
    model = APEX1Model(config)
    adapter_path = tmp_path / "adapter.pt"
    save_lora_adapters(model, adapter_path, peft_config=config.peft)

    infer_config = get_tiny_lora_config()
    result = load_lora_model_for_inference(
        config=infer_config,
        adapter=adapter_path,
        checkpoint=None,
        device="cpu",
        merge_for_runtime=True,
    )

    assert result.merged_for_runtime is True
    assert count_lora_modules(result.model) > 0
    assert all(module.merged for _name, module in result.model.named_modules() if hasattr(module, "merged"))


def test_merge_and_unload_removes_lora_wrappers():
    config = get_tiny_lora_config()
    model = APEX1Model(config)

    before = count_lora_modules(model)
    assert before > 0

    merge_and_unload_lora_weights(model)

    after = count_lora_modules(model)
    assert after == 0
    assert not has_lora_adapters(model)

    input_ids = torch.randint(0, config.model.vocab_size, (1, 8))
    output = model(input_ids)
    assert output["logits"].shape == (1, 8, config.model.vocab_size)


def test_merged_state_dict_loads_into_plain_model(tmp_path):
    lora_config = get_tiny_lora_config()
    lora_model = APEX1Model(lora_config)
    merge_and_unload_lora_weights(lora_model)

    checkpoint_path = tmp_path / "merged.pt"
    torch.save({"model_state_dict": lora_model.state_dict()}, checkpoint_path)

    plain_config = get_tiny_config()
    plain_config.peft.enabled = False
    plain_model = APEX1Model(plain_config)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    plain_model.load_state_dict(checkpoint["model_state_dict"], strict=True)


def test_load_merge_and_unload_helper_returns_plain_model(tmp_path):
    config = get_tiny_lora_config()
    model = APEX1Model(config)
    adapter_path = tmp_path / "adapter.pt"
    save_lora_adapters(model, adapter_path, peft_config=config.peft)

    merge_config = get_tiny_lora_config()
    result = load_merge_and_unload_lora_model(
        config=merge_config,
        adapter=adapter_path,
        checkpoint=None,
        device="cpu",
    )

    assert result.unloaded is True
    assert result.merged_for_runtime is True
    assert result.lora_modules == 0
    assert not has_lora_adapters(result.model)
    assert merge_config.peft.enabled is False
