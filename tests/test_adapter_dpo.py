from __future__ import annotations

import copy
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from apex.alignment.adapter_dpo import (
    AdapterDPOTrainer,
    PreferenceJSONLDataset,
    adapter_dpo_loss,
    format_preference_example,
    preference_collate,
)
from apex.config import get_tiny_adapter_dpo_config
from apex.model.apex_model import APEX1Model
from apex.model.lora import count_lora_modules, count_trainable_parameters
from apex.tokenizer import APEX1Tokenizer


def _make_models(method: str = "lora"):
    config = get_tiny_adapter_dpo_config(method=method)
    config.adapter_dpo.max_prompt_len = 32
    config.adapter_dpo.max_response_len = 32
    config.training.max_steps = 2
    config.validate()

    policy = APEX1Model(config)

    ref_config = copy.deepcopy(config)
    ref_config.peft.enabled = False
    ref_config.adapter_dpo.enabled = False
    reference = APEX1Model(ref_config)

    for p in reference.parameters():
        p.requires_grad = False

    return config, policy, reference


def test_format_preference_example_shapes():
    tokenizer = APEX1Tokenizer()
    ex = format_preference_example(
        tokenizer,
        prompt="Explain ownership",
        chosen="One owner controls a value.",
        rejected="It is random.",
        max_prompt_len=32,
        max_response_len=16,
    )

    assert ex.prompt_ids.dim() == 1
    assert ex.chosen_ids.dim() == 1
    assert ex.rejected_ids.dim() == 1
    assert ex.prompt_len == ex.prompt_ids.numel()
    assert ex.chosen_ids.numel() > ex.prompt_len
    assert ex.rejected_ids.numel() > ex.prompt_len


def test_adapter_dpo_loss_is_finite():
    config, policy, reference = _make_models("lora")
    tokenizer = APEX1Tokenizer()
    ex = format_preference_example(
        tokenizer,
        prompt="What is LoRA?",
        chosen="LoRA trains small adapter matrices.",
        rejected="LoRA trains nothing.",
        max_prompt_len=32,
        max_response_len=16,
    )

    loss, metrics = adapter_dpo_loss(
        policy,
        reference,
        chosen_ids=ex.chosen_ids,
        rejected_ids=ex.rejected_ids,
        prompt_len=ex.prompt_len,
        beta=config.adapter_dpo.beta,
    )

    assert torch.isfinite(loss)
    assert "reward_margin" in metrics
    assert "accuracy" in metrics


def test_adapter_dpo_backward_trains_only_adapters():
    _config, policy, reference = _make_models("lora")
    tokenizer = APEX1Tokenizer()
    ex = format_preference_example(
        tokenizer,
        prompt="Explain Rust",
        chosen="Rust is a systems language focused on safety.",
        rejected="Rust is a database.",
        max_prompt_len=32,
        max_response_len=16,
    )

    loss, _metrics = adapter_dpo_loss(policy, reference, ex.chosen_ids, ex.rejected_ids, ex.prompt_len)
    loss.backward()

    trainable_names = [name for name, p in policy.named_parameters() if p.requires_grad]
    assert trainable_names
    assert all(("lora_A" in name or "lora_B" in name or "dora_magnitude" in name) for name in trainable_names)

    grad_names = [name for name, p in policy.named_parameters() if p.grad is not None]
    assert any("lora_A" in name or "lora_B" in name for name in grad_names)


def test_reference_model_is_frozen():
    _config, _policy, reference = _make_models("lora")
    assert all(not p.requires_grad for p in reference.parameters())


def test_adapter_dpo_trainer_saves_adapter(tmp_path: Path):
    config, policy, reference = _make_models("lora")
    tokenizer = APEX1Tokenizer()

    data_path = tmp_path / "prefs.jsonl"
    rows = [
        {
            "prompt": "Explain ownership",
            "chosen": "Ownership gives each value one owner.",
            "rejected": "Ownership is unrelated to memory safety.",
        },
        {
            "prompt": "What is DPO?",
            "chosen": "DPO trains on chosen and rejected response pairs.",
            "rejected": "DPO requires a separate reward model every time.",
        },
    ]
    with data_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    dataset = PreferenceJSONLDataset(data_path, tokenizer, max_prompt_len=32, max_response_len=16)
    loader = DataLoader(dataset, batch_size=1, collate_fn=preference_collate)

    trainer = AdapterDPOTrainer(policy, reference, config, loader)
    result = trainer.train(max_steps=2, output_dir=tmp_path / "out")

    adapter_path = Path(str(result["adapter_path"]))
    assert adapter_path.exists()
    assert result["steps"] == 2
    assert count_lora_modules(policy) > 0
    assert count_trainable_parameters(policy) > 0


def test_dora_adapter_dpo_loss_is_finite():
    config, policy, reference = _make_models("dora")
    tokenizer = APEX1Tokenizer()
    ex = format_preference_example(
        tokenizer,
        prompt="What is DoRA?",
        chosen="DoRA separates direction and magnitude in adapter fine-tuning.",
        rejected="DoRA is not related to adapters.",
        max_prompt_len=32,
        max_response_len=16,
    )

    loss, metrics = adapter_dpo_loss(
        policy,
        reference,
        chosen_ids=ex.chosen_ids,
        rejected_ids=ex.rejected_ids,
        prompt_len=ex.prompt_len,
        beta=config.adapter_dpo.beta,
    )

    assert torch.isfinite(loss)
    assert isinstance(metrics["reward_margin"], float)
