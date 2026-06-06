"""
APEX-1 QLoRA / 4-bit PEFT CPU smoke demo.

This demo proves the v2.7.0 workflow:
1. Build tiny APEX-1 with QLoRA enabled
2. Quantize targeted base Linear layers to 4-bit NF4-style storage
3. Train only LoRA adapter matrices
4. Save an adapter-only checkpoint
5. Merge and unload into a plain model checkpoint-compatible shape
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from apex.config import get_tiny_qlora_config
from apex.model.apex_model import APEX1Model
from apex.model.lora import (
    count_lora_modules,
    count_qlora_modules,
    merge_and_unload_lora_weights,
    peft_parameter_summary,
    print_peft_parameter_summary,
    qlora_storage_summary,
    save_lora_adapters,
)
from apex.training.losses import compute_sft_loss


def main() -> None:
    torch.manual_seed(42)

    config = get_tiny_qlora_config()
    config.validate()
    model = APEX1Model(config)

    print(f"LoRA/QLoRA modules inserted: {count_lora_modules(model)}")
    print(f"QLoRA modules inserted: {count_qlora_modules(model)}")
    print(f"QLoRA storage summary: {qlora_storage_summary(model)}")
    print_peft_parameter_summary(model)

    model.train()
    token_ids = torch.randint(0, config.model.vocab_size, (2, config.training.seq_len))
    token_types = torch.ones_like(token_ids) * 2

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=config.training.peak_lr)

    output = model(token_ids)
    loss, _metrics = compute_sft_loss(
        output["logits"],
        token_ids,
        token_types,
        config.model.vocab_size,
    )
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    print(f"One-step QLoRA loss: {loss.item():.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        adapter_path = Path(tmp) / "adapter_final.pt"
        save_lora_adapters(model, adapter_path, config.peft)
        print(f"Saved QLoRA adapter: {adapter_path}")

        merge_and_unload_lora_weights(model)
        print(f"LoRA/QLoRA modules after merge+unload: {count_lora_modules(model)}")

    stats = peft_parameter_summary(model)
    print(f"Final trainable params after unload: {stats['trainable']}")


if __name__ == "__main__":
    main()
