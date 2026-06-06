"""
APEX-1 DoRA PEFT CPU smoke demo.

This demo proves the v2.8.0 workflow:
1. Build tiny APEX-1 with DoRA enabled
2. Freeze base model weights
3. Train low-rank direction adapters plus DoRA magnitude vectors
4. Save an adapter-only checkpoint
5. Merge and unload into a plain model checkpoint-compatible shape
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from apex.config import get_tiny_config, get_tiny_dora_config
from apex.model.apex_model import APEX1Model
from apex.model.lora import (
    count_dora_modules,
    count_lora_modules,
    load_lora_adapters,
    merge_and_unload_lora_weights,
    peft_parameter_summary,
    print_peft_parameter_summary,
    save_lora_adapters,
)
from apex.training.losses import compute_sft_loss


def main() -> None:
    torch.manual_seed(42)

    config = get_tiny_dora_config()
    config.validate()
    model = APEX1Model(config)

    print(f"Adapter modules inserted: {count_lora_modules(model)}")
    print(f"DoRA modules inserted: {count_dora_modules(model)}")
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

    print(f"One-step DoRA loss: {loss.item():.4f}")

    with tempfile.TemporaryDirectory() as tmp:
        adapter_path = Path(tmp) / "adapter_final.pt"
        save_lora_adapters(model, adapter_path, config.peft)
        print(f"Saved DoRA adapter: {adapter_path}")

        fresh = APEX1Model(config)
        info = load_lora_adapters(fresh, adapter_path, strict=False)
        print(f"Loaded adapter method: {info['method']}")

        merge_and_unload_lora_weights(fresh)
        print(f"Adapter modules after merge+unload: {count_lora_modules(fresh)}")

        plain = APEX1Model(get_tiny_config())
        incompatible = plain.load_state_dict(fresh.state_dict(), strict=True)
        print(f"Plain model load missing keys: {len(incompatible.missing_keys)}")
        print(f"Plain model load unexpected keys: {len(incompatible.unexpected_keys)}")

    stats = peft_parameter_summary(model)
    print(f"Trainable params during DoRA: {stats['trainable']}")


if __name__ == "__main__":
    main()
