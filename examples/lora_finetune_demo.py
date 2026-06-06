"""
Tiny LoRA fine-tuning demo for APEX-1.

This runs fully on CPU with random SFT-style data and proves:
- LoRA adapters are inserted
- base model is frozen
- only adapter parameters train
- adapter-only checkpoint is saved
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from apex.config import get_tiny_lora_config
from apex.data.data_loader import create_sft_loader
from apex.data.dataset import SFTDataset
from apex.model.apex_model import APEX1Model
from apex.model.lora import iter_lora_modules, peft_parameter_summary
from apex.training.peft import PEFTSFTTrainer


def make_samples(vocab_size: int, seq_len: int, n_samples: int = 16):
    samples = []
    for _ in range(n_samples):
        ids = torch.randint(0, vocab_size, (seq_len,)).tolist()
        types = [0] * 8 + [1] * 16 + [2] * (seq_len - 24)
        samples.append({"input_ids": ids, "token_types": types})
    return samples


def main() -> None:
    config = get_tiny_lora_config()
    config.validate()

    model = APEX1Model(config)
    lora_modules = list(iter_lora_modules(model))
    stats = peft_parameter_summary(model)

    print(f"LoRA modules inserted: {len(lora_modules)}")
    print(f"Trainable params: {stats['trainable']:,}")
    print(f"Trainable percent: {stats['trainable_percent']:.4f}%")

    dataset = SFTDataset(
        make_samples(config.model.vocab_size, config.training.seq_len),
        max_seq_len=config.training.seq_len,
    )
    loader = create_sft_loader(dataset, batch_size=2, num_workers=0)

    with tempfile.TemporaryDirectory() as tmp:
        trainer = PEFTSFTTrainer(model, config, loader)
        result = trainer.train(max_steps=2, output_dir=Path(tmp), checkpoint_interval=1)
        print(f"Saved adapter: {result['adapter_path']}")


if __name__ == "__main__":
    main()
