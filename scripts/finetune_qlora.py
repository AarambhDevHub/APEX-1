"""
APEX-1 QLoRA / 4-bit PEFT fine-tuning CLI.

v2.7.0 adds an educational QLoRA path:

    frozen 4-bit quantized base model + trainable LoRA adapters

Examples:

    python scripts/finetune_qlora.py \
      --config configs/apex1_tiny_qlora.yaml \
      --data data/samples/tiny_sft.jsonl \
      --output-dir outputs/qlora-test \
      --max-steps 20

Dry run:

    python scripts/finetune_qlora.py \
      --config configs/apex1_tiny_qlora.yaml \
      --dry-run \
      --max-steps 5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from apex.config import APEXConfig
from apex.data.data_loader import create_sft_loader
from apex.data.dataset import SFTDataset
from apex.model.apex_model import APEX1Model
from apex.model.lora import (
    apply_lora_adapters,
    count_qlora_modules,
    load_lora_adapters,
    qlora_storage_summary,
)
from apex.tokenizer.tokenizer import APEX1Tokenizer
from apex.training.checkpoint import load_checkpoint
from apex.training.peft import PEFTSFTTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def create_dummy_sft_samples(vocab_size: int, seq_len: int, n_samples: int = 64):
    """Create synthetic SFT samples for CPU smoke tests."""
    samples = []
    for _ in range(n_samples):
        ids = torch.randint(0, vocab_size, (seq_len,)).tolist()
        types = [0] * (seq_len // 4) + [1] * (seq_len // 4) + [2] * (seq_len - seq_len // 2)
        samples.append({"input_ids": ids, "token_types": types})
    return samples


def build_model_with_optional_base_checkpoint(
    config: APEXConfig,
    checkpoint: str | None,
) -> APEX1Model:
    """Load a base checkpoint safely, then inject QLoRA adapters."""
    peft_enabled = config.peft.enabled
    config.peft.enabled = False
    model = APEX1Model(config)

    if checkpoint:
        info = load_checkpoint(checkpoint, model)
        logger.info("Loaded base checkpoint from step %s", info.get("step", "unknown"))

    config.peft.enabled = peft_enabled
    config.peft.method = "qlora"
    apply_lora_adapters(model, config.peft)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune APEX-1 with QLoRA-style 4-bit PEFT")
    parser.add_argument("--config", type=str, default="configs/apex1_tiny_qlora.yaml")
    parser.add_argument("--data", type=str, default=None, help="SFT JSONL data path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer.json path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional base checkpoint")
    parser.add_argument("--adapter", type=str, default=None, help="Optional adapter checkpoint to resume")
    parser.add_argument("--output-dir", type=str, default="outputs/qlora-test", help="Adapter save dir")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max train steps")
    parser.add_argument("--dry-run", action="store_true", help="Use synthetic data")
    args = parser.parse_args()

    config = APEXConfig.from_yaml(args.config)
    config.peft.enabled = True
    config.peft.method = "qlora"
    config.validate()

    model = build_model_with_optional_base_checkpoint(config, args.checkpoint)

    if args.adapter:
        load_lora_adapters(model, args.adapter, strict=False)

    logger.info("QLoRA modules inserted: %d", count_qlora_modules(model))
    logger.info("QLoRA storage summary: %s", qlora_storage_summary(model))

    if args.dry_run or args.data is None:
        logger.info("Using dummy SFT data")
        samples = create_dummy_sft_samples(
            vocab_size=config.model.vocab_size,
            seq_len=config.training.seq_len,
        )
        dataset = SFTDataset(samples, max_seq_len=config.training.seq_len)
    else:
        tokenizer = APEX1Tokenizer(args.tokenizer) if args.tokenizer else APEX1Tokenizer()
        dataset = SFTDataset.from_jsonl(
            Path(args.data),
            tokenizer=tokenizer,
            max_seq_len=config.training.seq_len,
        )

    loader = create_sft_loader(
        dataset,
        batch_size=config.training.batch_size,
        num_workers=0,
        shuffle=True,
    )

    trainer = PEFTSFTTrainer(model, config, loader)
    result = trainer.train(
        max_steps=args.max_steps or config.training.max_steps,
        output_dir=args.output_dir,
    )

    logger.info("Done. QLoRA adapter saved to %s", result["adapter_path"])


if __name__ == "__main__":
    main()
