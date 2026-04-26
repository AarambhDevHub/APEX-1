"""
APEX-1 Training CLI.

Launch pretraining or SFT from the command line with a config file.

Usage:
    python scripts/train.py --config configs/apex1_tiny.yaml --mode pretrain
    python scripts/train.py --config configs/apex1_small.yaml --mode sft --checkpoint checkpoints/pretrained.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

from apex.config import APEXConfig
from apex.data.data_loader import create_pretrain_loader, create_sft_loader
from apex.data.dataset import PretrainDataset, SFTDataset
from apex.model.apex_model import APEX1Model
from apex.training.checkpoint import load_checkpoint
from apex.training.trainer import PreTrainer, SFTTrainer
from apex.utils.param_counter import print_parameter_summary

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("training.log")],
)
logger = logging.getLogger(__name__)


def create_dummy_data(config: APEXConfig, n_tokens: int = 100000) -> torch.Tensor:
    """Create random token data for testing the training pipeline."""
    return torch.randint(0, config.model.vocab_size, (n_tokens,))


def main():
    parser = argparse.ArgumentParser(description="APEX-1 Training CLI")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--mode", choices=["pretrain", "sft"], default="pretrain")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints/", help="Save dir")
    parser.add_argument("--max-steps", type=int, default=None, help="Override max steps")
    parser.add_argument("--data", type=str, default=None, help="Path to training data")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")
    parser.add_argument("--dry-run", action="store_true", help="Quick test with dummy data")
    args = parser.parse_args()

    # Load config
    config = APEXConfig.from_yaml(args.config)
    config.validate()
    logger.info("Config loaded from %s", args.config)

    # Create model
    model = APEX1Model(config)
    print_parameter_summary(model)

    # Resume from checkpoint
    if args.checkpoint:
        info = load_checkpoint(args.checkpoint, model)
        logger.info("Resumed from step %d", info["step"])

    # Set up data
    if args.dry_run or args.data is None:
        logger.info("Using dummy data (dry run)")
        token_ids = create_dummy_data(config, n_tokens=50000)
    else:
        logger.info("Loading data from %s", args.data)
        # In production: load real tokenized data
        token_ids = create_dummy_data(config)

    # WandB
    wandb_run = None
    if args.wandb:
        try:
            import wandb

            wandb_run = wandb.init(project="apex-1", config=vars(config))
        except ImportError:
            logger.warning("wandb not installed, skipping")

    if args.mode == "pretrain":
        dataset = PretrainDataset(token_ids, seq_len=config.training.seq_len)
        loader = create_pretrain_loader(dataset, batch_size=config.training.batch_size)

        trainer = PreTrainer(model, config, loader)
        trainer.train(
            max_steps=args.max_steps or config.training.max_steps,
            checkpoint_dir=args.checkpoint_dir,
            wandb_run=wandb_run,
        )

    elif args.mode == "sft":
        # For SFT, create dummy instruction data
        samples = []
        seq_len = config.training.seq_len
        for _ in range(1000):
            ids = torch.randint(0, config.model.vocab_size, (seq_len,)).tolist()
            types = (
                [0] * (seq_len // 3) + [1] * (seq_len // 3) + [2] * (seq_len - 2 * (seq_len // 3))
            )
            samples.append({"input_ids": ids, "token_types": types})

        dataset = SFTDataset(samples, max_seq_len=seq_len)
        loader = create_sft_loader(dataset, batch_size=config.training.batch_size)

        trainer = SFTTrainer(model, config, loader)
        trainer.train(
            max_steps=args.max_steps or 5000,
            checkpoint_dir=args.checkpoint_dir,
            wandb_run=wandb_run,
        )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
