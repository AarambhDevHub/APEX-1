#!/usr/bin/env python
"""
APEX-1 Adapter-DPO fine-tuning CLI.

Examples:

python scripts/finetune_adapter_dpo.py \
  --config configs/apex1_tiny_lora_dpo.yaml \
  --data data/samples/tiny_preference.jsonl \
  --output-dir outputs/adapter-dpo-test \
  --max-steps 10

python scripts/finetune_adapter_dpo.py \
  --config configs/apex1_tiny_dora_dpo.yaml \
  --method dora \
  --data data/samples/tiny_preference.jsonl \
  --output-dir outputs/dora-dpo-test \
  --max-steps 10
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from apex.alignment.adapter_dpo import AdapterDPOTrainer, PreferenceJSONLDataset, preference_collate
from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model
from apex.model.lora import load_lora_adapters
from apex.tokenizer import APEX1Tokenizer
from apex.training.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="APEX-1 Adapter-DPO alignment")
    parser.add_argument("--config", type=str, default="configs/apex1_tiny_lora_dpo.yaml")
    parser.add_argument("--data", type=str, required=True, help="Preference JSONL path")
    parser.add_argument("--output-dir", type=str, default="runs/adapter_dpo")
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional base/SFT checkpoint")
    parser.add_argument("--adapter", type=str, default=None, help="Optional adapter checkpoint to continue from")
    parser.add_argument("--method", type=str, default=None, choices=["lora", "qlora", "dora", "qdora"])
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--reference-free", action="store_true")
    parser.add_argument("--length-normalize", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    config = APEXConfig.from_yaml(args.config)
    config.peft.enabled = True
    config.adapter_dpo.enabled = True

    if args.method:
        config.peft.method = args.method
    if args.beta is not None:
        config.adapter_dpo.beta = args.beta
    if args.reference_free:
        config.adapter_dpo.reference_free = True
    if args.length_normalize:
        config.adapter_dpo.length_normalize = True
    if args.max_steps is not None:
        config.training.max_steps = args.max_steps

    config.validate()

    tokenizer = APEX1Tokenizer(args.tokenizer)

    # Policy model: PEFT enabled.
    policy_model = APEX1Model(config)

    # Reference model: adapter-free frozen base.
    reference_config = copy.deepcopy(config)
    reference_config.peft.enabled = False
    reference_config.adapter_dpo.enabled = False
    reference_model = APEX1Model(reference_config)

    if args.checkpoint:
        load_checkpoint(args.checkpoint, reference_model, strict=False)
        load_checkpoint(args.checkpoint, policy_model, strict=False)
        logger.info("Loaded base/SFT checkpoint into policy and reference: %s", args.checkpoint)

    if args.adapter:
        load_lora_adapters(policy_model, args.adapter, strict=False)
        logger.info("Loaded starting adapter: %s", args.adapter)

    dataset = PreferenceJSONLDataset(
        args.data,
        tokenizer=tokenizer,
        max_prompt_len=config.adapter_dpo.max_prompt_len,
        max_response_len=config.adapter_dpo.max_response_len,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=preference_collate)

    device = torch.device(args.device) if args.device else None
    trainer = AdapterDPOTrainer(
        policy_model=policy_model,
        reference_model=reference_model,
        config=config,
        train_loader=loader,
        device=device,
    )

    result = trainer.train(max_steps=config.training.max_steps, output_dir=args.output_dir)

    print("\n" + "=" * 70)
    print("APEX-1 Adapter-DPO Complete")
    print("=" * 70)
    print(f"Method:       {config.peft.method}")
    print(f"Steps:        {result['steps']}")
    print(f"Final loss:   {result.get('loss', 0.0):.4f}")
    print(f"Reward margin:{result.get('reward_margin', 0.0):.4f}")
    print(f"Adapter:      {result['adapter_path']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
