"""
APEX-1 LoRA/QLoRA Merge CLI.

v2.6.0 feature: merge a trained LoRA/QLoRA adapter into the base model and save a
normal APEX checkpoint for deployment-style inference.

Example:

    python scripts/merge_lora.py \
      --config configs/apex1_tiny_lora.yaml \
      --adapter outputs/lora-test/adapter_final.pt \
      --output outputs/merged-apex-lora.pt

With a trained base checkpoint:

    python scripts/merge_lora.py \
      --config configs/apex1_tiny_lora.yaml \
      --checkpoint checkpoints/base.pt \
      --adapter outputs/lora-test/adapter_final.pt \
      --output outputs/merged-apex-lora.pt

Then generate with the normal script:

    python scripts/generate.py \
      --config configs/apex1_tiny.yaml \
      --checkpoint outputs/merged-apex-lora.pt \
      --prompt "Hello"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from apex.config import APEXConfig
from apex.model.lora import (
    count_lora_modules,
    load_lora_adapters,
    merge_and_unload_lora_weights,
    merge_lora_weights,
    save_merged_lora_checkpoint,
)
from apex.model.lora_inference import build_base_model_then_apply_lora

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge an APEX-1 LoRA/QLoRA adapter into base weights")
    parser.add_argument("--config", type=str, required=True, help="APEX YAML config path")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA/QLoRA adapter checkpoint path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional base checkpoint")
    parser.add_argument("--output", type=str, required=True, help="Merged checkpoint output path")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, cuda:0")
    parser.add_argument(
        "--keep-lora-wrappers",
        action="store_true",
        help="Merge weights but keep LoRA/QLoRA wrapper modules in the saved checkpoint",
    )
    parser.add_argument(
        "--strict-base",
        action="store_true",
        help="Strictly load base checkpoint keys when --checkpoint is provided",
    )
    parser.add_argument(
        "--strict-adapter",
        action="store_true",
        help="Strictly load adapter checkpoint keys",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    adapter_path = Path(args.adapter)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter checkpoint not found: {adapter_path}")

    config = APEXConfig.from_yaml(args.config)
    config.peft.enabled = True
    config.validate()

    model = build_base_model_then_apply_lora(
        config=config,
        checkpoint=args.checkpoint,
        device=args.device,
        strict_base=args.strict_base,
    )
    adapter_info = load_lora_adapters(
        model,
        adapter_path,
        strict=args.strict_adapter,
        map_location="cpu",
    )

    before = count_lora_modules(model)
    if args.keep_lora_wrappers:
        merge_lora_weights(model)
        after = count_lora_modules(model)
        contains_wrappers = True
    else:
        merge_and_unload_lora_weights(model)
        config.peft.enabled = False
        after = count_lora_modules(model)
        contains_wrappers = False

    output_path = Path(args.output)
    save_merged_lora_checkpoint(
        model,
        output_path,
        config=config,
        extra={
            "source_adapter": str(adapter_path),
            "source_checkpoint": args.checkpoint,
            "adapter_info": adapter_info,
            "lora_modules_before_merge": before,
            "lora_modules_after_merge": after,
            "contains_lora_wrappers": contains_wrappers,
        },
    )

    print("\n" + "=" * 70)
    print("APEX-1 LoRA/QLoRA Merge Complete")
    print("=" * 70)
    print(f"Adapter: {adapter_path}")
    print(f"Base checkpoint: {args.checkpoint or 'random/base config weights'}")
    print(f"LoRA/QLoRA modules before merge: {before}")
    print(f"LoRA/QLoRA modules after merge:  {after}")
    print(f"Contains LoRA/QLoRA wrappers:    {contains_wrappers}")
    print(f"Saved merged checkpoint:   {output_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
