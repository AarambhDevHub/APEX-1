"""
APEX-1 LoRA Adapter Generation CLI.

v2.6.0 feature: generate text with a trained LoRA adapter.

Examples:

    python scripts/generate_with_lora.py \
      --config configs/apex1_tiny_lora.yaml \
      --adapter outputs/lora-test/adapter_final.pt \
      --prompt "Explain Rust ownership simply" \
      --max-tokens 64

With a trained base checkpoint:

    python scripts/generate_with_lora.py \
      --config configs/apex1_tiny_lora.yaml \
      --checkpoint checkpoints/base.pt \
      --adapter outputs/lora-test/adapter_final.pt \
      --prompt "Write a short answer"

For slightly faster runtime, merge adapter weights into wrapped base layers
before generation:

    python scripts/generate_with_lora.py \
      --config configs/apex1_tiny_lora.yaml \
      --adapter outputs/lora-test/adapter_final.pt \
      --merge-before-generate \
      --prompt "Hello"
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

from apex.config import APEXConfig
from apex.generation.generator import APEX1Generator, GenerationConfig
from apex.model.lora import print_peft_parameter_summary
from apex.model.lora_inference import load_lora_model_for_inference
from apex.tokenizer import APEX1Tokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text with an APEX-1 LoRA adapter")
    parser.add_argument("--config", type=str, required=True, help="APEX YAML config path")
    parser.add_argument("--adapter", type=str, required=True, help="LoRA adapter checkpoint path")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional base model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None, help="Optional tokenizer.json path")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Prompt text")
    parser.add_argument("--max-tokens", type=int, default=128, help="Maximum new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus threshold")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking-mode token behavior")
    parser.add_argument("--device", type=str, default=None, help="Device: cpu, cuda, cuda:0")
    parser.add_argument(
        "--merge-before-generate",
        action="store_true",
        help="Merge LoRA weights into wrapped base layers before generation",
    )
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not print PEFT parameter summary",
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

    result = load_lora_model_for_inference(
        config=config,
        adapter=adapter_path,
        checkpoint=args.checkpoint,
        device=args.device,
        strict_base=args.strict_base,
        strict_adapter=args.strict_adapter,
        merge_for_runtime=args.merge_before_generate,
    )
    model = result.model
    device = result.device

    logger.info(
        "LoRA inference ready: modules=%d, merged_for_runtime=%s, adapter_version=%s",
        result.lora_modules,
        result.merged_for_runtime,
        result.adapter_info.get("version", "unknown"),
    )

    if not args.no_summary:
        print_peft_parameter_summary(model)

    tokenizer = APEX1Tokenizer(args.tokenizer)
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        enable_thinking=args.thinking,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        thinking_start_id=tokenizer.thinking_start_id,
        thinking_end_id=tokenizer.thinking_end_id,
    )
    generator = APEX1Generator(model, gen_config)

    input_ids = torch.tensor(
        [tokenizer.encode(args.prompt, add_special_tokens=True)],
        device=device,
        dtype=torch.long,
    )

    output = generator.generate(input_ids)
    generated_text = tokenizer.decode(output.token_ids)

    print("\n" + "=" * 70)
    print("APEX-1 LoRA Generation")
    print("=" * 70)
    print(f"Prompt:\n{args.prompt}\n")
    print(f"Generated ({output.total_tokens} tokens):\n")
    print(generated_text)
    print("\n" + "=" * 70)
    print(f"Finished EOS: {output.finished}")
    print(f"Adapter: {adapter_path}")
    if args.checkpoint:
        print(f"Base checkpoint: {args.checkpoint}")
    print("=" * 70)


if __name__ == "__main__":
    main()
