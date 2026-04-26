"""
APEX-1 Text Generation CLI.

Generate text from a trained APEX-1 model checkpoint.

Usage:
    python scripts/generate.py --checkpoint checkpoints/model.pt --prompt "Hello world"
    python scripts/generate.py --checkpoint checkpoints/model.pt --interactive
    python scripts/generate.py --config configs/apex1_tiny.yaml --random --prompt "Test"
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch

from apex.config import APEXConfig
from apex.generation.generator import APEX1Generator, GenerationConfig
from apex.model.apex_model import APEX1Model
from apex.tokenizer import APEX1Tokenizer
from apex.training.checkpoint import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="APEX-1 Text Generation CLI")
    parser.add_argument("--config", type=str, default="configs/apex1_tiny.yaml")
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer JSON path")
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Input prompt")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p (nucleus) threshold")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering")
    parser.add_argument("--repetition-penalty", type=float, default=1.1, help="Repetition penalty")
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode")
    parser.add_argument("--speculative", action="store_true", help="Enable speculative decoding")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat mode")
    parser.add_argument("--random", action="store_true", help="Use random weights (no checkpoint)")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    args = parser.parse_args()

    # Load config
    config = APEXConfig.from_yaml(args.config)
    config.validate()

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Load model
    model = APEX1Model(config).to(device)
    if args.checkpoint and not args.random:
        load_checkpoint(args.checkpoint, model)
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        logger.info("Using random weights (no checkpoint loaded)")

    # Load tokenizer
    tokenizer = APEX1Tokenizer(args.tokenizer)

    # Generation config
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        enable_thinking=args.thinking,
        use_speculative=args.speculative,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        thinking_start_id=tokenizer.thinking_start_id,
        thinking_end_id=tokenizer.thinking_end_id,
    )

    generator = APEX1Generator(model, gen_config)

    if args.interactive:
        _interactive_loop(generator, tokenizer, gen_config, device)
    else:
        _single_generation(args.prompt, generator, tokenizer, device)


def _single_generation(prompt, generator, tokenizer, device):
    """Generate from a single prompt."""
    print(f"\n{'='*60}")
    print(f"Prompt: {prompt}")
    print(f"{'='*60}\n")

    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    output = generator.generate(input_ids)

    generated_text = tokenizer.decode(output.token_ids)
    print(f"Generated ({output.total_tokens} tokens):\n")
    print(generated_text)
    print(f"\n{'='*60}")
    print(f"Thinking tokens: {output.thinking_tokens}")
    print(f"Finished (EOS): {output.finished}")


def _interactive_loop(generator, tokenizer, gen_config, device):
    """Interactive chat loop."""
    print("\n🔺 APEX-1 Interactive Chat")
    print("Type 'quit' to exit, 'clear' to reset history.\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() == "quit":
            break
        if user_input.lower() == "clear":
            messages = []
            print("History cleared.\n")
            continue
        if not user_input:
            continue

        messages.append({"role": "user", "content": user_input})
        chat_text = tokenizer.format_chat(messages, add_generation_prompt=True)
        input_ids = torch.tensor(
            [tokenizer.encode(chat_text, add_special_tokens=False)],
            device=device,
        )

        output = generator.generate(input_ids)
        response_text = tokenizer.decode(output.token_ids)

        print(f"APEX-1: {response_text}\n")
        messages.append({"role": "assistant", "content": response_text})


if __name__ == "__main__":
    main()
