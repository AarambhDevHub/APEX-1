"""Example: Thinking mode demo with budget enforcement."""

import torch

from apex.config import get_tiny_config
from apex.generation.generator import APEX1Generator, GenerationConfig
from apex.model.apex_model import APEX1Model


def main():
    print("=== APEX-1 Thinking Mode Demo ===\n")
    config = get_tiny_config()
    model = APEX1Model(config)
    model.eval()
    gen_config = GenerationConfig(
        max_new_tokens=64,
        temperature=0.6,
        top_p=0.95,
        enable_thinking=True,
        max_thinking_tokens=20,
        thinking_temperature=0.6,
        output_temperature=0.3,
        thinking_start_id=6,
        thinking_end_id=7,
        eos_token_id=2,
    )
    generator = APEX1Generator(model, gen_config)
    input_ids = torch.randint(0, config.model.vocab_size, (1, 8))
    print(f"Input: {input_ids[0].tolist()}")
    print(f"Thinking budget: {gen_config.max_thinking_tokens} tokens")
    output = generator.generate(input_ids, prefix_len=4)
    print(f"Generated {output.total_tokens} tokens")
    print(f"Thinking tokens used: {output.thinking_tokens}")
    print(f"Budget enforced: {output.thinking_tokens <= gen_config.max_thinking_tokens}")
    print("\n✓ Thinking mode works!")


if __name__ == "__main__":
    main()
