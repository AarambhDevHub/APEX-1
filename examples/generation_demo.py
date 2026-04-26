"""Example: Text generation with random weights (shows pipeline works)."""

import torch

from apex.config import get_tiny_config
from apex.generation.generator import APEX1Generator, GenerationConfig
from apex.model.apex_model import APEX1Model


def main():
    print("=== APEX-1 Text Generation Demo ===\n")
    config = get_tiny_config()
    model = APEX1Model(config)
    model.eval()
    gen_config = GenerationConfig(
        max_new_tokens=32,
        temperature=0.8,
        top_p=0.9,
        top_k=50,
        eos_token_id=2,
    )
    generator = APEX1Generator(model, gen_config)
    input_ids = torch.randint(0, config.model.vocab_size, (1, 8))
    print(f"Input token IDs: {input_ids[0].tolist()}")
    output = generator.generate(input_ids, prefix_len=4)
    print(f"Generated {output.total_tokens} tokens: {output.token_ids[:20]}...")
    print(f"Finished: {output.finished}")
    print("\n✓ Generation pipeline works!")


if __name__ == "__main__":
    main()
