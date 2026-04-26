"""Example: APEX-1 forward pass with random weights."""
import torch
from apex.config import get_tiny_config
from apex.model.apex_model import APEX1Model
from apex.utils.param_counter import print_parameter_summary

def main():
    print("=== APEX-1 Forward Pass Demo ===\n")
    config = get_tiny_config()
    config.validate()
    model = APEX1Model(config)
    model.eval()
    print_parameter_summary(model)
    token_ids = torch.randint(0, config.model.vocab_size, (1, 32))
    print(f"\nInput shape: {token_ids.shape}")
    with torch.no_grad():
        output = model(token_ids, prefix_len=16)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"Spec logits: {len(output['spec_logits'])} heads")
    print(f"KV caches: {len(output['kv_caches'])} layers")
    next_logits = output["logits"][0, -1, :]
    top5 = torch.topk(next_logits, 5)
    print(f"\nTop-5 next token predictions:")
    for i, (val, idx) in enumerate(zip(top5.values, top5.indices)):
        print(f"  {i+1}. Token {idx.item()} (logit: {val.item():.4f})")
    print("\n✓ Forward pass completed successfully!")

if __name__ == "__main__":
    main()
