"""Minimal APEX-1 vision forward pass demo.

Run:
    python examples/vision_forward_demo.py
"""

import torch

from apex.config import get_tiny_vision_config
from apex.model.apex_vision_model import APEX1VisionModel


def main() -> None:
    cfg = get_tiny_vision_config()
    cfg.validate()

    model = APEX1VisionModel(cfg)
    model.eval()

    # Token 8 is the tokenizer fallback for <|img|> in APEX1Tokenizer.
    token_ids = torch.tensor([[1, 4, 8, 20, 21, 5, 30]], dtype=torch.long)
    pixel_values = torch.randn(1, 3, cfg.vision.image_size, cfg.vision.image_size)

    with torch.no_grad():
        out = model(token_ids=token_ids, pixel_values=pixel_values, return_hidden=True)

    print("Input text tokens:", tuple(token_ids.shape))
    print("Visual tokens inserted:", out["visual_token_count"])
    print("Logits:", tuple(out["logits"].shape))
    print("Hidden states:", tuple(out["hidden_states"].shape))
    print("KV cache layers:", len(out["kv_caches"]))


if __name__ == "__main__":
    main()
