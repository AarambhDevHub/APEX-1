"""Tiny greedy vision chat demo.

This demonstrates the *plumbing* for image-conditioned generation. The tiny
random model is not trained, so the text will be nonsense until you train or
load weights.

Run:
    python examples/vision_chat_demo.py
"""

import torch

from apex.config import get_tiny_vision_config
from apex.model.apex_vision_model import APEX1VisionModel
from apex.tokenizer.tokenizer import APEX1Tokenizer, SPECIAL_TOKENS


def greedy_generate_with_image(
    model: APEX1VisionModel,
    tokenizer: APEX1Tokenizer,
    pixel_values: torch.Tensor,
    question: str,
    max_new_tokens: int = 16,
) -> list[int]:
    prompt = (
        f"{SPECIAL_TOKENS['bos']}"
        f"{SPECIAL_TOKENS['user']}\n"
        f"{SPECIAL_TOKENS['img']}\n{question}\n"
        f"{SPECIAL_TOKENS['assistant']}\n"
    )
    token_ids = torch.tensor([tokenizer.encode(prompt, add_special_tokens=False)], dtype=torch.long)

    generated: list[int] = []
    kv_caches = None
    with torch.no_grad():
        # First pass includes image tokens.
        out = model(token_ids=token_ids, pixel_values=pixel_values, kv_caches=None)
        next_token = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
        kv_caches = out["kv_caches"]
        generated.append(int(next_token.item()))

        # Later passes reuse cached image+prompt context and pass text only.
        for _ in range(max_new_tokens - 1):
            out = model(token_ids=next_token, kv_caches=kv_caches)
            next_token = out["logits"][:, -1].argmax(dim=-1, keepdim=True)
            kv_caches = out["kv_caches"]
            token = int(next_token.item())
            generated.append(token)
            if token == tokenizer.eos_token_id:
                break

    return generated


def main() -> None:
    cfg = get_tiny_vision_config()
    model = APEX1VisionModel(cfg).eval()
    tokenizer = APEX1Tokenizer()
    pixel_values = torch.randn(1, 3, cfg.vision.image_size, cfg.vision.image_size)

    ids = greedy_generate_with_image(
        model=model,
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question="What is in this image?",
        max_new_tokens=8,
    )
    print("Generated token IDs:", ids)
    print("Decoded:", tokenizer.decode(ids, skip_special_tokens=True))


if __name__ == "__main__":
    main()
