from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch

from apex.config import get_tiny_config
from apex.eval import compute_perplexity, evaluate_generated_texts, next_token_accuracy
from apex.model.apex_model import APEX1Model


def main() -> None:
    torch.manual_seed(0)
    cfg = get_tiny_config()
    cfg.validate()
    model = APEX1Model(cfg)

    input_ids = torch.randint(0, cfg.model.vocab_size, (2, 16))
    batches = [{"input_ids": input_ids}]
    ppl = compute_perplexity(model, batches)
    print("Perplexity result:", ppl.as_dict())

    with torch.no_grad():
        logits = model(input_ids)["logits"][:, :-1, :]
    labels = input_ids[:, 1:]
    acc = next_token_accuracy(logits, labels)
    print("Next-token accuracy:", acc)

    samples = [
        "APEX one teaches attention and evaluation",
        "Vision tokens enter the transformer context",
    ]
    quality = evaluate_generated_texts(samples)
    print("Generation quality:", quality.as_dict())


if __name__ == "__main__":
    main()
