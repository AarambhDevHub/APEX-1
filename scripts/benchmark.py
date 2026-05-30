from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch

from apex.config import APEXConfig, get_tiny_config, get_tiny_vision_config
from apex.eval.benchmark import run_forward_benchmark
from apex.model.apex_model import APEX1Model
from apex.model.apex_vision_model import APEX1VisionModel


def load_config(path: str | None, vision: bool) -> APEXConfig:
    if path:
        cfg = APEXConfig.from_yaml(path)
    else:
        cfg = get_tiny_vision_config() if vision else get_tiny_config()
    if vision:
        cfg.vision.enabled = True
    cfg.validate()
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny APEX-1 forward-pass benchmark.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path. Defaults to tiny config.")
    parser.add_argument("--vision", action="store_true", help="Benchmark APEX1VisionModel.")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    cfg = load_config(args.config, args.vision)
    device = torch.device(args.device)
    model = APEX1VisionModel(cfg) if args.vision else APEX1Model(cfg)

    input_ids = torch.randint(0, cfg.model.vocab_size, (args.batch_size, args.seq_len))
    pixel_values = None
    if args.vision:
        # Ensure one image placeholder is present so visual tokens replace it.
        input_ids[:, 1] = cfg.vision.image_token_id
        pixel_values = torch.randn(
            args.batch_size,
            cfg.vision.in_channels,
            cfg.vision.image_size,
            cfg.vision.image_size,
        )

    result = run_forward_benchmark(
        model,
        input_ids=input_ids,
        pixel_values=pixel_values,
        warmup=args.warmup,
        repeats=args.repeats,
        device=device,
    )
    print(result.to_markdown())


if __name__ == "__main__":
    main()
