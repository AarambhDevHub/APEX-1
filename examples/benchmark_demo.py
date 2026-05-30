from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch

from apex.config import get_tiny_config
from apex.eval.benchmark import run_forward_benchmark
from apex.model.apex_model import APEX1Model


def main() -> None:
    torch.manual_seed(0)
    cfg = get_tiny_config()
    cfg.validate()
    model = APEX1Model(cfg)
    input_ids = torch.randint(0, cfg.model.vocab_size, (1, 16))
    result = run_forward_benchmark(model, input_ids, warmup=1, repeats=3)
    print(result.to_markdown())


if __name__ == "__main__":
    main()
