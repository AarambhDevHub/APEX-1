"""
Learning Rate Scheduler for APEX-1.

Implements the cosine warmup + decay schedule from Section 12a:
    Warmup:       0 → warmup_steps    linear ramp 0 → peak_lr
    Cosine decay: warmup → max_steps  cosine from peak_lr → min_lr

Peak LR by model size:
    Small (100M):  3e-4
    Medium (7B):   1e-4
    Large (900B):  3e-5
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR


def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    peak_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute learning rate at a given step.

    Uses linear warmup followed by cosine decay to min_lr.

    Args:
        step: Current training step.
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        peak_lr: Maximum learning rate after warmup.
        min_lr_ratio: Minimum LR as fraction of peak_lr (default: 0.1).

    Returns:
        Learning rate for the current step.
    """
    if step < warmup_steps:
        # Linear warmup
        return peak_lr * (step / max(warmup_steps, 1))

    if step >= max_steps:
        return peak_lr * min_lr_ratio

    # Cosine decay
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine)


class CosineWarmupScheduler(LambdaLR):
    """PyTorch LR scheduler with linear warmup and cosine decay.

    Wraps the ``get_lr`` function into a proper PyTorch scheduler
    compatible with the training loop.

    Args:
        optimizer: PyTorch optimizer.
        warmup_steps: Number of warmup steps.
        max_steps: Total training steps.
        min_lr_ratio: Minimum LR as fraction of initial LR.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio

        def lr_lambda(step: int) -> float:
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            if step >= max_steps:
                return min_lr_ratio
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
            return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

        super().__init__(optimizer, lr_lambda)
