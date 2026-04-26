"""
APEX-1 Training package.

Provides pretraining, SFT, and alignment training loops with
mixed precision, distributed training, gradient accumulation,
and checkpoint management.
"""

from apex.training.trainer import PreTrainer, SFTTrainer
from apex.training.scheduler import get_lr, CosineWarmupScheduler
from apex.training.checkpoint import save_checkpoint, load_checkpoint

__all__ = [
    "PreTrainer",
    "SFTTrainer",
    "get_lr",
    "CosineWarmupScheduler",
    "save_checkpoint",
    "load_checkpoint",
]
