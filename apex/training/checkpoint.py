"""
Checkpoint Save/Load for APEX-1.

Handles saving and restoring complete training state:
- Model parameters
- Optimizer state
- LR scheduler state
- Training step count
- Load balancer state
- RNG states for reproducibility

Fix BUG-13: Both "python" and "cpu" entries previously stored the same
``torch.random.get_rng_state()`` value.  "python" now correctly stores
the Python ``random`` module state via ``random.getstate()``.
"""

from __future__ import annotations

import logging
import random  # BUG-13 FIX: needed for Python RNG state
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    step: int = 0,
    epoch: int = 0,
    loss: float = 0.0,
    load_balancer_state: Optional[dict] = None,
    extra: Optional[dict] = None,
) -> None:
    """Save a training checkpoint.

    Args:
        path: File path to save the checkpoint.
        model: The model to save.
        optimizer: Optimizer state to save.
        scheduler: LR scheduler state to save.
        step: Current training step.
        epoch: Current epoch.
        loss: Current loss value.
        load_balancer_state: Load balancer state dict.
        extra: Any extra metadata to save.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: dict[str, Any] = {
        "step": step,
        "epoch": epoch,
        "loss": loss,
        "model_state_dict": model.state_dict(),
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    if load_balancer_state is not None:
        checkpoint["load_balancer_state"] = load_balancer_state

    # BUG-13 FIX: "python" now stores the Python random module state
    # (random.getstate()), not a duplicate of the PyTorch CPU RNG state.
    checkpoint["rng_states"] = {
        "python": random.getstate(),  # ← Python random module
        "cpu": torch.random.get_rng_state(),  # ← PyTorch CPU RNG
    }
    if torch.cuda.is_available():
        checkpoint["rng_states"]["cuda"] = torch.cuda.get_rng_state_all()

    if extra is not None:
        checkpoint["extra"] = extra

    torch.save(checkpoint, path)
    logger.info(
        "Checkpoint saved to %s (step=%d, epoch=%d, loss=%.4f)",
        path,
        step,
        epoch,
        loss,
    )


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    map_location: Optional[str] = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Load a training checkpoint.

    Args:
        path: Path to the checkpoint file.
        model: Model to load state into.
        optimizer: Optimizer to restore state.
        scheduler: Scheduler to restore state.
        map_location: Device mapping for torch.load.
        strict: Whether to strictly enforce state_dict key matching.

    Returns:
        Dict with checkpoint metadata (step, epoch, loss, etc.).

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if map_location is None:
        map_location = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = torch.load(path, map_location=map_location, weights_only=False)

    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"], strict=strict)

    # Load optimizer state
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # Load scheduler state
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    # BUG-13 FIX: Restore Python random state separately from PyTorch CPU state.
    if "rng_states" in checkpoint:
        rng = checkpoint["rng_states"]
        if "python" in rng:
            # Handle both old checkpoints (torch Tensor) and new ones (tuple)
            py_state = rng["python"]
            if isinstance(py_state, tuple):
                random.setstate(py_state)
            # Old checkpoints stored a torch state here — skip silently
        if "cpu" in rng:
            cpu_state = rng["cpu"]
            if isinstance(cpu_state, torch.Tensor):
                torch.random.set_rng_state(cpu_state)
        if "cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

    logger.info(
        "Checkpoint loaded from %s (step=%d, epoch=%d, loss=%.4f)",
        path,
        checkpoint.get("step", 0),
        checkpoint.get("epoch", 0),
        checkpoint.get("loss", 0.0),
    )

    return {
        "step": checkpoint.get("step", 0),
        "epoch": checkpoint.get("epoch", 0),
        "loss": checkpoint.get("loss", 0.0),
        "load_balancer_state": checkpoint.get("load_balancer_state"),
        "extra": checkpoint.get("extra"),
    }
