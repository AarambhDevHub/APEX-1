# 19 — Checkpointing: Saving and Restoring Progress

> **Difficulty:** ⭐⭐☆☆☆ Beginner-Intermediate  
> **Source file:** `apex/training/checkpoint.py`  
> **You will learn:** What a checkpoint saves, why RNG state matters (BUG-13), and how to resume training exactly.

---

## 1. Why Checkpoints Are Essential

Training a large LLM takes days or weeks. If the machine crashes at step 50,000, you do not want to restart from step 0. A **checkpoint** saves everything needed to resume training from exactly where you left off.

Think of it like a **save point in a video game** — you can always reload and continue from there.

---

## 2. What Must Be Saved

To fully resume training:

| Saved Item | Why It Is Needed |
|---|---|
| Model weights | The learned parameters |
| Optimizer state | Momentum and variance estimates (m and v in AdamW) |
| Scheduler state | Current step, warmup progress |
| Load balancer biases | Expert balance state |
| Global step | Where to resume counting |
| Epoch number | Where to resume in the dataset |
| Best loss | To track improvement |
| PyTorch RNG state | For reproducible dropout, data shuffling |
| Python RNG state | For reproducible data shuffling (random.shuffle) |
| CUDA RNG state | For reproducible GPU operations |

---

## 3. BUG-13: Python RNG State Was Wrong

The original checkpoint code saved:

```python
"python": torch.get_rng_state()   # BUG: this is torch's RNG, not Python's!
```

So `checkpoint["rng"]["python"]` was actually a **second copy of PyTorch's** RNG state. When restoring, the Python `random` module was never properly restored. This meant that data shuffling was not reproducible after a resume.

**Fix:**

```python
import random
"python": random.getstate()   # Python's own RNG state
"torch": torch.get_rng_state()   # Correct PyTorch state
```

---

## 4. Full Annotated Source: `apex/training/checkpoint.py`

```python
"""
Checkpoint utilities for APEX-1.

Saves and restores:
  - Model state dict
  - Optimizer state dict
  - Scheduler state
  - Load balancer biases
  - Training metadata (step, epoch, loss)
  - Full RNG state (Python, PyTorch, CUDA)

BUG-13 FIX: Python RNG state is now saved with random.getstate()
instead of torch.get_rng_state() (which saved a duplicate PyTorch state).
"""

import logging
import random
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def save_checkpoint(
    checkpoint_dir: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,                           # CosineWarmupScheduler
    global_step: int,
    epoch: int,
    loss: float,
    load_balancer_states: Optional[dict] = None,
    best_loss: float = float("inf"),
    tag: str = "latest",
) -> Path:
    """Save a complete training checkpoint.
    
    Args:
        checkpoint_dir: Directory to save the checkpoint.
        model:          The model being trained.
        optimizer:      The optimizer.
        scheduler:      The LR scheduler.
        global_step:    Current global step count.
        epoch:          Current epoch.
        loss:           Current loss.
        load_balancer_states: Dict of {layer_idx: LoadBalancer.state_dict()}.
        best_loss:      Best loss seen so far.
        tag:            Checkpoint name suffix (e.g., 'latest', 'best', 'step10000').
    
    Returns:
        Path to the saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        # ── Model ─────────────────────────────────────────────────────
        # state_dict() is a dict of {parameter_name: tensor}
        "model_state_dict": model.state_dict(),

        # ── Optimizer ─────────────────────────────────────────────────
        # Contains momentum (m) and variance (v) for every parameter
        "optimizer_state_dict": optimizer.state_dict(),

        # ── Scheduler ─────────────────────────────────────────────────
        "scheduler_state_dict": scheduler.state_dict(),

        # ── Load Balancers ────────────────────────────────────────────
        "load_balancer_states": load_balancer_states or {},

        # ── Training Metadata ─────────────────────────────────────────
        "global_step": global_step,
        "epoch": epoch,
        "loss": loss,
        "best_loss": best_loss,

        # ── RNG States ────────────────────────────────────────────────
        # MUST save all three for fully reproducible resume
        "rng": {
            # BUG-13 FIX: Use random.getstate() for Python's RNG
            # Previously saved torch.get_rng_state() by mistake
            "python": random.getstate(),          # Python random module state
            "torch": torch.get_rng_state(),       # PyTorch CPU RNG state
            "cuda": (                             # GPU RNG state(s)
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available() else None
            ),
        },
    }

    # Save to disk
    save_path = checkpoint_dir / f"checkpoint_{tag}.pt"
    torch.save(checkpoint, save_path)

    logger.info(
        "Checkpoint saved: %s (step=%d, loss=%.4f)",
        save_path, global_step, loss
    )
    return save_path


def load_checkpoint(
    checkpoint_path: str | Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    load_balancers: Optional[dict] = None,
    map_location: str = "cpu",
) -> dict:
    """Restore a checkpoint.
    
    Args:
        checkpoint_path: Path to .pt checkpoint file.
        model:           Model to load weights into.
        optimizer:       Optimizer to restore state (or None for inference).
        scheduler:       Scheduler to restore (or None for inference).
        load_balancers:  Dict {layer_idx: LoadBalancer} to restore biases.
        map_location:    Device to load tensors onto ('cpu' or 'cuda:0', etc.)
    
    Returns:
        The full checkpoint dict (for accessing metadata).
    """
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info("Loading checkpoint: %s", path)

    # Load checkpoint dict from disk
    # map_location allows loading GPU checkpoints on CPU (and vice versa)
    checkpoint = torch.load(path, map_location=map_location)

    # ── Restore Model Weights ──────────────────────────────────────────
    # strict=True: all keys must match exactly (default)
    # strict=False: allows partial loading (for transfer learning)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    logger.info("Model weights loaded.")

    # ── Restore Optimizer State ────────────────────────────────────────
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        logger.info("Optimizer state restored.")

    # ── Restore Scheduler ─────────────────────────────────────────────
    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        logger.info("Scheduler state restored.")

    # ── Restore Load Balancers ─────────────────────────────────────────
    if load_balancers is not None and "load_balancer_states" in checkpoint:
        for layer_idx_str, state in checkpoint["load_balancer_states"].items():
            layer_idx = int(layer_idx_str)
            if layer_idx in load_balancers:
                load_balancers[layer_idx].load_state_dict(state)
        logger.info("Load balancer states restored.")

    # ── Restore RNG State ─────────────────────────────────────────────
    if "rng" in checkpoint:
        rng = checkpoint["rng"]

        # Python random module
        if "python" in rng and rng["python"] is not None:
            random.setstate(rng["python"])

        # PyTorch CPU RNG
        if "torch" in rng and rng["torch"] is not None:
            torch.set_rng_state(rng["torch"])

        # CUDA RNG (if training on GPU)
        if "cuda" in rng and rng["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["cuda"])

        logger.info("RNG states restored for reproducible training.")

    step = checkpoint.get("global_step", 0)
    loss = checkpoint.get("loss", float("inf"))
    logger.info("Resumed from step %d (loss=%.4f)", step, loss)

    return checkpoint
```

---

## 5. How Checkpointing Fits the Training Loop

```python
# In PreTrainer.train():
for batch in train_loader:
    # ... training step ...

    # Save periodically
    if global_step % save_every_n_steps == 0:
        save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            global_step=global_step,
            epoch=epoch,
            loss=current_loss,
            tag="latest",
        )

    # Save best model separately
    if current_loss < best_loss:
        best_loss = current_loss
        save_checkpoint(..., tag="best")
```

---

## 6. Resuming Training

```python
# Start or resume training:
checkpoint_path = "checkpoints/checkpoint_latest.pt"

if Path(checkpoint_path).exists():
    ckpt = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    # Continue from where we left off
    start_step = ckpt["global_step"]
    print(f"Resuming from step {start_step}")
else:
    print("Starting from scratch")
    start_step = 0
```

---

**Next:** [20 — Datasets →](20-datasets.md)
