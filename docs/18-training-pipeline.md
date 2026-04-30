# 18 — Training Pipeline: The Training Loop

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/training/trainer.py`  
> **You will learn:** How the full training loop works step-by-step, mixed precision, gradient accumulation, DDP, and the BUG-11 load balancer fix.

---

## 1. What Happens in One Training Step

```
1. Get a batch of token sequences from the DataLoader
2. Forward pass: model(input) → logits
3. Compute loss: cross_entropy(logits, targets)
4. Backward pass: loss.backward() → compute gradients
5. Clip gradients (prevent exploding)
6. optimizer.step() → update weights
7. scheduler.step() → update learning rate
8. load_balancer.update() → update expert biases
9. optimizer.zero_grad() → clear gradients for next step
```

This loop repeats for `max_steps` iterations.

---

## 2. Mixed Precision Training

Normally, PyTorch uses 32-bit floats (float32 = 4 bytes per number). With a 7B parameter model, storing all parameters and gradients in float32 requires ~56 GB — far more than most GPUs have.

**Mixed precision** uses 16-bit floats (float16 or bfloat16 = 2 bytes) for most computations:

```
Weights in memory: float32 (stable "master copy")
Forward/backward:  float16 (fast, uses half the memory)
Gradient update:   float32 (maintain precision for small updates)
```

**GradScaler:** When using float16, tiny gradients can underflow to zero. PyTorch's `GradScaler` multiplies the loss by a large scale factor (e.g., 65536) before backward, then divides it back after — keeping gradient magnitudes in the float16 representable range.

---

## 3. Gradient Accumulation

**Problem:** A batch size of 32 might not fit in GPU memory for a large model.

**Solution:** Process `N` small micro-batches, accumulate gradients without stepping, then update once. This simulates a batch of `N × micro_batch_size`.

```python
for step in range(gradient_accumulation_steps):
    loss = model(micro_batch) / gradient_accumulation_steps
    loss.backward()   # accumulate gradients

optimizer.step()      # update once after all micro-batches
```

Dividing by `gradient_accumulation_steps` ensures the effective loss scale is the same as a real large batch.

---

## 4. Gradient Clipping

Sometimes a bad batch produces very large gradients that would cause an enormous weight update, destabilising training. **Gradient clipping** scales down the gradient vector if its norm exceeds a threshold:

$$g \leftarrow g \times \frac{\text{clip\_norm}}{\max(\|g\|, \text{clip\_norm})}$$

For APEX-1: `grad_clip = 1.0`. If the gradient norm > 1.0, it is rescaled to exactly 1.0.

---

## 5. The BUG-11 Fix: Per-Layer Expert Count

When creating load balancers, the original code used `config.moe.n_experts` (a global default). But some layers might have different expert counts. The fix:

```python
# BUG-11 FIX: use moe_ffn.n_experts — the actual count for this layer
for layer_idx, block in enumerate(model.blocks):
    if hasattr(block.ffn, "n_experts"):
        moe_ffn = block.ffn
        # Use the actual expert count from the built MoEFFN, not config
        load_balancers[layer_idx] = LoadBalancer(
            n_experts=moe_ffn.n_experts,   # ← real count
            alpha=config.moe.balancer_alpha,
        )
```

---

## 6. Full Annotated Source: `PreTrainer`

```python
"""
Training infrastructure for APEX-1.

PreTrainer:  Pretraining on raw token sequences.
SFTTrainer:  Supervised fine-tuning on chat conversations.

BUG-11 FIX: LoadBalancers use per-layer n_experts.
"""

import logging
from typing import Optional
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model
from apex.model.load_balancer import LoadBalancer
from apex.training.losses import compute_pretrain_loss
from apex.training.scheduler import CosineWarmupScheduler

logger = logging.getLogger(__name__)


class PreTrainer:
    """Pretraining pipeline for APEX-1.
    
    Args:
        model:      APEX1Model to train.
        config:     Full APEXConfig.
        train_loader: DataLoader providing batches of token_ids.
        device:     Training device ('cuda' or 'cpu').
        checkpoint_dir: Where to save checkpoints.
    """

    def __init__(self, model, config, train_loader, device="cuda", checkpoint_dir="checkpoints"):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        t = config.training

        # ── Optimizer ────────────────────────────────────────────────────
        # Separate parameter groups: no weight decay for biases and norms
        decay_params = []
        no_decay_params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                if "bias" in name or "norm" in name or "embedding" in name:
                    no_decay_params.append(param)
                else:
                    decay_params.append(param)

        self.optimizer = torch.optim.AdamW(
            [
                {"params": decay_params, "weight_decay": t.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=t.peak_lr,
            betas=(t.beta1, t.beta2),
            eps=t.eps,
        )

        # ── Scheduler ────────────────────────────────────────────────────
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            peak_lr=t.peak_lr,
            warmup_steps=t.warmup_steps,
            max_steps=t.max_steps,
            min_lr_ratio=config.training.min_lr_ratio,
        )

        # ── Mixed Precision ───────────────────────────────────────────────
        use_amp = t.mixed_precision in ("fp16", "bf16") and device == "cuda"
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16 if t.mixed_precision == "bf16" else torch.float16
        # GradScaler prevents gradient underflow in fp16
        self.scaler = GradScaler(enabled=use_amp and t.mixed_precision == "fp16")

        # ── Load Balancers (BUG-11 FIX) ──────────────────────────────────
        # One LoadBalancer per MoE layer, using ACTUAL per-layer n_experts
        self.load_balancers: dict[int, LoadBalancer] = {}
        for layer_idx, block in enumerate(model.blocks):
            if hasattr(block.ffn, "n_experts"):
                moe_ffn = block.ffn
                # BUG-11: use moe_ffn.n_experts, not config.moe.n_experts
                self.load_balancers[layer_idx] = LoadBalancer(
                    n_experts=moe_ffn.n_experts,
                    alpha=config.moe.balancer_alpha,
                )

        self.global_step = 0

    def train(self, max_steps: Optional[int] = None) -> None:
        """Run the training loop.
        
        Args:
            max_steps: Stop after this many steps (or config.training.max_steps).
        """
        max_steps = max_steps or self.config.training.max_steps
        t = self.config.training
        accum_steps = t.gradient_accumulation_steps

        self.model.train()
        self.model.to(self.device)

        logger.info("Starting pretraining for %d steps", max_steps)
        running_loss = 0.0

        for batch in self.train_loader:
            if self.global_step >= max_steps:
                break

            # Move batch to device
            token_ids = batch["input_ids"].to(self.device)
            attention_mask = batch.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)

            # ── Forward pass (with optional AMP) ─────────────────────────
            with autocast(dtype=self.amp_dtype, enabled=self.use_amp):
                output = self.model(token_ids)
                loss, metrics = compute_pretrain_loss(
                    logits=output["logits"],
                    token_ids=token_ids,
                    spec_logits=output.get("spec_logits"),
                    attention_mask=attention_mask,
                    lambda_spec=self.config.multi_token_head.lambda_spec,
                )
                # Divide by accumulation steps for gradient accumulation
                loss = loss / accum_steps

            # ── Backward pass ─────────────────────────────────────────────
            # GradScaler scales up loss to prevent fp16 underflow
            self.scaler.scale(loss).backward()

            running_loss += loss.item() * accum_steps

            # ── Optimizer step (every accum_steps batches) ────────────────
            if (self.global_step + 1) % accum_steps == 0:
                # Unscale gradients (needed before clip)
                self.scaler.unscale_(self.optimizer)

                # Gradient clipping: prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.grad_clip,
                )

                # Update weights
                self.scaler.step(self.optimizer)
                self.scaler.update()

                # Update LR schedule
                self.scheduler.step()

                # Clear gradients for next accumulation window
                self.optimizer.zero_grad()

                # ── Load balancer update (AFTER optimizer.step) ──────────
                # No gradient should flow through this!
                for layer_idx, balancer in self.load_balancers.items():
                    # In a real implementation, track expert_indices during forward
                    # Here: placeholder (real code passes expert routing indices)
                    # balancer.update(expert_indices[layer_idx])
                    # balancer.bias is then pushed to the MoE layer:
                    self.model.blocks[layer_idx].ffn.set_expert_bias(balancer.bias)

                # Logging
                if self.global_step % 100 == 0:
                    avg_loss = running_loss / 100
                    lr = self.scheduler.get_lr()
                    logger.info(
                        "Step %d | loss=%.4f | lr=%.2e",
                        self.global_step, avg_loss, lr
                    )
                    running_loss = 0.0

            self.global_step += 1
```

---

## 7. SFTTrainer Differences

`SFTTrainer` is almost identical to `PreTrainer` but with two key differences:

1. **Loss:** Uses `compute_sft_loss()` (only on assistant tokens)
2. **Learning Rate Cap:** `peak_lr = min(config_lr, 1e-5)` — SFT uses a lower LR to avoid catastrophic forgetting

```python
class SFTTrainer(PreTrainer):
    """Supervised Fine-Tuning trainer."""

    def __init__(self, model, config, train_loader, device="cuda", ...):
        super().__init__(model, config, train_loader, device, ...)
        
        # SFT uses lower LR to preserve pretrained knowledge
        sft_lr = min(config.training.peak_lr, 1e-5)
        for pg in self.optimizer.param_groups:
            pg["lr"] = sft_lr
        # Also update scheduler's peak_lr
        self.scheduler.peak_lr = sft_lr
        self.scheduler.min_lr = sft_lr * config.training.min_lr_ratio
```

---

**Next:** [19 — Checkpointing →](19-checkpointing.md)
