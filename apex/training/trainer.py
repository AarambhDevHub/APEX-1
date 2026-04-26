"""
APEX-1 Training Loops.

Implements complete training pipelines:
1. PreTrainer — Phase 1 pretraining with multi-token auxiliary loss
2. SFTTrainer — Phase 2 supervised fine-tuning with assistant-only loss

Both support:
- AdamW optimizer with betas (0.9, 0.95), weight decay 0.1
- Cosine LR schedule with warmup
- Gradient clipping at 1.0
- LoadBalancer.update() after each optimizer step
- Mixed precision training (AMP)
- Gradient accumulation
- Distributed training (DDP/FSDP) support
- Checkpoint save/load
- WandB logging
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model
from apex.model.load_balancer import LoadBalancer
from apex.training.checkpoint import load_checkpoint, save_checkpoint
from apex.training.losses import compute_pretrain_loss, compute_sft_loss
from apex.training.scheduler import CosineWarmupScheduler

logger = logging.getLogger(__name__)


class PreTrainer:
    """Phase 1 Pretraining Trainer.

    Trains the model on next-token prediction with multi-token auxiliary
    loss. Supports mixed precision, gradient accumulation, distributed
    training, and checkpoint management.

    Args:
        model: APEX-1 model instance.
        config: APEXConfig with training hyperparameters.
        train_loader: DataLoader yielding batches of token IDs.
        val_loader: Optional validation DataLoader.
        device: Training device.
        rank: Process rank for distributed training (0 for single-GPU).
        world_size: Total number of processes.
    """

    def __init__(
        self,
        model: APEX1Model,
        config: APEXConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.rank = rank
        self.world_size = world_size

        # Device setup
        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{rank}")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        # Optimizer — AdamW with exact betas from architecture doc
        tc = config.training
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=tc.peak_lr,
            betas=(tc.beta1, tc.beta2),
            eps=tc.eps,
            weight_decay=tc.weight_decay,
        )

        # Scheduler — cosine warmup
        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=tc.warmup_steps,
            max_steps=tc.max_steps,
            min_lr_ratio=tc.min_lr_ratio,
        )

        # Mixed precision
        self.use_amp = tc.mixed_precision in ("fp16", "bf16") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.amp_dtype = torch.float16 if tc.mixed_precision == "fp16" else torch.bfloat16

        # Load balancers for MoE layers
        self.load_balancers: list[LoadBalancer] = []
        moe_layers = model.get_moe_layers()
        for layer_idx, moe_ffn in moe_layers:
            lb = LoadBalancer(
                n_experts=config.moe.n_experts,
                alpha=config.moe.balancer_alpha,
            )
            self.load_balancers.append(lb)

        # Gradient accumulation
        self.grad_accum_steps = tc.gradient_accumulation_steps

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float("inf")

        # DDP wrapper
        self.ddp_model = None
        if world_size > 1:
            self._setup_ddp()

        logger.info(
            "PreTrainer initialized: device=%s, amp=%s, grad_accum=%d, world_size=%d",
            self.device, self.use_amp, self.grad_accum_steps, world_size,
        )

    def _setup_ddp(self) -> None:
        """Set up DistributedDataParallel wrapper."""
        try:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.ddp_model = DDP(
                self.model,
                device_ids=[self.rank],
                output_device=self.rank,
            )
            logger.info("DDP initialized on rank %d", self.rank)
        except Exception as e:
            logger.warning("DDP setup failed: %s. Using single-GPU.", e)
            self.ddp_model = None

    def _get_model(self) -> nn.Module:
        """Return DDP-wrapped model if available, else raw model."""
        return self.ddp_model if self.ddp_model is not None else self.model

    def train(
        self,
        max_steps: Optional[int] = None,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 1000,
        log_interval: int = 10,
        val_interval: int = 500,
        wandb_run: Optional[Any] = None,
    ) -> dict[str, float]:
        """Run the pretraining loop.

        Args:
            max_steps: Override max training steps.
            checkpoint_dir: Directory to save checkpoints.
            checkpoint_interval: Save checkpoint every N steps.
            log_interval: Log metrics every N steps.
            val_interval: Run validation every N steps.
            wandb_run: WandB run object for logging.

        Returns:
            Dict of final training metrics.
        """
        tc = self.config.training
        max_steps = max_steps or tc.max_steps
        model = self._get_model()
        model.train()

        if checkpoint_dir:
            Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        running_loss = 0.0
        step_times: list[float] = []
        self.optimizer.zero_grad()

        logger.info("Starting pretraining for %d steps", max_steps)

        while self.global_step < max_steps:
            self.epoch += 1

            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break

                t0 = time.time()

                # Move batch to device
                token_ids = batch["input_ids"].to(self.device)

                # Forward pass with mixed precision
                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    output = model(token_ids)
                    loss, metrics = compute_pretrain_loss(
                        logits_main=output["logits"],
                        logits_speculative=output.get("spec_logits"),
                        token_ids=token_ids,
                        vocab_size=self.config.model.vocab_size,
                        lambda_spec=self.config.multi_token_head.lambda_spec,
                    )
                    loss = loss / self.grad_accum_steps

                # Backward pass
                self.scaler.scale(loss).backward()

                # Gradient accumulation
                if (self.global_step + 1) % self.grad_accum_steps == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), tc.grad_clip
                    )

                    # Optimizer step
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    # Update load balancers AFTER optimizer step
                    self._update_load_balancers()

                self.global_step += 1
                running_loss += loss.item() * self.grad_accum_steps
                step_times.append(time.time() - t0)

                # Logging
                if self.global_step % log_interval == 0 and self.rank == 0:
                    avg_loss = running_loss / log_interval
                    avg_time = sum(step_times[-log_interval:]) / min(
                        log_interval, len(step_times)
                    )
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "Step %d/%d | Loss: %.4f | LR: %.2e | Time: %.3fs/step",
                        self.global_step, max_steps, avg_loss, lr, avg_time,
                    )

                    if wandb_run is not None:
                        wandb_run.log({
                            "train/loss": avg_loss,
                            "train/lr": lr,
                            "train/step_time": avg_time,
                            "train/step": self.global_step,
                            **{f"train/{k}": v for k, v in metrics.items()},
                        })

                    running_loss = 0.0

                # Validation
                if (
                    self.val_loader is not None
                    and self.global_step % val_interval == 0
                    and self.rank == 0
                ):
                    val_loss = self._validate()
                    logger.info(
                        "Validation at step %d: loss=%.4f",
                        self.global_step, val_loss,
                    )
                    if wandb_run is not None:
                        wandb_run.log({
                            "val/loss": val_loss,
                            "train/step": self.global_step,
                        })

                # Checkpoint
                if (
                    checkpoint_dir
                    and self.global_step % checkpoint_interval == 0
                    and self.rank == 0
                ):
                    ckpt_path = Path(checkpoint_dir) / f"step_{self.global_step}.pt"
                    lb_states = [lb.state_dict() for lb in self.load_balancers]
                    save_checkpoint(
                        ckpt_path,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        step=self.global_step,
                        epoch=self.epoch,
                        loss=metrics.get("loss_total", 0.0),
                        load_balancer_state={"balancers": lb_states},
                    )

        logger.info("Pretraining complete at step %d", self.global_step)
        return {"final_step": self.global_step, "final_loss": running_loss}

    def _update_load_balancers(self) -> None:
        """Update load balancers for all MoE layers."""
        moe_layers = self.model.get_moe_layers()
        for (layer_idx, moe_ffn), lb in zip(moe_layers, self.load_balancers):
            routing_idx = moe_ffn.get_last_routing_indices()
            if routing_idx is not None:
                lb.update(routing_idx)
                moe_ffn.set_expert_bias(lb.get_bias())

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation and return average loss."""
        model = self._get_model()
        model.eval()
        total_loss = 0.0
        n_batches = 0

        for batch in self.val_loader:
            token_ids = batch["input_ids"].to(self.device)

            with torch.amp.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                output = model(token_ids)
                loss, _ = compute_pretrain_loss(
                    output["logits"],
                    output.get("spec_logits"),
                    token_ids,
                    self.config.model.vocab_size,
                )

            total_loss += loss.item()
            n_batches += 1

        model.train()
        return total_loss / max(n_batches, 1)


class SFTTrainer:
    """Phase 2 Supervised Fine-Tuning Trainer.

    Trains on instruction/response pairs with loss computed only on
    assistant tokens. Uses lower learning rate (1e-5) and fewer steps.

    Args:
        model: APEX-1 model instance (pretrained).
        config: APEXConfig with training hyperparameters.
        train_loader: DataLoader yielding batches with input_ids and token_types.
        val_loader: Optional validation DataLoader.
        device: Training device.
    """

    def __init__(
        self,
        model: APEX1Model,
        config: APEXConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.model = self.model.to(self.device)

        # SFT uses lower LR
        tc = config.training
        sft_lr = min(tc.peak_lr, 1e-5)

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=sft_lr,
            betas=(tc.beta1, tc.beta2),
            eps=tc.eps,
            weight_decay=tc.weight_decay,
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=100,
            max_steps=tc.max_steps,
            min_lr_ratio=tc.min_lr_ratio,
        )

        self.use_amp = tc.mixed_precision in ("fp16", "bf16") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.amp_dtype = torch.float16 if tc.mixed_precision == "fp16" else torch.bfloat16

        self.global_step = 0

    def train(
        self,
        max_steps: int = 5000,
        checkpoint_dir: Optional[str] = None,
        checkpoint_interval: int = 500,
        log_interval: int = 10,
        wandb_run: Optional[Any] = None,
    ) -> dict[str, float]:
        """Run the SFT training loop.

        Args:
            max_steps: Maximum training steps.
            checkpoint_dir: Directory for checkpoints.
            checkpoint_interval: Checkpoint every N steps.
            log_interval: Log every N steps.
            wandb_run: WandB run for logging.

        Returns:
            Final training metrics.
        """
        self.model.train()
        running_loss = 0.0

        logger.info("Starting SFT training for %d steps", max_steps)

        while self.global_step < max_steps:
            for batch in self.train_loader:
                if self.global_step >= max_steps:
                    break

                token_ids = batch["input_ids"].to(self.device)
                token_types = batch["token_types"].to(self.device)

                with torch.amp.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    output = self.model(token_ids)
                    loss, metrics = compute_sft_loss(
                        output["logits"],
                        token_ids,
                        token_types,
                        self.config.model.vocab_size,
                    )

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.training.grad_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                running_loss += loss.item()

                if self.global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "SFT Step %d/%d | Loss: %.4f | LR: %.2e",
                        self.global_step, max_steps, avg_loss, lr,
                    )
                    if wandb_run is not None:
                        wandb_run.log({
                            "sft/loss": avg_loss,
                            "sft/lr": lr,
                            "sft/step": self.global_step,
                        })
                    running_loss = 0.0

                if (
                    checkpoint_dir
                    and self.global_step % checkpoint_interval == 0
                ):
                    ckpt_path = Path(checkpoint_dir) / f"sft_step_{self.global_step}.pt"
                    save_checkpoint(
                        ckpt_path, self.model, self.optimizer, self.scheduler,
                        step=self.global_step, loss=metrics.get("loss_sft", 0.0),
                    )

        logger.info("SFT training complete at step %d", self.global_step)
        return {"final_step": self.global_step}
