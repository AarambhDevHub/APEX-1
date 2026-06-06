"""
PEFT fine-tuning trainer for APEX-1.

This trainer is intentionally separate from the normal SFT trainer so learners
can see the PEFT workflow clearly:

1. Load/freeze base model
2. Inject PEFT adapters
3. Train only adapter parameters
4. Save adapter-only checkpoint
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model
from apex.model.lora import (
    count_trainable_parameters,
    print_peft_parameter_summary,
    save_lora_adapters,
)
from apex.training.losses import compute_sft_loss
from apex.training.scheduler import CosineWarmupScheduler

logger = logging.getLogger(__name__)


class PEFTSFTTrainer:
    """Supervised fine-tuning trainer for PEFT adapters."""

    def __init__(
        self,
        model: APEX1Model,
        config: APEXConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        if not config.peft.enabled:
            raise ValueError("PEFTSFTTrainer requires config.peft.enabled=True")

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
        print_peft_parameter_summary(self.model)

        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable PEFT parameters found")

        tc = config.training
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=tc.peak_lr,
            betas=(tc.beta1, tc.beta2),
            eps=tc.eps,
            weight_decay=tc.weight_decay,
        )

        self.scheduler = CosineWarmupScheduler(
            self.optimizer,
            warmup_steps=tc.warmup_steps,
            max_steps=tc.max_steps,
            min_lr_ratio=tc.min_lr_ratio,
        )

        self.use_amp = tc.mixed_precision in ("fp16", "bf16") and torch.cuda.is_available()
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.use_amp)
        self.amp_dtype = torch.float16 if tc.mixed_precision == "fp16" else torch.bfloat16
        self.global_step = 0

        logger.info(
            "PEFTSFTTrainer initialized: device=%s, trainable_params=%d",
            self.device,
            count_trainable_parameters(self.model),
        )

    def train(
        self,
        max_steps: Optional[int] = None,
        output_dir: str | Path = "checkpoints/lora",
        checkpoint_interval: int = 100,
        log_interval: int = 10,
        wandb_run: Optional[Any] = None,
    ) -> dict[str, float]:
        """Run PEFT SFT training."""
        max_steps = max_steps or self.config.training.max_steps
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.model.train()
        running_loss = 0.0
        last_loss = 0.0

        logger.info("Starting PEFT SFT for %d steps", max_steps)

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
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.training.grad_clip,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self.optimizer.zero_grad()

                self.global_step += 1
                last_loss = float(loss.item())
                running_loss += last_loss

                if self.global_step % log_interval == 0:
                    avg_loss = running_loss / log_interval
                    lr = self.optimizer.param_groups[0]["lr"]
                    logger.info(
                        "PEFT SFT Step %d/%d | Loss: %.4f | LR: %.2e",
                        self.global_step,
                        max_steps,
                        avg_loss,
                        lr,
                    )
                    if wandb_run is not None:
                        wandb_run.log(
                            {
                                "peft_sft/loss": avg_loss,
                                "peft_sft/lr": lr,
                                "peft_sft/step": self.global_step,
                                **{f"peft_sft/{k}": v for k, v in metrics.items()},
                            }
                        )
                    running_loss = 0.0

                if self.global_step % checkpoint_interval == 0:
                    adapter_path = output_dir / f"adapter_step_{self.global_step}.pt"
                    save_lora_adapters(self.model, adapter_path, self.config.peft)

        final_path = output_dir / "adapter_final.pt"
        save_lora_adapters(self.model, final_path, self.config.peft)

        logger.info("PEFT SFT complete at step %d", self.global_step)
        return {
            "final_step": float(self.global_step),
            "final_loss": float(last_loss),
            "adapter_path": str(final_path),
        }
