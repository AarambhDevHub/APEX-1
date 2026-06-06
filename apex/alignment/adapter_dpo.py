"""
Adapter-based Direct Preference Optimization for APEX-1.

v2.9.0 adds the alignment step that naturally follows adapter fine-tuning:

    train base/SFT model -> train PEFT adapter -> align adapter with DPO

The base model remains frozen. Only LoRA/QLoRA/DoRA/QDoRA adapter parameters are
updated. The reference model stays frozen and adapter-free by default.

This module is intentionally educational:
- no external TRL dependency
- readable DPO loss
- JSONL preference dataset
- CPU-friendly tiny demos/tests
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from apex.config import APEXConfig
from apex.model.lora import count_trainable_parameters, print_peft_parameter_summary, save_lora_adapters

logger = logging.getLogger(__name__)


@dataclass
class PreferenceTensors:
    """Tokenized preference pair."""

    prompt_ids: torch.Tensor
    chosen_ids: torch.Tensor
    rejected_ids: torch.Tensor
    prompt_len: int


def _to_2d_long(ids: torch.Tensor, device: torch.device | None = None) -> torch.Tensor:
    """Normalize token IDs to ``[1, seq_len]`` long tensor."""
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2:
        raise ValueError("token IDs must have shape [seq] or [1, seq]")
    ids = ids.to(dtype=torch.long)
    if device is not None:
        ids = ids.to(device)
    return ids


def compute_response_logprob(
    model: nn.Module,
    token_ids: torch.Tensor,
    response_start_idx: int,
    prefix_len: Optional[int] = None,
    length_normalize: bool = False,
) -> torch.Tensor:
    """Compute log-probability of response tokens only.

    Args:
        model: Policy or reference model.
        token_ids: Full prompt+response IDs, shape ``[seq]`` or ``[1, seq]``.
        response_start_idx: Index where the response starts in the unshifted sequence.
        prefix_len: Optional prefix length passed to APEX attention masks.
        length_normalize: If true, return mean log-probability instead of sum.

    Returns:
        Tensor with shape ``[batch]``.
    """
    token_ids = _to_2d_long(token_ids, next(model.parameters()).device)
    if token_ids.shape[1] < 2:
        raise ValueError("DPO sequence must contain at least two tokens")

    prefix_len = int(prefix_len if prefix_len is not None else response_start_idx)
    output = model(token_ids, prefix_len=prefix_len)
    logits = output["logits"]

    shift_logits = logits[:, :-1, :]
    shift_targets = token_ids[:, 1:]
    log_probs = F.log_softmax(shift_logits, dim=-1)
    token_log_probs = log_probs.gather(2, shift_targets.unsqueeze(-1)).squeeze(-1)

    start = max(0, int(response_start_idx) - 1)
    response_log_probs = token_log_probs[:, start:]
    summed = response_log_probs.sum(dim=-1)

    if length_normalize:
        denom = max(int(response_log_probs.shape[-1]), 1)
        return summed / denom
    return summed


def adapter_dpo_loss(
    policy_model: nn.Module,
    reference_model: nn.Module,
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    prompt_len: int,
    beta: float = 0.1,
    label_smoothing: float = 0.0,
    reference_free: bool = False,
    length_normalize: bool = False,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DPO loss for PEFT adapter alignment.

    DPO compares the policy model against a frozen reference model:

        log_ratio_chosen   = log π(chosen|prompt)   - log π_ref(chosen|prompt)
        log_ratio_rejected = log π(rejected|prompt) - log π_ref(rejected|prompt)
        loss = -log σ(beta * (log_ratio_chosen - log_ratio_rejected))

    With PEFT enabled, gradients only update adapter parameters.
    """
    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    if not 0.0 <= label_smoothing < 0.5:
        raise ValueError("label_smoothing must be in [0.0, 0.5)")

    policy_device = next(policy_model.parameters()).device
    chosen_ids = _to_2d_long(chosen_ids, policy_device)
    rejected_ids = _to_2d_long(rejected_ids, policy_device)

    log_pi_chosen = compute_response_logprob(
        policy_model,
        chosen_ids,
        response_start_idx=prompt_len,
        prefix_len=prompt_len,
        length_normalize=length_normalize,
    )
    log_pi_rejected = compute_response_logprob(
        policy_model,
        rejected_ids,
        response_start_idx=prompt_len,
        prefix_len=prompt_len,
        length_normalize=length_normalize,
    )

    if reference_free:
        log_ref_chosen = torch.zeros_like(log_pi_chosen)
        log_ref_rejected = torch.zeros_like(log_pi_rejected)
    else:
        ref_device = next(reference_model.parameters()).device
        with torch.no_grad():
            log_ref_chosen = compute_response_logprob(
                reference_model,
                chosen_ids.to(ref_device),
                response_start_idx=prompt_len,
                prefix_len=prompt_len,
                length_normalize=length_normalize,
            ).to(policy_device)
            log_ref_rejected = compute_response_logprob(
                reference_model,
                rejected_ids.to(ref_device),
                response_start_idx=prompt_len,
                prefix_len=prompt_len,
                length_normalize=length_normalize,
            ).to(policy_device)

    chosen_logratio = log_pi_chosen - log_ref_chosen
    rejected_logratio = log_pi_rejected - log_ref_rejected
    logits = beta * (chosen_logratio - rejected_logratio)

    losses = (
        -(1.0 - label_smoothing) * F.logsigmoid(logits)
        - label_smoothing * F.logsigmoid(-logits)
    )
    loss = losses.mean()

    reward_chosen = beta * chosen_logratio.detach()
    reward_rejected = beta * rejected_logratio.detach()
    reward_margin = reward_chosen - reward_rejected

    metrics = {
        "dpo_loss": float(loss.detach().cpu()),
        "reward_chosen": float(reward_chosen.mean().cpu()),
        "reward_rejected": float(reward_rejected.mean().cpu()),
        "reward_margin": float(reward_margin.mean().cpu()),
        "log_pi_chosen": float(log_pi_chosen.detach().mean().cpu()),
        "log_pi_rejected": float(log_pi_rejected.detach().mean().cpu()),
        "log_ref_chosen": float(log_ref_chosen.detach().mean().cpu()),
        "log_ref_rejected": float(log_ref_rejected.detach().mean().cpu()),
        "accuracy": float((reward_chosen > reward_rejected).float().mean().cpu()),
    }
    return loss, metrics


def format_preference_example(
    tokenizer: Any,
    prompt: str,
    chosen: str,
    rejected: str,
    max_prompt_len: int = 128,
    max_response_len: int = 128,
    add_chat_template: bool = True,
) -> PreferenceTensors:
    """Convert one prompt/chosen/rejected example into tensors."""
    if add_chat_template and hasattr(tokenizer, "format_chat"):
        prompt_text = tokenizer.format_chat(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    else:
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=True)

    prompt_ids = prompt_ids[-max_prompt_len:]

    chosen_ids = tokenizer.encode(chosen, add_special_tokens=False)[:max_response_len]
    rejected_ids = tokenizer.encode(rejected, add_special_tokens=False)[:max_response_len]

    eos_id = getattr(tokenizer, "eos_token_id", None)
    if eos_id is not None:
        chosen_ids = chosen_ids + [int(eos_id)]
        rejected_ids = rejected_ids + [int(eos_id)]

    full_chosen = prompt_ids + chosen_ids
    full_rejected = prompt_ids + rejected_ids

    if len(full_chosen) < 2 or len(full_rejected) < 2:
        raise ValueError("Preference example produced too few tokens")

    return PreferenceTensors(
        prompt_ids=torch.tensor(prompt_ids, dtype=torch.long),
        chosen_ids=torch.tensor(full_chosen, dtype=torch.long),
        rejected_ids=torch.tensor(full_rejected, dtype=torch.long),
        prompt_len=len(prompt_ids),
    )


class PreferenceJSONLDataset(Dataset):
    """JSONL preference dataset.

    Supported key names:
    - prompt: ``prompt`` / ``instruction`` / ``question``
    - chosen: ``chosen`` / ``accepted`` / ``preferred``
    - rejected: ``rejected`` / ``rejected_response`` / ``dispreferred``
    """

    def __init__(
        self,
        path: str | Path,
        tokenizer: Any,
        max_prompt_len: int = 128,
        max_response_len: int = 128,
        add_chat_template: bool = True,
    ) -> None:
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"Preference data not found: {self.path}")

        self.tokenizer = tokenizer
        self.max_prompt_len = max_prompt_len
        self.max_response_len = max_response_len
        self.add_chat_template = add_chat_template
        self.rows: list[dict[str, str]] = []

        with self.path.open("r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                prompt = raw.get("prompt") or raw.get("instruction") or raw.get("question")
                chosen = raw.get("chosen") or raw.get("accepted") or raw.get("preferred")
                rejected = raw.get("rejected") or raw.get("rejected_response") or raw.get("dispreferred")
                if not prompt or not chosen or not rejected:
                    raise ValueError(
                        f"{self.path}:{line_no} must contain prompt, chosen, and rejected fields"
                    )
                self.rows.append(
                    {
                        "prompt": str(prompt),
                        "chosen": str(chosen),
                        "rejected": str(rejected),
                    }
                )

        if not self.rows:
            raise ValueError(f"No preference examples found in {self.path}")

        logger.info("Loaded %d preference examples from %s", len(self.rows), self.path)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> PreferenceTensors:
        row = self.rows[idx]
        return format_preference_example(
            self.tokenizer,
            prompt=row["prompt"],
            chosen=row["chosen"],
            rejected=row["rejected"],
            max_prompt_len=self.max_prompt_len,
            max_response_len=self.max_response_len,
            add_chat_template=self.add_chat_template,
        )


def preference_collate(batch: Iterable[PreferenceTensors]) -> list[PreferenceTensors]:
    """Keep variable-length preference examples as a list."""
    return list(batch)


def freeze_reference_model(reference_model: nn.Module) -> nn.Module:
    """Freeze and eval the DPO reference model."""
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False
    return reference_model


class AdapterDPOTrainer:
    """Train PEFT adapters with Direct Preference Optimization."""

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        config: APEXConfig,
        train_loader: DataLoader,
        device: Optional[torch.device] = None,
    ) -> None:
        if not config.peft.enabled:
            raise ValueError("AdapterDPOTrainer requires config.peft.enabled=True")
        if not config.adapter_dpo.enabled:
            raise ValueError("AdapterDPOTrainer requires config.adapter_dpo.enabled=True")

        self.config = config
        self.train_loader = train_loader

        if device is not None:
            self.device = device
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.policy_model = policy_model.to(self.device)
        self.reference_model = freeze_reference_model(reference_model.to(self.device))

        print_peft_parameter_summary(self.policy_model)

        trainable_params = [p for p in self.policy_model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable adapter parameters found for DPO")

        tc = config.training
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(tc.peak_lr),
            betas=(float(tc.beta1), float(tc.beta2)),
            eps=float(tc.eps),
            weight_decay=float(tc.weight_decay),
        )

        logger.info(
            "AdapterDPOTrainer initialized: method=%s, trainable_params=%d, beta=%.4f",
            config.peft.method,
            count_trainable_parameters(self.policy_model),
            config.adapter_dpo.beta,
        )

    def train(
        self,
        max_steps: Optional[int] = None,
        output_dir: str | Path = "runs/adapter_dpo",
    ) -> dict[str, float | int | str]:
        """Run adapter-DPO training and save ``adapter_final.pt``."""
        dpo_cfg = self.config.adapter_dpo
        max_steps = int(max_steps or self.config.training.max_steps)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        step = 0
        last_metrics: dict[str, float] = {}

        self.policy_model.train()

        while step < max_steps:
            for batch in self.train_loader:
                examples = batch if isinstance(batch, list) else [batch]

                losses = []
                metric_list = []
                for ex in examples:
                    loss, metrics = adapter_dpo_loss(
                        self.policy_model,
                        self.reference_model,
                        chosen_ids=ex.chosen_ids,
                        rejected_ids=ex.rejected_ids,
                        prompt_len=ex.prompt_len,
                        beta=dpo_cfg.beta,
                        label_smoothing=dpo_cfg.label_smoothing,
                        reference_free=dpo_cfg.reference_free,
                        length_normalize=dpo_cfg.length_normalize,
                    )
                    losses.append(loss)
                    metric_list.append(metrics)

                batch_loss = torch.stack(losses).mean()
                self.optimizer.zero_grad(set_to_none=True)
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.policy_model.parameters() if p.requires_grad],
                    self.config.training.grad_clip,
                )
                self.optimizer.step()

                step += 1
                last_metrics = _mean_metrics(metric_list)
                last_metrics["loss"] = float(batch_loss.detach().cpu())

                if step % 10 == 0 or step == 1 or step == max_steps:
                    logger.info(
                        "DPO step %d/%d | loss=%.4f | margin=%.4f | acc=%.2f",
                        step,
                        max_steps,
                        last_metrics.get("loss", 0.0),
                        last_metrics.get("reward_margin", 0.0),
                        last_metrics.get("accuracy", 0.0),
                    )

                if dpo_cfg.save_every_steps and step % dpo_cfg.save_every_steps == 0:
                    save_lora_adapters(
                        self.policy_model,
                        output_dir / f"adapter_step_{step}.pt",
                        peft_config=self.config.peft,
                    )

                if step >= max_steps:
                    break

        final_path = output_dir / "adapter_final.pt"
        save_lora_adapters(self.policy_model, final_path, peft_config=self.config.peft)

        return {
            "steps": step,
            "adapter_path": str(final_path),
            **last_metrics,
        }


def _mean_metrics(metrics: list[dict[str, float]]) -> dict[str, float]:
    if not metrics:
        return {}
    keys = metrics[0].keys()
    return {key: float(sum(m[key] for m in metrics) / len(metrics)) for key in keys}
