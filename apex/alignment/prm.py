"""
Process Reward Model (PRM) for APEX-1.

Evaluates each reasoning step independently, not just the final answer.
Trained on human annotations of (step, correctness_label) pairs.

Used in GRPO reward (Section 13c) and as inference-time verifier.
Each step is scored in cumulative context (conditioning on all prior steps).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ProcessRewardModel(nn.Module):
    """Process Reward Model for step-level reasoning evaluation.

    Takes a prompt and a list of reasoning steps, returning a quality
    score for each step independently. Each score conditions on all
    previous steps (cumulative context).

    Args:
        backbone: APEX-1 model (used as feature extractor).
        d_model: Hidden dimension.
        freeze_backbone: Whether to freeze backbone parameters.
    """

    def __init__(
        self,
        backbone: nn.Module,
        d_model: int,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.step_head = nn.Linear(d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.d_model = d_model

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Compute step-level score for an input sequence.

        Args:
            input_ids: Token IDs ``[1, seq_len]`` containing prompt and steps.

        Returns:
            Scalar score in [0, 1].
        """
        output = self.backbone(input_ids, return_hidden=True)
        hidden = output["hidden_states"]  # [1, seq, d_model]
        score = self.sigmoid(self.step_head(hidden[:, -1, :]))
        return score.squeeze()

    def score_steps(
        self,
        prompt_ids: torch.Tensor,
        step_ids_list: list[torch.Tensor],
    ) -> list[float]:
        """Score each reasoning step independently.

        Each step is scored with cumulative context — conditioning on
        the prompt and all previous steps.

        Args:
            prompt_ids: Prompt token IDs ``[1, prompt_len]``.
            step_ids_list: List of token ID tensors for each step.

        Returns:
            List of float scores in [0, 1], one per step.
        """
        scores: list[float] = []
        context = prompt_ids  # [1, current_len]

        for step_ids in step_ids_list:
            # Append this step to context
            if step_ids.dim() == 1:
                step_ids = step_ids.unsqueeze(0)
            context = torch.cat([context, step_ids], dim=1)

            # Score with cumulative context
            with torch.no_grad():
                score = self.forward(context)
            scores.append(score.item())

        return scores

    def score_steps_from_text(
        self,
        prompt: str,
        steps: list[str],
        tokenizer: object,
    ) -> list[float]:
        """Score reasoning steps from text strings.

        Convenience method that handles tokenization internally.

        Args:
            prompt: Prompt text.
            steps: List of reasoning step strings.
            tokenizer: Tokenizer with encode() method.

        Returns:
            List of float scores in [0, 1].
        """
        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],
            device=next(self.parameters()).device,
        )

        step_ids_list = []
        for step in steps:
            ids = tokenizer.encode("\n" + step, add_special_tokens=False)
            step_ids_list.append(torch.tensor(ids, device=prompt_ids.device))

        return self.score_steps(prompt_ids, step_ids_list)
