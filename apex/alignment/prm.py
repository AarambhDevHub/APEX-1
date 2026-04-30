"""
Process Reward Model (PRM) for APEX-1.

Fix BUG-06: ``score_steps_from_text`` previously called
``tokenizer.encode()`` unconditionally, crashing with ``AttributeError:
'NoneType' object has no attribute 'encode'`` when ``None`` was passed
(as in ``combined_reward.py``).

The method now accepts a ``None`` tokenizer and raises a clear
``ValueError`` with guidance.  A new companion method
``score_steps_from_text_pretokenized`` allows callers to pass
pre-tokenised step IDs directly, which is what ``combined_reward.py``
should use.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ProcessRewardModel(nn.Module):
    """Process Reward Model for step-level reasoning evaluation.

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

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute step-level score for an input sequence.

        Args:
            input_ids: Token IDs ``[1, seq_len]``.

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

        Args:
            prompt_ids: Prompt token IDs ``[1, prompt_len]``.
            step_ids_list: List of token ID tensors for each step.

        Returns:
            List of float scores in [0, 1].
        """
        scores: list[float] = []
        context = prompt_ids

        for step_ids in step_ids_list:
            if step_ids.dim() == 1:
                step_ids = step_ids.unsqueeze(0)
            context = torch.cat([context, step_ids], dim=1)

            with torch.no_grad():
                score = self.forward(context)
            scores.append(score.item())

        return scores

    def score_steps_from_text(
        self,
        prompt: str,
        steps: list[str],
        tokenizer: Optional[object],
    ) -> list[float]:
        """Score reasoning steps from text strings.

        BUG-06 FIX: When ``tokenizer`` is ``None`` this method now raises a
        clear ``ValueError`` instead of crashing with ``AttributeError``.
        Callers that do not have a tokenizer at hand should use
        ``score_steps_from_text_pretokenized`` or pass a real tokenizer.

        Args:
            prompt: Prompt text.
            steps: List of reasoning step strings.
            tokenizer: Tokenizer with ``encode()`` method, or ``None``
                (raises ``ValueError`` — see note above).

        Returns:
            List of float scores in [0, 1].

        Raises:
            ValueError: If ``tokenizer`` is ``None``.
        """
        # BUG-06 FIX: guard against None tokenizer with an informative message.
        if tokenizer is None:
            raise ValueError(
                "score_steps_from_text requires a real tokenizer. "
                "Pass a tokenizer instance or call score_steps_from_text_pretokenized "
                "with pre-encoded IDs instead."
            )

        device = next(self.parameters()).device

        prompt_ids = torch.tensor(
            [tokenizer.encode(prompt, add_special_tokens=False)],  # type: ignore[attr-defined]
            device=device,
        )

        step_ids_list = []
        for step in steps:
            ids = tokenizer.encode(  # type: ignore[attr-defined]
                "\n" + step, add_special_tokens=False
            )
            step_ids_list.append(torch.tensor(ids, device=device))

        return self.score_steps(prompt_ids, step_ids_list)

    def score_steps_from_text_pretokenized(
        self,
        prompt_ids: torch.Tensor,
        step_texts: list[str],
        tokenizer: object,
    ) -> list[float]:
        """Score reasoning steps from text using a caller-supplied tokenizer.

        Convenience wrapper that tokenises each step string and calls
        ``score_steps``.  This is the recommended path for code that
        already has a tokenizer (e.g. ``combined_reward.py``).

        Args:
            prompt_ids: Pre-tokenised prompt ``[1, prompt_len]``.
            step_texts: List of reasoning step strings.
            tokenizer: Tokenizer with ``encode()`` method.

        Returns:
            List of float scores in [0, 1].
        """
        device = next(self.parameters()).device
        step_ids_list = []
        for step in step_texts:
            ids = tokenizer.encode(  # type: ignore[attr-defined]
                "\n" + step, add_special_tokens=False
            )
            step_ids_list.append(torch.tensor(ids, device=device))
        return self.score_steps(prompt_ids.to(device), step_ids_list)
