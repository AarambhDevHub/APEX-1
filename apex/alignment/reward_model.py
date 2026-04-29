"""
Reward Model for APEX-1 RLHF Alignment.

Implements a reward model that scores responses on a scalar scale.
The model replaces the LM head with a regression head and is trained
on human preference data using the Bradley-Terry loss.

Architecture: SFT backbone + Linear(d_model, 1) reward head
Loss: -log(sigmoid(r_chosen - r_rejected))

Fix BUG-05: ``Optional`` is now imported at the top of the file instead
of at the bottom, which previously caused a ``NameError`` when
``forward()`` was called before the import was reached.
"""

from __future__ import annotations

# BUG-05 FIX: ``Optional`` must be imported BEFORE it is used in the
# ``forward()`` signature below.  The original code placed this import at
# the very end of the file, causing a ``NameError`` at class-definition
# time in Python 3.10+ (annotations are evaluated eagerly in some paths).
from typing import Optional

import torch
import torch.nn as nn

from apex.model.apex_model import APEX1Model


class RewardModel(nn.Module):
    """Reward model for RLHF preference learning.

    Takes an APEX-1 model backbone and adds a scalar reward head.
    Trained with Bradley-Terry preference loss on (chosen, rejected) pairs.

    Args:
        backbone: APEX-1 model (LM head is not used).
        d_model: Hidden dimension of the backbone.
        freeze_backbone: If True, freeze backbone weights during training.
    """

    def __init__(
        self,
        backbone: APEX1Model,
        d_model: int,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.reward_head = nn.Linear(d_model, 1, bias=False)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute scalar reward for input sequences.

        Uses the last non-padded token's hidden state as the
        sequence representation.

        Args:
            input_ids: Token IDs ``[batch, seq_len]``.
            attention_mask: Optional mask ``[batch, seq_len]``.

        Returns:
            Scalar rewards ``[batch]``.
        """
        output = self.backbone(input_ids, return_hidden=True)
        hidden = output["hidden_states"]  # [batch, seq, d_model]

        # Use the last non-padded token's hidden state
        if attention_mask is not None:
            # Find last non-zero position per sequence
            lengths = attention_mask.sum(dim=1).long() - 1
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            last_hidden = hidden[batch_idx, lengths, :]
        else:
            last_hidden = hidden[:, -1, :]

        reward = self.reward_head(last_hidden).squeeze(-1)  # [batch]
        return reward


def reward_model_loss(
    reward_chosen: torch.Tensor,
    reward_rejected: torch.Tensor,
) -> torch.Tensor:
    """Bradley-Terry preference loss.

    The chosen response should score higher than the rejected one.

    Args:
        reward_chosen: Rewards for chosen responses ``[batch]``.
        reward_rejected: Rewards for rejected responses ``[batch]``.

    Returns:
        Scalar loss value.
    """
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()