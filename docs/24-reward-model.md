# 24 — Reward Model: What Humans Prefer

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/alignment/reward_model.py`  
> **You will learn:** Why RLHF needs a reward model, the Bradley-Terry loss, and BUG-05.

---

## 1. The Alignment Problem

After pretraining and SFT, the model can generate fluent text and follow instructions. But it does not know what humans actually *prefer* — it might still generate responses that are technically accurate but unhelpful, verbose, or unsafe.

**RLHF (Reinforcement Learning from Human Feedback)** teaches the model to maximise human preferences. The core component is a **reward model** that scores any response.

---

## 2. Training Data: Human Preference Pairs

Human annotators compare pairs of model responses and indicate which is better:

```
Prompt:   "Explain recursion."
Response A: "Recursion is when a function calls itself..."
Response B: "In simple terms, imagine Russian nesting dolls..."

Human preference: B > A  (clearer, more accessible)
```

The reward model learns to assign higher scores to preferred responses.

---

## 3. Architecture

The reward model is built on top of a trained language model backbone:

```
Input: prompt + response (as token IDs)
         ↓
   APEX1Model backbone (frozen or lightly trained)
         ↓
   Last token's hidden state [d_model]
         ↓
   Linear(d_model, 1)  ← the reward head
         ↓
   Scalar reward score (higher = better response)
```

The **last token's hidden state** represents the entire sequence — it has seen all previous tokens through attention and encodes a summary.

---

## 4. The Bradley-Terry Loss

The Bradley-Terry model from statistics says: given two items $i$ and $j$ with scores $r_i$ and $r_j$, the probability that $i$ is preferred over $j$ is:

$$P(i \succ j) = \sigma(r_i - r_j) = \frac{1}{1 + e^{-(r_i - r_j)}}$$

The loss is the negative log-likelihood:

$$L = -\log P(\text{chosen} \succ \text{rejected}) = -\log \sigma(r_{\text{chosen}} - r_{\text{rejected}})$$

**When does this loss equal zero?**
- When $r_{chosen} \gg r_{rejected}$ — we correctly score chosen much higher than rejected

**When is the loss large?**
- When $r_{chosen} \approx r_{rejected}$ — we cannot tell which is better
- When $r_{chosen} < r_{rejected}$ — we incorrectly prefer the rejected response

---

## 5. BUG-05: Optional Import at the Bottom

The original `reward_model.py` had:

```python
class RewardModel(nn.Module):
    def forward(self, input_ids, attention_mask: Optional[torch.Tensor] = None):
        # ... code uses Optional ...

from typing import Optional   # BUG-05: this import is AFTER the class definition!
```

In Python 3.10+, type annotations in function signatures are evaluated at class definition time. `Optional` was referenced before it was imported → `NameError` when the module was loaded.

**Fix:** Move all imports to the top of the file (standard Python practice).

---

## 6. Full Annotated Source: `apex/alignment/reward_model.py`

```python
"""
Reward Model for RLHF Alignment.

Architecture: SFT backbone + Linear(d_model, 1) reward head.
Loss: Bradley-Terry preference loss.

BUG-05 FIX: Optional is now imported at the TOP of the file,
before it is used in the class definition.
"""

from __future__ import annotations
from typing import Optional   # BUG-05 FIX: import at top!

import torch
import torch.nn as nn
from apex.model.apex_model import APEX1Model


class RewardModel(nn.Module):
    """Reward model for RLHF preference learning.
    
    Wraps an APEX-1 backbone with a scalar reward head.
    Trained on (chosen, rejected) pairs with Bradley-Terry loss.
    
    Args:
        backbone:         Pre-trained APEX1Model (SFT checkpoint).
        d_model:          Hidden dimension of the backbone.
        freeze_backbone:  If True, only train the reward head (faster).
    """

    def __init__(self, backbone: APEX1Model, d_model: int, freeze_backbone: bool = False):
        super().__init__()
        self.backbone = backbone

        # Reward head: maps last hidden state to a scalar score
        # No bias (the scale is set by training data, not an offset)
        self.reward_head = nn.Linear(d_model, 1, bias=False)

        if freeze_backbone:
            # Freeze all backbone parameters (only train reward_head)
            # Useful for fast reward model training on small datasets
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute a scalar reward for each sequence in the batch.
        
        Args:
            input_ids:      Token IDs [batch, seq_len].
            attention_mask: Optional mask [batch, seq_len] (1=real, 0=pad).
        
        Returns:
            Scalar rewards [batch]. Higher = more preferred.
        """
        # Get hidden states from the backbone
        # return_hidden=True adds hidden_states to the output dict
        output = self.backbone(input_ids, return_hidden=True)
        hidden = output["hidden_states"]   # [batch, seq, d_model]

        # Use the LAST non-padded token's hidden state as the sequence summary
        # This is a standard approach for sequence classification tasks
        if attention_mask is not None:
            # Find the last real token position in each sequence
            # attention_mask.sum(dim=1): count of real tokens per sequence
            # - 1: convert count to index of last token
            lengths = attention_mask.sum(dim=1).long() - 1   # [batch]
            batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
            last_hidden = hidden[batch_idx, lengths, :]   # [batch, d_model]
        else:
            # No mask: assume all tokens are real, use the last one
            last_hidden = hidden[:, -1, :]   # [batch, d_model]

        # Project to scalar reward
        reward = self.reward_head(last_hidden).squeeze(-1)   # [batch]
        return reward


def reward_model_loss(
    reward_chosen: torch.Tensor,    # [batch] — rewards for chosen responses
    reward_rejected: torch.Tensor,  # [batch] — rewards for rejected responses
) -> torch.Tensor:
    """Compute Bradley-Terry preference loss.
    
    The model should assign higher reward to chosen responses.
    
    Loss = -log(σ(r_chosen - r_rejected))
    
    Perfect:   r_chosen >> r_rejected → loss ≈ 0
    Confused:  r_chosen ≈ r_rejected  → loss ≈ log(2) ≈ 0.693
    Wrong:     r_chosen << r_rejected → loss → ∞
    
    Args:
        reward_chosen:   Reward scores for preferred responses.
        reward_rejected: Reward scores for rejected responses.
    
    Returns:
        Scalar mean loss.
    """
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

---

## 7. Training the Reward Model

```python
# Example reward model training step
reward_model = RewardModel(backbone=sft_model, d_model=512)
optimizer = torch.optim.AdamW(reward_model.parameters(), lr=1e-5)

for batch in preference_loader:
    # batch contains: prompt_ids, chosen_ids, rejected_ids

    # Score both responses
    r_chosen = reward_model(batch["chosen_ids"])
    r_rejected = reward_model(batch["rejected_ids"])

    # Bradley-Terry loss: chosen should score higher
    loss = reward_model_loss(r_chosen, r_rejected)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    # Monitor: how often does the model correctly prefer chosen?
    accuracy = (r_chosen > r_rejected).float().mean()
    print(f"Reward accuracy: {accuracy:.2%}")
```

**Goal:** Achieve > 70–80% accuracy on a held-out preference set before using the reward model in RLHF.

---

**Next:** [25 — Direct Preference Optimization (DPO) →](25-dpo.md)
