# 13 — Multi-Token Prediction Head

> **Difficulty:** ⭐⭐☆☆☆ Intermediate  
> **Source file:** `apex/model/multi_token_head.py`  
> **You will learn:** Why predicting multiple tokens helps training, how speculative decoding uses the heads, and the auxiliary loss weight.

---

## 1. Standard Training: Predict One Token at a Time

Standard language model training: given tokens $[t_1, t_2, \ldots, t_n]$, predict $[t_2, t_3, \ldots, t_{n+1}]$.

Each position in the sequence produces one gradient signal. This is fine, but wasteful — the model only receives one signal per token position.

---

## 2. Multi-Token Prediction: 4 Signals Per Position

APEX-1 adds 4 extra "speculative" prediction heads (from DeepSeek-V3). Each head $k$ predicts the token at offset $k$ ahead of the current position:

- Head 1: predicts $t_{i+1}$ (same as standard LM head)
- Head 2: predicts $t_{i+2}$
- Head 3: predicts $t_{i+3}$
- Head 4: predicts $t_{i+4}$

**Benefits:**
1. **Richer gradient signal**: 4× more feedback per training example
2. **Better long-range planning**: the model learns to "plan ahead"
3. **Speculative decoding**: at inference, the heads draft 4 tokens at once, which can then be verified in a single forward pass — 2-3× throughput boost

---

## 3. The Training Loss

The speculative heads contribute an **auxiliary loss**:

$$L_{total} = L_{main} + \lambda_{spec} \times \frac{1}{4} \sum_{k=1}^{4} L_k$$

where:
- $L_{main}$ = standard LM cross-entropy (predicting $t_{i+1}$)
- $L_k$ = cross-entropy for head $k$ predicting $t_{i+k}$
- $\lambda_{spec} = 0.1$ (10% weight — heads improve training but do not dominate)

---

## 4. Full Annotated Source: `apex/model/multi_token_head.py`

```python
"""
Multi-Token Prediction Head for APEX-1.

4 linear heads predict the next 1-4 tokens simultaneously.
Training: auxiliary loss adds richer gradient signal.
Inference: draft_tokens() produces 4 speculative token IDs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiTokenHead(nn.Module):
    """Multi-token prediction heads for speculative decoding.
    
    Args:
        d_model:   Model hidden dimension.
        vocab_size: Vocabulary size (output logits).
        n_predict:  How many future tokens to predict (default: 4).
    """

    def __init__(self, d_model: int, vocab_size: int, n_predict: int = 4) -> None:
        super().__init__()
        self.n_predict = n_predict
        self.vocab_size = vocab_size

        # n_predict independent linear heads
        # Each head k predicts the token k positions ahead
        # All share the same d_model input but have independent weights
        self.heads = nn.ModuleList([
            nn.Linear(d_model, vocab_size, bias=False)
            for _ in range(n_predict)
        ])

    def forward(
        self, hidden_states: torch.Tensor
    ) -> list[torch.Tensor]:
        """Compute speculative logits for all heads.
        
        Args:
            hidden_states: [batch, seq_len, d_model] — last layer hidden states.
        
        Returns:
            List of n_predict tensors, each [batch, seq_len, vocab_size].
            Element k is the logits for predicting the token k+1 positions ahead.
        """
        # Apply each head independently to the same hidden states
        spec_logits = []
        for head in self.heads:
            # [batch, seq_len, vocab_size]
            logits = head(hidden_states)
            spec_logits.append(logits)
        return spec_logits

    @torch.no_grad()
    def draft_tokens(
        self,
        hidden: torch.Tensor,     # [1, 1, d_model] — last token's hidden state
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Draft n_predict tokens for speculative decoding.
        
        This is the INFERENCE path. Given the hidden state of the most recent
        token, each head independently predicts the next k tokens.
        
        Args:
            hidden:      Hidden state of the most recently generated token.
            temperature: Sampling temperature (higher = more diverse).
        
        Returns:
            [1, n_predict] tensor of drafted token IDs.
        """
        spec_logits = self.forward(hidden)   # List of [1, 1, vocab_size]
        draft_ids = []

        for k, logits_k in enumerate(spec_logits):
            # logits_k: [1, 1, vocab_size]
            # Take the last position's logits → [1, vocab_size]
            logits_last = logits_k[:, -1, :]

            if temperature <= 0:
                # Greedy: always pick highest probability
                token_id = logits_last.argmax(dim=-1)   # [1]
            else:
                # Temperature sampling
                probs = torch.softmax(logits_last / temperature, dim=-1)
                token_id = torch.multinomial(probs, num_samples=1).squeeze(-1)

            draft_ids.append(token_id)

        # Stack all drafted token IDs into [1, n_predict]
        return torch.stack(draft_ids, dim=1)
```

---

## 5. Computing the Auxiliary Loss

In `apex/training/losses.py`:

```python
def compute_pretrain_loss(
    logits: torch.Tensor,              # [B, S, vocab] — main LM head
    spec_logits: list[torch.Tensor],   # List of [B, S, vocab] — speculative heads
    token_ids: torch.Tensor,           # [B, S] — ground truth
    lambda_spec: float = 0.1,
) -> torch.Tensor:
    """Compute combined pretrain loss: main + speculative auxiliary."""
    B, S, V = logits.shape

    # Standard LM loss: shift by 1 (predict next token)
    # logits[:, :-1, :] predicts for positions 0..S-2
    # token_ids[:, 1:] is the ground truth for positions 1..S-1
    main_loss = F.cross_entropy(
        logits[:, :-1, :].reshape(-1, V),
        token_ids[:, 1:].reshape(-1),
        ignore_index=-100,
    )

    if not spec_logits:
        return main_loss

    spec_loss_total = 0.0
    valid_heads = 0

    for k, sl in enumerate(spec_logits, start=1):
        # BUG-12 FIX: guard against empty slice
        # Head k predicts token at offset k+1 from position i
        # We need at least (k+1) tokens in the sequence
        if S - k < 1:
            continue   # Sequence too short for this head — skip!
            # (Previously this caused cross_entropy on empty tensor → NaN)

        # logits for head k: positions 0..S-k-1 predict tokens at 1+k..S-1
        pred_logits = sl[:, : S - k, :].reshape(-1, V)
        targets = token_ids[:, 1 + k :].reshape(-1)

        spec_loss_total += F.cross_entropy(pred_logits, targets, ignore_index=-100)
        valid_heads += 1

    if valid_heads > 0:
        spec_loss = spec_loss_total / valid_heads
        return main_loss + lambda_spec * spec_loss

    return main_loss
```

---

## 6. Speculative Decoding at Inference

At inference, the multi-token heads are used in `APEX1Generator.generate_with_speculative()`:

```
Step 1: Generate token t (main model)
Step 2: draft_tokens(hidden) → [t+1_draft, t+2_draft, t+3_draft, t+4_draft]
Step 3: Verify ALL 4 drafts in ONE forward pass
Step 4: Accept drafts that match target distribution (probabilistic)
Step 5: Jump forward by 1+accepted steps instead of just 1
```

Result: instead of 1 new token per forward pass, the model often accepts 3-4, giving 2-3× throughput with identical output quality.

---

## 7. Parameter Count

The multi-token head adds $n_{predict} \times d_{model} \times vocab_{size}$ parameters:

For Small ($n=4$, $d=512$, $v=151643$):
$$4 \times 512 \times 151643 \approx 310M$$

These are not weight-tied with the embedding (each head needs independent weights to predict different offsets). They are a significant parameter cost but provide large training and inference benefits.

---

**Next:** [14 — Transformer Block →](14-transformer-block.md)
