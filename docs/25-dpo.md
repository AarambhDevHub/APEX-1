# 25 — DPO: Direct Preference Optimization

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/alignment/dpo.py`  
> **You will learn:** How DPO skips the reward model, the implicit reward derivation, and BUG-16 (bidirectional prefix attention).

---

## 1. The Problem with RLHF

Standard RLHF requires:
1. Train a reward model
2. Use RL (PPO) to optimise the policy against the reward model
3. Manage a reference model to prevent "reward hacking"

This is complex, unstable, and computationally expensive.

---

## 2. DPO's Insight: The Implicit Reward

Rafailov et al. (2023) showed that the RLHF objective has a closed-form solution. The optimal policy implicitly defines a reward:

$$r^*(x, y) = \beta \log\frac{\pi^*(y|x)}{\pi_{ref}(y|x)} + \beta \log Z(x)$$

where $Z(x)$ is a partition function that cancels out when comparing two responses.

The **implicit reward** difference between chosen and rejected is:

$$r_{chosen} - r_{rejected} = \beta\!\left(\log\frac{\pi(y_c|x)}{\pi_{ref}(y_c|x)} - \log\frac{\pi(y_r|x)}{\pi_{ref}(y_r|x)}\right)$$

This can be computed directly from log-probabilities — **no reward model needed**!

---

## 3. The DPO Loss

Plugging into the Bradley-Terry formulation:

$$L_{DPO} = -\mathbb{E}\!\left[\log\sigma\!\left(\beta\!\left(\log\frac{\pi(y_c|x)}{\pi_{ref}(y_c|x)} - \log\frac{\pi(y_r|x)}{\pi_{ref}(y_r|x)}\right)\right)\right]$$

**Breaking it down:**
- $\log \pi(y|x)$: log-probability that the policy model assigns to the full response $y$
- $\log \pi_{ref}(y|x)$: log-probability from the frozen reference model
- The difference $\log\pi - \log\pi_{ref}$ is how much the policy has moved away from reference
- $\beta$ controls how far we allow the policy to move (higher = more conservative)
- The whole thing is pushed through Bradley-Terry ($\sigma$): chosen should be higher

---

## 4. Computing Sequence Log-Probability

$$\log P(y|x) = \sum_{t=1}^{T} \log P(y_t | x, y_{<t})$$

This sums the log-probability of each response token given all previous tokens:

```python
log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
token_log_probs = log_probs.gather(2, targets[:, 1:].unsqueeze(-1)).squeeze(-1)
response_log_prob = token_log_probs[:, response_start:].sum(dim=-1)
```

---

## 5. BUG-16: Bidirectional Prompt Attention

The prompt (user's question) is static — the model knows all of it simultaneously. Treating the prompt causally (each prompt token only sees past tokens) gives a weaker contextual representation than bidirectional attention.

**BUG-16:** The original `dpo_loss()` called `model(chosen_ids)` without `prefix_len`, meaning the prompt was processed causally:

```python
# ORIGINAL (wrong):
chosen_logits = model(chosen_ids)["logits"]   # prefix_len defaults to 0 (causal)
```

**Fix:** Pass `prefix_len=prompt_len` to enable bidirectional attention on the prompt:

```python
# FIXED:
chosen_logits = model(chosen_ids, prefix_len=prompt_len)["logits"]
```

This gives the model a richer understanding of the prompt context, improving DPO training quality.

---

## 6. Full Annotated Source: `apex/alignment/dpo.py`

```python
"""
DPO — Direct Preference Optimization.

Loss: -log(σ(β(log π/π_ref)_chosen - β(log π/π_ref)_rejected))

BUG-16 FIX: prefix_len=prompt_len ensures prompt tokens get
bidirectional attention (GLM-4 style) for richer context.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_sequence_logprob(
    logits: torch.Tensor,          # [1, seq_len, vocab_size]
    token_ids: torch.Tensor,       # [1, seq_len]
    response_start_idx: int,       # Where the response starts
) -> torch.Tensor:
    """Compute log-probability of response tokens only.
    
    We sum only over the response portion, not the prompt.
    This is correct because the model should be evaluated on
    how likely it makes the response given the prompt, not
    on how well it predicts the prompt itself.
    
    Returns:
        Scalar log-probability tensor.
    """
    # Shift: logits at position i predict token at i+1
    shift_logits = logits[:, :-1, :]     # [1, seq-1, vocab]
    shift_targets = token_ids[:, 1:]     # [1, seq-1]

    # Per-token log-probs
    log_probs = F.log_softmax(shift_logits, dim=-1)   # [1, seq-1, vocab]
    # Gather the log-prob of the actual token at each position
    token_log_probs = log_probs.gather(
        2, shift_targets.unsqueeze(-1)   # [1, seq-1, 1]
    ).squeeze(-1)   # [1, seq-1]

    # Sum only over response tokens (start-1 due to the shift)
    start = max(0, response_start_idx - 1)
    response_log_prob = token_log_probs[:, start:].sum(dim=-1)   # [1]
    return response_log_prob


def dpo_loss(
    model: nn.Module,
    reference_model: nn.Module,
    prompt_ids: torch.Tensor,       # [1, prompt_len]
    chosen_ids: torch.Tensor,       # [1, prompt+chosen_len]
    rejected_ids: torch.Tensor,     # [1, prompt+rejected_len]
    prompt_len: int,                # Length of the prompt section
    beta: float = 0.1,             # KL penalty (higher = more conservative)
) -> tuple[torch.Tensor, dict]:
    """Compute DPO loss for one preference pair.
    
    Args:
        model:           Policy being trained (gradients flow through).
        reference_model: Frozen SFT model (no gradients).
        prompt_ids:      Prompt tokens (for reference only, not used directly).
        chosen_ids:      Full sequence (prompt + good response).
        rejected_ids:    Full sequence (prompt + bad response).
        prompt_len:      Where the response starts in chosen/rejected_ids.
        beta:            KL constraint coefficient.
    
    Returns:
        (loss, metrics_dict)
    """
    # ── Policy log-probs ─────────────────────────────────────────────────
    # BUG-16 FIX: prefix_len=prompt_len gives bidirectional attention
    # on the prompt, producing a richer context representation
    chosen_logits = model(chosen_ids, prefix_len=prompt_len)["logits"]
    rejected_logits = model(rejected_ids, prefix_len=prompt_len)["logits"]

    # Log P_policy(y_chosen | x) — sum over response tokens
    log_pi_chosen = compute_sequence_logprob(chosen_logits, chosen_ids, prompt_len)
    log_pi_rejected = compute_sequence_logprob(rejected_logits, rejected_ids, prompt_len)

    # ── Reference log-probs (no gradient) ────────────────────────────────
    with torch.no_grad():
        ref_chosen_logits = reference_model(chosen_ids, prefix_len=prompt_len)["logits"]
        ref_rejected_logits = reference_model(rejected_ids, prefix_len=prompt_len)["logits"]

    log_ref_chosen = compute_sequence_logprob(ref_chosen_logits, chosen_ids, prompt_len)
    log_ref_rejected = compute_sequence_logprob(ref_rejected_logits, rejected_ids, prompt_len)

    # ── Implicit rewards ──────────────────────────────────────────────────
    # r = β × (log π - log π_ref)
    # How much the policy has "moved" from reference (positive = more likely under policy)
    reward_chosen = beta * (log_pi_chosen - log_ref_chosen)
    reward_rejected = beta * (log_pi_rejected - log_ref_rejected)

    # ── DPO loss ──────────────────────────────────────────────────────────
    # Chosen should have higher implicit reward than rejected
    # = -log σ(reward_chosen - reward_rejected)
    loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()

    metrics = {
        "dpo_loss": loss.item(),
        "reward_chosen": reward_chosen.mean().item(),
        "reward_rejected": reward_rejected.mean().item(),
        "reward_margin": (reward_chosen - reward_rejected).mean().item(),
        "log_pi_chosen": log_pi_chosen.mean().item(),
        "log_pi_rejected": log_pi_rejected.mean().item(),
    }

    return loss, metrics
```

---

## 7. DPO vs Standard RLHF

| Aspect | RLHF (PPO) | DPO |
|---|---|---|
| Reward model needed? | Yes | No |
| Training stability | Notoriously tricky | Stable supervised training |
| Compute overhead | 2× (policy + reference + reward) | ~1.5× (policy + reference) |
| Quality | Can be higher with good RM | Competitive, often better |
| Implementation complexity | Very high | Simple |

DPO is now the preferred approach in most modern fine-tuning pipelines (Llama 3, Qwen2.5, etc.).

---

**Next:** [26 — GRPO →](26-grpo.md)
