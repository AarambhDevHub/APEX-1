# 16 — Training Losses: How the Model Learns

> **Difficulty:** ⭐⭐☆☆☆ Intermediate  
> **Source file:** `apex/training/losses.py`  
> **You will learn:** What cross-entropy loss is, how pretraining and SFT losses differ, and the BUG-12 NaN fix.

---

## 1. What Is a Loss Function?

A **loss function** measures how wrong the model is. During training:

1. Model sees input tokens and predicts the next token
2. Loss function compares prediction to the actual next token
3. Loss = 0 → perfect prediction; Loss = 5 → very wrong
4. Gradient of loss → tells us how to adjust weights to do better

**The goal of training:** make the loss as small as possible over all training examples.

---

## 2. Cross-Entropy Loss

The most common loss for language models. For each position $i$, if the model assigns probability $p$ to the correct next token:

$$L_i = -\log(p_{\text{correct}})$$

**Why negative log?**
- If $p = 1.0$ (perfect): $-\log(1) = 0$ ✓
- If $p = 0.5$: $-\log(0.5) \approx 0.693$ (some penalty)
- If $p = 0.01$ (near zero): $-\log(0.01) \approx 4.6$ (big penalty)
- If $p \to 0$: $-\log(0) \to \infty$ (infinite penalty)

The function is convex, smooth, and penalises confident wrong predictions very harshly.

The model outputs raw scores (**logits**) for all 151,643 tokens, which are converted to probabilities via softmax. In practice, PyTorch's `F.cross_entropy` combines softmax + log + negative in one numerically stable operation.

---

## 3. Pretraining Loss

During pretraining, the model sees sequences of text and must predict each next token:

Given tokens $[t_0, t_1, t_2, \ldots, t_{n-1}]$:
- At position 0: predict $t_1$
- At position 1: predict $t_2$
- ...
- At position $n-2$: predict $t_{n-1}$

**The shift-by-1 trick:**
```python
# logits[:, :-1, :] — positions 0 to n-2 (each predicts one ahead)
# token_ids[:, 1:]  — positions 1 to n-1 (the ground truth "next token")
```

With the multi-token heads, 4 additional heads also contribute:

$$L = L_{\text{main}} + \lambda \times \frac{1}{K}\sum_{k=1}^{K} L_k$$

where $K = 4$ and $\lambda = 0.1$.

---

## 4. SFT Loss: Only Learn From Assistant Tokens

During Supervised Fine-Tuning, the model is trained on conversations:
```
<|system|>You are helpful.<|eom|>
<|user|>What is 2+2?<|eom|>
<|assistant|>The answer is 4.<|eom|>
```

We only want the model to learn to generate the **assistant's response**. The system prompt and user message are context — not targets.

**Implementation:** Set labels to `-100` for non-assistant tokens. PyTorch ignores `-100` in cross-entropy:

```python
labels = token_ids.clone()
labels[token_types != 2] = -100   # -100 = ignore_index
loss = F.cross_entropy(logits, labels, ignore_index=-100)
```

---

## 5. The BUG-12 Story: NaN on Short Sequences

The original speculative head loss code:

```python
for k, sl in enumerate(spec_logits, start=1):
    pred_logits = sl[:, :-k-1, :]   # BUG: could be empty!
    targets = token_ids[:, k+1:]    # BUG: could be empty!
    spec_loss += F.cross_entropy(pred_logits, targets)
    # F.cross_entropy on an empty tensor → NaN!
```

For a sequence of length 5 and head k=4: `sl[:, :-5, :]` → empty tensor → `cross_entropy` returns `nan` → `nan` propagates to the main loss → all gradients become `nan` → model stops learning.

**Fix:** Guard against empty slices:

```python
if S - k < 1:
    continue   # Skip this head for short sequences
```

---

## 6. Full Annotated Source: `apex/training/losses.py`

```python
"""
Loss Functions for APEX-1.

BUG-12 FIX: Speculative head loss now skips heads where
the sequence is too short (S - k < 1) to avoid NaN from
cross_entropy on empty tensors.
"""

import torch
import torch.nn.functional as F


def compute_pretrain_loss(
    logits: torch.Tensor,              # [B, S, vocab] — main LM head
    token_ids: torch.Tensor,           # [B, S] — ground truth token IDs
    spec_logits: list[torch.Tensor] | None = None,  # Speculative head logits
    attention_mask: torch.Tensor | None = None,     # [B, S] — 1=real, 0=pad
    lambda_spec: float = 0.1,         # Weight for speculative loss
) -> tuple[torch.Tensor, dict]:
    """Compute combined pretraining loss.
    
    Args:
        logits:         Main LM head logits [B, S, V].
        token_ids:      Ground truth token IDs [B, S].
        spec_logits:    List of speculative head logits (or None).
        attention_mask: Optional mask to exclude padding tokens from loss.
        lambda_spec:    Weight for speculative auxiliary loss.
    
    Returns:
        (total_loss, metrics_dict)
    """
    B, S, V = logits.shape

    # ── Main LM Loss ─────────────────────────────────────────────────────
    # Shift: logits at position i predict token at position i+1
    shifted_logits = logits[:, :-1, :].contiguous()   # [B, S-1, V]
    shifted_targets = token_ids[:, 1:].contiguous()    # [B, S-1]

    # If we have an attention_mask, mask out padding tokens
    # Padding tokens should not contribute to the loss
    if attention_mask is not None:
        # Shift mask same as logits (align with shifted targets)
        shifted_mask = attention_mask[:, 1:].contiguous()   # [B, S-1]
        # Set targets to -100 where attention_mask == 0 (padding)
        shifted_targets = shifted_targets.masked_fill(shifted_mask == 0, -100)

    main_loss = F.cross_entropy(
        shifted_logits.view(-1, V),     # [B*(S-1), V]
        shifted_targets.view(-1),        # [B*(S-1)]
        ignore_index=-100,
    )

    # ── Speculative Head Loss ─────────────────────────────────────────────
    metrics = {"main_loss": main_loss.item()}

    if not spec_logits:
        return main_loss, metrics

    spec_loss_total = torch.tensor(0.0, device=logits.device)
    valid_heads = 0

    for k, sl in enumerate(spec_logits, start=1):
        # BUG-12 FIX: guard against empty slices
        # Head k at position i predicts token i+k+1
        # We need S - k - 1 >= 1 (at least one prediction)
        if S - k < 1:
            metrics[f"spec_loss_head_{k}"] = float("nan")  # skipped
            continue   # Sequence too short for this offset

        # Logits for positions 0..S-k-1, predicting tokens k+1..S-1
        pred_logits = sl[:, : S - k - 1, :]    # [B, S-k-1, V]
        targets = token_ids[:, 1 + k : S]       # [B, S-k-1]

        # Guard: if still empty (e.g., S=2, k=1: S-k-1 = 0), skip
        if pred_logits.shape[1] == 0:
            continue

        loss_k = F.cross_entropy(
            pred_logits.contiguous().view(-1, V),
            targets.contiguous().view(-1),
            ignore_index=-100,
        )

        spec_loss_total = spec_loss_total + loss_k
        valid_heads += 1
        metrics[f"spec_loss_head_{k}"] = loss_k.item()

    if valid_heads > 0:
        spec_loss = spec_loss_total / valid_heads
        total_loss = main_loss + lambda_spec * spec_loss
        metrics["spec_loss"] = spec_loss.item()
        metrics["total_loss"] = total_loss.item()
        return total_loss, metrics

    return main_loss, metrics


def compute_sft_loss(
    logits: torch.Tensor,       # [B, S, vocab] — model output
    token_ids: torch.Tensor,    # [B, S] — ground truth
    token_types: torch.Tensor,  # [B, S] — 0=system, 1=user, 2=assistant
) -> tuple[torch.Tensor, dict]:
    """Compute SFT loss — only on assistant tokens.
    
    The model should learn to generate assistant responses,
    not to predict user or system tokens.
    
    Args:
        logits:      Model logits [B, S, V].
        token_ids:   Ground truth token IDs [B, S].
        token_types: Token type labels [B, S]. 2 = assistant.
    
    Returns:
        (loss, metrics_dict)
    """
    B, S, V = logits.shape

    # Build labels: -100 for all non-assistant positions
    labels = token_ids.clone()   # Start with ground truth
    # Where type != 2 (assistant), set to -100 (ignore in CE)
    labels[token_types != 2] = -100

    # Shift: logits at i predict token i+1
    shifted_logits = logits[:, :-1, :].contiguous()   # [B, S-1, V]
    shifted_labels = labels[:, 1:].contiguous()         # [B, S-1]

    # How many assistant tokens are being trained?
    n_assistant_tokens = (shifted_labels != -100).sum().item()

    if n_assistant_tokens == 0:
        # No assistant tokens in this batch — return zero loss
        # (prevents nan/inf from empty cross entropy)
        zero = shifted_logits.new_tensor(0.0, requires_grad=True)
        return zero, {"sft_loss": 0.0, "n_assistant_tokens": 0}

    loss = F.cross_entropy(
        shifted_logits.view(-1, V),
        shifted_labels.view(-1),
        ignore_index=-100,
    )

    metrics = {
        "sft_loss": loss.item(),
        "n_assistant_tokens": int(n_assistant_tokens),
        "assistant_token_ratio": n_assistant_tokens / (B * (S - 1)),
    }

    return loss, metrics
```

---

## 7. Loss Values to Expect

| Stage | Typical Loss | Interpretation |
|---|---|---|
| Random init | ~12 (= ln(151643)) | Uniform distribution over vocab |
| After 1K steps pretrain | 4–6 | Learning basic patterns |
| After 100K steps pretrain | 2–3 | Good language model |
| After SFT | 0.5–1.5 | Following instructions well |
| After GRPO | 0.3–1.0 | Aligned and helpful |

---

**Next:** [17 — Optimizer & Learning Rate Scheduler →](17-scheduler-and-optimizer.md)
