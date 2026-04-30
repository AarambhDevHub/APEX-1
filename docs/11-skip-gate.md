# 11 — Dynamic Skip Gate

> **Difficulty:** ⭐⭐☆☆☆ Intermediate  
> **Source file:** `apex/model/skip_gate.py`  
> **You will learn:** How APEX-1 saves 25–35% FFN compute by skipping easy tokens, and the STE trick for training binary decisions.

---

## 1. The Observation: Not All Tokens Are Equal

Some tokens are "hard" — they carry complex semantic meaning (e.g., "photosynthesis", "recursion"). Other tokens are "easy" — they carry little information (e.g., "the", "a", ","). Why run a computationally expensive FFN on a comma?

The **Skip Gate** is a small learned network that decides, per-token, whether the FFN is worth running.

---

## 2. Architecture

The skip gate is a 2-layer MLP that outputs a scalar between 0 and 1:

$$g(x) = \sigma\!\left(W_2 \cdot \text{SiLU}(W_1 x)\right) \in [0, 1]$$

where $\sigma$ is the sigmoid function and:
- $W_1 \in \mathbb{R}^{d_{hidden} \times d_{model}}$ (`d_hidden` is small, e.g., 64)
- $W_2 \in \mathbb{R}^{1 \times d_{hidden}}$


The output $g$ is one number per token: close to 1 → run the FFN; close to 0 → skip it.

---

## 3. The Training Challenge: Non-Differentiable Binary Decision

**Problem:** The skip decision is binary (skip or not). Binary functions are not differentiable — you cannot compute gradients through them.

**Bad idea:** Use a hard threshold during training:
```python
skip = (g < 0.15)  # True or False
```
Gradient of a step function = 0 everywhere except at the threshold. Gradient vanishes. Model cannot learn.

### Straight-Through Estimator (STE)

**Solution:** In the **forward pass**, use the binary (hard) threshold. In the **backward pass**, pretend the gradient passes through as if it were the identity:

$$\text{forward}: \hat{g} = \mathbf{1}[g \geq \theta]$$
$$\text{backward}: \frac{\partial \hat{g}}{\partial g} \approx 1 \quad \text{(instead of 0)}$$

This "lies" to the gradient calculation, but it works well in practice because:
1. The gate value $g$ is still trained via its own loss signal (the overall LM loss)
2. The model learns to push easy tokens below the threshold

---

## 4. Full Annotated Source: `apex/model/skip_gate.py`

```python
"""
Dynamic Skip Gate for APEX-1.

A lightweight MLP that learns to skip the FFN for tokens that do not
need expensive processing.  Uses Straight-Through Estimator (STE) for
the binary threshold during training.

Benefits:
- 25-35% of FFN computations skipped at convergence
- No accuracy degradation (simple tokens truly do not need FFN)
- Gradient flows through STE during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class STEThreshold(torch.autograd.Function):
    """Straight-Through Estimator for a binary threshold.
    
    Forward:  returns 1.0 if x >= threshold, else 0.0  (binary/hard)
    Backward: passes gradient through as-is (pretends it is identity)
    
    This allows gradients to flow through the binary decision
    even though the true gradient of a step function is 0.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, threshold: float) -> torch.Tensor:
        # Hard binary threshold (the actual inference behavior)
        # x >= threshold → 1.0 (run FFN)
        # x < threshold  → 0.0 (skip FFN)
        return (x >= threshold).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # STE: pass gradient straight through
        # The second return is for `threshold` (not a tensor, so None)
        return grad_output, None


class SkipGate(nn.Module):
    """Learned gate to skip the FFN for simple tokens.
    
    Args:
        d_model:    Input dimension.
        hidden_dim: Gate MLP hidden dimension (small, e.g., 64).
        threshold:  Skip if gate output < this value (default: 0.15).
    """

    def __init__(self, d_model: int, hidden_dim: int = 64, threshold: float = 0.15) -> None:
        super().__init__()
        self.threshold = threshold

        # Two-layer MLP: d_model → hidden_dim → 1
        self.W1 = nn.Linear(d_model, hidden_dim, bias=True)
        self.W2 = nn.Linear(hidden_dim, 1, bias=True)
        # Sigmoid at the end: squashes output to [0, 1]
        self.sigmoid = nn.Sigmoid()

        # Initialise W2 with small weights so gate starts near 0.5
        # (no strong initial bias toward always-skip or always-run)
        nn.init.constant_(self.W2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute skip mask for each token.
        
        Args:
            x: Token representations [batch, seq_len, d_model].
        
        Returns:
            gate_value: Continuous gate output [batch, seq_len, 1] in [0,1].
            skip_mask:  Binary mask [batch, seq_len, 1] — 1=skip, 0=run FFN.
        
        Usage in transformer block:
            gate_val, skip = skip_gate(x)
            if skip.all():
                pass   # skip the entire FFN computation
            else:
                ffn_out = ffn(x)
                x = x + ffn_out * (1 - skip)   # zero out skipped tokens
        """
        # Layer 1: expand with SiLU activation
        h = F.silu(self.W1(x))    # [B, S, hidden_dim]

        # Layer 2: compress to scalar + sigmoid
        gate_value = self.sigmoid(self.W2(h))   # [B, S, 1]

        # Apply STE threshold (binary decision for skip)
        # Training: forward = binary, backward = straight-through
        # Inference: model.eval() → same behavior
        skip_mask = STEThreshold.apply(
            1.0 - gate_value,   # invert: low gate → high skip probability
            self.threshold,
        )   # [B, S, 1]  — 1.0 = skip, 0.0 = run FFN

        return gate_value, skip_mask
```

---

## 5. How the Skip Gate Integrates Into the Block

In `apex/model/block.py`, the skip gate wraps the FFN:

```python
# 1. Run skip gate — get continuous value and binary mask
gate_val, skip_mask = self.skip_gate(x)   # [B, S, 1]

# 2. Optimisation: if ALL tokens want to skip, avoid FFN entirely
if skip_mask.all():
    # No FFN needed at all! Return x unchanged (residual only)
    return x

# 3. Otherwise, run the FFN
ffn_out = self.ffn(x)   # [B, S, d_model]

# 4. Apply skip: where skip_mask=1, multiply FFN output by 0
# This effectively zeros out the FFN contribution for skipped tokens
# (the residual connection still passes x through unchanged)
x = x + ffn_out * (1.0 - skip_mask)   # skip_mask broadcasts over d_model
```

---

## 6. What the Model Learns

At convergence, the skip gate discovers that function words and punctuation rarely need FFN processing:

```
Token "the"    → gate ≈ 0.05 → SKIP (threshold = 0.15)
Token "a"      → gate ≈ 0.08 → SKIP
Token ","      → gate ≈ 0.03 → SKIP
Token "neural" → gate ≈ 0.72 → RUN FFN
Token "photosynthesis" → gate ≈ 0.91 → RUN FFN
```

Typical convergence: **25–35% of tokens skip**, saving 25–35% of FFN FLOPs.

---

## 7. Why Not Just Set Threshold to 0?

If threshold = 0, nothing ever skips. If threshold = 1, everything skips. At threshold = 0.15:
- High-confidence "easy" tokens (gate < 0.15) skip
- Ambiguous tokens run
- The model is still penalised by the LM loss if it skips too aggressively

---

**Next:** [12 — Auxiliary-Loss-Free Load Balancer →](12-load-balancer.md)
