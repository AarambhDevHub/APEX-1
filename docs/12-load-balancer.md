# 12 — Auxiliary-Loss-Free Load Balancer

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/model/load_balancer.py`  
> **You will learn:** Why MoE experts collapse, how traditional aux loss hurts quality, and how APEX-1's bias-only approach solves both.

---

## 1. The Expert Collapse Problem

When training MoE models, without any balancing, a few experts receive all the tokens and the others receive none. This is called **expert collapse**.

**Why does this happen?** Once an expert gets slightly better (by chance), the router assigns more tokens to it. It becomes better still. The other experts receive fewer tokens and stop improving. Eventually, you effectively have only 2-3 experts doing all the work, and the other 253 are idle.

**Why is this bad?** You spent compute and memory for 256 experts but only use 2-3. All that capacity is wasted.

---

## 2. Traditional Fix: Auxiliary Loss (and Its Problems)

The standard solution adds an "auxiliary loss" that penalises imbalanced routing:

$$L_{aux} = \alpha \times \frac{n_{experts}}{n_{tokens}} \sum_{i=1}^{n_{experts}} f_i \cdot P_i$$

where $f_i$ is the fraction of tokens routed to expert $i$ and $P_i$ is the average routing probability for expert $i$.

**Problem:** This auxiliary loss **competes** with the main language model loss. The model is trying to:
1. Predict the next token well (main loss)
2. Balance expert load (auxiliary loss)

These objectives sometimes conflict — the model sacrifices prediction quality to achieve balance. Studies show $0.5–1.0\%$ perplexity degradation from auxiliary loss.

---

## 3. APEX-1's Solution: Bias-Only Balancing

Inspired by DeepSeek-V3, APEX-1 uses a completely **gradient-free** approach:

Each expert $i$ has a learnable bias $b_i$ that is added to its routing score:

$$s_i = W_{router} x + b_i$$

After each training step, the bias is adjusted based on how over/under-utilised the expert was:

$$b_i \leftarrow b_i + \alpha \cdot \text{sign}(r_{target} - r_i)$$

where:
- $r_{target} = 1/n_{experts}$ — ideal fraction of tokens (uniform distribution)
- $r_i$ = actual fraction of tokens expert $i$ received this step
- $\alpha$ = step size (e.g., 0.001)
- $\text{sign}(\cdot)$ = +1 or -1

**Why `sign()` not exact gradient?**
- $\text{sign}()$ gives constant-magnitude updates regardless of how far off we are
- Avoids oscillation that can come from proportional feedback
- Simple, robust, proven in practice

**Why is this better?**
- Zero gradient flows to the main model — no interference with LM loss
- The bias is just a small additive nudge to routing scores
- Expert utilisation smoothly converges to uniform

---

## 4. Full Annotated Source: `apex/model/load_balancer.py`

```python
"""
Auxiliary-Loss-Free MoE Load Balancer for APEX-1.

Uses DeepSeek-V3-style bias updates instead of auxiliary loss.
No gradient flows through the balancer — zero interference with LM training.
"""

import torch


class LoadBalancer:
    """Auxiliary-loss-free load balancer for MoE layers.
    
    Maintains per-expert bias terms that nudge routing scores
    toward uniform distribution without any auxiliary loss.
    
    IMPORTANT: This is NOT an nn.Module because it has NO parameters
    that should be trained by the optimizer. The bias is updated by
    a separate sign-gradient rule after each optimizer.step().
    
    Args:
        n_experts:       Number of experts to balance.
        alpha:           Step size for bias updates (e.g., 0.001).
        target_fraction: Ideal fraction per expert (default = 1/n_experts).
        bias_clamp:      Maximum absolute value of bias (prevent runaway).
    """

    def __init__(
        self,
        n_experts: int,
        alpha: float = 0.001,
        target_fraction: float | None = None,
        bias_clamp: float = 1.0,
    ) -> None:
        self.n_experts = n_experts
        self.alpha = alpha
        # Target: every expert should handle 1/n_experts of all tokens
        self.target_fraction = target_fraction or (1.0 / n_experts)
        self.bias_clamp = bias_clamp

        # Expert bias terms: start at zero (no initial preference)
        # Not a Parameter — updated by our own rule, not by optimizer
        self.bias = torch.zeros(n_experts)

        # Statistics tracked per-step for monitoring
        self._last_fractions: torch.Tensor | None = None

    def update(self, expert_indices: torch.Tensor) -> dict[str, float]:
        """Update expert biases after one forward pass.
        
        Should be called AFTER optimizer.step() to avoid interfering
        with gradient computation.
        
        Args:
            expert_indices: [N, n_active] — which experts each token selected.
                N = batch_size × seq_len.
        
        Returns:
            Dict of monitoring metrics (fractions, bias norms, etc.)
        """
        n_tokens = expert_indices.shape[0]

        # Count how many tokens each expert received
        # Flatten: [N, n_active] → [N * n_active] — one entry per assignment
        flat_indices = expert_indices.flatten()

        # torch.bincount: fast vectorised histogram
        # Result: [n_experts] — count of tokens per expert
        counts = torch.bincount(
            flat_indices.long(),
            minlength=self.n_experts,
        ).float()

        # Fraction = count / total assignments
        # (total assignments = n_tokens × n_active)
        total_assignments = flat_indices.numel()
        fractions = counts / max(total_assignments, 1)
        self._last_fractions = fractions

        # Sign-gradient update:
        # Expert over-loaded (fraction > target) → decrease bias → fewer tokens
        # Expert under-loaded (fraction < target) → increase bias → more tokens
        delta = self.alpha * torch.sign(self.target_fraction - fractions)
        # Note: sign(target - actual):
        #   actual < target → sign > 0 → bias increases → more tokens next time
        #   actual > target → sign < 0 → bias decreases → fewer tokens next time

        # Update bias (note: bias lives on CPU, moved to device by MoEFFN.set_expert_bias)
        self.bias = self.bias + delta

        # Clamp to prevent extreme bias values
        self.bias = self.bias.clamp(-self.bias_clamp, self.bias_clamp)

        return {
            "fraction_mean": fractions.mean().item(),
            "fraction_std": fractions.std().item(),
            "fraction_max": fractions.max().item(),
            "fraction_min": fractions.min().item(),
            "bias_norm": self.bias.norm().item(),
        }

    def state_dict(self) -> dict:
        """Serialize balancer state for checkpointing."""
        return {
            "bias": self.bias.tolist(),         # Save as plain list (JSON-safe)
            "n_experts": self.n_experts,
            "alpha": self.alpha,
            "target_fraction": self.target_fraction,
            "bias_clamp": self.bias_clamp,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore balancer state from a checkpoint."""
        self.bias = torch.tensor(state["bias"])
        self.n_experts = state["n_experts"]
        self.alpha = state["alpha"]
        self.target_fraction = state["target_fraction"]
        self.bias_clamp = state.get("bias_clamp", 1.0)
```

---

## 5. How the Bias Is Applied

In `MoEFFN.forward()`:

```python
# Router scores: [N, n_experts]
scores = self.router(x_flat)

# Add load balancer bias (no gradient!)
# self.expert_bias was set by trainer.py after optimizer.step()
scores = scores + self.expert_bias.unsqueeze(0)

# TopK selection on the adjusted scores
topk_vals, topk_idx = scores.topk(self.n_active, dim=-1)
```

---

## 6. The Trainer Hooks It Up (BUG-11 Fix)

In `apex/training/trainer.py`, after each optimizer step:

```python
# AFTER optimizer.step() — gradients are already applied
optimizer.step()
optimizer.zero_grad()

# BUG-11 FIX: use moe_ffn.n_experts (actual expert count for this layer)
# NOT config.moe.n_experts (global default, which might be wrong for some layers)
for layer_idx, balancer in self.load_balancers.items():
    # Update the bias based on this step's routing choices
    metrics = balancer.update(expert_indices[layer_idx])
    
    # Push the updated bias to the MoE layer on the right device
    moe_layer = self.model.blocks[layer_idx].ffn
    moe_layer.set_expert_bias(balancer.bias)
```

---

## 7. Convergence Behaviour

```
Step    0: Expert fractions ≈ [0.3, 0.05, 0.4, 0.25]  (imbalanced!)
Step  100: Expert fractions ≈ [0.28, 0.12, 0.32, 0.28]
Step 1000: Expert fractions ≈ [0.26, 0.24, 0.25, 0.25]  (nearly uniform)
Step 5000: Expert fractions ≈ [0.25, 0.25, 0.25, 0.25]  (balanced!)
```

The balancer nudges routing gently and consistently until all experts share the load equally — without touching the LM loss at all.

---

**Next:** [13 — Multi-Token Prediction Head →](13-multi-token-head.md)
