# 09 — Feed-Forward Networks & SwiGLU

> **Difficulty:** ⭐⭐☆☆☆ Beginner-Intermediate  
> **Source file:** `apex/model/ffn.py` — class `DenseFFN`  
> **You will learn:** What the FFN does, why SwiGLU outperforms ReLU, and the 3-matrix design.

---

## 1. What Does the FFN Do?

After the attention layer decides **which tokens to look at**, the Feed-Forward Network (FFN) processes the **information gathered** and transforms it through learned patterns.

### The "Fact Database" Analogy

Think of attention as a research librarian who gathers relevant books. The FFN is the scholar who reads those books and synthesises the information — applying learned knowledge about grammar, facts, logic, and language patterns.

The FFN operates **independently on each token position** (no interaction between positions — that is attention's job). It is a position-wise MLP.

---

## 2. Standard FFN

A basic two-layer FFN:

$$\text{FFN}(x) = W_2 \cdot \text{ReLU}(W_1 x)$$

where:
- $W_1 \in \mathbb{R}^{d_{ffn} \times d_{model}}$ expands the vector (projects to wider space)
- $\text{ReLU}(z) = \max(0, z)$ introduces non-linearity
- $W_2 \in \mathbb{R}^{d_{model} \times d_{ffn}}$ compresses back down

The expansion ratio $d_{ffn}/d_{model}$ is typically 4.

---

## 3. SwiGLU — A Better Activation

SwiGLU (Noam Shazeer, 2020; used in PaLM, Llama, DeepSeek, APEX-1) uses a **gating mechanism**:

$$\text{SwiGLU}(x) = W_{down}\!\left(\text{SiLU}(W_{gate}\, x) \odot W_{up}\, x\right)$$

where $\odot$ is element-wise multiplication, and:

$$\text{SiLU}(z) = z \cdot \sigma(z) = \frac{z}{1 + e^{-z}}$$

(SiLU = Sigmoid Linear Unit; sometimes called Swish)

**Breaking it down:**
1. $W_{gate}\, x$ → "gating scores" (which information to let through)
2. $\text{SiLU}(\cdot)$ → smooth, differentiable gating (unlike ReLU's hard 0/1 cut)
3. $W_{up}\, x$ → the actual information to pass
4. $\odot$ → the gate **modulates** the information (element-wise product)
5. $W_{down}\, (\cdot)$ → project back to d_model

### Why Better Than ReLU?

- **ReLU is dead:** Once a ReLU unit outputs 0, the gradient is 0 too — the neuron "dies" and stops learning.
- **SiLU is smooth:** Small negative inputs still pass a little signal through, keeping gradients alive.
- **Gating adds expressivity:** The model learns both what information to retain and how much.
- **Empirical gain:** ~1-2% lower perplexity for no extra compute (the gate replaces the bias)

---

## 4. The 3-Matrix Design and d_ffn

Standard FFN has 2 matrices ($W_1$, $W_2$). SwiGLU needs 3 ($W_{gate}$, $W_{up}$, $W_{down}$).

To keep FLOPs equal to a 2-matrix FFN with ratio 4, the expansion is reduced:

$$d_{ffn} = \frac{8}{3} \times d_{model} \approx 2.67 \times d_{model}$$

For APEX-1 Small: $d_{ffn} = \frac{8}{3} \times 512 \approx 1376$. This ensures SwiGLU costs no more than ReLU FFN.

---

## 5. Full Annotated Source: `apex/model/ffn.py` — `DenseFFN`

```python
"""
Feed-Forward Networks for APEX-1.

DenseFFN: SwiGLU activation (gated linear unit).
MoEFFN:   Mixture-of-Experts with shared + routed experts.

BUG-08 FIX: MoEFFN expert dispatch now correctly handles n_e > 1
by reshaping token-expert batches as [1, n_e, d_model].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseFFN(nn.Module):
    """SwiGLU Feed-Forward Network.
    
    Args:
        d_model: Model dimension (input and output size).
        d_ffn:   FFN hidden dimension (typically 8/3 * d_model).
        dropout: Dropout probability.
    """

    def __init__(self, d_model: int, d_ffn: int, dropout: float = 0.0) -> None:
        super().__init__()

        # Three linear projections for SwiGLU:
        # W_gate: produces the "gate" signal
        self.W_gate = nn.Linear(d_model, d_ffn, bias=False)

        # W_up: produces the "value" signal
        self.W_up = nn.Linear(d_model, d_ffn, bias=False)

        # W_down: projects back from d_ffn to d_model
        self.W_down = nn.Linear(d_ffn, d_model, bias=False)

        # Optional dropout (usually 0 for large models)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input of shape [batch, seq_len, d_model]
               (or [batch, n_experts, d_model] when called from MoEFFN)
        
        Returns:
            Output of the same shape as input.
        """
        # Compute gate values (which info to let through)
        # Shape: [batch, seq, d_ffn]
        gate = self.W_gate(x)

        # Compute value signal (the actual information)
        # Shape: [batch, seq, d_ffn]
        value = self.W_up(x)

        # Apply SwiGLU: gate signal modulates value signal
        # F.silu is the Sigmoid Linear Unit: silu(z) = z * sigmoid(z)
        activated = F.silu(gate) * value  # element-wise product

        # Apply optional dropout to the activated values
        activated = self.dropout(activated)

        # Project back to model dimension
        # Shape: [batch, seq, d_model]
        return self.W_down(activated)
```

---

## 6. ReLU vs SiLU — Visual Comparison

```
ReLU:   f(x) = max(0, x)
         ____/
        /
   ____/
   -3  -2  -1   0   1   2   3
       ^ Dead zone: gradient = 0 for x < 0

SiLU:   f(x) = x * sigmoid(x)
          ___/
        _/
   ___/ ← small negative output keeps gradient alive
   -3  -2  -1   0   1   2   3
```

---

## 7. FLOPs Breakdown

For one token through the DenseFFN:

| Operation | FLOPs |
|---|---|
| $W_{gate}$: $(d_{model} \to d_{ffn})$ | $2 \times d_{model} \times d_{ffn}$ |
| $W_{up}$: $(d_{model} \to d_{ffn})$ | $2 \times d_{model} \times d_{ffn}$ |
| $\text{SiLU}(gate) \odot value$ | $d_{ffn}$ (BUG-17 fix — was missing!) |
| $W_{down}$: $(d_{ffn} \to d_{model})$ | $2 \times d_{ffn} \times d_{model}$ |
| **Total** | $6 \times d_{model} \times d_{ffn} + d_{ffn}$ |

---

## 8. Where FFN Sits in the Architecture

Each transformer block has one FFN (either Dense or MoE):

```
Input x
  │
  ├── RMSNorm
  ├── Attention
  └── + x  (residual)
  │
  ├── SkipGate decides: skip?
  ├── (if not skip)
  │   ├── RMSNorm
  │   ├── DenseFFN  ← you are here
  │   └── + x  (residual)
  └── Output x
```

---

**Next:** [10 — Mixture of Experts (MoE) →](10-mixture-of-experts.md)
