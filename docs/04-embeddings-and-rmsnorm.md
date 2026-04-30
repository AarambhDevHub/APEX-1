# 04 — Embeddings & RMSNorm: From Token IDs to Vectors

> **Difficulty:** ⭐⭐☆☆☆ Beginner-Intermediate  
> **Source files:** `apex/model/apex_model.py` (embedding), `apex/model/norm.py`  
> **You will learn:** How a token ID becomes a rich vector of numbers, and why we normalise activations.

---

## 1. Embeddings — The Word Coordinate System

### The Analogy

Imagine a map of a city. Every location has coordinates: (latitude, longitude). Two nearby coffee shops have similar coordinates; a coffee shop and an airport have very different coordinates.

An **embedding** does the same thing for words. Every token gets a unique vector of numbers (its "coordinates") in a high-dimensional space. Words with similar meanings end up near each other.

For example (simplified to 3D for illustration):
```
"cat"   → [0.8, 0.2, 0.9]
"dog"   → [0.7, 0.3, 0.8]   ← similar to "cat"
"Paris" → [0.1, 0.9, 0.2]   ← very different
```

In APEX-1, the embedding vectors are `d_model` = 512 to 7,168 dimensional — much richer than 3D.

---

## 2. The Embedding Table

An embedding is implemented as a simple **lookup table**. Think of a dictionary where every token ID maps to a vector:

```
Token ID 9906 ("Hello") → [0.231, -0.012, 0.897, ... (512 numbers)]
Token ID   11 (",")     → [0.011,  0.234, -0.005, ...]
Token ID 1917 ("world") → [0.452, -0.341, 0.128, ...]
```

In PyTorch:
```python
embedding = nn.Embedding(vocab_size, d_model)
# vocab_size = 151,643  (rows)
# d_model    = 512       (columns)
# Total size = 151,643 × 512 = ~77.6M numbers
```

When you call `embedding(token_ids)`:
1. Each token ID selects a row from the table
2. The result is a matrix `[batch_size, seq_len, d_model]`

---

## 3. The Embedding Scale Factor

APEX-1 multiplies the embedding by $\sqrt{d_{\text{model}}}$:

$$\mathbf{x} = \text{Embedding}(\texttt{token\_ids}) \times \sqrt{d_{\text{model}}}$$

**Why?** Without scaling, the embedding values are tiny (initialised with standard deviation ~0.02). As $d_{\text{model}}$ grows, the values stay small and get overwhelmed by the other components (like positional encoding). Multiplying by $\sqrt{d_{\text{model}}}$ keeps the scale stable across different model sizes.

For `d_model = 512`: multiply by $\sqrt{512} \approx 22.6$  
For `d_model = 7168`: multiply by $\sqrt{7168} \approx 84.7$

---

## 4. Weight Tying — Sharing Embedding with LM Head

The **LM head** is the final layer that converts hidden states back into vocabulary scores (logits). Instead of having a separate weight matrix, APEX-1 **reuses the embedding matrix**:

$$\text{logits} = \mathbf{x} \cdot \mathbf{W}_{\text{embed}}^T$$

where $\mathbf{W}_{\text{embed}}^T$ is the **transpose** of the embedding table.

**Why this is smart:**
- Saves 77.6M parameters (huge for the Small model!)
- The embedding and LM head represent the same tokens — it makes sense to tie them
- Well-established practice (used in GPT-2, Llama, DeepSeek, etc.)

---

## 5. RMSNorm — Why We Normalise

### The Problem Without Normalisation

As a signal passes through dozens of layers, the numbers can grow very large or very small. This is called **internal covariate shift** and it makes training unstable.

**Analogy:** Imagine a chain of amplifiers. Each one amplifies the signal by a different amount. After 72 amplifiers, the output could be billions of times louder than the input — or near zero. You need a volume normaliser at each stage.

---

## 6. The RMSNorm Formula

Standard **LayerNorm** subtracts the mean and divides by the standard deviation:

$$\text{LayerNorm}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sigma} \cdot \gamma + \beta$$

**RMSNorm** (Root Mean Square Norm) skips the mean subtraction:

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \gamma$$

where:

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \varepsilon}$$

**Breaking it down:**
- $x_i$ are the individual elements of the vector
- $\frac{1}{d} \sum x_i^2$ is the mean of the **squares** (not the values)
- $\sqrt{\cdot + \varepsilon}$ is the root mean square ($\varepsilon$ prevents division by zero)
- $\gamma$ is a learned **scale** parameter (initialized to 1.0)

**Why RMSNorm instead of LayerNorm?**
- 20–40% fewer operations (no mean subtraction, no bias $\beta$)
- Equal or better quality (Llama, DeepSeek, Qwen3 all use it)
- The mean subtraction in LayerNorm was not adding value

---

## 7. Full Annotated Source: `apex/model/norm.py`

```python
"""
RMSNorm — Root Mean Square Layer Normalization.

Formula:
    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x²) + ε)
    γ is a learned per-dimension scale parameter initialized to 1.0
"""

from __future__ import annotations
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.
    
    Args:
        d_model: The dimension of the input features.
        eps: Small constant for numerical stability. Default: 1e-6.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        
        # γ is a learnable scale vector, one value per dimension.
        # Initialized to 1.0 so at the start, RMSNorm is close to identity.
        self.weight = nn.Parameter(torch.ones(d_model))
        
        self.eps = eps          # Small number to avoid dividing by zero
        self.d_model = d_model  # Stored for the extra_repr() method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the input tensor.
        
        Args:
            x: Input tensor of shape [batch, seq_len, d_model].
        
        Returns:
            Normalized tensor of the same shape.
        """
        # Step 1: Square every element: x² → shape [batch, seq, d_model]
        # Step 2: Average across the last dimension (d_model): shape [batch, seq, 1]
        # Step 3: Add epsilon for safety
        # Step 4: Take square root → RMS value per position
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        
        # Step 5: Divide x by its RMS (scales all values to ~unit norm)
        # Step 6: Multiply by the learned scale γ (self.weight)
        # Broadcasting: weight [d_model] applies to every [batch, seq] position
        return (x / rms) * self.weight

    def extra_repr(self) -> str:
        """Used by PyTorch's print(model) for a readable summary."""
        return f"d_model={self.d_model}, eps={self.eps}"
```

---

## 8. Where RMSNorm Is Used in APEX-1

RMSNorm is applied at three places in every transformer block:

```
Input x
  │
  ├─ RMSNorm ─→ Attention ─→ + x   (pre-norm before attention)
  │
  ├─ RMSNorm ─→ FFN       ─→ + x   (pre-norm before FFN)
  │
  └─ (after all blocks) Final RMSNorm before LM head
```

This **pre-norm** placement (normalise before each sub-layer, not after) is more stable than the original transformer's post-norm design.

---

## 9. Full Embedding Code in `apex_model.py`

```python
class APEX1Model(nn.Module):
    def __init__(self, config: APEXConfig) -> None:
        super().__init__()
        m = config.model

        # The embedding table: vocab_size rows, d_model columns
        # Each row is the learnable vector for one token
        self.embedding = nn.Embedding(m.vocab_size, m.d_model)
        
        # Scale factor: √d_model
        # math.sqrt is used (not torch.sqrt) because this is a constant scalar
        self.embed_scale = math.sqrt(m.d_model)
        
        # ... (blocks, norm, etc.) ...

    def forward(self, token_ids, ...):
        # Look up embedding for each token, then scale
        # token_ids: [batch, seq_len]
        # x: [batch, seq_len, d_model]
        x = self.embedding(token_ids) * self.embed_scale
        
        # ... (pass through layers) ...
        
        x = self.final_norm(x)   # Final RMSNorm
        
        # LM head: multiply by embedding weights transposed
        # x: [batch, seq, d_model] × embedding.weight.T [d_model, vocab]
        # → logits: [batch, seq, vocab_size]
        logits = torch.matmul(x, self.embedding.weight.T)
        
        return {"logits": logits, ...}
```

---

## 10. Summary

| Concept | Formula | Purpose |
|---|---|---|
| Embedding lookup | $\mathbf{x} = \mathbf{W}_e[\texttt{token\_id}]$ | Token ID → vector |
| Embedding scale | $\mathbf{x} = \mathbf{x} \times \sqrt{d_{\text{model}}}$ | Stable magnitudes |
| Weight tying | $\text{logits} = \mathbf{x} \mathbf{W}_e^T$ | Reuse embedding as LM head |
| RMSNorm | $\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d}\sum x_i^2 + \varepsilon}$, then $\mathbf{x}/\text{RMS} \times \gamma$ | Stable signal magnitude |

---

**Next:** [05 — Positional Encoding (RoPE & YaRN) →](05-positional-encoding-rope.md)
