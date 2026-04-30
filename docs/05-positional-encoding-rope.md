# 05 — Positional Encoding: RoPE & YaRN

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/model/rope.py`  
> **You will learn:** Why position matters, how RoPE encodes it via rotation, and how YaRN extends context length.

---

## 1. The Problem: Order Matters

Consider these two sentences:
- "The dog bit the man."
- "The man bit the dog."

Same words, completely different meanings. Without knowing the **position** of each word, the model cannot tell them apart — because the attention mechanism treats all tokens equally regardless of order.

We must somehow inject **position information** into the token vectors.

---

## 2. Early Approach: Fixed Sinusoidal Encoding

The original 2017 Transformer used **sinusoidal positional encodings** — fixed mathematical patterns added to the embedding:

$$PE(pos, 2i) = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \quad PE(pos, 2i+1) = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

This worked but had a major flaw: the model cannot generalise beyond the training context length.

---

## 3. RoPE — Rotary Positional Encoding

**RoPE** (Su et al., 2021) takes a different approach: instead of **adding** position to the vector, it **rotates** the query and key vectors based on position. This is smarter because:

1. Rotation preserves the **dot product** relationship between Q and K (the core of attention)
2. The **relative** position between two tokens is automatically encoded
3. The model learns from relative positions, which generalises better

### The "Clock Hands" Analogy

Think of each pair of dimensions in a vector as a clock hand. At position 0, all hands point to 12 o'clock. At position 1, each hand rotates by a different angle. At position 2, it rotates twice as much.

Different dimension pairs rotate at different speeds — high-frequency dimensions (fast rotation) encode **short-range** information; low-frequency dimensions (slow rotation) encode **long-range** structure.

---

## 4. The RoPE Math

### Step 1: Build the Frequency Table

For a head of dimension $d$, we define $d/2$ frequency values:

$$\theta_i = \frac{1}{10000^{2i/d}}, \quad i = 0, 1, \ldots, \frac{d}{2}-1$$

For positions $0, 1, 2, \ldots$, we compute angles:

$$\phi_{pos,i} = pos \times \theta_i$$

Then the rotation tables:
$$\cos\_\text{cache}[pos] = [\cos\phi_{pos,0},\cos\phi_{pos,0}, \cos\phi_{pos,1}, \cos\phi_{pos,1}, \ldots]$$
$$\sin\_\text{cache}[pos] = [\sin\phi_{pos,0},\sin\phi_{pos,0}, \sin\phi_{pos,1}, \sin\phi_{pos,1}, \ldots]$$

(Each value is repeated twice — once for the even index, once for the odd index.)

### Step 2: Rotate a Vector

Given a query vector $\mathbf{q}$, we apply the rotation:

$$\mathbf{q}_{\text{rot}} = \mathbf{q} \odot \cos + \text{rotate\_half}(\mathbf{q}) \odot \sin$$

where $\odot$ is element-wise multiplication, and `rotate_half` rearranges pairs:

$$\text{rotate\_half}([x_0, x_1, x_2, x_3, \ldots]) = [-x_1, x_0, -x_3, x_2, \ldots]$$

This is a **2D rotation** applied independently to each pair of dimensions:

$$\begin{pmatrix} x_0' \\ x_1' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x_0 \\ x_1 \end{pmatrix}$$

---

## 5. YaRN — Extending Context Without Retraining

A model trained on sequences of length 8,192 will fail on sequences of length 65,536 — the positions it has never seen lead to garbage output.

**YaRN** (Yet Another RoPE extensioN) scales the frequency table so the model can handle longer sequences:

### Three-Regime Scaling

For each frequency dimension $i$, YaRN checks the **wavelength** $\lambda_i = 2\pi / \theta_i$:

$$\theta_i^{\text{scaled}} = \begin{cases}
\theta_i & \text{if } \lambda_i < \beta_{\text{fast}} \quad \text{(high-freq: no scaling)} \\
\theta_i / s & \text{if } \lambda_i > s \cdot \beta_{\text{slow}} \quad \text{(low-freq: full scaling)} \\
\theta_i / (t \cdot s + (1-t)) & \text{otherwise} \quad \text{(mid-freq: smooth blend)}
\end{cases}$$

where $s$ is the scale factor (e.g., $s=4$ means extending 4× beyond training length), and:

$$t = \frac{\lambda_i/\beta_{\text{slow}} - 1}{s - 1}$$

**Intuition:**
- **High-frequency** dimensions encode local patterns (nearby tokens). Local patterns do not change with longer context — no scaling needed.
- **Low-frequency** dimensions encode global structure. With 4× longer sequences, each unit of position should "feel" the same — so divide by 4.
- **Mid-frequency** dimensions get a smooth blend.

### Temperature Correction

YaRN also adjusts the attention scale to prevent entropy collapse:

$$\text{attn\_factor} = 0.1 \times \ln(s) + 1.0$$

---

## 6. Full Annotated Source: `apex/model/rope.py`

```python
"""
Rotary Positional Encoding (RoPE) with YaRN Extension.

BUG-22 FIX: apply_yarn_scaling is now fully vectorised using
torch.where instead of a Python for loop over dimensions.
"""

import math
from typing import Optional
import torch


def precompute_rope_cache(
    d_head: int,
    max_seq_len: int,
    rope_base: float = 10000.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute sin/cos rotation tables for RoPE.
    
    Returns:
        (cos_cache, sin_cache) each of shape [max_seq_len, d_head]
    """
    # Build frequency vector θ_i = 1 / (base^(2i/d))
    # arange(0, d_head, 2) gives [0, 2, 4, ..., d_head-2]
    i = torch.arange(0, d_head, 2, dtype=dtype, device=device)
    theta = 1.0 / (rope_base ** (i / d_head))   # shape: [d_head/2]

    # Outer product: positions [0..max_seq_len-1] × frequencies
    # positions shape: [max_seq_len]
    # angles shape:    [max_seq_len, d_head/2]
    positions = torch.arange(max_seq_len, dtype=dtype, device=device)
    angles = torch.outer(positions, theta)

    # Each frequency is used for two dimensions (even and odd index)
    # repeat_interleave(2): [θ₀,θ₁,...] → [θ₀,θ₀,θ₁,θ₁,...]
    cos_cache = torch.cos(angles).repeat_interleave(2, dim=-1)  # [max_seq, d_head]
    sin_cache = torch.sin(angles).repeat_interleave(2, dim=-1)  # [max_seq, d_head]

    return cos_cache, sin_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate alternate dimension pairs: [-x₁, x₀, -x₃, x₂, ...]
    
    This implements the 'rotate by 90 degrees' part of the rotation matrix.
    
    Args:
        x: Input tensor of shape [..., d_head] where d_head is even.
    
    Returns:
        Rotated tensor of the same shape.
    """
    x1 = x[..., ::2]    # Take every even index: [x₀, x₂, x₄, ...]
    x2 = x[..., 1::2]   # Take every odd index:  [x₁, x₃, x₅, ...]
    # Stack [-x₁, x₀] then flatten the last two dims
    return torch.stack([-x2, x1], dim=-1).flatten(-2)
    # Result: [-x₁, x₀, -x₃, x₂, -x₅, x₄, ...]


def apply_rope(
    q: torch.Tensor,        # Query: [batch, n_heads, seq_len, d_head]
    k: torch.Tensor,        # Key:   [batch, n_heads, seq_len, d_head]
    cos_cache: torch.Tensor,# [max_seq_len, d_head]
    sin_cache: torch.Tensor,# [max_seq_len, d_head]
    positions: torch.Tensor,# [seq_len] — absolute position of each token
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to Query and Key tensors."""
    
    # Gather the cos/sin values for our specific positions
    # cos_cache[positions]: [seq_len, d_head]
    # .unsqueeze(0).unsqueeze(0): [1, 1, seq_len, d_head]  (for broadcasting)
    cos = cos_cache[positions].unsqueeze(0).unsqueeze(0)
    sin = sin_cache[positions].unsqueeze(0).unsqueeze(0)

    # Trim to actual head dimension (cache may be larger)
    d = q.shape[-1]
    cos = cos[..., :d]
    sin = sin[..., :d]

    # Apply rotation: q_rot = q * cos + rotate_half(q) * sin
    # This is equivalent to rotating each pair (q[2i], q[2i+1]) by angle θ_i
    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


def apply_yarn_scaling(
    theta: torch.Tensor,   # Original frequencies [d_head/2]
    scale_factor: float,   # Target context / training context ratio
    d_head: int,
    beta_fast: float = 32.0,   # High-frequency wavelength cutoff
    beta_slow: float = 1.0,    # Low-frequency multiplier
) -> tuple[torch.Tensor, float]:
    """Apply YaRN frequency scaling for context extension.
    
    BUG-22 FIX: This was a Python for loop (O(d_head) Python calls).
    Now it is fully vectorised with torch.where — a single fused kernel.
    """
    if scale_factor <= 1.0:
        return theta.clone(), 1.0

    # Wavelength for each dimension: λ_i = 2π / θ_i
    wavelength = 2.0 * math.pi / theta.clamp(min=1e-30)

    # Mid-frequency interpolation blend factor
    # t = 0 → pure original; t = 1 → full scaling
    t = (wavelength / beta_slow - 1.0) / (scale_factor - 1.0)
    mid_divisor = t * scale_factor + (1.0 - t)

    # Start with mid-frequency scaling for all dimensions
    scaled_theta = theta / mid_divisor

    # Override: low-frequency → divide by scale_factor (full scaling)
    low_freq_mask = wavelength > beta_slow * scale_factor
    scaled_theta = torch.where(low_freq_mask, theta / scale_factor, scaled_theta)

    # Override: high-frequency → keep original (no scaling)
    # Applied LAST for highest priority
    high_freq_mask = wavelength < beta_fast
    scaled_theta = torch.where(high_freq_mask, theta, scaled_theta)

    # Temperature correction to prevent attention entropy collapse
    attn_factor = 0.1 * math.log(scale_factor) + 1.0

    return scaled_theta, attn_factor


def precompute_rope_cache_with_yarn(
    d_head: int,
    max_seq_len: int,
    rope_base: float = 10000.0,
    scale_factor: float = 1.0,  # 1.0 = plain RoPE, >1.0 = YaRN
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Precompute RoPE cache with optional YaRN scaling."""
    
    # Base frequencies
    i = torch.arange(0, d_head, 2, dtype=dtype, device=device)
    theta = 1.0 / (rope_base ** (i / d_head))

    attn_factor = 1.0
    if scale_factor > 1.0:
        # Apply YaRN frequency scaling
        theta, attn_factor = apply_yarn_scaling(theta, scale_factor, d_head)
        if device is not None:
            theta = theta.to(device)

    # Build cos/sin tables from scaled frequencies
    positions = torch.arange(max_seq_len, dtype=dtype, device=device)
    angles = torch.outer(positions, theta)
    cos_cache = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin_cache = torch.sin(angles).repeat_interleave(2, dim=-1)

    return cos_cache, sin_cache, attn_factor
```

---

## 7. The Two RoPE Caches in APEX-1

APEX-1 has **two types of attention layers** that use RoPE differently:

| Layer Type | RoPE Dimension | Cache Name |
|---|---|---|
| GQA (local layers) | `d_head` (full head dim) | `cos_cache`, `sin_cache` |
| MLA (global layers) | `d_head_rope` (smaller) | `cos_cache_rope`, `sin_cache_rope` |

Both caches are precomputed once in `APEX1Model.__init__()` and reused for every layer. This was a **critical bug** (BUG-07): originally only one cache was used for both types, causing a shape mismatch crash.

---

## 8. Visual Summary

```
Position 0: vector not rotated (all at 0°)
Position 1: each pair rotated by θ_i degrees
Position 2: each pair rotated by 2×θ_i degrees
...

High-freq dims: rotate fast → good for nearby token relationships
Low-freq dims:  rotate slow → good for long-range relationships

YaRN at 4× scale: 
  High-freq dims: same rotation (local info unchanged)
  Low-freq dims:  slower rotation (long-range still works at 4× length)
```

---

**Next:** [06 — Attention Masks →](06-attention-masks.md)
