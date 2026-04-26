<div align="center">

# 🔺 APEX-1 Mathematical Reference — Part 1

### Foundations: Embedding, RMSNorm, RoPE & YaRN

</div>

---

## Table of Contents — Part 1

| # | Topic | Section |
|---|---|---|
| 1 | Notation & Symbol Table | §1 |
| 2 | Input Embedding with Scaling | §2 |
| 3 | RMSNorm (Root Mean Square Normalization) | §3 |
| 4 | RoPE (Rotary Position Embedding) | §4 |
| 5 | YaRN (Yet another RoPE extensioN) | §5 |

---

## §1 — Notation & Symbol Table

Every formula in this reference uses the following consistent notation:

| Symbol | Meaning | Typical Value |
|---|---|---|
| $B$ | Batch size | 1–64 |
| $S$ | Sequence length (number of tokens) | 2048–131072 |
| $d$ | Model hidden dimension (`d_model`) | 512–7168 |
| $d_h$ | Per-head dimension (`d_head`) | 64–128 |
| $d_r$ | RoPE head dimension (`d_head_rope`) | 32–64 |
| $d_{ff}$ | FFN intermediate dimension | $\frac{8}{3} \times d$ |
| $d_{kv}$ | KV compressed dimension (MLA) | 256–512 |
| $d_q$ | Q compressed dimension (MLA) | 192–1536 |
| $H_q$ | Number of query heads | 8–128 |
| $H_{kv}$ | Number of KV heads | 2–8 |
| $L$ | Number of transformer layers | 6–72 |
| $V$ | Vocabulary size | 151,643 |
| $E$ | Number of MoE experts | 4–256 |
| $K$ | Number of active experts per token | 2–8 |
| $W$ | Sliding window size | 512–4096 |
| $\epsilon$ | Numerical stability constant | $10^{-6}$ |
| $\gamma$ | Learnable scale parameter (RMSNorm) | init: 1.0 |
| $\theta_i$ | RoPE frequency for dimension pair $i$ | varies |
| $m$ | Token position index | 0 to $S-1$ |

### Tensor Shape Convention

All tensors follow the shape `[Batch, Sequence, Dimension]` for 3D and `[Batch, Heads, Sequence, HeadDim]` for 4D attention tensors.

---

## §2 — Input Embedding with Scaling

### 2.1 What It Does

Converts discrete token IDs (integers) into continuous vectors that the neural network can process. APEX-1 scales the embedding by $\sqrt{d}$ to stabilize variance early in the network.

### 2.2 Formula

$$\mathbf{X} = \text{Embed}(\text{tokens}) \times \sqrt{d}$$

Where:
- $\text{Embed}: \mathbb{Z}^{B \times S} \rightarrow \mathbb{R}^{B \times S \times d}$ is a lookup table of shape $[V, d]$
- The $\sqrt{d}$ scaling ensures the initial activations have variance ≈ 1.0

### 2.3 Why Scale by √d?

Without scaling, embedding vectors have variance ≈ $\frac{1}{d}$ (since each element is ~$\mathcal{N}(0, 0.02^2)$). Multiplying by $\sqrt{d}$ restores variance to ~1.0, which:
- Prevents vanishing activations in early layers
- Matches the expected input scale of RMSNorm
- Used by: T5, PaLM, Gemma, APEX-1

### 2.4 Numerical Example

```
d_model = 4 (tiny example)
Vocabulary = {0: "the", 1: "cat", 2: "sat"}

Embedding table (randomly initialized):
  Token 0 → [0.01, -0.03, 0.02, 0.01]
  Token 1 → [-0.02, 0.04, -0.01, 0.03]
  Token 2 → [0.03, 0.01, -0.04, 0.02]

Input: [1, 0, 2] → "cat the sat"

Step 1 — Lookup:
  [[−0.02, 0.04, −0.01, 0.03],
   [ 0.01,−0.03,  0.02, 0.01],
   [ 0.03, 0.01, −0.04, 0.02]]

Step 2 — Scale by √4 = 2.0:
  [[−0.04, 0.08, −0.02, 0.06],
   [ 0.02,−0.06,  0.04, 0.02],
   [ 0.06, 0.02, −0.08, 0.04]]
```

### 2.5 Weight Tying

The embedding table $\mathbf{W}_E \in \mathbb{R}^{V \times d}$ is **shared** with the LM head:

$$\text{logits} = \mathbf{h} \cdot \mathbf{W}_E^T$$

This saves $V \times d$ parameters (e.g., 151,643 × 7,168 = **1.09B params** saved for Large).

---

## §3 — RMSNorm (Root Mean Square Normalization)

### 3.1 What It Does

Normalizes activations to stabilize training. Unlike LayerNorm, RMSNorm **does not center** (no mean subtraction), making it 20-40% faster with equal quality.

### 3.2 Formula

$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \cdot \gamma$$

Where:

$$\text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}$$

- $\gamma \in \mathbb{R}^d$ is a **learnable scale** parameter (initialized to 1.0)
- $\epsilon = 10^{-6}$ prevents division by zero

### 3.3 Comparison: LayerNorm vs RMSNorm

| Property | LayerNorm | RMSNorm |
|---|---|---|
| Mean centering | ✅ Yes ($x - \mu$) | ❌ No |
| Variance normalization | ✅ Yes | ✅ Yes (via RMS) |
| Learnable parameters | $\gamma, \beta$ (2d params) | $\gamma$ only (d params) |
| Speed | Baseline | **20-40% faster** |
| Quality | Baseline | Equal or better |
| Used by | GPT-2, BERT | Llama, DeepSeek, Qwen, APEX-1 |

### 3.4 Step-by-Step Example

```
Input: x = [2.0, -1.0, 0.5, 1.5]    (d = 4)
γ = [1.0, 1.0, 1.0, 1.0]            (initial)
ε = 1e-6

Step 1 — Compute mean of squares:
  mean(x²) = (4.0 + 1.0 + 0.25 + 2.25) / 4 = 7.5 / 4 = 1.875

Step 2 — Compute RMS:
  RMS = √(1.875 + 1e-6) = √1.875001 ≈ 1.3693

Step 3 — Normalize:
  x / RMS = [2.0/1.3693, -1.0/1.3693, 0.5/1.3693, 1.5/1.3693]
          = [1.4606, -0.7303, 0.3651, 1.0954]

Step 4 — Scale by γ:
  output = [1.4606, -0.7303, 0.3651, 1.0954] × [1, 1, 1, 1]
         = [1.4606, -0.7303, 0.3651, 1.0954]

Verification: mean(output²) = (2.133 + 0.533 + 0.133 + 1.200)/4 ≈ 1.0 ✓
```

### 3.5 Where Used in APEX-1

RMSNorm appears **twice** in every transformer block:
1. **Pre-attention norm**: `RMSNorm(x)` → Attention
2. **Pre-FFN norm**: `RMSNorm(x')` → FFN

Plus once at the end: **Final norm** before the LM head.

Total: $2L + 1$ RMSNorm layers ($L$ = number of layers).

---

## §4 — RoPE (Rotary Position Embedding)

### 4.1 What It Does

Encodes **position information** into Query and Key vectors by **rotating** pairs of dimensions. The rotation angle depends on position, so the model learns relative distances between tokens.

### 4.2 Core Idea

For each pair of dimensions $(2i, 2i+1)$, apply a 2D rotation by angle $m \cdot \theta_i$:

$$\begin{bmatrix} x'_{2i} \\ x'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(m\theta_i) & -\sin(m\theta_i) \\ \sin(m\theta_i) & \cos(m\theta_i) \end{bmatrix} \begin{bmatrix} x_{2i} \\ x_{2i+1} \end{bmatrix}$$

Where:
- $m$ = position index (0, 1, 2, ...)
- $\theta_i = \frac{1}{b^{2i/d_h}}$ = frequency for dimension pair $i$
- $b = 10{,}000$ = base frequency (standard)

### 4.3 Frequency Formula

$$\theta_i = \frac{1}{10000^{2i / d_h}} \quad \text{for } i = 0, 1, \ldots, \frac{d_h}{2} - 1$$

| Dim pair $i$ | $\theta_i$ ($d_h=8$) | Wavelength | What it encodes |
|---|---|---|---|
| 0 | 1.0000 | 6.28 | Very local (adjacent tokens) |
| 1 | 0.0316 | 198.7 | Short-range (phrases) |
| 2 | 0.0010 | 6,283 | Medium-range (paragraphs) |
| 3 | 0.00003 | 198,692 | Long-range (document-level) |

**Key insight**: Low dimensions rotate fast (local patterns), high dimensions rotate slow (global patterns).

### 4.4 Compact Form

The full rotation can be written compactly as:

$$\text{RoPE}(\mathbf{x}, m) = \mathbf{x} \odot \cos(\mathbf{m\Theta}) + \text{rotate-half}(\mathbf{x}) \odot \sin(\mathbf{m\Theta})$$

Where $\text{rotate-half}$ swaps and negates adjacent pairs: $[x_0, x_1, x_2, x_3] \rightarrow [-x_1, x_0, -x_3, x_2]$

### 4.5 Why Rotation Encodes Relative Position

The **inner product** of two rotated vectors depends only on the **relative distance**:

$$\langle \text{RoPE}(\mathbf{q}, m), \text{RoPE}(\mathbf{k}, n) \rangle = f(\mathbf{q}, \mathbf{k}, m - n)$$

This means: attention scores naturally depend on *how far apart* two tokens are, not their absolute position. This is crucial for generalization to longer sequences.

### 4.6 Step-by-Step Example

```
d_head = 4, base = 10000
Position m = 3

Step 1 — Compute frequencies:
  θ₀ = 1 / 10000^(0/4) = 1.0
  θ₁ = 1 / 10000^(2/4) = 1 / 100 = 0.01

Step 2 — Compute angles at position 3:
  angle₀ = 3 × 1.0  = 3.0 rad
  angle₁ = 3 × 0.01 = 0.03 rad

Step 3 — Compute cos/sin:
  cos = [cos(3.0), cos(3.0), cos(0.03), cos(0.03)]
      = [−0.990, −0.990, 0.9996, 0.9996]
  sin = [sin(3.0), sin(3.0), sin(0.03), sin(0.03)]
      = [0.141, 0.141, 0.030, 0.030]

Step 4 — Apply to query q = [1.0, 0.5, 0.3, -0.2]:
  rotate_half(q) = [-0.5, 1.0, 0.2, 0.3]

  q' = q × cos + rotate_half(q) × sin
     = [1.0×(−0.990) + (−0.5)×0.141,
        0.5×(−0.990) + 1.0×0.141,
        0.3×0.9996 + 0.2×0.030,
        (−0.2)×0.9996 + 0.3×0.030]
     = [−1.061, −0.354, 0.306, −0.191]

Note: magnitude is preserved: ||q|| = ||q'|| ≈ 1.187 ✓
```

### 4.7 Precomputation

RoPE cos/sin tables are computed **once** at model init and reused:

```python
# Precompute for all positions up to max_seq_len
cos_cache[m, 2i] = cos_cache[m, 2i+1] = cos(m × θᵢ)
sin_cache[m, 2i] = sin_cache[m, 2i+1] = sin(m × θᵢ)
# Shape: [max_seq_len, d_head]
```

---

## §5 — YaRN (Yet another RoPE extensioN)

### 5.1 What It Does

Extends a model trained at context length $C$ to work at $C' = s \times C$ (e.g., 4K → 128K) **without retraining**. It selectively scales RoPE frequencies.

### 5.2 The Problem YaRN Solves

Naive extrapolation fails because:
- High-frequency dimensions see angles they never trained on → noise
- Simply dividing all frequencies by $s$ breaks local syntax patterns

YaRN's insight: **different dimensions need different treatment**.

### 5.3 Three Frequency Regimes

| Regime | Condition | Action | Rationale |
|---|---|---|---|
| **High-frequency** | wavelength $< \beta_{\text{fast}}$ | No scaling | Local syntax works at any length |
| **Low-frequency** | wavelength $> \beta_{\text{slow}} \times s$ | Full scaling: $\theta'_i = \theta_i / s$ | Long-range position needs compression |
| **Mid-frequency** | Between the two | Smooth interpolation | Gradual transition |

Default: $\beta_{\text{fast}} = 32$, $\beta_{\text{slow}} = 1$

### 5.4 Interpolation Formula

For mid-frequency dimensions:

$$t = \frac{\lambda_i / \beta_{\text{slow}} - 1}{s - 1}$$

$$\theta'_i = \frac{\theta_i}{t \cdot s + (1 - t)}$$

Where $\lambda_i = \frac{2\pi}{\theta_i}$ is the wavelength.

### 5.5 Attention Temperature Correction

At long context, attention entropy increases (scores become more uniform). YaRN counteracts this with a temperature correction:

$$\text{attn-factor} = 0.1 \times \ln(s) + 1.0$$

This is multiplied into attention scores to maintain sharpness.

### 5.6 Example: Extending 4K to 128K

```
Scale factor s = 128K / 4K = 32
β_fast = 32, β_slow = 1

Dimension pair i=0: θ = 1.0, λ = 6.28
  λ = 6.28 < 32 (β_fast) → HIGH-FREQ → no scaling
  θ'₀ = 1.0 (unchanged)

Dimension pair i=15: θ = 0.0001, λ = 62,832
  λ = 62,832 > 1 × 32 = 32 (β_slow × s) → LOW-FREQ → full scaling
  θ'₁₅ = 0.0001 / 32 = 0.000003125

Dimension pair i=8: θ = 0.01, λ = 628
  628 > 32 but 628 < 32 → MID-FREQ → interpolate
  t = (628/1 - 1) / (32 - 1) = 627/31 ≈ 20.2 → clamped
  (In practice, interpolation weights depend on exact values)

Temperature correction:
  attn_factor = 0.1 × ln(32) + 1.0 = 0.1 × 3.466 + 1.0 = 1.347
```

### 5.7 Context Extension Capability

| Model | Training Context | Extended Context | Method |
|---|---|---|---|
| Llama 2 | 4K | 32K | Position interpolation |
| KIMI | 128K | 1M+ | YaRN |
| DeepSeek-V3 | 128K | 128K | Native long (b=1M) |
| APEX-1 Small | 8K | 32K | YaRN (s=4) |
| APEX-1 Large | 128K | 1M+ | YaRN (s=8+) |

---

## Summary — Part 1 Formula Quick Reference

| Component | Formula | Params |
|---|---|---|
| **Embedding** | $\mathbf{X} = \text{Embed}(\text{tok}) \times \sqrt{d}$ | $V \times d$ |
| **RMSNorm** | $\frac{\mathbf{x}}{\sqrt{\text{mean}(\mathbf{x}^2) + \epsilon}} \cdot \gamma$ | $d$ |
| **RoPE freq** | $\theta_i = 10000^{-2i/d_h}$ | 0 (fixed) |
| **RoPE apply** | $\mathbf{x}' = \mathbf{x} \cos(m\theta) + \text{rot}(\mathbf{x}) \sin(m\theta)$ | 0 (fixed) |
| **YaRN scale** | $\theta'_i = \theta_i / f(s, \lambda_i, \beta)$ | 0 (fixed) |
| **YaRN temp** | $\text{factor} = 0.1 \ln(s) + 1$ | 0 (fixed) |

---

*Continue to **Part 2**: Attention (MLA, GQA+SW, Masks) →*
