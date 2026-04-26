<div align="center">

# 🔺 APEX-1 Mathematical Reference — Part 2

### Attention: MLA, GQA+SW, Masks & Scaled Dot-Product

</div>

---

## Table of Contents — Part 2

| # | Topic | Section |
|---|---|---|
| 6 | Scaled Dot-Product Attention | §6 |
| 7 | Multi-Head Attention (MHA) | §7 |
| 8 | Grouped Query Attention (GQA) | §8 |
| 9 | Multi-Head Latent Attention (MLA) | §9 |
| 10 | Sliding Window Attention | §10 |
| 11 | Attention Mask Builder | §11 |

---

## §6 — Scaled Dot-Product Attention

### 6.1 What It Does

The core operation of all transformers. Each token "queries" all other tokens, computes a relevance score, and takes a weighted average of their values.

### 6.2 Formula

$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\!\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_h}}\right) \mathbf{V}$$

Where:
- $\mathbf{Q} \in \mathbb{R}^{S_q \times d_h}$ — Query matrix (what am I looking for?)
- $\mathbf{K} \in \mathbb{R}^{S_k \times d_h}$ — Key matrix (what do I contain?)
- $\mathbf{V} \in \mathbb{R}^{S_k \times d_h}$ — Value matrix (what do I output?)
- $\sqrt{d_h}$ — scaling factor to prevent softmax saturation

### 6.3 Why Scale by √d_h?

Without scaling, dot products grow proportionally to $d_h$. For large $d_h$:
- Dot products become very large → softmax pushes to one-hot → gradients vanish
- Scaling keeps variance ≈ 1.0 regardless of $d_h$

Proof: if $q_i, k_i \sim \mathcal{N}(0,1)$, then $\text{Var}(\mathbf{q} \cdot \mathbf{k}) = d_h$, so dividing by $\sqrt{d_h}$ gives $\text{Var} = 1$.

### 6.4 Step-by-Step Example

```
Q = [[1, 0],     K = [[1, 0],     V = [[1, 2],
     [0, 1]]          [0, 1],          [3, 4],
                       [1, 1]]          [5, 6]]
d_h = 2

Step 1 — Compute QK^T:
  [[1×1+0×0, 1×0+0×1, 1×1+0×1],   [[1, 0, 1],
   [0×1+1×0, 0×0+1×1, 0×1+1×1]] =  [0, 1, 1]]

Step 2 — Scale by √2 ≈ 1.414:
  [[0.707, 0.000, 0.707],
   [0.000, 0.707, 0.707]]

Step 3 — Softmax (per row):
  Row 0: exp([0.707, 0.000, 0.707]) = [2.028, 1.000, 2.028]
         → normalize: [0.401, 0.198, 0.401]
  Row 1: exp([0.000, 0.707, 0.707]) = [1.000, 2.028, 2.028]
         → normalize: [0.198, 0.401, 0.401]

Step 4 — Multiply by V:
  Row 0: 0.401×[1,2] + 0.198×[3,4] + 0.401×[5,6]
       = [0.401+0.594+2.005, 0.802+0.792+2.406]
       = [3.000, 4.000]
  Row 1: 0.198×[1,2] + 0.401×[3,4] + 0.401×[5,6]
       = [0.198+1.203+2.005, 0.396+1.604+2.406]
       = [3.406, 4.406]

Output: [[3.000, 4.000],
         [3.406, 4.406]]
```

---

## §7 — Multi-Head Attention (MHA)

### 7.1 What It Does

Runs $H$ parallel attention operations, each focusing on different aspects of the input (syntax, semantics, coreference, etc.).

### 7.2 Formula

$$\text{MHA}(\mathbf{x}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \cdot \mathbf{W}_O$$

$$\text{head}_i = \text{Attention}(\mathbf{x}\mathbf{W}_{Q_i}, \mathbf{x}\mathbf{W}_{K_i}, \mathbf{x}\mathbf{W}_{V_i})$$

### 7.3 Parameter Count

| Matrix | Shape | Count |
|---|---|---|
| $\mathbf{W}_{Q}$ | $[d, H_q \times d_h]$ | $d \times H_q \times d_h$ |
| $\mathbf{W}_{K}$ | $[d, H_q \times d_h]$ | $d \times H_q \times d_h$ |
| $\mathbf{W}_{V}$ | $[d, H_q \times d_h]$ | $d \times H_q \times d_h$ |
| $\mathbf{W}_{O}$ | $[H_q \times d_h, d]$ | $H_q \times d_h \times d$ |
| **Total** | | $4 \times d \times H_q \times d_h = 4d^2$ |

### 7.4 KV Cache Problem

During autoregressive generation, K and V must be **cached** for all previous tokens. Memory cost per layer:

$$\text{KV cache} = 2 \times B \times S \times H_q \times d_h \times \text{bytes}$$

For a 128K context with 128 heads × 128 dim in FP16: **4 GB per layer** → motivates GQA and MLA.

---

## §8 — Grouped Query Attention (GQA)

### 8.1 What It Does

Reduces KV cache by sharing KV heads across groups of Q heads. APEX-1 uses GQA on **local** (non-global) layers.

### 8.2 Formula

$$G = \frac{H_q}{H_{kv}}$$

Each KV head is shared by $G$ query heads:

$$\text{head}_{i} = \text{Attention}\!\left(\mathbf{x}\mathbf{W}_{Q_i},\; \mathbf{x}\mathbf{W}_{K_{\lfloor i/G \rfloor}},\; \mathbf{x}\mathbf{W}_{V_{\lfloor i/G \rfloor}}\right)$$

### 8.3 Memory Savings

| Method | KV heads | KV cache size | Relative |
|---|---|---|---|
| MHA | $H_q = 128$ | $2 \times 128 \times d_h \times S$ | 1.0× |
| GQA ($H_{kv}=8$) | 8 | $2 \times 8 \times d_h \times S$ | **0.0625×** (16× smaller) |
| MQA ($H_{kv}=1$) | 1 | $2 \times 1 \times d_h \times S$ | 0.0078× |

APEX-1 Large: $H_q = 128$, $H_{kv} = 8$ → $G = 16$ Q heads per KV head.

### 8.4 Example

```
H_q = 4, H_kv = 2, G = 4/2 = 2

Q heads: [Q₀, Q₁, Q₂, Q₃]
K heads: [K₀, K₁]
V heads: [V₀, V₁]

Assignment:
  Q₀, Q₁ → share K₀, V₀  (group 0)
  Q₂, Q₃ → share K₁, V₁  (group 1)

Each Q head computes attention with its assigned K,V:
  head₀ = Attn(Q₀, K₀, V₀)
  head₁ = Attn(Q₁, K₀, V₀)  ← same K,V as head₀
  head₂ = Attn(Q₂, K₁, V₁)
  head₃ = Attn(Q₃, K₁, V₁)  ← same K,V as head₂
```

---

## §9 — Multi-Head Latent Attention (MLA)

### 9.1 What It Does

Compresses all KV information into a tiny **latent vector** $\mathbf{c}_{kv}$, caching only this instead of full K and V. Used on APEX-1's **global** layers.

### 9.2 Compression/Decompression Formulas

**KV Compression** (input → latent):

$$\mathbf{c}_{kv} = \mathbf{x} \cdot \mathbf{W}_{DKV} \quad \in \mathbb{R}^{S \times d_{kv}}$$

**KV Decompression** (latent → full K, V):

$$\mathbf{K} = \mathbf{c}_{kv} \cdot \mathbf{W}_{UK} \quad \in \mathbb{R}^{S \times H_{kv} \times d_h}$$
$$\mathbf{V} = \mathbf{c}_{kv} \cdot \mathbf{W}_{UV} \quad \in \mathbb{R}^{S \times H_{kv} \times d_h}$$

**Q Compression** (similar):

$$\mathbf{c}_q = \mathbf{x} \cdot \mathbf{W}_{DQ} \quad \in \mathbb{R}^{S \times d_q}$$
$$\mathbf{Q} = \mathbf{c}_q \cdot \mathbf{W}_{UQ} \quad \in \mathbb{R}^{S \times H_q \times d_h}$$

### 9.3 Decoupled RoPE

Positional encoding is applied via **separate** projections, not through the compressed latent (which would lose position info):

$$\mathbf{Q}_{\text{rope}} = \text{RoPE}(\mathbf{x} \cdot \mathbf{W}_{QR}, m)$$
$$\mathbf{K}_{\text{rope}} = \text{RoPE}(\mathbf{x} \cdot \mathbf{W}_{KR}, m)$$

Final Q and K are concatenated:

$$\mathbf{Q}_{\text{final}} = [\mathbf{Q}_{\text{content}} \| \mathbf{Q}_{\text{rope}}] \quad \in \mathbb{R}^{S \times H_q \times (d_h + d_r)}$$
$$\mathbf{K}_{\text{final}} = [\mathbf{K}_{\text{content}} \| \mathbf{K}_{\text{rope}}] \quad \in \mathbb{R}^{S \times H_{kv} \times (d_h + d_r)}$$

### 9.4 Cache Size Comparison

| Method | What is cached | Size per token | Relative |
|---|---|---|---|
| Standard MHA | Full K + V | $2 \times H_q \times d_h$ | 1.0× |
| GQA | K + V (fewer heads) | $2 \times H_{kv} \times d_h$ | $H_{kv}/H_q$ |
| **MLA** | Only $\mathbf{c}_{kv}$ | $d_{kv}$ | **$d_{kv} / (2 H_q d_h)$** |

**APEX-1 Large**: $d_{kv} = 512$, $H_q = 128$, $d_h = 128$

$$\text{MLA ratio} = \frac{512}{2 \times 128 \times 128} = \frac{512}{32768} = 1.6\% \quad (\textbf{98.4\% reduction!})$$

### 9.5 MLA Parameter Count

| Matrix | Shape | Purpose |
|---|---|---|
| $\mathbf{W}_{DKV}$ | $[d, d_{kv}]$ | Compress to KV latent |
| $\mathbf{W}_{UK}$ | $[d_{kv}, H_{kv} \times d_h]$ | Decompress to K |
| $\mathbf{W}_{UV}$ | $[d_{kv}, H_{kv} \times d_h]$ | Decompress to V |
| $\mathbf{W}_{DQ}$ | $[d, d_q]$ | Compress to Q latent |
| $\mathbf{W}_{UQ}$ | $[d_q, H_q \times d_h]$ | Decompress to Q |
| $\mathbf{W}_{QR}$ | $[d, H_q \times d_r]$ | Q RoPE projection |
| $\mathbf{W}_{KR}$ | $[d, H_{kv} \times d_r]$ | K RoPE projection |
| $\mathbf{W}_O$ | $[H_q \times d_h, d]$ | Output projection |

---

## §10 — Sliding Window Attention

### 10.1 What It Does

Limits attention to the most recent $W$ tokens instead of the full sequence. Makes local layers $O(S \times W)$ instead of $O(S^2)$.

### 10.2 Mask Formula

$$\text{mask}[i, j] = \begin{cases} 1 & \text{if } 0 \leq i - j < W \text{ and } j \leq i \\ 0 & \text{otherwise} \end{cases}$$

### 10.3 Complexity Comparison

| Method | Time complexity | Memory | Best for |
|---|---|---|---|
| Full causal | $O(S^2 \cdot d_h)$ | $O(S^2)$ | Global reasoning |
| Sliding window | $O(S \cdot W \cdot d_h)$ | $O(S \cdot W)$ | Local syntax |
| Ratio | | $W/S$ smaller | $W \ll S$ |

For APEX-1 Large: $S = 128\text{K}$, $W = 4\text{K}$ → sliding window is **32× cheaper**.

---

## §11 — APEX-1 Attention Mask Builder

### 11.1 Three Attention Regimes

APEX-1 combines three masking strategies:

| Regime | Positions | Behavior |
|---|---|---|
| **Prefix bidirectional** | $[0, P)$ | All prefix tokens attend to all prefix tokens |
| **Global causal** | $[P, S)$ on global layers | Full causal attention over entire history |
| **Local causal + window** | $[P, S)$ on local layers | Causal limited to window $W$ |

### 11.2 Layer Assignment Rule

$$\text{is\_global}(l) = \left( l \bmod F_g \right) = \left( F_g - 1 \right)$$

With $F_g = 6$: layers 5, 11, 17, 23, ... are **global** (MLA), all others are **local** (GQA+SW).

| Layer | 0 | 1 | 2 | 3 | 4 | **5** | 6 | 7 | 8 | 9 | 10 | **11** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Type | L | L | L | L | L | **G** | L | L | L | L | L | **G** |

### 11.3 Visual: How Information Flows

```
                    Global layer (full causal)
Position:  0  1  2  3  4  5  6  7  8  9
    0:     █  █  █  █  ·  ·  ·  ·  ·  ·   ← prefix bidir
    1:     █  █  █  █  ·  ·  ·  ·  ·  ·
    2:     █  █  █  █  ·  ·  ·  ·  ·  ·
    3:     █  █  █  █  ·  ·  ·  ·  ·  ·
    4:     █  █  █  █  █  ·  ·  ·  ·  ·   ← full causal
    5:     █  █  █  █  █  █  ·  ·  ·  ·
    9:     █  █  █  █  █  █  █  █  █  █

                    Local layer (window=4)
Position:  0  1  2  3  4  5  6  7  8  9
    0:     █  █  █  █  ·  ·  ·  ·  ·  ·   ← prefix bidir
    3:     █  █  █  █  ·  ·  ·  ·  ·  ·
    4:     ·  █  █  █  █  ·  ·  ·  ·  ·   ← window only
    7:     ·  ·  ·  ·  █  █  █  █  ·  ·
    9:     ·  ·  ·  ·  ·  ·  █  █  █  █
```

---

## Summary — Part 2 Formula Quick Reference

| Component | Formula | Cache Size |
|---|---|---|
| **SDPA** | $\text{softmax}(\mathbf{QK}^T / \sqrt{d_h}) \mathbf{V}$ | — |
| **GQA** | Share KV across $G = H_q / H_{kv}$ groups | $2 H_{kv} d_h S$ |
| **MLA compress** | $\mathbf{c}_{kv} = \mathbf{x} \mathbf{W}_{DKV}$ | $d_{kv} S$ |
| **MLA decompress** | $\mathbf{K} = \mathbf{c}_{kv} \mathbf{W}_{UK}$ | on-the-fly |
| **Sliding window** | mask: $0 \leq i-j < W$ | $2 H_{kv} d_h W$ |
| **Global/local rule** | $l \bmod 6 = 5$ → global | — |

---

*← Part 1: Foundations | Continue to **Part 3**: FFN, MoE, Skip Gate, Multi-Token →*
