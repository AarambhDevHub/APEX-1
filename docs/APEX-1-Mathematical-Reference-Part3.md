<div align="center">

# 🔺 APEX-1 Mathematical Reference — Part 3

### FFN, MoE, Skip Gate & Multi-Token Prediction

</div>

---

## Table of Contents — Part 3

| # | Topic | Section |
|---|---|---|
| 12 | SwiGLU Feed-Forward Network | §12 |
| 13 | Mixture of Experts (MoE) | §13 |
| 14 | Auxiliary-Loss-Free Load Balancing | §14 |
| 15 | Dynamic Skip Gate | §15 |
| 16 | Multi-Token Prediction | §16 |

---

## §12 — SwiGLU Feed-Forward Network

### 12.1 What It Does

The FFN transforms each token's representation through a non-linear projection. SwiGLU uses **gating** — one branch decides what to keep, another provides the values.

### 12.2 Formula

$$\text{FFN}(\mathbf{x}) = \mathbf{W}_{\text{down}} \left( \text{SiLU}(\mathbf{x} \mathbf{W}_{\text{gate}}) \odot (\mathbf{x} \mathbf{W}_{\text{up}}) \right)$$

Where:
- $\mathbf{W}_{\text{gate}} \in \mathbb{R}^{d \times d_{ff}}$ — gate branch
- $\mathbf{W}_{\text{up}} \in \mathbb{R}^{d \times d_{ff}}$ — value branch
- $\mathbf{W}_{\text{down}} \in \mathbb{R}^{d_{ff} \times d}$ — projection back
- $\text{SiLU}(x) = x \cdot \sigma(x)$ — Sigmoid Linear Unit
- $\odot$ — elementwise multiplication (gating)
- $d_{ff} = \frac{8}{3} d$ — intermediate dimension

### 12.3 SiLU Activation

$$\text{SiLU}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$$

| $x$ | $\sigma(x)$ | $\text{SiLU}(x)$ | Behavior |
|---|---|---|---|
| -3.0 | 0.047 | -0.143 | Near zero (suppressed) |
| -1.0 | 0.269 | -0.269 | Slightly negative |
| 0.0 | 0.500 | 0.000 | Zero |
| 1.0 | 0.731 | 0.731 | Slightly below identity |
| 3.0 | 0.953 | 2.858 | Near identity |

### 12.4 Why SwiGLU Beats ReLU/GELU

| FFN Type | Formula | Params | Perplexity |
|---|---|---|---|
| ReLU FFN | $W_2 \cdot \text{ReLU}(W_1 x)$ | $2d \cdot d_{ff}$ | Baseline |
| GELU FFN | $W_2 \cdot \text{GELU}(W_1 x)$ | $2d \cdot d_{ff}$ | -0.5% |
| **SwiGLU** | $W_{\text{down}}(\text{SiLU}(W_g x) \odot W_u x)$ | $3d \cdot d_{ff}$ | **-1.5%** |

SwiGLU uses 50% more parameters but wins ~1-2% perplexity. The FFN dimension is reduced to $\frac{8}{3}d$ (from $4d$) to compensate, keeping FLOPs similar.

### 12.5 Step-by-Step Example

```
d = 4, d_ff = 3
x = [1.0, -0.5, 0.3, 0.8]

W_gate (4×3):        W_up (4×3):
[[0.5, -0.3, 0.1],   [[0.2, 0.4, -0.1],
 [0.2,  0.4, 0.3],    [0.1, -0.2, 0.5],
 [-0.1, 0.2, 0.5],    [0.3, 0.1, 0.2],
 [0.4, -0.1, 0.2]]    [-0.2, 0.3, 0.1]]

Step 1 — Gate branch: x @ W_gate
  [1×0.5+(-0.5)×0.2+0.3×(-0.1)+0.8×0.4, ...]
  = [0.69, -0.36, 0.31]

Step 2 — Apply SiLU:
  SiLU([0.69, -0.36, 0.31]) = [0.474, -0.163, 0.186]

Step 3 — Value branch: x @ W_up
  = [0.01, 0.39, -0.23]

Step 4 — Gate × Value (elementwise):
  [0.474×0.01, -0.163×0.39, 0.186×(-0.23)]
  = [0.005, -0.064, -0.043]

Step 5 — Project back: result @ W_down → [d] output
```

---

## §13 — Mixture of Experts (MoE)

### 13.1 What It Does

Replaces a single large FFN with $E$ smaller expert FFNs. A **router** selects the top-$K$ experts per token, so only a fraction of parameters are active.

### 13.2 Architecture

$$\text{MoE}(\mathbf{x}) = \underbrace{\sum_{s=1}^{E_{\text{shared}}} \text{Expert}_s(\mathbf{x})}_{\text{shared (always active)}} + \underbrace{\sum_{k=1}^{K} g_k \cdot \text{Expert}_{e_k}(\mathbf{x})}_{\text{routed (top-K selected)}}$$

Where:
- Router: $\mathbf{r} = \mathbf{x} \cdot \mathbf{W}_{\text{router}} + \mathbf{b}_{\text{balance}} \in \mathbb{R}^E$
- Top-K selection: $(e_1, \ldots, e_K) = \text{top-}K(\mathbf{r})$
- Routing weights: $g_k = \text{softmax}(r_{e_1}, \ldots, r_{e_K})_k$

### 13.3 APEX-1 MoE Configuration

| Parameter | Small | Medium | Large |
|---|---|---|---|
| Total experts $E$ | 4 | 64 | 256 |
| Active experts $K$ | 2 | 8 | 8 |
| Shared experts | 1 | 2 | 2 |
| MoE layer frequency | Every odd layer | Every odd layer | Every odd layer |
| Active/Total ratio | 50% | 12.5% | 3.1% |

### 13.4 Why MoE is Efficient

$$\text{Active FLOPs} = (E_{\text{shared}} + K) \times \text{FLOPs}_{\text{per expert}}$$
$$\text{Total params} = (E_{\text{shared}} + E) \times \text{Params}_{\text{per expert}}$$

**APEX-1 Large**: 256 experts, 8 active → 900B total params but only 45B active per token. **20× parameter efficiency**.

### 13.5 Router Example

```
x = [0.5, -0.3]  (d=2, simplified)
W_router (2×4):   (4 experts)
b_balance = [0.0, 0.0, 0.0, 0.0]  (initially)

Step 1 — Router logits:
  r = x @ W_router + b_balance
  r = [1.2, -0.3, 0.8, 0.1]

Step 2 — Top-K (K=2):
  Top-2 indices: [0, 2]  (values 1.2 and 0.8)

Step 3 — Routing weights (softmax over selected):
  softmax([1.2, 0.8]) = [exp(1.2)/(exp(1.2)+exp(0.8)),
                          exp(0.8)/(exp(1.2)+exp(0.8))]
                       = [0.599, 0.401]

Step 4 — Combine expert outputs:
  output = 0.599 × Expert_0(x) + 0.401 × Expert_2(x)
  (Experts 1 and 3 are NOT computed — zero FLOPs for them)
```

---

## §14 — Auxiliary-Loss-Free Load Balancing

### 14.1 The Problem

Without balancing, routers develop "winner-take-all" patterns — a few experts get most tokens while others are starved.

### 14.2 Traditional Solution (Problems)

Add auxiliary loss: $L = L_{\text{LM}} + \alpha \cdot L_{\text{balance}}$

Problem: $L_{\text{balance}}$ interferes with language modeling gradients → worse perplexity.

### 14.3 APEX-1 Solution: Bias-Based Balancing

Update a per-expert bias **outside** the gradient computation:

$$\mathbf{r}_{\text{biased}} = \mathbf{x} \cdot \mathbf{W}_{\text{router}} + \mathbf{b}_{\text{expert}}$$

After each optimizer step:
1. Count tokens per expert: $c_e = |\{t : e \in \text{top-K}(t)\}|$
2. Observed load: $f_e = c_e / (N \times K)$
3. Target load: $\tau = 1/E$
4. Update: $b_e \leftarrow b_e + \alpha \cdot \text{sign}(\tau - f_e)$
5. Clamp: $b_e \leftarrow \text{clamp}(b_e, -1, 1)$

### 14.4 How the Bias Works

| Expert Status | $f_e$ vs $\tau$ | $\text{sign}(\tau - f_e)$ | Bias change | Effect |
|---|---|---|---|---|
| **Overloaded** | $f_e > \tau$ | $-1$ | $b_e$ decreases | Router scores this expert lower |
| **Underloaded** | $f_e < \tau$ | $+1$ | $b_e$ increases | Router scores this expert higher |
| **Balanced** | $f_e = \tau$ | $0$ | No change | Equilibrium |

### 14.5 Example

```
E = 4 experts, K = 2, N = 8 tokens, α = 0.1
Target rate τ = 1/4 = 0.25
Initial bias = [0, 0, 0, 0]

After step 1:
  Token assignments: Expert 0 got 5 tokens, Expert 1 got 4,
                     Expert 2 got 3, Expert 3 got 4
  Total assignments = 8 × 2 = 16
  Observed rates: [5/16, 4/16, 3/16, 4/16] = [0.313, 0.250, 0.188, 0.250]

  Deltas (τ - f_e):    [-0.063, 0.000, 0.062, 0.000]
  Signs:                [-1, 0, +1, 0]
  Bias update (α=0.1): [-0.1, 0.0, +0.1, 0.0]
  New bias:             [-0.1, 0.0, +0.1, 0.0]

Effect: Expert 0's router scores drop by 0.1 → fewer tokens routed
        Expert 2's router scores increase by 0.1 → more tokens routed
```

---

## §15 — Dynamic Skip Gate

### 15.1 What It Does

A learned gate that decides per-token whether to **skip the FFN entirely**. Simple tokens (punctuation, articles) don't need the FFN.

### 15.2 Formula

$$g(\mathbf{x}) = \sigma(\mathbf{W}_2 \cdot \text{SiLU}(\mathbf{W}_1 \cdot \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2)$$

$$\text{output} = \begin{cases} \mathbf{x} + \text{FFN}(\text{RMSNorm}(\mathbf{x})) & \text{if } g(\mathbf{x}) \geq \tau \\ \mathbf{x} & \text{if } g(\mathbf{x}) < \tau \quad \text{(skip)} \end{cases}$$

Where $\tau = 0.15$ is the skip threshold.

### 15.3 Training: Straight-Through Estimator (STE)

Problem: The threshold comparison $g < \tau$ is non-differentiable.

Solution: STE — forward uses hard threshold, backward passes gradient through as identity:

$$\text{Forward}: \quad \text{skip} = \mathbb{1}[g < \tau]$$
$$\text{Backward}: \quad \frac{\partial \text{skip}}{\partial g} = 1 \quad \text{(identity, not 0)}$$

### 15.4 Typical Skip Rates

| Token Type | Gate Value | Skipped? | Why |
|---|---|---|---|
| "the" | 0.05 | ✅ Yes | Article — no complex semantics |
| "," | 0.03 | ✅ Yes | Punctuation — trivial |
| "however" | 0.42 | ❌ No | Discourse marker — changes meaning |
| "mitochondria" | 0.89 | ❌ No | Domain knowledge required |
| "\n\n" | 0.08 | ✅ Yes | Formatting — trivial |

At convergence: **25-35% of tokens skip** with < 0.3% quality loss.

### 15.5 Compute Savings

$$\text{Savings} = \text{skip\_rate} \times \text{FFN\_FLOPs}$$

FFN is ~67% of each block's compute. With 30% skip rate:

$$\text{Block speedup} = \frac{1}{1 - 0.30 \times 0.67} = \frac{1}{0.80} = 1.25\times$$

---

## §16 — Multi-Token Prediction

### 16.1 What It Does

Instead of predicting only the next token, APEX-1 predicts the next $N$ tokens simultaneously using separate linear heads.

### 16.2 Training Loss

$$L = L_{\text{main}} + \lambda \sum_{k=1}^{N} L_{\text{offset}_k}$$

Where:
- $L_{\text{main}} = \text{CE}(\text{logits}_{t}, \text{token}_{t+1})$ — standard next-token
- $L_{\text{offset}_k} = \text{CE}(\text{head}_k(\mathbf{h}_t), \text{token}_{t+k})$ — predict $k$ steps ahead
- $\lambda = 0.1$ — speculative heads contribute 10% gradient weight
- $N = 4$ — predict 4 tokens ahead

### 16.3 Architecture

$$\text{head}_k(\mathbf{h}) = \mathbf{h} \cdot \mathbf{W}_k \quad \text{for } k = 1, 2, 3, 4$$

Each $\mathbf{W}_k \in \mathbb{R}^{d \times V}$ — a simple linear projection.

### 16.4 Speculative Decoding (Inference)

At inference, the speculative heads **draft** tokens that the main model verifies:

```
Step 1: Main model generates token t₁
Step 2: Spec heads draft [t₂', t₃', t₄', t₅'] from hidden state
Step 3: Main model verifies all 4 drafts in ONE forward pass
Step 4: Accept matching prefix, resample at first mismatch

If 3 out of 4 drafts match → 4 tokens generated in ~1.3 forward passes
Normal: 4 tokens = 4 forward passes
Speedup: 4 / 1.3 ≈ 3× throughput
```

### 16.5 Acceptance Rate by Task

| Task | Acceptance Rate | Effective Speedup |
|---|---|---|
| Code completion | 70-85% | 2.5-3.5× |
| Factual Q&A | 50-65% | 2.0-2.5× |
| Creative writing | 30-45% | 1.5-2.0× |
| Translation | 55-70% | 2.0-3.0× |

---

## Summary — Part 3 Formula Quick Reference

| Component | Formula | Params |
|---|---|---|
| **SwiGLU** | $W_d(\text{SiLU}(xW_g) \odot xW_u)$ | $3 d \cdot d_{ff}$ |
| **MoE output** | $\sum_s E_s(x) + \sum_k g_k E_{e_k}(x)$ | $(E_s + E) \times 3d \cdot d_{ff}$ |
| **Router** | $r = xW_r + b$; top-K | $d \times E + E$ |
| **Load balance** | $b_e += \alpha \cdot \text{sign}(\tau - f_e)$ | 0 (no gradient) |
| **Skip gate** | $\sigma(W_2 \text{SiLU}(W_1 x + b_1) + b_2)$ | $d \times h + h$ |
| **Multi-token** | $L = L_0 + 0.1 \sum_k L_k$ | $N \times d \times V$ |

---

*← Part 2: Attention | Continue to **Part 4**: Training, Alignment & Complete Pipeline →*
