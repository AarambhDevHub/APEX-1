<div align="center">

# 🔺 APEX-1 Mathematical Reference — Part 4

### Training, Alignment & Complete Forward Pass

</div>

---

## Table of Contents — Part 4

| # | Topic | Section |
|---|---|---|
| 17 | Optimizer: AdamW | §17 |
| 18 | Learning Rate Schedule | §18 |
| 19 | DPO (Direct Preference Optimization) | §19 |
| 20 | GRPO (Group Relative Policy Optimization) | §20 |
| 21 | Combined Reward Function | §21 |
| 22 | Sampling Strategies | §22 |
| 23 | Complete Forward Pass | §23 |
| 24 | Master Formula Table | §24 |

---

## §17 — Optimizer: AdamW

### 17.1 What It Does

AdamW is a momentum-based optimizer with decoupled weight decay. It tracks two running averages — the mean and variance of gradients — to adapt learning rates per parameter.

### 17.2 Update Rule

At each step $t$:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \qquad \text{(first moment — momentum)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \qquad \text{(second moment — variance)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \qquad \text{(bias correction)}$$
$$\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)$$

Where the second term $\lambda \theta_{t-1}$ is **decoupled weight decay** (not L2 regularization).

### 17.3 APEX-1 Hyperparameters

| Parameter | Symbol | Value | Why |
|---|---|---|---|
| Peak LR (Small) | $\eta$ | $3 \times 10^{-4}$ | Smaller models tolerate higher LR |
| Peak LR (Medium) | $\eta$ | $1 \times 10^{-4}$ | Standard for 7B scale |
| Peak LR (Large) | $\eta$ | $3 \times 10^{-5}$ | Large models need lower LR |
| Beta 1 | $\beta_1$ | 0.9 | Standard momentum |
| Beta 2 | $\beta_2$ | 0.95 | Higher than default 0.999 for stability |
| Epsilon | $\epsilon$ | $10^{-8}$ | Numerical stability |
| Weight decay | $\lambda$ | 0.1 | Regularization |
| Gradient clip | | 1.0 | Prevents exploding gradients |

### 17.4 Gradient Clipping

Before the optimizer step, all gradients are clipped by global norm:

$$\text{if } \|\mathbf{g}\|_2 > c: \quad \mathbf{g} \leftarrow c \cdot \frac{\mathbf{g}}{\|\mathbf{g}\|_2}$$

Where $c = 1.0$. This prevents training instability from rare large-gradient batches.

---

## §18 — Learning Rate Schedule

### 18.1 Formula: Cosine Warmup + Decay

$$\eta(t) = \begin{cases} \eta_{\text{peak}} \cdot \frac{t}{T_w} & \text{if } t < T_w \quad \text{(linear warmup)} \\ \eta_{\text{min}} + \frac{\eta_{\text{peak}} - \eta_{\text{min}}}{2} \left(1 + \cos\!\left(\pi \cdot \frac{t - T_w}{T_{\max} - T_w}\right)\right) & \text{if } t \geq T_w \quad \text{(cosine decay)} \end{cases}$$

Where:
- $T_w$ = warmup steps
- $T_{\max}$ = total training steps
- $\eta_{\text{min}} = 0.1 \times \eta_{\text{peak}}$

### 18.2 Schedule Visualization

```
LR
  ↑
η_peak ─ ─ ─ ─ ─ ╮
  │             ╱   ╲
  │           ╱       ╲
  │         ╱           ╲
  │       ╱               ╲
  │     ╱                   ╲
η_min ╱─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ╲─ ─
  │
  └───┬───────┬───────────────┬──→ Steps
      0     T_w              T_max
      Warmup    Cosine Decay
```

### 18.3 Numerical Example

```
η_peak = 1e-4, T_w = 2000, T_max = 100000, η_min = 1e-5

Step 0:     η = 1e-4 × 0/2000 = 0
Step 1000:  η = 1e-4 × 1000/2000 = 5e-5
Step 2000:  η = 1e-4 (peak)
Step 51000: progress = (51000-2000)/(100000-2000) = 0.5
            η = 1e-5 + (1e-4 - 1e-5)/2 × (1 + cos(π × 0.5))
            η = 1e-5 + 4.5e-5 × (1 + 0) = 5.5e-5
Step 100000: η = 1e-5 (minimum)
```

---

## §19 — DPO (Direct Preference Optimization)

### 19.1 What It Does

Trains the model directly on preference pairs (chosen vs rejected) without needing a separate reward model. Uses log-probability ratios as implicit rewards.

### 19.2 Implicit Reward

$$r(\mathbf{y} | \mathbf{x}) = \beta \left( \log \pi_\theta(\mathbf{y} | \mathbf{x}) - \log \pi_{\text{ref}}(\mathbf{y} | \mathbf{x}) \right)$$

Where:
- $\pi_\theta$ = current policy (model being trained)
- $\pi_{\text{ref}}$ = frozen reference model (SFT checkpoint)
- $\beta = 0.1$ = KL penalty coefficient

### 19.3 DPO Loss

$$L_{\text{DPO}} = -\log \sigma\!\left( r(\mathbf{y}_w | \mathbf{x}) - r(\mathbf{y}_l | \mathbf{x}) \right)$$

Expanded:

$$L_{\text{DPO}} = -\log \sigma\!\left( \beta \left[ \log \frac{\pi_\theta(\mathbf{y}_w | \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_w | \mathbf{x})} - \log \frac{\pi_\theta(\mathbf{y}_l | \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_l | \mathbf{x})} \right] \right)$$

Where $\mathbf{y}_w$ = chosen, $\mathbf{y}_l$ = rejected.

### 19.4 Intuition

| Scenario | $r_w - r_l$ | $\sigma(\cdot)$ | Loss | Gradient |
|---|---|---|---|---|
| Model strongly prefers chosen | +3.0 | 0.95 | 0.05 | Small (good!) |
| Model is uncertain | 0.0 | 0.50 | 0.69 | Medium |
| Model prefers rejected | -2.0 | 0.12 | 2.12 | **Large (fix this!)** |

### 19.5 Sequence Log-Probability

$$\log \pi(\mathbf{y} | \mathbf{x}) = \sum_{t=|\mathbf{x}|}^{|\mathbf{x}|+|\mathbf{y}|} \log P(y_t | y_{\lt t}, \mathbf{x})$$

Only response tokens contribute — prompt tokens are excluded.

---

## §20 — GRPO (Group Relative Policy Optimization)

### 20.1 What It Does

DeepSeek-R1's key innovation. For each prompt, samples $G$ responses, ranks them by reward, and uses **group-relative** advantages as the training signal. **No critic/value network needed**.

### 20.2 Algorithm

For each prompt $\mathbf{x}$:

**Step 1 — Sample G responses:**
$$\mathbf{y}_1, \ldots, \mathbf{y}_G \sim \pi_\theta(\cdot | \mathbf{x})$$

**Step 2 — Score with reward function:**
$$r_i = R(\mathbf{x}, \mathbf{y}_i) \quad \text{for } i = 1, \ldots, G$$

**Step 3 — Group-normalize advantages:**
$$A_i = \frac{r_i - \text{mean}(\mathbf{r})}{\text{std}(\mathbf{r})}$$

**Step 4 — Clipped surrogate objective:**
$$\rho_i = \frac{\pi_\theta(\mathbf{y}_i | \mathbf{x})}{\pi_{\text{ref}}(\mathbf{y}_i | \mathbf{x})}$$

$$L_i = -\min\!\left( \rho_i A_i,\; \text{clip}(\rho_i, 1-\epsilon, 1+\epsilon) A_i \right) + \beta \cdot D_{\text{KL}}[\pi_\theta \| \pi_{\text{ref}}]$$

### 20.3 Why Group-Relative?

| Method | Advantage Source | Needs Critic? | Stability |
|---|---|---|---|
| PPO | Value network $V(s)$ | ✅ Yes (billions of params) | Moderate |
| REINFORCE | Baseline $b$ | Optional | Low |
| **GRPO** | Group mean/std | ❌ No | **High** |

GRPO's $A_i$ is a **z-score** — mean 0, std 1 — which provides automatic variance normalization.

### 20.4 Clipping Explained

The clip prevents the policy from changing too much in one step:

$$\text{clip}(\rho, 1-\epsilon, 1+\epsilon) = \begin{cases} 1-\epsilon & \text{if } \rho < 1-\epsilon \\ \rho & \text{if } 1-\epsilon \leq \rho \leq 1+\epsilon \\ 1+\epsilon & \text{if } \rho > 1+\epsilon \end{cases}$$

With $\epsilon = 0.2$: policy ratio is bounded in $[0.8, 1.2]$.

### 20.5 Numerical Example

```
Prompt: "What is 2+2?"
G = 4 responses:

  y₁: "2+2=4"                     → reward r₁ = 1.0 (correct)
  y₂: "The answer is 4"           → reward r₂ = 0.9 (correct, verbose)
  y₃: "2+2=5"                     → reward r₃ = 0.0 (wrong)
  y₄: "Let me think... 2+2=4"     → reward r₄ = 0.8 (correct, thinking)

Group statistics:
  mean(r) = (1.0 + 0.9 + 0.0 + 0.8) / 4 = 0.675
  std(r)  = 0.396

Advantages (z-scores):
  A₁ = (1.0 - 0.675) / 0.396 = +0.82   ← reinforce concise correct
  A₂ = (0.9 - 0.675) / 0.396 = +0.57   ← mildly reinforce
  A₃ = (0.0 - 0.675) / 0.396 = -1.70   ← strongly suppress wrong answer
  A₄ = (0.8 - 0.675) / 0.396 = +0.32   ← slightly reinforce

The gradient pushes the model toward y₁ and away from y₃.
```

---

## §21 — Combined Reward Function

### 21.1 What It Does

APEX-1's dual-signal alignment combines **three** reward signals into one score:

$$R(\mathbf{x}, \mathbf{y}) = w_o \cdot R_{\text{outcome}} + w_p \cdot R_{\text{process}} + w_c \cdot R_{\text{constitutional}}$$

### 21.2 Signal Weights

| Signal | Weight | Range | What it measures |
|---|---|---|---|
| Outcome reward $R_o$ | $w_o = 0.5$ | [0, 1] | Is the final answer correct? |
| Process reward $R_p$ | $w_p = 0.2$ | [0, 1] | Are reasoning steps valid? |
| Constitutional $R_c$ | $w_c = 0.3$ | [0, 1] | Does response follow safety principles? |

### 21.3 Process Reward Model (PRM)

Scores each reasoning step independently with cumulative context:

$$R_{\text{process}} = \frac{1}{N} \sum_{i=1}^{N} \text{PRM}(\mathbf{x}, s_1, \ldots, s_i)$$

Where $s_i$ is reasoning step $i$, scored in context of all prior steps.

### 21.4 Example

```
Prompt: "Solve: 3x + 5 = 20"
Response:
  Step 1: "Subtract 5 from both sides: 3x = 15"  → PRM: 0.95
  Step 2: "Divide by 3: x = 5"                    → PRM: 0.98
  Answer: "x = 5"                                 → Outcome: 1.0
  Constitutional: no violations                    → Score: 1.0

Combined reward:
  R = 0.5 × 1.0 + 0.2 × (0.95+0.98)/2 + 0.3 × 1.0
  R = 0.5 + 0.193 + 0.3
  R = 0.993
```

---

## §22 — Sampling Strategies

### 22.1 Temperature Scaling

$$P(t) = \frac{\exp(z_t / T)}{\sum_j \exp(z_j / T)}$$

| Temperature $T$ | Effect | Use Case |
|---|---|---|
| $T \to 0$ | Greedy (argmax) | Code, math |
| $T = 0.3$ | Sharp, deterministic | Factual Q&A |
| $T = 0.7$ | Balanced | General chat |
| $T = 1.0$ | Standard distribution | Training |
| $T > 1.0$ | Flat, random | Brainstorming |

### 22.2 Top-p (Nucleus Sampling)

Keep smallest set of tokens whose cumulative probability ≥ $p$:

$$V_p = \arg\min_{V'} \left\\{ |V'| : \sum_{t \in V'} P(t) \geq p \right\\}$$

```
Sorted probs: [0.40, 0.25, 0.15, 0.10, 0.05, 0.03, 0.02]
Top-p = 0.9:
  Cumulative:  [0.40, 0.65, 0.80, 0.90, ...]
                                    ^ stop here
  Keep tokens: [0, 1, 2, 3]  (4 tokens)
  Re-normalize and sample from these 4
```

### 22.3 Top-k Sampling

Keep only the $k$ highest-probability tokens:

$$V_k = \text{argtop}_k(P)$$

### 22.4 Repetition Penalty

$$z'_t = \begin{cases} z_t / \alpha & \text{if } z_t > 0 \text{ and } t \in \text{generated} \\ z_t \times \alpha & \text{if } z_t < 0 \text{ and } t \in \text{generated} \\ z_t & \text{otherwise} \end{cases}$$

With $\alpha = 1.1$: previously generated tokens are slightly penalized.

---

## §23 — Complete Forward Pass

### 23.1 Full Pipeline (One Forward Pass)

```
Input: token_ids [B, S]
         │
    ┌────▼─────┐
    │ Embedding │  X = Embed(tokens) × √d        Shape: [B, S, d]
    └────┬─────┘
         │
    ┌────▼──────────────────────────────────────┐
    │  FOR layer l = 0 to L-1:                  │
    │                                           │
    │  1. h = Attention(RMSNorm(x))             │  [B, S, d]
    │     • layer 5,11,17,...: MLA (global)      │
    │     • other layers:     GQA+SW (local)     │
    │                                           │
    │  2. x = x + h                (residual)   │  [B, S, d]
    │                                           │
    │  3. gate = SkipGate(x)       [B, S, 1]    │
    │     IF gate ≥ 0.15:                       │
    │       f = FFN(RMSNorm(x))                 │  [B, S, d]
    │       • even layers: Dense SwiGLU          │
    │       • odd layers:  MoE (top-K routing)   │
    │       x = x + f              (residual)   │
    │     ELSE:                                 │
    │       x = x                  (skip FFN)   │
    │                                           │
    └────┬──────────────────────────────────────┘
         │
    ┌────▼──────┐
    │ RMSNorm   │                                 [B, S, d]
    └────┬──────┘
         │
    ┌────▼──────────┐
    │ LM Head       │  logits = h × W_E^T         [B, S, V]
    │ Spec Heads    │  spec_k = h × W_k           [B, S, V] × 4
    └───────────────┘
```

### 23.2 Parameter Budget by Component

| Component | % of Total | APEX-1 Large |
|---|---|---|
| Embedding | 5-8% | ~54B |
| Attention (all layers) | 15-20% | ~135B |
| FFN / MoE experts | 65-75% | ~675B |
| Skip gates | < 0.1% | ~0.1B |
| Multi-token heads | 2-5% | ~27B |
| Norms | < 0.01% | ~0.01B |
| **Total** | 100% | **~900B** |
| **Active per token** | ~5% | **~45B** |

---

## §24 — Master Formula Table

Every mathematical formula used in APEX-1, in one reference:

### Normalization & Embedding

| # | Name | Formula |
|---|---|---|
| F1 | Embedding | $\mathbf{X} = \text{Embed}(\text{tok}) \times \sqrt{d}$ |
| F2 | RMSNorm | $\frac{\mathbf{x}}{\sqrt{\text{mean}(\mathbf{x}^2) + \epsilon}} \cdot \gamma$ |

### Positional Encoding

| # | Name | Formula |
|---|---|---|
| F3 | RoPE frequency | $\theta_i = 10000^{-2i/d_h}$ |
| F4 | RoPE rotation | $\mathbf{x}' = \mathbf{x} \cos(m\theta) + \text{rot}(\mathbf{x}) \sin(m\theta)$ |
| F5 | rotate_half | $[x_0,x_1,x_2,x_3] \to [-x_1,x_0,-x_3,x_2]$ |
| F6 | YaRN hi-freq | $\theta'_i = \theta_i$ if $\lambda_i < \beta_f$ |
| F7 | YaRN lo-freq | $\theta'_i = \theta_i / s$ if $\lambda_i > \beta_s \cdot s$ |
| F8 | YaRN temperature | $\text{factor} = 0.1 \ln(s) + 1$ |

### Attention

| # | Name | Formula |
|---|---|---|
| F9 | Scaled dot-product | $\text{softmax}(\mathbf{QK}^T / \sqrt{d_h}) \mathbf{V}$ |
| F10 | GQA grouping | $G = H_q / H_{kv}$; head $i$ uses KV head $\lfloor i/G \rfloor$ |
| F11 | MLA compress KV | $\mathbf{c}_{kv} = \mathbf{x} \mathbf{W}_{DKV}$ |
| F12 | MLA decompress | $\mathbf{K} = \mathbf{c}_{kv} \mathbf{W}_{UK}$; $\mathbf{V} = \mathbf{c}_{kv} \mathbf{W}_{UV}$ |
| F13 | MLA decoupled RoPE | $\mathbf{Q}_f = [\mathbf{Q}_c \| \text{RoPE}(\mathbf{x}\mathbf{W}_{QR})]$ |
| F14 | Sliding window mask | $\text{mask}[i,j] = \mathbb{1}[0 \leq i-j < W]$ |
| F15 | Global/local rule | $\text{global} \iff l \bmod 6 = 5$ |

### FFN & MoE

| # | Name | Formula |
|---|---|---|
| F16 | SiLU | $\text{SiLU}(x) = x \cdot \sigma(x)$ |
| F17 | SwiGLU FFN | $\mathbf{W}_d(\text{SiLU}(\mathbf{x}\mathbf{W}_g) \odot \mathbf{x}\mathbf{W}_u)$ |
| F18 | MoE routing | $\mathbf{r} = \mathbf{x}\mathbf{W}_r + \mathbf{b}$; select top-$K$ |
| F19 | MoE output | $\sum_s E_s(\mathbf{x}) + \sum_k g_k E_{e_k}(\mathbf{x})$ |
| F20 | Load balance | $b_e \mathrel{+}= \alpha \cdot \text{sign}(\tau - f_e)$ |
| F21 | Skip gate | $g = \sigma(\mathbf{W}_2 \text{SiLU}(\mathbf{W}_1 \mathbf{x}))$; skip if $g < 0.15$ |

### Training & Alignment

| # | Name | Formula |
|---|---|---|
| F22 | Pretrain loss | $L = \text{CE}(\text{logits}, \text{next-tok})$ |
| F23 | Multi-token aux | $L_{\text{total}} = L_0 + 0.1 \sum_{k=1}^4 L_k$ |
| F24 | SFT loss | $L = \text{CE}(\text{logits}, \text{labels})$; labels[non-asst] = $-100$ |
| F25 | AdamW | $\theta -= \eta(\hat{m}/(\sqrt{\hat{v}}+\epsilon) + \lambda\theta)$ |
| F26 | Cosine warmup LR | $\eta_{\min} + \frac{\eta_p - \eta_{\min}}{2}(1+\cos(\pi \cdot \text{progress}))$ |
| F27 | DPO loss | $-\log\sigma(\beta[\log\frac{\pi_\theta(y_w)}{\pi_{\text{ref}}(y_w)} - \log\frac{\pi_\theta(y_l)}{\pi_{\text{ref}}(y_l)}])$ |
| F28 | GRPO advantage | $A_i = (r_i - \bar{r}) / \text{std}(r)$ |
| F29 | GRPO objective | $-\min(\rho A, \text{clip}(\rho) A) + \beta D_{\text{KL}}$ |
| F30 | Combined reward | $R = 0.5 R_o + 0.2 R_p + 0.3 R_c$ |

### Sampling

| # | Name | Formula |
|---|---|---|
| F31 | Temperature | $P(t) = \exp(z_t/T) / \sum \exp(z_j/T)$ |
| F32 | Top-p | Keep smallest $V'$ where $\sum_{V'} P \geq p$ |
| F33 | Top-k | Keep top-$k$ by probability |
| F34 | Repetition penalty | $z'_t = z_t / \alpha$ if $t \in \text{generated}$ |

---

<div align="center">

### 📚 Complete Mathematical Reference

| Part | File | Topics |
|---|---|---|
| **Part 1** | `Part1.md` | Embedding, RMSNorm, RoPE, YaRN |
| **Part 2** | `Part2.md` | SDPA, MHA, GQA, MLA, Sliding Window, Masks |
| **Part 3** | `Part3.md` | SwiGLU, MoE, Load Balancing, Skip Gate, Multi-Token |
| **Part 4** | `Part4.md` | AdamW, LR Schedule, DPO, GRPO, Sampling, Full Pipeline |

**Total: 34 formulas · 24 sections · 4 parts**

*Copyright 2024-2026 Aarambh Dev Hub · Apache 2.0*

</div>
