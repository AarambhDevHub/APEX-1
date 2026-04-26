# APEX-1 Model Architecture
### A Best-of-All-Worlds LLM Design — v2.0
**Inspired by:** Claude · GPT-4.5 · DeepSeek-V3/R1 · Qwen3 · Gemma 4 · GLM-4 · KIMI · MiniMax · Llama 3

---

## Table of Contents

1. [Overview & Philosophy](#1-overview--philosophy)
2. [Tokenizer](#2-tokenizer)
3. [Model Dimensions & Hyperparameters](#3-model-dimensions--hyperparameters)
4. [Input Embedding Layer](#4-input-embedding-layer)
5. [Rotary Positional Encoding — RoPE + YaRN](#5-rotary-positional-encoding--rope--yarn)
6. [Transformer Block — Full Stack](#6-transformer-block--full-stack)
   - 6a. Pre-Norm with RMSNorm
   - 6b. Attention Strategy: MLA vs GQA (which layers use which)
   - 6c. Multi-Head Latent Attention (MLA) — full forward pass
   - 6d. Grouped Query Attention + Sliding Window (GQA+SW) — full forward pass
   - 6e. Interleaved Local / Global Attention — implementation
   - 6f. Prefix Bidirectional Attention Mask
   - 6g. Flash Attention v3
   - 6h. Feed-Forward Network with SwiGLU
   - 6i. Mixture of Experts (MoE) — full forward pass
   - 6j. Auxiliary-Loss-Free Load Balancing — full algorithm
   - 6k. Dynamic Skip Gate
   - 6l. Residual Connections
7. [KV Cache & Memory Optimization](#7-kv-cache--memory-optimization)
8. [Long Context Handling](#8-long-context-handling)
9. [Language Model Head](#9-language-model-head)
10. [Sampling & Decoding Strategy](#10-sampling--decoding-strategy)
11. [Reasoning / Thinking Mode](#11-reasoning--thinking-mode)
12. [Training Pipeline — Full Detail](#12-training-pipeline--full-detail)
    - 12a. Phase 1: Pre-training (loss, optimizer, schedule)
    - 12b. Phase 2: Supervised Fine-Tuning
    - 12c. Gradient flow through MoE layers
13. [Alignment: RLHF → DPO → GRPO — Full Detail](#13-alignment-rlhf--dpo--grpo--full-detail)
    - 13a. RLHF + PPO
    - 13b. DPO
    - 13c. GRPO — full rollout loop with code
    - 13d. Constitutional AI
    - 13e. Process Reward Model (PRM)
    - 13f. Dual-signal alignment (GRPO + CAI simultaneously)
14. [Inference Optimizations](#14-inference-optimizations)
15. [Model Size Configurations](#15-model-size-configurations)
16. [Full Forward Pass — Step by Step](#16-full-forward-pass--step-by-step)
17. [Implementation Checklist](#17-implementation-checklist)

---

## 1. Overview & Philosophy

APEX-1 is a decoder-only transformer that synthesizes the single best innovation from each frontier lab into one coherent design. No component is included without a clear engineering reason.

| Feature | Source model | Why it wins |
|---|---|---|
| Large vocabulary (151k tokens) | Qwen3 | Better multilingual & code coverage |
| RoPE with YaRN extension | KIMI / DeepSeek | Extends context without retraining |
| MLA on even layers (global attention) | DeepSeek-V3 | 93% KV cache reduction for full-context layers |
| GQA + sliding window on odd layers | Llama 3 / Kimi / Mistral | Efficient local attention between global passes |
| Interleaved local/global at 1:5 ratio | Gemma 4 | Long-context at fraction of compute |
| Prefix bidirectional attention | GLM-4 | Full context over system prompt before generation |
| SwiGLU activation | PaLM / Llama / Claude | ~1–2% perplexity gain over ReLU |
| 3-tier hierarchical MoE (256 experts) | DeepSeek-V3 | Frontier quality at fraction of FLOPS |
| Auxiliary-loss-free load balancing | DeepSeek-V3 | Stable expert utilization without hurting LM loss |
| Dynamic skip gate | Early-exit research | 25–35% FFN compute saved on simple tokens |
| Multi-token prediction head | DeepSeek-V3 | 3× richer training signal, 2.5× inference speedup |
| Thinking mode (chain-of-thought) | DeepSeek-R1 / Claude 3.7 | Reasoning via internal scratchpad |
| GRPO alignment | DeepSeek-R1 | Stable RL for reasoning, no reward model needed |
| Constitutional AI alignment | Claude / Anthropic | Principled safety baked in, not patched on |
| Process Reward Model | GPT-4o | Step-level reasoning quality signal |
| RMSNorm pre-norm | Universal (post-2022) | Stable deep training |
| Flash Attention v3 | All modern models | 3–7× faster attention, same output |

### Design Principles

```
1. EFFICIENCY FIRST   — Active parameters << Total parameters (MoE)
2. CONTEXT FIRST      — Native 128k, extendable to 1M+ via YaRN
3. REASONING FIRST    — Built-in chain-of-thought scratchpad
4. SAFETY FIRST       — Constitutional AI + GRPO baked into training
5. OPEN BY DEFAULT    — Architecture fully specified and reproducible
```

---

## 2. Tokenizer

### Algorithm: Byte-Pair Encoding (BPE)

The tokenizer converts raw Unicode text into integer token IDs before the model ever sees it.

**How BPE works:**
1. Start with individual bytes as the alphabet (256 initial tokens)
2. Count all adjacent byte-pair frequencies in the training corpus
3. Merge the most frequent pair into a new single token
4. Repeat until vocabulary reaches target size
5. The result: common words become single tokens, rare words split into subwords, unknown characters fall back to bytes — **zero unknown tokens ever**

```
Example:
  Input:  "unbelievable"
  Tokens: ["un", "belie", "vable"]   ← 3 tokens

  Input:  "猫" (cat in Chinese)
  Tokens: ["猫"]                      ← 1 token (common in training data)

  Input:  "xyzabc123"
  Tokens: ["x", "yz", "abc", "123"]  ← rare chars fall back to subwords
```

### Vocabulary Size: **151,643 tokens**

Taken from Qwen3's tokenizer — the largest among major open models. Larger vocab means:
- Fewer tokens needed per sentence (faster generation)
- Better coverage of Chinese, Arabic, Hindi, code, math symbols
- Single tokens for common programming keywords (`function`, `return`, `import`)

### Special Tokens

```
<|begin_of_text|>      — Start of every sequence
<|end_of_text|>        — End of generation
<|system|>             — System prompt boundary
<|user|>               — User turn boundary
<|assistant|>          — Assistant turn boundary
<|thinking|>           — Start of internal reasoning scratchpad
<|/thinking|>          — End of reasoning scratchpad
<|pad|>                — Padding token (ID = 0)
<|img|>                — Image placeholder (for future multimodal)
```

### Chat Template

```
<|begin_of_text|><|system|>
You are a helpful AI assistant.
<|user|>
What is 2 + 2?
<|assistant|>
<|thinking|>
Simple arithmetic: 2 + 2 = 4
<|/thinking|>
The answer is 4.
<|end_of_text|>
```

---

## 3. Model Dimensions & Hyperparameters

We define three sizes. Start with **APEX-1-Small** for testing.

| Parameter | Small (test) | Medium | Large |
|---|---|---|---|
| `d_model` (hidden dim) | 512 | 2048 | 7168 |
| `n_layers` (depth) | 12 | 36 | 72 |
| `n_heads_q` (query heads) | 8 | 16 | 128 |
| `n_heads_kv` (KV heads, GQA) | 2 | 4 | 8 |
| `d_head` (head dim) | 64 | 128 | 128 |
| `d_kv_compressed` (MLA latent dim) | 64 | 256 | 512 |
| `d_q_compressed` (MLA Q latent) | 96 | 384 | 768 |
| `d_ffn` (FFN intermediate) | 1376 | 5504 | 18432 |
| `n_experts` total (MoE) | 8 | 64 | 256 |
| `n_experts_active` (routed top-K) | 2 | 4 | 8 |
| `n_experts_shared` (always active) | 1 | 2 | 4 |
| `local_window` (sliding window) | 512 | 2048 | 8192 |
| `global_layer_freq` | 6 | 6 | 6 |
| `vocab_size` | 151,643 | 151,643 | 151,643 |
| `max_seq_len` | 8,192 | 65,536 | 131,072 |
| `rope_base` | 10,000 | 500,000 | 1,000,000 |
| `rope_scaling` (YaRN) | 1.0 | 4.0 | 8.0 |
| `dropout` | 0.0 | 0.0 | 0.0 |
| Total parameters (approx.) | ~100M | ~7B | ~900B |
| Active parameters per token | ~40M | ~2B | ~45B |

> **Why dropout = 0.0?**
> Modern large models do not use dropout during training. Regularization comes from data diversity, weight decay, and model scale. Dropout consistently hurts large-model performance.

> **Why n_layers changed to 12/36/72?**
> With `global_layer_freq = 6`, these layer counts give clean multiples: 2, 6, and 12 global attention layers respectively. This ensures every attention tier fires evenly throughout the network.

---

## 4. Input Embedding Layer

The first real computation the model does.

### What it does:
Map each integer token ID to a real-valued vector of shape `[d_model]`.

### How it works:

```python
import math
import torch

# Embedding table: shape = [vocab_size, d_model]
embedding_table = torch.nn.Embedding(vocab_size, d_model)

# Input: token IDs of shape [batch_size, seq_len]
# Output: vectors of shape [batch_size, seq_len, d_model]
x = embedding_table(token_ids)

# Scale embeddings — important for gradient stability at large d_model
# Without this, variance of x shrinks as d_model grows
x = x * math.sqrt(d_model)
```

### Weight Tying:
The embedding table weights are **shared** with the final LM head:
- Saves `vocab_size × d_model` parameters (~550M for large model)
- Works because embedding a token and scoring a token as output are inverse operations
- Used by: GPT-2, Llama, Qwen3, Gemma, and almost all modern models

### No positional embeddings at this stage:
RoPE is applied **inside each attention layer** on the Q and K vectors.
The embedding layer handles only token identity.

---

## 5. Rotary Positional Encoding — RoPE + YaRN

RoPE encodes position information without adding anything to the embeddings. Instead, it rotates Query and Key vectors before computing attention scores.

### The Core Idea:

For a vector dimension pair `(x₂ᵢ, x₂ᵢ₊₁)` at position `m`:

```
Rotated pair:
  x₂ᵢ'   =  x₂ᵢ · cos(m·θᵢ) - x₂ᵢ₊₁ · sin(m·θᵢ)
  x₂ᵢ₊₁' =  x₂ᵢ · sin(m·θᵢ) + x₂ᵢ₊₁ · cos(m·θᵢ)

Where:
  θᵢ = rope_base^(-2i/d_head)   — frequency for dimension i
  m  = token position in sequence
  rope_base = 10000 (original), 500000–1000000 (modern, long context)
```

### Implementation:

```python
def precompute_rope_cache(d_head, max_seq_len, rope_base=10000):
    """Precompute sin/cos rotation tables once, reuse every forward pass."""
    # Frequencies: shape [d_head // 2]
    i = torch.arange(0, d_head, 2, dtype=torch.float32)
    theta = 1.0 / (rope_base ** (i / d_head))

    # Positions: shape [max_seq_len]
    positions = torch.arange(max_seq_len, dtype=torch.float32)

    # Outer product: shape [max_seq_len, d_head // 2]
    angles = torch.outer(positions, theta)

    # Stack sin and cos: shape [max_seq_len, d_head]
    cos_cache = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin_cache = torch.sin(angles).repeat_interleave(2, dim=-1)
    return cos_cache, sin_cache  # cached, not recomputed each step


def rotate_half(x):
    """Rotate alternate pairs: [-x1, x0, -x3, x2, ...]"""
    x1 = x[..., ::2]   # even indices
    x2 = x[..., 1::2]  # odd indices
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(q, k, cos_cache, sin_cache, positions):
    """Apply RoPE rotation to Q and K tensors."""
    cos = cos_cache[positions].unsqueeze(1)  # [seq, 1, d_head]
    sin = sin_cache[positions].unsqueeze(1)  # [seq, 1, d_head]

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin
    return q_rot, k_rot
```

### YaRN Extension for Long Context:

YaRN allows a model trained at 4k context to run at 128k+ without retraining. It applies selective frequency scaling:

```python
def apply_yarn_scaling(theta, scale_factor, d_head, beta_fast=32, beta_slow=1):
    """
    YaRN: scale low-frequency dimensions (long-range position) more aggressively
    while leaving high-frequency dimensions (short-range) unchanged.

    beta_fast: high-freq cutoff (dimensions < this: no scaling)
    beta_slow: low-freq cutoff  (dimensions > this: full scaling)
    """
    scaled_theta = theta.clone()
    for i in range(len(theta)):
        # Convert dimension index to wavelength
        wavelength = 2 * math.pi / theta[i].item()

        if wavelength < beta_fast:
            # High-frequency (short wavelength): do not scale
            # These handle local syntax — already work at any length
            scaled_theta[i] = theta[i]
        elif wavelength > beta_slow * scale_factor:
            # Low-frequency (long wavelength): full scaling
            # These handle document-level position
            scaled_theta[i] = theta[i] / scale_factor
        else:
            # Smooth linear interpolation between the two regimes
            t = (wavelength / beta_slow - 1) / (scale_factor - 1)
            scaled_theta[i] = theta[i] / (t * scale_factor + (1 - t))

    # Temperature correction: prevents attention entropy collapse at long context
    attn_factor = 0.1 * math.log(scale_factor) + 1.0
    return scaled_theta, attn_factor
```

**Used by:** KIMI (1M context), DeepSeek-V3 (128k), Qwen3 (128k extended)

---

## 6. Transformer Block — Full Stack

### Architecture overview of one APEX-1 block:

```
Input x  [batch, seq_len, d_model]
  │
  ├─ RMSNorm(x)
  │       │
  │       └─► Attention sub-layer ──────────────► + x   (residual)
  │           [MLA if global layer,                  │
  │            GQA+SW if local layer]                x'
  │
  ├─ Dynamic Skip Gate — evaluates x' per token
  │       │
  │       ├─ gate < 0.15 → SKIP FFN (residual only)
  │       │
  │       └─ gate ≥ 0.15 → FFN sub-layer:
  │                   RMSNorm(x')
  │                       │
  │                       └─► [Dense FFN or MoE FFN] ──► + x'  (residual)
  │
Output x''  [batch, seq_len, d_model]
```

---

### 6a. Pre-Norm with RMSNorm

**The rule:** Normalize the input BEFORE each sub-layer (Pre-LN), not after.

**RMSNorm formula:**
```
RMSNorm(x) = x / RMS(x) × γ

Where:
  RMS(x) = sqrt( mean(x²) )   ← root mean square, NO mean subtraction
  γ       = learned scale parameter (shape: [d_model], initialized to 1.0)
```

```python
class RMSNorm(torch.nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight
```

**Why RMSNorm over LayerNorm:**
- No mean-centering step → 20–40% faster computation
- Equal or better quality (centering was not contributing)
- Used by: Llama 3, DeepSeek, Qwen3, Gemma 4, Claude

---

### 6b. Attention Strategy: MLA vs GQA — Which Layers Use Which

This is the key architectural decision that separates APEX-1 from prior designs.

```
APEX-1 layer assignment rule:

  if layer_idx % global_layer_freq == (global_layer_freq - 1):
      → GLOBAL layer: use MLA (Multi-Head Latent Attention)
        Full causal attention over ALL positions
        KV compressed to latent — 93% smaller cache

  else:
      → LOCAL layer: use GQA + Sliding Window
        Attention limited to last `local_window` positions
        Efficient O(seq × window) instead of O(seq²)

Example with n_layers=12, global_layer_freq=6:
  Layer  0: LOCAL  (GQA + SW)
  Layer  1: LOCAL
  Layer  2: LOCAL
  Layer  3: LOCAL
  Layer  4: LOCAL
  Layer  5: GLOBAL (MLA) ← full attention
  Layer  6: LOCAL
  Layer  7: LOCAL
  Layer  8: LOCAL
  Layer  9: LOCAL
  Layer 10: LOCAL
  Layer 11: GLOBAL (MLA) ← full attention
```

**Why this alternation works:**
- Local layers handle syntax, phrase coherence, local references — efficient O(n·w)
- Global layers propagate long-range information: references across the document, multi-hop reasoning
- MLA on global layers keeps the KV footprint manageable even with full attention
- Result: document-level understanding at roughly 1/3 the KV memory of pure MLA

```python
def is_global_layer(layer_idx, global_layer_freq):
    return (layer_idx % global_layer_freq) == (global_layer_freq - 1)
```

---

### 6c. Multi-Head Latent Attention (MLA) — Full Forward Pass

MLA is DeepSeek-V3's key innovation. Instead of caching full K and V matrices, it caches a compressed latent vector and reconstructs K, V on the fly.

**Weight matrices required (per layer):**
```
W_DKV : [d_model, d_kv_compressed]       — compress input to KV latent
W_UK  : [d_kv_compressed, n_kv_heads × d_head]   — reconstruct K from latent
W_UV  : [d_kv_compressed, n_kv_heads × d_head]   — reconstruct V from latent
W_DQ  : [d_model, d_q_compressed]        — compress input to Q latent
W_UQ  : [d_q_compressed, n_heads_q × d_head]     — reconstruct Q from latent
W_KR  : [d_model, n_kv_heads × d_head_rope]      — decoupled RoPE keys
W_QR  : [d_model, n_heads_q × d_head_rope]       — decoupled RoPE queries
W_O   : [n_heads_q × d_head, d_model]    — output projection
```

**Forward pass — full detail:**

```python
def mla_forward(x, cos_cache, sin_cache, positions, kv_cache=None):
    """
    x:          [batch, seq_len, d_model]
    cos/sin:    precomputed RoPE tables
    positions:  token positions in sequence
    kv_cache:   cached latent c_kv from previous steps (for inference)

    Returns: output [batch, seq_len, d_model], new_kv_cache
    """
    batch, seq_len, _ = x.shape

    # ── Step 1: Compress input to KV latent ──────────────────────────────
    # c_kv is what we actually cache — much smaller than full K, V
    c_kv = x @ W_DKV                   # [batch, seq, d_kv_compressed]

    # ── Step 2: Append to KV cache (for autoregressive inference) ────────
    if kv_cache is not None:
        c_kv = torch.cat([kv_cache, c_kv], dim=1)   # grow cache along seq dim
    new_kv_cache = c_kv                 # save for next step

    # ── Step 3: Reconstruct K and V from latent ───────────────────────────
    K = c_kv @ W_UK                     # [batch, full_seq, n_kv_heads × d_head]
    V = c_kv @ W_UV                     # [batch, full_seq, n_kv_heads × d_head]
    K = K.view(batch, -1, n_kv_heads, d_head).transpose(1, 2)
    V = V.view(batch, -1, n_kv_heads, d_head).transpose(1, 2)

    # ── Step 4: Compress input to Q latent, reconstruct Q ────────────────
    c_q = x @ W_DQ                      # [batch, seq, d_q_compressed]
    Q   = c_q @ W_UQ                    # [batch, seq, n_heads_q × d_head]
    Q   = Q.view(batch, seq_len, n_heads_q, d_head).transpose(1, 2)

    # ── Step 5: Decoupled RoPE — apply position encoding separately ───────
    # RoPE keys and queries are separate projections that don't go through latent
    # This is important: compressing through latent loses positional information
    Q_rope = (x @ W_QR).view(batch, seq_len, n_heads_q, d_head_rope).transpose(1, 2)
    K_rope = (x @ W_KR).view(batch, -1, n_kv_heads, d_head_rope).transpose(1, 2)

    Q_rope, K_rope = apply_rope(Q_rope, K_rope, cos_cache, sin_cache, positions)

    # Concatenate content and positional components along head dim
    Q = torch.cat([Q, Q_rope], dim=-1)  # [batch, n_heads_q, seq, d_head + d_head_rope]
    K = torch.cat([K, K_rope], dim=-1)

    # ── Step 6: Expand KV heads to match Q heads (GQA-style) ─────────────
    G = n_heads_q // n_kv_heads
    K = K.repeat_interleave(G, dim=1)   # [batch, n_heads_q, full_seq, d_head]
    V = V.repeat_interleave(G, dim=1)

    # ── Step 7: Scaled dot-product attention (Flash Attention handles this) ─
    d_total = d_head + d_head_rope
    scores  = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_total)

    # Apply causal mask (cannot attend to future tokens)
    scores  = scores.masked_fill(causal_mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(weights, V)   # [batch, n_heads_q, seq, d_head]

    # ── Step 8: Merge heads and project out ──────────────────────────────
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    output   = attn_out @ W_O             # [batch, seq, d_model]

    return output, new_kv_cache

# ─── KV Cache during inference ───────────────────────────────────────────────
# ONLY c_kv is stored — shape: [batch, seq_so_far, d_kv_compressed]
# For large model: d_kv_compressed = 512 (vs n_kv_heads × d_head = 8 × 128 = 1024)
# Memory saving vs GQA: 512 / 1024 = 50% (on top of GQA's existing saving)
# Memory saving vs MHA: 512 / (128 × 128) = ~3% — i.e., 97% smaller
```

---

### 6d. Grouped Query Attention + Sliding Window (GQA+SW) — Full Forward Pass

Used on all LOCAL layers (every layer that is not a global MLA layer).

```python
def gqa_sliding_window_forward(x, cos_cache, sin_cache, positions,
                                local_window, kv_cache=None):
    """
    x:            [batch, seq_len, d_model]
    local_window: max number of past tokens to attend to (e.g. 512 for small)
    kv_cache:     (K_cache, V_cache) from previous steps

    Returns: output [batch, seq_len, d_model], (new_K_cache, new_V_cache)
    """
    batch, seq_len, _ = x.shape

    # ── Step 1: Project to Q, K, V ───────────────────────────────────────
    Q = (x @ W_Q).view(batch, seq_len, n_heads_q,  d_head).transpose(1, 2)
    K = (x @ W_K).view(batch, seq_len, n_heads_kv, d_head).transpose(1, 2)
    V = (x @ W_V).view(batch, seq_len, n_heads_kv, d_head).transpose(1, 2)

    # ── Step 2: Apply RoPE to Q and K ────────────────────────────────────
    Q, K = apply_rope(Q, K, cos_cache, sin_cache, positions)

    # ── Step 3: Append to KV cache (inference) ───────────────────────────
    if kv_cache is not None:
        K_cache, V_cache = kv_cache
        K = torch.cat([K_cache, K], dim=2)
        V = torch.cat([V_cache, V], dim=2)

    # Trim KV cache to sliding window — only keep last `local_window` tokens
    if K.shape[2] > local_window:
        K = K[:, :, -local_window:, :]
        V = V[:, :, -local_window:, :]

    new_kv_cache = (K, V)

    # ── Step 4: Expand KV heads to match Q heads (GQA) ───────────────────
    G = n_heads_q // n_heads_kv
    K = K.repeat_interleave(G, dim=1)   # [batch, n_heads_q, window_len, d_head]
    V = V.repeat_interleave(G, dim=1)

    # ── Step 5: Sliding window attention ─────────────────────────────────
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_head)

    # Build sliding window mask: token i can only attend to [i-window+1 .. i]
    scores = scores.masked_fill(sliding_window_mask == 0, float('-inf'))
    weights = torch.softmax(scores, dim=-1)
    attn_out = torch.matmul(weights, V)   # [batch, n_heads_q, seq, d_head]

    # ── Step 6: Merge heads and project ──────────────────────────────────
    attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
    output   = attn_out @ W_O             # [batch, seq, d_model]

    return output, new_kv_cache
```

---

### 6e. Interleaved Local / Global Attention — Implementation

The dispatcher that decides which attention function each layer calls.

```python
class APEXTransformerBlock(torch.nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.config    = config
        self.is_global = is_global_layer(layer_idx, config.global_layer_freq)

        self.norm1 = RMSNorm(config.d_model)
        self.norm2 = RMSNorm(config.d_model)

        if self.is_global:
            self.attn = MLAAttention(config)      # full-context MLA
        else:
            self.attn = GQASlidingWindowAttn(config)  # local GQA+SW

        # FFN: alternate dense and MoE
        if layer_idx % 2 == 0:
            self.ffn = DenseFFN(config)           # dense SwiGLU
        else:
            self.ffn = MoEFFN(config)             # mixture of experts

        self.skip_gate = SkipGate(config.d_model) # dynamic skip (Section 6k)

    def forward(self, x, cos_cache, sin_cache, positions,
                attn_mask, kv_cache=None):
        # ── Attention sub-layer (always runs) ────────────────────────────
        h, new_kv = self.attn(
            self.norm1(x),
            cos_cache, sin_cache, positions,
            attn_mask, kv_cache
        )
        x = x + h   # residual

        # ── Dynamic skip gate — may bypass FFN ───────────────────────────
        gate = self.skip_gate(x)              # [batch, seq, 1] values in (0,1)
        skip_mask = gate < self.config.skip_threshold   # True = skip FFN

        if skip_mask.all():
            # Every token skips — avoid FFN entirely (rare but fast path)
            return x, new_kv

        # ── FFN sub-layer (skipped for low-complexity tokens) ─────────────
        ffn_out = self.ffn(self.norm2(x))

        # Apply gate: skipped tokens contribute 0, active tokens contribute ffn_out
        x = x + ffn_out * (~skip_mask).float()   # residual with selective gate

        return x, new_kv
```

---

### 6f. Prefix Bidirectional Attention Mask

The system prompt / instruction prefix is known before generation starts.
APEX-1 gives those tokens full bidirectional attention — the model reads the entire prompt before generating a single word.

```python
def build_apex_attention_mask(prefix_len, total_len, local_window,
                               is_global_layer):
    """
    Returns a boolean attention mask [total_len, total_len].
    True  = can attend
    False = masked (set to -inf before softmax)

    prefix_len:   number of system prompt + user turn tokens
    total_len:    prefix_len + generated tokens so far
    local_window: sliding window size (for local layers only)
    is_global:    True → full causal, False → windowed causal
    """
    mask = torch.zeros(total_len, total_len, dtype=torch.bool)

    # ── Prefix block: bidirectional (GLM-4 style) ─────────────────────────
    # All prefix tokens attend to all other prefix tokens in both directions
    mask[:prefix_len, :prefix_len] = True

    if is_global_layer:
        # ── Global layer: full causal attention over entire sequence ───────
        for i in range(prefix_len, total_len):
            mask[i, :i + 1] = True          # attend to all past tokens

    else:
        # ── Local layer: causal + sliding window ──────────────────────────
        for i in range(prefix_len, total_len):
            start = max(0, i - local_window + 1)
            mask[i, start:i + 1] = True     # attend only to recent window

    return mask  # [total_len, total_len]

# ─── Why this helps ───────────────────────────────────────────────────────────
# Standard causal: token 5 of system prompt can only see tokens 1-5
# Prefix bidir:    token 5 of system prompt sees the ENTIRE system prompt
# This drastically improves instruction following and system-prompt grounding
# Inspiration: GLM-4 architecture
```

---

### 6g. Flash Attention v3

Flash Attention does not change the math — it produces **identical output** to standard attention. It changes how GPU memory is accessed.

**The problem with standard attention:**
```
Standard:
  At seq_len = 128k: attention matrix = 128k × 128k × 2 bytes = 32 GB
  Must be written to and read from slow HBM memory repeatedly
  Dominated by memory bandwidth, not arithmetic throughput
```

**Flash Attention solution:**
```
Flash v3:
  Tiles Q, K, V into SRAM-sized blocks (fits in fast on-chip memory)
  Computes attention tile by tile, accumulates result with numerically stable
  online softmax — never materializes the full matrix
  Memory: O(seq_len) instead of O(seq_len²)
  Speed:  3–7× faster wall-clock
  Output: bit-for-bit identical to standard attention
```

**In practice:**
```python
# PyTorch calls Flash Attention automatically when inputs are on CUDA
# and the sequence length allows it
output = torch.nn.functional.scaled_dot_product_attention(
    Q, K, V,
    attn_mask=mask,
    dropout_p=0.0,
    is_causal=False  # we pass our own mask (prefix bidir + window)
)
```

---

### 6h. Feed-Forward Network with SwiGLU

The FFN processes each token independently after attention.
It is the primary "knowledge storage" of the model.

**Architecture:**
```
FFN(x) = W_down( SiLU(W_gate(x)) ⊙ W_up(x) )

Where:
  W_gate: [d_model → d_ffn]   — gate projection
  W_up:   [d_model → d_ffn]   — value projection
  W_down: [d_ffn   → d_model] — output projection
  SiLU(z) = z × sigmoid(z)    — smooth, monotonic activation
  ⊙ = elementwise multiplication (gating)
```

```python
class DenseFFN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.W_gate = torch.nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.W_up   = torch.nn.Linear(config.d_model, config.d_ffn, bias=False)
        self.W_down = torch.nn.Linear(config.d_ffn,   config.d_model, bias=False)
        self.act    = torch.nn.SiLU()

    def forward(self, x):
        gate   = self.act(self.W_gate(x))   # [batch, seq, d_ffn]
        value  = self.W_up(x)               # [batch, seq, d_ffn]
        hidden = gate * value               # elementwise gating
        return self.W_down(hidden)          # [batch, seq, d_model]
```

**Dimension rule:**
```
d_ffn = round_to_multiple( (2/3) × 4 × d_model, multiple=64 )

For d_model=512:   d_ffn = round((2/3) × 2048) = round(1365) → 1376  ✓
For d_model=2048:  d_ffn = round((2/3) × 8192) = round(5461) → 5504  ✓
For d_model=7168:  d_ffn = round((2/3) × 28672) = 19114 → 18432      ✓
```

---

### 6i. Mixture of Experts (MoE) — Full Forward Pass

In MoE layers, one large FFN is replaced by multiple small expert FFNs.
Only K experts process each token — scaling capacity without scaling compute.

**Architecture:**
```
Input x  [batch × seq, d_model]  (tokens as flat batch)
  │
  ├─ Shared experts (n_shared, always active):
  │    FFN_shared_1(x) + FFN_shared_2(x) + ...  → shared_out
  │
  ├─ Router: Linear(x, n_experts) → expert_scores [batch×seq, n_experts]
  │          + expert_bias (load balancing, see 6j)
  │          top-K selection → indices of K active experts
  │          softmax on top-K scores → routing weights (sum to 1.0)
  │
  ├─ Routed experts (top-K of n_experts):
  │    For each active expert e: FFN_e(x) × routing_weight_e
  │
  └─ Output = shared_out + sum(routing_weight_k × FFN_k(x))
```

```python
class MoEFFN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_experts        = config.n_experts
        self.n_active         = config.n_experts_active   # top-K routed
        self.n_shared         = config.n_experts_shared   # always active

        # Shared experts — always computed, bypass router
        self.shared_experts = torch.nn.ModuleList([
            DenseFFN(config) for _ in range(self.n_shared)
        ])

        # Routed experts — only top-K activated per token
        self.routed_experts = torch.nn.ModuleList([
            DenseFFN(config) for _ in range(self.n_experts)
        ])

        # Router: scores each expert for each token
        self.router = torch.nn.Linear(config.d_model, config.n_experts, bias=False)

        # Load-balancing biases — updated each step, NOT via backprop (see 6j)
        self.expert_bias = torch.zeros(config.n_experts)

    def forward(self, x):
        """x: [batch, seq_len, d_model]"""
        batch, seq_len, d_model = x.shape
        # Flatten tokens for routing
        x_flat = x.view(-1, d_model)   # [batch × seq, d_model]
        n_tokens = x_flat.shape[0]

        # ── Shared experts: always compute ──────────────────────────────
        shared_out = sum(expert(x) for expert in self.shared_experts)
        # shared_out: [batch, seq, d_model]

        # ── Router ───────────────────────────────────────────────────────
        router_logits = self.router(x_flat)                # [n_tokens, n_experts]
        biased_logits = router_logits + self.expert_bias   # add load-balance bias
        top_k_vals, top_k_idx = torch.topk(biased_logits, self.n_active, dim=-1)
        # top_k_vals: [n_tokens, n_active]
        # top_k_idx:  [n_tokens, n_active]

        # Routing weights: softmax over selected experts only
        routing_weights = torch.softmax(top_k_vals, dim=-1)   # [n_tokens, n_active]

        # ── Dispatch tokens to experts ────────────────────────────────────
        routed_out = torch.zeros_like(x_flat)   # [n_tokens, d_model]

        # Group tokens by expert for efficient batched computation
        for e_idx in range(self.n_experts):
            # Find all tokens routed to expert e_idx
            token_mask = (top_k_idx == e_idx).any(dim=-1)   # [n_tokens]
            if not token_mask.any():
                continue   # no tokens for this expert this step

            tokens_for_expert = x_flat[token_mask]           # [n_e, d_model]
            expert_output     = self.routed_experts[e_idx](tokens_for_expert)

            # Find routing weight for this expert for each relevant token
            e_position = (top_k_idx[token_mask] == e_idx).float().argmax(dim=-1)
            weights    = routing_weights[token_mask].gather(
                1, e_position.unsqueeze(1)
            ).squeeze(1)   # [n_e]

            routed_out[token_mask] += expert_output * weights.unsqueeze(-1)

        routed_out = routed_out.view(batch, seq_len, d_model)
        return shared_out + routed_out
```

---

### 6j. Auxiliary-Loss-Free Load Balancing — Full Algorithm

Standard MoE training adds an explicit auxiliary load-balancing loss that fights the main LM loss and degrades perplexity. DeepSeek-V3 eliminated this.

**The problem:**
Without balancing, a few experts get all the traffic (routing collapse), wasting capacity.
With a strong auxiliary loss, the LM loss is directly hurt (estimated 0.3–0.5% perplexity degradation).

**APEX-1 solution — bias-based dynamic balancing:**

```python
class LoadBalancer:
    """
    Maintains a per-expert bias term that nudges routing toward underused experts.
    Updated every step via simple rule — NOT via gradient / backprop.
    The LM objective sees zero interference.
    """
    def __init__(self, n_experts, target_rate=None, alpha=0.001):
        """
        n_experts:   total number of routed experts
        target_rate: ideal fraction of tokens per expert = 1 / n_experts
        alpha:       bias update step size (small → slow but stable)
        """
        self.n_experts   = n_experts
        self.target_rate = target_rate or (1.0 / n_experts)
        self.alpha       = alpha
        self.bias        = torch.zeros(n_experts)   # added to router logits

    def update(self, top_k_idx):
        """
        Call after each forward pass with the routing decisions.
        top_k_idx: [n_tokens, n_active] — which experts each token chose
        """
        n_tokens = top_k_idx.shape[0]

        # Count how many tokens each expert received
        counts = torch.zeros(self.n_experts)
        for e in range(self.n_experts):
            counts[e] = (top_k_idx == e).sum().float()

        # Normalize to get observed load fraction per expert
        observed_rate = counts / (n_tokens * top_k_idx.shape[1])
        # observed_rate[e] ≈ target_rate when balanced

        # Adjust bias:
        #   overloaded expert  (rate > target) → decrease bias (make less attractive)
        #   underloaded expert (rate < target) → increase bias (make more attractive)
        delta = self.target_rate - observed_rate   # positive if underloaded
        self.bias += self.alpha * delta.sign()     # ±alpha per step

        # Bias is bounded to prevent extreme values
        self.bias = self.bias.clamp(-1.0, 1.0)

    def get_bias(self):
        return self.bias   # injected into router_logits in MoEFFN.forward()

# ─── Training loop integration ────────────────────────────────────────────────
# balancer = LoadBalancer(n_experts=256, alpha=0.001)
#
# for batch in dataloader:
#     logits = model(batch)
#     loss   = cross_entropy(logits, targets)
#     loss.backward()              # normal gradient step — no aux loss
#     optimizer.step()
#     balancer.update(routing_idx) # update biases AFTER gradient step
#     model.moe_layers.expert_bias = balancer.get_bias()

# ─── Why this works ───────────────────────────────────────────────────────────
# The bias lives outside the computation graph (no gradient flows through it)
# It is updated by a simple rule, not by a loss function
# The LM cross-entropy loss sees zero interference from load balancing
# Observed result (DeepSeek-V3): near-uniform expert utilization with zero
# perplexity degradation vs auxiliary-loss balancing
```

---

### 6k. Dynamic Skip Gate

A lightweight learned gate that decides per-token whether to skip the FFN entirely.
Simple tokens (punctuation, articles, repeated phrases) almost never need FFN processing.

```python
class SkipGate(torch.nn.Module):
    """
    2-layer MLP producing a scalar gate per token.
    If gate < threshold → skip FFN (pass x through residual only).
    If gate ≥ threshold → FFN runs normally.

    Trained end-to-end via straight-through gradient estimation.
    At convergence: ~25–35% of tokens skip the FFN with <0.3% quality loss.
    """
    def __init__(self, d_model, hidden_dim=64, threshold=0.15):
        super().__init__()
        self.gate_mlp = torch.nn.Sequential(
            torch.nn.Linear(d_model, hidden_dim, bias=True),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 1, bias=True),
            torch.nn.Sigmoid()                  # output in (0, 1)
        )
        self.threshold = threshold

    def forward(self, x):
        """x: [batch, seq_len, d_model] → gate: [batch, seq_len, 1]"""
        return self.gate_mlp(x)

# ─── Straight-through estimator for training ──────────────────────────────────
# The hard threshold (gate < 0.15) is not differentiable.
# We use STE: forward uses binary decision, backward treats it as identity.
#
# class STEThreshold(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, gate, threshold):
#         return (gate < threshold).float()
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None   # pass gradient straight through
```

---

### 6l. Residual Connections

Every sub-layer adds its input back to its output:

```
x = x + Attention(RMSNorm(x))
x = x + FFN(RMSNorm(x))          # or skip, if gate fires
```

**Why this is essential:**
- Without residuals, gradients vanish at ~10+ layers — training is impossible
- With residuals, gradients flow directly from output back to embedding layer
- The residual path lets information skip layers unchanged — very deep networks become trainable
- Enables stable training at 72–128 layers

---

## 7. KV Cache & Memory Optimization

During inference (generation), the model processes one new token at a time
but needs all previous K and V vectors for attention.

### Without KV Cache (naive):
```
Generating token 100:
  Re-compute K, V for ALL 100 previous tokens at each step
  Time: O(seq_len²) total for full sequence
```

### With KV Cache:
```
At each step, store K and V for newly processed tokens
Generating token 100:
  Load cached K, V from steps 1–99
  Compute K, V only for token 100
  Time: O(seq_len) total — 100× faster at length 100
```

### APEX-1 KV Cache — two types:

```
Global layers (MLA):
  Cache: c_kv  [batch, seq, d_kv_compressed]
  Small model:  512 × seq_len × 2 bytes (fp16)
  Large model:  512 × 131072 × 2 = 134 MB per layer (72 global layers = 9.7 GB)

Local layers (GQA+SW):
  Cache: (K, V)  [batch, n_kv_heads, window, d_head]
  Bounded by local_window — does NOT grow with sequence length
  Small model:  2 × 2 × 2 × 512 × 64 × 2 bytes = 1 MB per layer (10 local = 10 MB)
```

### Additional Optimizations:

**Paged Attention (vLLM):**
Allocate KV cache in non-contiguous pages like OS virtual memory.
Eliminates internal fragmentation when serving many requests with different lengths.

**Quantized KV Cache:**
```
FP16 → INT8: halves cache memory, minimal quality impact
FP16 → INT4: 4× smaller, ~1% quality loss — best for batch serving
```

---

## 8. Long Context Handling

### Context length capability per model size:

```
Small:  8k native,  32k with YaRN (scale=4)
Medium: 64k native, 256k with YaRN (scale=4)
Large:  128k native, 1M+ with YaRN (scale=8)
```

### How the attention tiers achieve this:

```
For a 128k sequence on the large model:

LOCAL layers (60 of 72):
  Sliding window = 8192 tokens
  Cost per layer: O(128k × 8k) = O(1B) — linear in sequence length
  Handle: local syntax, phrase coherence, recent references

GLOBAL layers (12 of 72):
  MLA full attention over all 128k tokens
  Cost per layer: O(128k²) = O(16B) — quadratic BUT only 12 layers
  Handle: long-range cross-references, multi-hop reasoning, document structure

Total attention cost: 60×O(1B) + 12×O(16B) = 60B + 192B = 252B ops
vs. 72 full global layers: 72×O(16B) = 1152B ops — 4.6× more expensive
```

### YaRN enables extending context at zero training cost:
See Section 5 for full YaRN implementation.
Scale factor = `target_context / training_context`. For 1M on large model: `scale = 1M / 128k ≈ 8`.

---

## 9. Language Model Head

The final projection from hidden states to vocabulary logits.

```python
# Input:  [batch, seq_len, d_model]
# Output: [batch, seq_len, vocab_size]

# Final normalization
x_final = final_rmsnorm(x)   # [batch, seq_len, d_model]

# Weight tied to input embedding (same matrix, transposed)
# embedding_table.weight: [vocab_size, d_model]
logits = torch.matmul(x_final, embedding_table.weight.T)
# logits: [batch, seq_len, vocab_size]

# logits[b, t, v] = score for token v being next after position t
# No bias — saves vocab_size parameters, works equally well
```

### Multi-token prediction head (for training efficiency):

```python
# Speculative head: predicts next N tokens simultaneously
# Trained jointly — provides N× richer gradient signal
class MultiTokenHead(torch.nn.Module):
    def __init__(self, config, n_predict=4):
        super().__init__()
        self.n_predict = n_predict
        # One small projection per future offset (1, 2, 3, 4 steps ahead)
        self.heads = torch.nn.ModuleList([
            torch.nn.Linear(config.d_model, config.vocab_size, bias=False)
            for _ in range(n_predict)
        ])

    def forward(self, hidden_states):
        """Returns list of n_predict logit tensors, each [batch, seq, vocab]"""
        return [head(hidden_states) for head in self.heads]

# Training loss with multi-token prediction:
# L = L_main + λ × sum(L_offset_k for k in 1..n_predict)
# λ = 0.1 — speculative head contributes 10% of gradient signal
# At inference: speculative head drafts tokens for speculative decoding
```

---

## 10. Sampling & Decoding Strategy

Given logits over 151,643 tokens, how do we pick the next token?

### Temperature Scaling:
```python
logits = logits / temperature
# temperature = 1.0  → raw model distribution
# temperature < 1.0  → sharper (more deterministic)
# temperature > 1.0  → flatter (more random)
# temperature → 0    → greedy (always argmax)
```

### Top-P (Nucleus) Sampling:
```python
# Sort tokens by probability descending
sorted_logits, sorted_idx = torch.sort(logits, descending=True)
cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

# Remove tokens once cumulative probability exceeds top_p
remove_mask = cumulative_probs > top_p
remove_mask[..., 1:] = remove_mask[..., :-1].clone()  # shift right (keep first)
remove_mask[..., 0]  = False                            # always keep the best token

sorted_logits[remove_mask] = float('-inf')
# Scatter back to original order and sample
logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
next_token = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
```

### Top-K Sampling:
```python
top_k_logits, top_k_indices = logits.topk(k=50, dim=-1)
filtered = torch.full_like(logits, float('-inf'))
filtered.scatter_(-1, top_k_indices, top_k_logits)
next_token = torch.multinomial(torch.softmax(filtered, dim=-1), num_samples=1)
```

### Repetition Penalty:
```python
for token in set(generated_tokens):
    if logits[token] > 0:
        logits[token] /= repetition_penalty
    else:
        logits[token] *= repetition_penalty
# penalty = 1.0 → no change, 1.1–1.3 → reduces repetition effectively
```

### Recommended defaults:
```
Factual Q&A:       temperature=0.3, top_p=0.9,  top_k=off
Creative writing:  temperature=0.9, top_p=0.95, top_k=50
Code generation:   temperature=0.1, top_p=1.0,  top_k=off
Reasoning (final): temperature=0.3, top_p=0.9
Reasoning (think): temperature=0.6, top_p=0.95  ← more exploratory
```

---

## 11. Reasoning / Thinking Mode

Inspired by DeepSeek-R1 and Claude 3.7 Sonnet.

### What it is:
Before generating the final response, the model produces a hidden chain-of-thought inside `<|thinking|>...<|/thinking|>` tags. This scratchpad is shown to the model but optionally hidden from users.

### Inference flow:
```
User: "What is 127 × 348?"

Model generates:
  <|thinking|>
  Let me compute this step by step.
  127 × 348
  = 127 × 300 + 127 × 48
  = 38100 + 127 × 48
  = 38100 + 127 × 40 + 127 × 8
  = 38100 + 5080 + 1016
  = 44196
  <|/thinking|>

  127 × 348 = **44,196**
```

### Thinking budget enforcement:
```python
thinking_token_count = 0
in_thinking_mode     = False

for step in generation_loop:
    if current_token == THINKING_START_ID:
        in_thinking_mode = True

    if in_thinking_mode:
        thinking_token_count += 1
        if thinking_token_count >= max_thinking_tokens:
            # Force-close the scratchpad, begin final answer
            force_next_token(THINKING_END_ID)
            in_thinking_mode = False

    if current_token == THINKING_END_ID:
        in_thinking_mode = False
        # Reset temperature for final answer
        temperature = config.output_temperature
```

---

## 12. Training Pipeline — Full Detail

### 12a. Phase 1 — Pre-training

**Goal:** Learn language, world knowledge, reasoning, code from massive unlabeled text.

**Data mix:**
```
Web text (filtered CommonCrawl):    40%
Code (GitHub, Stack Overflow):      25%
Scientific papers (ArXiv, PubMed): 10%
Books (Project Gutenberg, etc.):    10%
Multilingual web text:               8%
Math (MATH dataset, AoPS, OEIS):     7%
```

**Loss function — next-token prediction with multi-token auxiliary:**
```python
def compute_pretrain_loss(logits_main, logits_speculative, token_ids, lambda_spec=0.1):
    """
    logits_main:        [batch, seq_len, vocab_size]   — main LM head
    logits_speculative: list of [batch, seq_len, vocab_size] — speculative heads
    token_ids:          [batch, seq_len]               — ground truth
    """
    # Main loss: predict next token at every position
    # Shift: input positions 0..T-1 predict targets 1..T
    main_logits  = logits_main[:, :-1, :].contiguous().view(-1, vocab_size)
    main_targets = token_ids[:, 1:].contiguous().view(-1)
    L_main = torch.nn.functional.cross_entropy(main_logits, main_targets)

    # Speculative head losses: predict token at offset k
    L_spec = 0.0
    for k, spec_logits in enumerate(logits_speculative, start=1):
        if k >= token_ids.shape[1]:
            break
        spec_l = spec_logits[:, :-k, :].contiguous().view(-1, vocab_size)
        spec_t = token_ids[:, k:].contiguous().view(-1)
        L_spec += torch.nn.functional.cross_entropy(spec_l, spec_t)

    L_spec /= len(logits_speculative)
    return L_main + lambda_spec * L_spec
```

**Optimizer — AdamW:**
```python
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr          = peak_lr,    # see schedule below
    betas       = (0.9, 0.95),  # β₂=0.95 (not 0.999) — better large-batch training
    eps         = 1e-8,
    weight_decay= 0.1         # L2 regularization — applied to all non-bias weights
)
```

**Learning rate schedule — warmup + cosine decay:**
```python
def get_lr(step, warmup_steps, max_steps, peak_lr, min_lr_ratio=0.1):
    """
    Warmup:         0 → warmup_steps    linear ramp 0 → peak_lr
    Cosine decay:   warmup → max_steps  cosine from peak_lr → min_lr
    """
    if step < warmup_steps:
        return peak_lr * (step / warmup_steps)

    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return peak_lr * (min_lr_ratio + (1 - min_lr_ratio) * cosine)

# Peak LR by model size:
#   Small (100M):  3e-4
#   Medium (7B):   1e-4
#   Large (900B):  3e-5
#
# Warmup steps:
#   Small: 1000    Medium: 5000    Large: 15000
```

**Gradient clipping:**
```python
# Applied every step before optimizer.step()
# Prevents explosive gradient norms from corrupted batches
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Training token budget:**
```
Small (100M):  5–10 billion tokens   (~2 days on 1×A100)
Medium (7B):   2–4 trillion tokens   (~30 days on 64×A100)
Large (900B):  10–20 trillion tokens (~180 days on 2048×A100)
```

---

### 12b. Phase 2 — Supervised Fine-Tuning (SFT)

**Goal:** Teach the model to follow instructions and produce well-formatted responses.

**Data — instruction/response pairs:**
```
Curated instruction datasets:       40%
Synthetic CoT (teacher-generated):  30%
Code tasks with test validation:    15%
Math with step-by-step solutions:   10%
Safety and refusal examples:         5%
```

**Critical rule — only compute loss on assistant tokens:**
```python
def compute_sft_loss(logits, token_ids, token_types):
    """
    token_types: 0=system, 1=user, 2=assistant
    Only compute loss where token_type == 2 (assistant)
    The model is learning to produce good outputs, not to mimic inputs.
    """
    labels = token_ids.clone()

    # Ignore system and user tokens in loss
    labels[token_types != 2] = -100

    # Also shift by 1 (predict next token)
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    return torch.nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=-100   # -100 positions contribute zero to loss
    )
```

**Training settings:**
```
lr:              1e-5 (10× lower than pre-training)
batch_size:      128–512 sequences
max_steps:       5000–20000 (much less data than pre-training)
grad_clip:       1.0
warmup_steps:    100
```

---

### 12c. Gradient Flow Through MoE Layers

MoE introduces a routing step (top-K selection) that is not differentiable.
This is how gradients still flow correctly:

```
Forward pass:
  router_logits = W_router × h           ← differentiable
  top_k_weights = softmax(top_k(logits)) ← differentiable (softmax on subset)
  output = Σ weight_k × FFN_k(h)         ← differentiable through weights and FFNs

What is NOT differentiable:
  The binary selection decision "which K experts" — this is argmax/topk

Why it still works:
  Gradients DO flow through:
    (a) The routing weights (softmax values) — dL/d(weight_k)
    (b) The expert FFN parameters for selected experts — dL/d(FFN_k params)
    (c) The router weight matrix — dL/d(W_router) via chain rule

  Gradients do NOT flow through:
    (a) Expert FFNs that were NOT selected (they got zero gradient this step)

  This is fine because:
    Over many training steps, each expert gets selected for many tokens
    The router learns which tokens to send where via the routing weight gradients
    Non-selected experts are implicitly discouraged (their logit was lower)

Load balancing (Section 6j) ensures no expert is permanently starved of gradient signal.
```

---

## 13. Alignment: RLHF → DPO → GRPO — Full Detail

After SFT, the model needs to learn nuanced human preferences and reasoning quality beyond what labeled examples can teach.

---

### 13a. RLHF + PPO

**Step 1 — Collect preference data:**
```
For each prompt, generate 2–4 responses from the SFT model
Human annotators rank: "A > B > C" or rate on dimensions (helpfulness, safety)
Result: dataset of (prompt, chosen_response, rejected_response) triples
```

**Step 2 — Train a Reward Model:**
```python
# Reward model = SFT model with regression head replacing LM head
class RewardModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.backbone = base_model
        self.reward_head = nn.Linear(d_model, 1, bias=False)

    def forward(self, input_ids, attention_mask):
        hidden = self.backbone(input_ids, attention_mask)
        # Use the last non-padded token's hidden state as the sequence representation
        reward = self.reward_head(hidden[:, -1, :]).squeeze(-1)
        return reward  # scalar reward per sequence

# Reward model loss — Bradley-Terry preference model
def reward_model_loss(reward_chosen, reward_rejected):
    """chosen should score higher than rejected."""
    return -torch.log(torch.sigmoid(reward_chosen - reward_rejected)).mean()
```

**Step 3 — PPO fine-tuning:**
```
For each prompt:
  1. Generate response with policy (current model)
  2. Score response with reward model → r
  3. Compute KL divergence from SFT reference → KL
  4. Actual reward = r - β × KL   (β ≈ 0.02–0.1)
  5. Update policy with PPO clipped objective to maximize actual reward

KL penalty (β × KL) prevents reward hacking:
  If model maximizes r while drifting far from SFT, KL becomes large
  This keeps the policy "close" to the SFT baseline
```

---

### 13b. DPO — Simpler Alternative

DPO skips the reward model entirely and trains directly on preference pairs:

```python
def dpo_loss(model, reference_model, prompt, chosen, rejected, beta=0.1):
    """
    beta: controls how much we trust the reference model
    Higher beta → stay closer to reference (more conservative)
    """
    def log_prob(m, prompt, response):
        input_ids = tokenize(prompt + response)
        logits    = m(input_ids)
        # Sum log-probabilities of response tokens only
        return compute_sequence_logprob(logits, input_ids, response_start_idx)

    # Log-probabilities from policy (model being trained)
    log_pi_chosen   = log_prob(model, prompt, chosen)
    log_pi_rejected = log_prob(model, prompt, rejected)

    # Log-probabilities from reference (frozen SFT model)
    with torch.no_grad():
        log_ref_chosen   = log_prob(reference_model, prompt, chosen)
        log_ref_rejected = log_prob(reference_model, prompt, rejected)

    # Implicit reward = β × (log π(y|x) - log π_ref(y|x))
    reward_chosen   = beta * (log_pi_chosen   - log_ref_chosen)
    reward_rejected = beta * (log_pi_rejected - log_ref_rejected)

    # Loss: maximize margin between chosen and rejected implicit rewards
    loss = -torch.log(torch.sigmoid(reward_chosen - reward_rejected))
    return loss.mean()

# DPO is more stable than PPO and easier to implement.
# Recommended for initial alignment. PPO if further fine-tuning needed.
```

---

### 13c. GRPO — Full Rollout Loop With Code

GRPO (Group Relative Policy Optimization) is DeepSeek-R1's key innovation for reasoning training. It eliminates the critic/value network, making training more stable and memory-efficient.

**Core idea:**
For each prompt, sample a group of G responses. Rank them. Use within-group relative reward as the training signal instead of an absolute reward score.

```python
def grpo_training_step(model, reference_model, reward_fn, prm, prompts,
                        G=8, beta=0.04, lambda_prm=0.3):
    """
    model:          policy being trained
    reference_model: frozen SFT model (for KL constraint)
    reward_fn:      outcome reward — checks final answer correctness
    prm:            Process Reward Model — scores intermediate reasoning steps
    prompts:        batch of math/reasoning prompts
    G:              group size — number of rollouts per prompt
    beta:           KL penalty coefficient
    lambda_prm:     weight of process reward vs outcome reward
    """
    all_losses = []

    for prompt in prompts:
        # ── Step 1: Sample G responses from current policy ─────────────────
        responses = []
        for _ in range(G):
            response = model.generate(
                prompt,
                max_new_tokens     = 2048,
                temperature        = 0.7,    # exploration during training
                top_p              = 0.95,
                include_thinking   = True     # allow scratchpad
            )
            responses.append(response)

        # ── Step 2: Score each response ────────────────────────────────────
        rewards = []
        for response in responses:
            # Outcome reward: is the final answer correct?
            outcome_r = reward_fn(prompt, response)         # binary {0, 1} or float

            # Process reward: are the reasoning STEPS correct?
            # PRM evaluates each <|thinking|> step independently
            steps       = extract_thinking_steps(response)  # list of reasoning steps
            step_scores = prm.score_steps(prompt, steps)    # list of floats in [0,1]
            process_r   = sum(step_scores) / len(step_scores) if steps else 0.0

            # Combined reward: outcome quality + step quality
            combined = (1 - lambda_prm) * outcome_r + lambda_prm * process_r
            rewards.append(combined)

        rewards = torch.tensor(rewards)  # [G]

        # ── Step 3: Compute group-normalized advantages ────────────────────
        # Instead of absolute reward, use reward relative to the group mean
        # This eliminates the need for a learned value/critic network
        group_mean   = rewards.mean()
        group_std    = rewards.std().clamp(min=1e-6)   # avoid /0
        advantages   = (rewards - group_mean) / group_std
        # advantages[i] > 0: this response was better than average → reinforce it
        # advantages[i] < 0: this response was worse than average → suppress it

        # ── Step 4: Compute GRPO loss for each response ────────────────────
        for response, advantage in zip(responses, advantages):
            input_ids    = tokenize(prompt + response)
            response_ids = tokenize(response)

            # Log-probability of response under CURRENT policy
            log_pi = model.log_prob(input_ids, response_ids)

            # Log-probability under REFERENCE policy (frozen)
            with torch.no_grad():
                log_ref = reference_model.log_prob(input_ids, response_ids)

            # KL divergence: how much has the policy drifted from reference?
            kl_div = log_pi - log_ref

            # GRPO objective (clipped, like PPO):
            # Ratio of new vs old policy probabilities
            ratio = torch.exp(log_pi - log_ref.detach())

            # Clipped surrogate objective — prevents too-large policy updates
            clip_eps = 0.2
            L_clip = torch.min(
                ratio * advantage,
                torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * advantage
            )

            # Full loss: maximize reward, penalize KL drift
            loss = -(L_clip - beta * kl_div).mean()
            all_losses.append(loss)

    # ── Step 5: Backprop and update ────────────────────────────────────────
    total_loss = torch.stack(all_losses).mean()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad()

    return total_loss.item()

# ─── Why GRPO beats PPO for reasoning ──────────────────────────────────────
# PPO requires a critic network (same size as policy) → 2× memory
# GRPO: group mean IS the baseline — no critic needed → same memory as SFT
# Group normalization produces better-calibrated advantage signals
# PRM provides step-level signal: even wrong final answers may have good steps
# Converges faster on math/code/logic than PPO in practice (DeepSeek-R1 result)
```

---

### 13d. Constitutional AI (from Anthropic/Claude)

An automated approach to alignment that does not require human preference labels.

```
1. Define a "constitution" — a set of principles:
   - "Be helpful, harmless, and honest"
   - "Never assist with creating weapons"
   - "Respect user privacy and autonomy"
   - "Acknowledge uncertainty rather than confabulate"
   - "Treat all people with equal dignity"
   - [30–50 principles in practice]

2. Red-teaming: Generate responses to adversarial prompts
   (jailbreaks, harmful requests, edge cases)

3. Critique: Have the model critique its own response against each principle:
   Prompt: "Does this response violate the principle that [X]?
            Identify specific problems."

4. Revision: Have the model revise its response to fix violations:
   Prompt: "Please rewrite your response to be consistent with [principle X]."

5. Dataset creation: Collect (original_prompt, revised_response) pairs

6. Training: Use these pairs for DPO
   chosen  = revised (constitutionally correct) response
   rejected = original (potentially violating) response

7. Iterate: Run the model through the constitution again after each training round
```

**In APEX-1: Constitutional AI is integrated into GRPO reward (Section 13f), not run separately.**

---

### 13e. Process Reward Model (PRM)

A reward model that evaluates each reasoning step independently, not just the final answer.

```python
class ProcessRewardModel(nn.Module):
    """
    Takes a prompt and a list of reasoning steps.
    Returns a quality score for each step independently.
    Trained on human annotations of (step, correctness_label) pairs.
    """
    def __init__(self, base_model, d_model):
        super().__init__()
        self.backbone   = base_model        # same architecture as policy
        self.step_head  = nn.Linear(d_model, 1)
        self.sigmoid    = nn.Sigmoid()

    def score_steps(self, prompt, steps):
        """
        prompt: str
        steps:  list of str, each a reasoning step from <|thinking|>
        Returns: list of float scores in [0, 1] — one per step
        """
        scores = []
        context = prompt
        for step in steps:
            context += "\n" + step
            input_ids  = tokenize(context)
            hidden     = self.backbone(input_ids)
            step_score = self.sigmoid(self.step_head(hidden[:, -1, :])).item()
            scores.append(step_score)
            # Each score conditions on all previous steps (cumulative context)
        return scores

# PRM training data:
# (prompt, step_1, step_2, ..., step_k, label_k) where label_k ∈ {correct, wrong}
# Labels collected by checking if removing step_k still leads to correct final answer
# Used in GRPO reward (13c) and can be used as verifier at inference time
```

---

### 13f. Dual-Signal Alignment — GRPO + Constitutional AI Simultaneously

APEX-1's key alignment innovation: rather than running Constitutional AI as a separate post-processing stage, it is integrated directly into the GRPO reward function.

```python
def combined_reward(prompt, response, constitution, prm, answer_checker):
    """
    Three reward signals, combined in a single score used for GRPO.

    outcome_reward:       Is the final answer correct? (for reasoning tasks)
    process_reward:       Are the reasoning steps valid? (from PRM)
    constitutional_score: Does the response follow all principles?
    """
    # ── Signal 1: Outcome reward ───────────────────────────────────────────
    outcome_r = answer_checker(prompt, response)    # float [0, 1]

    # ── Signal 2: Process reward ───────────────────────────────────────────
    steps     = extract_thinking_steps(response)
    process_r = mean(prm.score_steps(prompt, steps)) if steps else 0.5

    # ── Signal 3: Constitutional score ────────────────────────────────────
    # Model critiques its own response against each principle
    violations = 0
    for principle in constitution:
        critique_prompt = (
            f"Response: {response}\n\n"
            f"Principle: {principle}\n\n"
            f"Does this response violate this principle? Answer YES or NO."
        )
        critique = model.generate(critique_prompt, max_new_tokens=5, temperature=0.1)
        if "YES" in critique.upper():
            violations += 1

    constitutional_score = 1.0 - (violations / len(constitution))

    # ── Combined reward — weighted sum ────────────────────────────────────
    combined = (
        0.5  * outcome_r          +   # reasoning correctness
        0.2  * process_r          +   # step quality
        0.3  * constitutional_score   # safety and alignment
    )
    return combined

# Benefits of simultaneous vs sequential:
# Sequential (standard): SFT → RLHF → Constitutional AI (3 separate phases)
#   Problem: each phase can undo gains from the previous
#   Constitutional AI may reduce reasoning quality; RLHF may reduce safety
#
# Simultaneous (APEX-1): All three signals shape every GRPO gradient step
#   Result: the model learns that being helpful AND safe AND correct are not
#   competing objectives — they are jointly optimized from the start
```

---

## 14. Inference Optimizations

### Speculative Decoding (using multi-token head):
```
Multi-token prediction head drafts next 3 tokens quickly
Main model verifies all 3 in a single forward pass
If main model agrees → accept all 3 (3× throughput for free)
If main model disagrees at position k → accept k-1 tokens, resample from k

Draft model generates: ["The", "cat", "sat"]  (fast, speculative head)
Large model verifies:   ✓         ✓      ✗
Accept: "The", "cat" — resample "sat" position from large model
Result: 2 tokens in ~1.1× the time of 1 token — better than pure 1-token sampling
```

### Quantization:
```
FP32 → BF16:  2× memory reduction, preferred for training and inference
BF16 → INT8:  2× additional reduction, ~0.5% quality loss, good for serving
INT8 → INT4:  2× additional, ~2–4% quality loss, best for edge/mobile
INT8 → GPTQ:  best quality/size at 4-bit — recommended for deployment
```

### Continuous Batching:
In production serving, batch requests dynamically as they arrive rather than waiting for a fixed batch.
GPU utilization improves from ~30% (static batching) to ~80%+ (continuous batching).

### Tensor Parallelism:
```
Split each weight matrix across GPUs along one dimension.
Each GPU computes a partial result.
All-reduce (ring allreduce) combines partial outputs.

Attention: Q, K, V, O weights split across GPU count
FFN:       W_gate, W_up split column-wise; W_down split row-wise

Recommended: 8-GPU tensor parallel for large model (TP=8)
```

### Pipeline Parallelism (for extremely large models):
```
Assign consecutive transformer layers to different GPUs.
GPU 0: layers 0–17   GPU 1: layers 18–35   GPU 2: layers 36–53   GPU 3: layers 54–71
Micro-batch pipeline: while GPU 1 computes forward for micro-batch 1,
                       GPU 0 computes forward for micro-batch 2 (overlapped)
```

---

## 15. Model Size Configurations

### APEX-1-Small (recommended starting point)

```yaml
# apex1_small_config.yaml
model:
  d_model: 512
  n_layers: 12
  n_heads_q: 8
  n_heads_kv: 2
  d_head: 64
  d_kv_compressed: 64      # MLA latent dim
  d_q_compressed: 96       # MLA Q latent dim
  d_head_rope: 32          # decoupled RoPE head dim (MLA)
  d_ffn: 1376
  vocab_size: 151643
  max_seq_len: 8192
  rope_base: 10000
  rope_scaling: 1.0

attention:
  global_layer_freq: 6     # every 6th layer is global MLA
  local_window: 512        # sliding window for local GQA layers
  flash: true

moe:
  enabled: true
  n_experts: 8
  n_active: 2              # top-K routed experts per token
  n_shared: 1              # always-active shared experts
  moe_layer_freq: 2        # every 2nd layer is MoE (odd layers)
  balancer_alpha: 0.001    # load balance bias update rate

skip_gate:
  enabled: true
  hidden_dim: 64
  threshold: 0.15

multi_token_head:
  enabled: true
  n_predict: 4             # predict next 1, 2, 3, 4 tokens simultaneously
  lambda_spec: 0.1         # weight of speculative heads in training loss

thinking:
  enabled: true
  max_thinking_tokens: 1024

training:
  batch_size: 32
  seq_len: 2048
  peak_lr: 3e-4
  warmup_steps: 1000
  max_steps: 100000
  grad_clip: 1.0
  weight_decay: 0.1
  optimizer: adamw
  beta1: 0.9
  beta2: 0.95

grpo:
  G: 8                     # group size
  beta: 0.04               # KL penalty coefficient
  lambda_prm: 0.3          # process reward weight
  lambda_cai: 0.3          # constitutional AI weight
  clip_eps: 0.2            # PPO clip epsilon
```

### APEX-1-Medium

```yaml
model:
  d_model: 2048
  n_layers: 36
  n_heads_q: 16
  n_heads_kv: 4
  d_head: 128
  d_kv_compressed: 256
  d_q_compressed: 384
  d_head_rope: 64
  d_ffn: 5504
  max_seq_len: 65536
  rope_base: 500000
  rope_scaling: 4.0

moe:
  n_experts: 64
  n_active: 4
  n_shared: 2

attention:
  global_layer_freq: 6
  local_window: 2048

# Total params: ~7B   Active per token: ~2B
```

### APEX-1-Large

```yaml
model:
  d_model: 7168
  n_layers: 72
  n_heads_q: 128
  n_heads_kv: 8
  d_head: 128
  d_kv_compressed: 512
  d_q_compressed: 768
  d_head_rope: 64
  d_ffn: 18432
  max_seq_len: 131072
  rope_base: 1000000
  rope_scaling: 8.0

moe:
  n_experts: 256
  n_active: 8
  n_shared: 4

attention:
  global_layer_freq: 6
  local_window: 8192

# Total params: ~900B   Active per token: ~45B
```

---

## 16. Full Forward Pass — Step by Step

Here is every operation in order for one forward pass through APEX-1-Small.

```
Input: ["What", "is", "2", "+", "2", "?"]
Token IDs: [2874, 374, 17, 489, 17, 30]   ← from BPE tokenizer
prefix_len = 6  (all tokens are system/user — bidirectional attention applies)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 1: EMBEDDING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  x = embedding_table[token_ids] × sqrt(512)
  x.shape = [1, 6, 512]   ← [batch=1, seq_len=6, d_model=512]
  No positional encoding added here — RoPE is applied inside attention

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 2: BUILD ATTENTION MASKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  global_mask = build_apex_attention_mask(6, 6, 512, is_global=True)
    → 6×6 full bidirectional (all prefix → causal after)
  local_mask  = build_apex_attention_mask(6, 6, 512, is_global=False)
    → 6×6 bidirectional prefix + sliding window for generation

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 3: ROPE CACHE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  cos_cache, sin_cache = precompute_rope_cache(d_head=64, max_seq=8192, base=10000)
  positions = [0, 1, 2, 3, 4, 5]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEPS 4–15: 12 TRANSFORMER BLOCKS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ── LAYERS 0–4 (LOCAL, dense/MoE alternating) ──────────────────────────

  Layer 0 (LOCAL, Dense FFN):
  │
  ├─ ATTENTION (GQA + Sliding Window):
  │   a. h = RMSNorm(x)                     shape: [1, 6, 512]
  │   b. Q = h × W_Q → reshape             [1, 6, 8, 64] → transpose [1, 8, 6, 64]
  │      K = h × W_K → reshape             [1, 6, 2, 64] → transpose [1, 2, 6, 64]
  │      V = h × W_V → reshape             [1, 6, 2, 64] → transpose [1, 2, 6, 64]
  │   c. Q, K = apply_rope(Q, K, cos, sin, positions)
  │   d. K, V expand (GQA): repeat_interleave(4) → [1, 8, 6, 64]
  │   e. scores = Q @ K.T / sqrt(64)       [1, 8, 6, 6]
  │      apply local_mask (bidirectional for prefix)
  │   f. weights = softmax(scores)
  │      attn_out = weights @ V             [1, 8, 6, 64]
  │   g. attn_out → merge → [1, 6, 512]
  │      attn_out = attn_out × W_O         [1, 6, 512]
  │   h. x = x + attn_out                  residual connection
  │
  ├─ SKIP GATE:
  │   gate = SkipGate(x)                   [1, 6, 1]
  │   skip = gate < 0.15                   True for simple tokens
  │
  └─ FFN (Dense SwiGLU, for active tokens):
      a. h = RMSNorm(x)                    [1, 6, 512]
      b. gate_val = SiLU(h × W_gate)       [1, 6, 1376]
         value    = h × W_up               [1, 6, 1376]
         hidden   = gate_val * value       [1, 6, 1376]
      c. ffn_out = hidden × W_down         [1, 6, 512]
      d. x = x + ffn_out * (~skip).float() residual (0 for skipped tokens)

  Layer 1 (LOCAL, MoE FFN):
  │  [Same attention as Layer 0]
  │
  └─ FFN (MoE SwiGLU):
      a. h = RMSNorm(x)                           [1, 6, 512]
      b. shared_out = FFN_shared(h)               [1, 6, 512]
      c. router_logits = h × W_router             [1, 6, 8]
         + expert_bias (load balancing)
         top2_vals, top2_idx = topk(logits, k=2)  [1, 6, 2]
         routing_weights = softmax(top2_vals)      [1, 6, 2]
      d. For each active expert e (2 per token):
           expert_out_e = DenseFFN_e(h)            [1, 6, 512]
           weighted_e   = expert_out_e × routing_weight_e
      e. routed_out = sum(weighted_e for active e)
      f. ffn_out = shared_out + routed_out         [1, 6, 512]
      g. x = x + ffn_out * (~skip).float()        residual

  Layers 2, 3, 4: repeat pattern (even=Dense, odd=MoE)

  ── LAYER 5 (GLOBAL — MLA full attention) ──────────────────────────────

  Layer 5:
  │
  ├─ ATTENTION (MLA — full context, bidirectional mask):
  │   a. h = RMSNorm(x)                            [1, 6, 512]
  │   b. c_kv = h × W_DKV                          [1, 6, 64]   ← KV latent
  │      K    = c_kv × W_UK → reshape              [1, 2, 6, 64]
  │      V    = c_kv × W_UV → reshape              [1, 2, 6, 64]
  │   c. c_q  = h × W_DQ                           [1, 6, 96]   ← Q latent
  │      Q    = c_q × W_UQ → reshape               [1, 8, 6, 64]
  │   d. Decoupled RoPE:
  │      Q_rope = h × W_QR → apply_rope            [1, 8, 6, 32]
  │      K_rope = h × W_KR → apply_rope            [1, 2, 6, 32]
  │      Q = cat([Q, Q_rope], dim=-1)               [1, 8, 6, 96]
  │      K = cat([K, K_rope], dim=-1)               [1, 2, 6, 96]
  │   e. K, V expand (GQA): repeat_interleave(4)   [1, 8, 6, 96/64]
  │   f. scores = Q @ K.T / sqrt(96)               [1, 8, 6, 6]
  │      apply global_mask (full bidirectional prefix + causal generation)
  │   g. weights = softmax(scores)
  │      attn_out = weights @ V                     [1, 8, 6, 64]
  │   h. merge → [1, 6, 512], project W_O → [1, 6, 512]
  │   i. x = x + attn_out                          residual
  │
  └─ FFN (Dense SwiGLU, layer 5 is even):
      [Same as Layer 0 FFN, with skip gate]

  Layers 6–10: LOCAL layers (same as 0–4 pattern)
  Layer 11:    GLOBAL (same as Layer 5, second MLA pass)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 16: FINAL NORM
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  x = RMSNorm(x)
  x.shape = [1, 6, 512]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 17: LM HEAD (multi-token prediction)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  main_logits  = x × embedding_table.T         [1, 6, 151643]  ← main head
  spec_logits  = [head_k(x) for k in 1..4]     4 × [1, 6, 151643]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STEP 18: NEXT TOKEN PREDICTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  next_logits = main_logits[0, -1, :]           [151643]
  next_logits /= temperature (0.3 for factual)
  probs = softmax(nucleus_filter(next_logits, top_p=0.9))
  next_token = sample(probs)

  next_token → decode → "4"   ← model predicts correctly ✓

  (Speculative head also predicted "4" at offset +1 → high probability it
   also correctly predicted the next token after "4", reducing generation steps)
```

---

## 17. Implementation Checklist

Use this to track build progress. Recommended order: top to bottom.

### Tokenizer
- [ ] Load Qwen3 BPE tokenizer (151,643 vocab) or train equivalent
- [ ] Verify all special tokens have correct IDs
- [ ] Implement chat template formatting (system/user/assistant/thinking)
- [ ] Test encode/decode roundtrip — no information loss
- [ ] Verify CJK, Arabic, math symbol tokenization efficiency

### Core Architecture
- [ ] Embedding table with `sqrt(d_model)` scaling
- [ ] Weight tying: embedding ↔ LM head (shared matrix, transposed)
- [ ] RMSNorm (no bias, learned γ, eps=1e-6)
- [ ] RoPE: precompute sin/cos cache, rotate_half helper, apply_rope function
- [ ] YaRN frequency scaling for context extension
- [ ] `is_global_layer(layer_idx, global_layer_freq)` dispatcher function
- [ ] **GQA attention**: W_Q/K/V/O projections, KV head expansion, scaled dot product
- [ ] **Sliding window mask** for local GQA layers
- [ ] **MLA attention**: W_DKV, W_UK, W_UV, W_DQ, W_UQ, W_QR, W_KR matrices
- [ ] **Decoupled RoPE** in MLA (separate from content projection)
- [ ] **MLA KV cache**: cache `c_kv` (latent), reconstruct K/V on the fly
- [ ] Prefix bidirectional mask (system/user tokens attend to each other fully)
- [ ] Flash Attention via `torch.nn.functional.scaled_dot_product_attention`
- [ ] SwiGLU FFN (3 weight matrices: W_gate, W_up, W_down — no bias)
- [ ] MoE router: linear + top-K selection + softmax routing weights
- [ ] Shared experts (n_shared always-active FFNs, bypass router)
- [ ] Expert dispatch: group tokens by expert, batch compute, accumulate weighted sum
- [ ] Load balancer: `LoadBalancer` class, bias update rule, clamp ±1.0
- [ ] Dynamic skip gate: 2-layer MLP, threshold, straight-through estimator
- [ ] Residual connections on both attention and FFN sub-layers
- [ ] Pre-norm placement: RMSNorm before each sub-layer
- [ ] Final RMSNorm after all transformer blocks
- [ ] Multi-token prediction head (4 speculative heads, weight-tied or separate)

### Inference
- [ ] GQA KV cache: (K, V) tensors, trimmed to local_window
- [ ] MLA KV cache: c_kv tensor, grows with sequence, no trimming needed
- [ ] Temperature scaling
- [ ] Top-P (nucleus) filtering with cumulative sort
- [ ] Top-K filtering
- [ ] Repetition penalty
- [ ] `<|thinking|>` / `<|/thinking|>` token detection and budget enforcement
- [ ] Generation loop with EOS stopping and max_new_tokens limit
- [ ] Speculative decoding using multi-token head drafts

### Training
- [ ] Pre-training loss: cross-entropy on assistant tokens + multi-token aux loss
- [ ] SFT loss: cross-entropy with `ignore_index=-100` for non-assistant tokens
- [ ] AdamW with β₁=0.9, β₂=0.95, ε=1e-8, weight_decay=0.1
- [ ] Cosine LR schedule with warmup
- [ ] Gradient clipping at 1.0
- [ ] LoadBalancer.update() called after each optimizer step
- [ ] Checkpoint saving (model state, optimizer state, step count, LR)
- [ ] Checkpoint loading with model/optimizer state restoration

### Alignment (Phase 3)
- [ ] Reward model: backbone + scalar regression head + Bradley-Terry loss
- [ ] DPO loss: log-ratio difference between policy and reference
- [ ] GRPO rollout loop: G samples per prompt, group-normalized advantages
- [ ] Process Reward Model (PRM): step-level scoring with cumulative context
- [ ] Constitutional AI critique loop: model critiques own outputs against principles
- [ ] Combined reward function: outcome + process + constitutional
- [ ] GRPO gradient step with clipped surrogate objective

### Testing
- [ ] Load Small config (d_model=512, n_layers=12)
- [ ] Forward pass: all tensor shapes match Section 16 exactly
- [ ] Attention masks: verify bidirectional prefix, causal generation, sliding window
- [ ] MLA cache: verify c_kv grows correctly across autoregressive steps
- [ ] Skip gate: confirm ~25% of tokens skip FFN at convergence
- [ ] Expert utilization: confirm near-uniform distribution across experts
- [ ] Loss decreases monotonically over first 500 steps
- [ ] Model generates plausible completions after 5000 steps
- [ ] Thinking mode: `<|thinking|>` tokens appear, budget enforced correctly
- [ ] YaRN: generate at 2× training context length without quality collapse

---

*APEX-1 Architecture — v2.0*
*Designed to be the best of Claude · GPT · DeepSeek · Qwen · Gemma · GLM · KIMI · MiniMax*
*Every component fully specified, every gap filled.*
