# 06 — Attention Masks: Who Can See What?

> **Difficulty:** ⭐⭐☆☆☆ Beginner-Intermediate  
> **Source file:** `apex/model/mask.py`  
> **You will learn:** Why attention needs a mask, how the APEX-1 mask combines three regimes, and the BUG-10 performance fix.

---

## 1. The Problem: Attention Can Cheat

In a transformer, every token can potentially **attend to** (look at) every other token. But this causes two problems:

1. **Future leakage**: When training, token at position 5 should not see tokens 6, 7, 8, ... (that would be cheating — the model would learn to copy the answer).
2. **Efficiency**: For very long sequences, attending to everything is expensive and unnecessary.

The **attention mask** is a matrix of True/False values that tells each query token which key tokens it is **allowed to look at**.

---

## 2. The Three Attention Regimes in APEX-1

APEX-1 combines three types of attention in one mask:

### Regime 1: Prefix Bidirectional (GLM-4 Style)
The **system prompt + user message** (prefix) can attend to each other in all directions — no restriction:

```
Prefix tokens: [sys_1, sys_2, user_1, user_2]
Can attend:    all ↔ all (full bidirectional)
```

**Why?** The system prompt and user message are static — they are fully known before generation begins. Letting them see each other bidirectionally gives richer contextual representations.

### Regime 2: Causal (for Global / MLA Layers)
Generated tokens can attend to all previous tokens but not future ones:

```
gen_1 can see: [sys_1, sys_2, user_1, user_2, gen_1]
gen_2 can see: [sys_1, sys_2, user_1, user_2, gen_1, gen_2]
gen_3 can see: [sys_1, sys_2, user_1, user_2, gen_1, gen_2, gen_3]
```

### Regime 3: Causal + Sliding Window (for Local / GQA Layers)
Like causal, but each query only sees the most recent `local_window` tokens:

```
local_window = 3:
gen_5 can see: [gen_3, gen_4, gen_5]   (only the last 3)
gen_6 can see: [gen_4, gen_5, gen_6]
```

---

## 3. Visualising the Combined Mask

For a sequence with `prefix_len=2`, `total_len=6`, `local_window=3`:

**Global layer mask** (`is_global_layer=True`):
```
        [p0] [p1] [g0] [g1] [g2] [g3]
  [p0]  [ T ]  T    F    F    F    F    ← prefix sees other prefix
  [p1]  [ T ]  T    F    F    F    F
  [g0]  [ T ]  T    T    F    F    F    ← gen sees prefix + past gen
  [g1]  [ T ]  T    T    T    F    F
  [g2]  [ T ]  T    T    T    T    F
  [g3]  [ T ]  T    T    T    T    T
```

**Local layer mask** (`is_global_layer=False`, `window=3`):
```
        [p0] [p1] [g0] [g1] [g2] [g3]
  [p0]  [ T ]  T    F    F    F    F
  [p1]  [ T ]  T    F    F    F    F
  [g0]  [ T ]  T    T    F    F    F    ← gen_0 sees prefix + itself
  [g1]  [ T ]  T    T    T    F    F
  [g2]    F    T    T    T    T    F    ← window cuts off p0 (too far)
  [g3]    F    F    T    T    T    T    ← window of 3: only g1,g2,g3
```

(`T` = True = can attend, `F` = False = masked out)

---

## 4. The BUG-10 Story: 128,000 Python Iterations

The original sliding-window mask was built with a Python `for` loop:

```python
# ORIGINAL (slow) — ran 128,000 times for a 128K sequence!
for i in range(prefix_len, total_len):         # outer loop
    for j in range(total_len):                  # inner loop
        q_pos = i
        k_pos = j
        if k_pos <= q_pos and q_pos - k_pos < local_window:
            mask[i, j] = True
```

For a sequence of 128,000 tokens, this executes **128,000 × 128,000 = 16 billion iterations** in Python — catastrophically slow.

**The fix** uses tensor broadcasting — a single mathematical operation:

```python
# FIXED: vectorised, O(1) Python calls regardless of sequence length
row_idx = torch.arange(gen_len).unsqueeze(1)   # [gen, 1]
col_idx = torch.arange(total_len).unsqueeze(0)  # [1, total]
abs_row = row_idx + prefix_len
# causal: col <= abs_row
# window: abs_row - col < local_window
window_mask = (col_idx <= abs_row) & ((abs_row - col_idx) < local_window)
mask[prefix_len:, :] = window_mask
```

Broadcasting makes PyTorch compute the entire mask matrix in C++ with a single call — thousands of times faster.

---

## 5. Full Annotated Source: `apex/model/mask.py`

```python
"""
Attention Mask Builder for APEX-1.

Combines three attention regimes:
1. Prefix bidirectional (GLM-4 style)
2. Full causal (global MLA layers)
3. Sliding-window causal (local GQA layers)

BUG-10 FIX: Vectorised with torch.arange broadcasting.
"""

import torch


def build_apex_attention_mask(
    prefix_len: int,      # Tokens with bidirectional attention
    total_len: int,       # Total sequence length
    local_window: int,    # Window size for sliding-window attention
    is_global_layer: bool,# True = full causal, False = windowed causal
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Build the APEX-1 attention mask.
    
    Returns:
        Boolean mask [total_len, total_len].
        True = allowed to attend, False = masked.
    """
    # Start with all False (no token can see anything)
    mask = torch.zeros(total_len, total_len, dtype=dtype, device=device)

    # ── Regime 1: Prefix block (bidirectional) ──────────────────────────
    # All prefix tokens see all other prefix tokens
    if prefix_len > 0:
        mask[:prefix_len, :prefix_len] = True

    if is_global_layer:
        # ── Regime 2: Global layers — full causal attention ─────────────
        # Each generated token (row) sees all previous tokens up to itself
        if prefix_len < total_len:
            gen_len = total_len - prefix_len
            # row_idx: which generated token are we? [gen, 1]
            row_idx = torch.arange(gen_len, device=device).unsqueeze(1)
            # col_idx: which key position? [1, total]
            col_idx = torch.arange(total_len, device=device).unsqueeze(0)
            # gen token at row i has absolute position (prefix_len + i)
            # it can see column j if j <= (prefix_len + i)
            gen_causal = col_idx <= (row_idx + prefix_len)  # [gen, total]
            mask[prefix_len:, :] = gen_causal
    else:
        # ── Regime 3: Local layers — sliding-window causal ──────────────
        gen_len = total_len - prefix_len
        if gen_len > 0:
            row_idx = torch.arange(gen_len, device=device).unsqueeze(1)
            col_idx = torch.arange(total_len, device=device).unsqueeze(0)
            abs_row = row_idx + prefix_len  # absolute position of each row

            # Two conditions must both be True:
            # 1. Causal: key position <= query position
            # 2. Window: distance < local_window
            window_mask = (col_idx <= abs_row) & ((abs_row - col_idx) < local_window)
            mask[prefix_len:, :] = window_mask

    return mask


def build_causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """Build a simple lower-triangular causal mask.
    
    Returns:
        [seq_len, seq_len] bool mask where True = can attend.
    """
    # torch.tril creates a lower-triangular matrix (True on and below diagonal)
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def is_global_layer(layer_idx: int, global_layer_freq: int) -> bool:
    """Determine if a layer uses global (MLA) or local (GQA+SW) attention.
    
    A layer is global if its index (0-based) is the LAST in each group
    of size global_layer_freq.
    
    With global_layer_freq=6:
      Layer 0,1,2,3,4 → local  (5 % 6 != 5)
      Layer 5         → global (5 % 6 == 5)
      Layer 6,7,8,9,10→ local
      Layer 11        → global
    
    Args:
        layer_idx: Zero-based layer index.
        global_layer_freq: How often a global layer appears.
    
    Returns:
        True if this layer uses MLA (global), False if GQA+SW (local).
    
    Examples:
        >>> is_global_layer(5, 6)
        True
        >>> is_global_layer(0, 6)
        False
    """
    return (layer_idx % global_layer_freq) == (global_layer_freq - 1)
```

---

## 6. How the Mask Is Used in Attention

In the attention computation, the mask is applied to the scores:

```python
scores = Q @ K.T / sqrt(d_head)   # [batch, heads, seq, seq]

# Wherever mask is False, set score to -infinity
# After softmax, -infinity becomes 0 probability (completely ignored)
scores = scores.masked_fill(~mask, float("-inf"))

weights = torch.softmax(scores, dim=-1)  # Masked positions → weight = 0
output = weights @ V
```

---

## 7. The 1:6 Interleaving Ratio

With `global_layer_freq=6` and 12 layers:

```
Layer  0: LOCAL  (GQA + sliding window, window=512)
Layer  1: LOCAL
Layer  2: LOCAL
Layer  3: LOCAL
Layer  4: LOCAL
Layer  5: GLOBAL (MLA, full causal)   ← every 6th layer
Layer  6: LOCAL
Layer  7: LOCAL
Layer  8: LOCAL
Layer  9: LOCAL
Layer 10: LOCAL
Layer 11: GLOBAL (MLA, full causal)
```

**Why this ratio?** Global attention is expensive — it scales as $O(n^2)$ where $n$ is sequence length. Local attention only looks at the nearest `window` tokens, which is $O(n \times w)$. By doing 5 local layers for every 1 global layer, APEX-1 gets the benefits of global context (from MLA layers) at a fraction of the compute cost.

---

**Next:** [07 — Multi-Head Latent Attention (MLA) →](07-attention-mla.md)
