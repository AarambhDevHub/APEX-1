# 08 — Grouped Query Attention + Sliding Window (GQA+SW)

> **Difficulty:** ⭐⭐☆☆☆ Intermediate  
> **Source file:** `apex/model/attention.py` — class `GQASlidingWindowAttention`  
> **You will learn:** How GQA saves memory vs multi-head attention, what a sliding window does, and why local layers dominate the stack.

---

## 1. Standard Multi-Head Attention — Memory Cost

In standard MHA, every query head has its own K and V heads. For APEX-1 Large with 128 query heads:

- K matrix: `128 heads × d_head × seq_len`
- V matrix: `128 heads × d_head × seq_len`

This costs enormous memory, especially for long sequences.

---

## 2. Grouped Query Attention (GQA)

**GQA** (Ainslie et al., 2023, from Google; adopted by Llama 3, Mistral) reduces the number of K and V heads while keeping all Q heads.

**Grouping:** Every $G = n_{heads\_q} / n_{heads\_kv}$ query heads share one K/V head pair.

For APEX-1 Small: $G = 8/2 = 4$ — every 4 query heads share 1 KV head.

**Memory saving:** KV cache shrinks by factor $G$ compared to standard MHA.

### The "Shared Secretary" Analogy

Imagine 8 managers (query heads) and 2 secretaries (KV heads). Manager 1–4 share secretary A; managers 5–8 share secretary B. Each manager still asks their own questions, but the information they pull from is shared.

### GQA Math

After computing Q, K, V we "expand" K and V to match the number of Q heads using `repeat_interleave`:

$$K_{expanded}[\text{head } i] = K[\text{head } \lfloor i/G \rfloor]$$

In code: `K.repeat_interleave(G, dim=1)` — repeats each KV head G times.

---

## 3. Sliding Window Attention

For the local layers, we do not need to attend to tokens far in the past. The **sliding window** restricts each query to only the most recent `local_window` tokens:

$$\text{Attend to positions } p \text{ where: } \text{pos}_{query} - \text{pos}_{key} < \text{local\_window}$$

**Why?**
- Language has strong **local dependencies** — nearby words are most relevant for understanding syntax and immediate semantics.
- Restricting to a window makes attention cost $O(n \times w)$ instead of $O(n^2)$.
- Long-range global context is handled by the MLA layers (every 6th layer).

### Visual Example

With `local_window = 3`:
```
Query position 7 can see: positions 5, 6, 7  (last 3)
Query position 8 can see: positions 6, 7, 8
Query position 9 can see: positions 7, 8, 9
```

---

## 4. Full Annotated Source: `GQASlidingWindowAttention`

```python
class GQASlidingWindowAttention(nn.Module):
    """Local attention: GQA + sliding window."""

    def __init__(self, config) -> None:
        super().__init__()
        m = config.model

        self.n_heads_q = m.n_heads_q
        self.n_heads_kv = m.n_heads_kv    # << n_heads_q (GQA)
        self.d_head = m.d_head
        self.local_window = config.attention.local_window  # e.g., 512
        self.use_flash = config.attention.flash

        # Standard Q, K, V projections (no compression like MLA)
        self.W_Q = nn.Linear(m.d_model, m.n_heads_q * m.d_head, bias=False)
        self.W_K = nn.Linear(m.d_model, m.n_heads_kv * m.d_head, bias=False)
        self.W_V = nn.Linear(m.d_model, m.n_heads_kv * m.d_head, bias=False)
        # Output projection
        self.W_O = nn.Linear(m.n_heads_q * m.d_head, m.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,              # [B, S, d_model]
        cos_cache: torch.Tensor,      # [max_seq, d_head]
        sin_cache: torch.Tensor,
        positions: torch.Tensor,      # [S] — absolute positions
        attn_mask=None,
        kv_cache=None,
    ):
        batch, seq_len, _ = x.shape

        # ── Step 1: Project to Q, K, V ──────────────────────────────────
        Q = (self.W_Q(x)
             .view(batch, seq_len, self.n_heads_q, self.d_head)
             .transpose(1, 2))   # [B, n_q, S, d_head]
        K = (self.W_K(x)
             .view(batch, seq_len, self.n_heads_kv, self.d_head)
             .transpose(1, 2))   # [B, n_kv, S, d_head]
        V = (self.W_V(x)
             .view(batch, seq_len, self.n_heads_kv, self.d_head)
             .transpose(1, 2))

        # ── Step 2: Apply RoPE to Q and K ───────────────────────────────
        # Uses the d_head-wide cache (not d_head_rope like MLA)
        Q, K = apply_rope(Q, K, cos_cache, sin_cache, positions)

        # ── Step 3: Append to KV cache ───────────────────────────────────
        # kv_cache is (K_prev, V_prev) for this local layer
        if kv_cache is not None:
            K_prev, V_prev = kv_cache
            K = torch.cat([K_prev, K], dim=2)   # [B, n_kv, prev+S, d_head]
            V = torch.cat([V_prev, V], dim=2)

        # ── Step 4: Sliding window truncation ────────────────────────────
        # Keep only the last 'local_window' positions
        # This enforces the locality constraint
        if K.shape[2] > self.local_window:
            K = K[:, :, -self.local_window:, :]   # trim old tokens
            V = V[:, :, -self.local_window:, :]

        # Detach from computation graph (KV cache is not trained through)
        new_kv_cache = (K.detach(), V.detach())

        # ── Step 5: GQA head expansion ───────────────────────────────────
        # Repeat each KV head G = (n_q / n_kv) times to match Q heads
        G = self.n_heads_q // self.n_heads_kv
        K_exp = K.repeat_interleave(G, dim=1)   # [B, n_q, window, d_head]
        V_exp = V.repeat_interleave(G, dim=1)

        kv_len = K_exp.shape[2]   # = min(prev+S, local_window)

        # ── Step 6: Scaled dot-product attention ─────────────────────────
        if self.use_flash and x.is_cuda:
            # Flash Attention (memory-efficient kernel)
            float_mask = None
            if attn_mask is not None:
                m_slice = attn_mask[:seq_len, :kv_len]
                float_mask = (torch.zeros(seq_len, kv_len, device=x.device, dtype=x.dtype)
                              .masked_fill(~m_slice, float("-inf"))
                              .unsqueeze(0).unsqueeze(0))
            attn_out = F.scaled_dot_product_attention(
                Q, K_exp, V_exp, attn_mask=float_mask, dropout_p=0.0, is_causal=False
            )
        else:
            # Manual attention (CPU path)
            scores = torch.matmul(Q, K_exp.transpose(-2, -1)) / math.sqrt(self.d_head)
            if attn_mask is not None:
                mask_2d = attn_mask[:seq_len, :kv_len]
                scores = scores.masked_fill(~mask_2d.unsqueeze(0).unsqueeze(0), float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(weights, V_exp)   # [B, n_q, S, d_head]

        # ── Step 7: Merge heads and project ──────────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.W_O(attn_out)    # [B, S, d_model]

        return output, new_kv_cache
```

---

## 5. GQA vs MLA Comparison

| Feature | GQA + Sliding Window | MLA (Full Causal) |
|---|---|---|
| Used in | Local layers (5 of 6) | Global layers (1 of 6) |
| KV cache format | `(K, V)` tuple | `(c_kv, K_rope)` tuple |
| Attention scope | Last `local_window` tokens | All previous tokens |
| Memory cost | $O(w)$ per layer | $O(d_{compressed})$ per layer |
| Compute cost | $O(S \times w)$ | $O(S^2)$ (within layer) |
| Position encoding | Standard RoPE on d_head | Decoupled RoPE on d_head_rope |

---

## 6. Why 5 Local : 1 Global?

Pure global attention costs $O(n^2)$ — for 128K tokens that is 16 billion operations per layer. Pure local attention misses long-range context.

The 5:1 ratio gives:
- **Long-range context** from MLA layers — the model can relate distant tokens
- **Local processing efficiency** from GQA layers — cheap and fast
- **Overall cost** dominated by the cheap local layers

This design was inspired by Gemma 4's interleaved attention pattern.

---

**Next:** [09 — FFN & SwiGLU →](09-ffn-swiglu.md)
