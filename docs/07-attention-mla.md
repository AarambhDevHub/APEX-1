# 07 — Multi-Head Latent Attention (MLA)

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/model/attention.py` — class `MLAAttention`  
> **You will learn:** What attention is, how MLA compresses the KV cache by 93%, BUG-01 and BUG-02.

---

## 1. What Is Attention?

### The Library Book Analogy

You (the **Query** Q) search a library. Each book has a title card (**Key** K) and content (**Value** V). You score every book by comparing your question to its title, then blend the content weighted by relevance.

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$ — dot product → relevance score for every token pair
- $/ \sqrt{d_k}$ — scale down to stabilise softmax gradients
- $\text{softmax}(\cdot)$ — normalise scores to probabilities
- $\times V$ — weighted blend of value vectors

**Multi-head** means we run this process $h$ times in parallel, each in a different learned subspace, then concatenate results:

$$\text{MHA} = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) W_O$$

---

## 2. The KV Cache Problem

At generation time, the model produces one token per step but must attend to all previous tokens. Recomputing K and V for all past tokens every step costs $O(n^2)$. The **KV cache** stores past K and V tensors.

**Problem:** For a 128K-token sequence with Large config (128 heads, `d_head=128`, 72 layers):
$$\text{Cache size} = 72 \times 2 \times 128 \times 128 \times 128 \approx 600\text{ MB per sequence}$$

This is unacceptable for serving many users simultaneously.

---

## 3. MLA: Compress the KV Cache

Instead of caching full K and V, MLA caches a small **latent vector** $c_{KV}$:

$$c_{KV} = W_{DKV}\, x \in \mathbb{R}^{d_{kv\_compressed}}$$

K and V are reconstructed on demand:

$$K_{content} = W_{UK}\, c_{KV}, \qquad V = W_{UV}\, c_{KV}$$

For Large config: $d_{kv\_compressed} = 512$ vs. $n_{kv} \times d_{head} = 8 \times 128 = 1024$ → **50% savings per layer** (the 93% figure comes from comparing against the naïve full `n_heads_q` KV cache).

---

## 4. Decoupled RoPE

In MLA, K is reconstructed from the latent — you cannot apply position info to the latent directly. The solution: add a **separate small rope projection** alongside the content:

$$K = [K_{content},\; K_{rope}], \quad Q = [Q_{content},\; Q_{rope}]$$

where $K_{rope} = W_{KR} x$ with RoPE applied, and has dimension $d_{head\_rope}$.

---

## 5. BUG-01: K_rope Was Always Zeros

The original KV cache stored only $c_{KV}$. At each step, $K_{rope}$ for **past** tokens was reset to zeros — meaning all cached tokens had wrong position info.

**Fix:** The cache is now a tuple `(c_kv, K_rope_cache)`. Both are stored and concatenated at each step.

## 6. BUG-02: Wrong W_O Dimension

The output projection $W_O$ maps the merged attention output back to `d_model`. The original code used `n_heads_q * (d_head + d_head_rope)` as input size. But `attn_weights @ V` only has `d_head` dimensions (V has no rope component). So the correct size is `n_heads_q * d_head`.

---

## 7. Full Annotated Source: `MLAAttention`

```python
MLACache = tuple[torch.Tensor, torch.Tensor]
#   c_kv:        [batch, seq_so_far, d_kv_compressed]
#   K_rope_cache:[batch, n_heads_kv, seq_so_far, d_head_rope]


class MLAAttention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        m = config.model
        # KV compression: x → small latent
        self.W_DKV = nn.Linear(m.d_model, m.d_kv_compressed, bias=False)
        # KV decompression: latent → K and V
        self.W_UK = nn.Linear(m.d_kv_compressed, m.n_heads_kv * m.d_head, bias=False)
        self.W_UV = nn.Linear(m.d_kv_compressed, m.n_heads_kv * m.d_head, bias=False)
        # Q compression/decompression
        self.W_DQ = nn.Linear(m.d_model, m.d_q_compressed, bias=False)
        self.W_UQ = nn.Linear(m.d_q_compressed, m.n_heads_q * m.d_head, bias=False)
        # Decoupled RoPE projections (only for positional info)
        self.W_KR = nn.Linear(m.d_model, m.n_heads_kv * m.d_head_rope, bias=False)
        self.W_QR = nn.Linear(m.d_model, m.n_heads_q * m.d_head_rope, bias=False)
        # BUG-02 FIX: input = n_heads_q * d_head (NOT including rope dim)
        self.W_O = nn.Linear(m.n_heads_q * m.d_head, m.d_model, bias=False)

    def forward(self, x, cos_cache, sin_cache, positions, attn_mask=None, kv_cache=None):
        batch, seq_len, _ = x.shape

        # Step 1: compress x to KV latent
        c_kv_new = self.W_DKV(x)   # [B, S, d_kv_compressed]

        # Step 2: compute K_rope for NEW tokens and rotate
        # BUG-01 FIX: compute K_rope here and store it in cache
        K_rope_new = (self.W_KR(x)
            .view(batch, seq_len, self.n_heads_kv, self.d_head_rope)
            .transpose(1, 2))  # [B, n_kv, S, d_rope]
        cos_k = cos_cache[positions].unsqueeze(0).unsqueeze(0)[..., :self.d_head_rope]
        sin_k = sin_cache[positions].unsqueeze(0).unsqueeze(0)[..., :self.d_head_rope]
        K_rope_new = K_rope_new * cos_k + rotate_half(K_rope_new) * sin_k

        # Step 3: append to cache
        if kv_cache is not None:
            c_kv_prev, K_rope_prev = kv_cache  # unpack tuple
            c_kv_full = torch.cat([c_kv_prev, c_kv_new], dim=1)
            K_rope_full = torch.cat([K_rope_prev, K_rope_new], dim=2)
        else:
            c_kv_full = c_kv_new
            K_rope_full = K_rope_new

        new_kv_cache: MLACache = (c_kv_full, K_rope_full)
        full_seq = c_kv_full.shape[1]

        # Step 4: reconstruct K_content and V from latent
        K_content = (self.W_UK(c_kv_full)
            .view(batch, full_seq, self.n_heads_kv, self.d_head)
            .transpose(1, 2))   # [B, n_kv, full_seq, d_head]
        V = (self.W_UV(c_kv_full)
            .view(batch, full_seq, self.n_heads_kv, self.d_head)
            .transpose(1, 2))   # [B, n_kv, full_seq, d_head]

        # Step 5: compute Q_content and Q_rope
        Q_content = (self.W_UQ(self.W_DQ(x))
            .view(batch, seq_len, self.n_heads_q, self.d_head)
            .transpose(1, 2))   # [B, n_q, S, d_head]
        Q_rope = (self.W_QR(x)
            .view(batch, seq_len, self.n_heads_q, self.d_head_rope)
            .transpose(1, 2))
        cos_q = cos_cache[positions].unsqueeze(0).unsqueeze(0)[..., :self.d_head_rope]
        sin_q = sin_cache[positions].unsqueeze(0).unsqueeze(0)[..., :self.d_head_rope]
        Q_rope = Q_rope * cos_q + rotate_half(Q_rope) * sin_q

        # Step 6: concatenate content + rope
        Q = torch.cat([Q_content, Q_rope], dim=-1)   # [B, n_q, S, d_head+d_rope]
        K = torch.cat([K_content, K_rope_full], dim=-1)

        # Step 7: GQA head expansion (each KV head shared by G query heads)
        G = self.n_heads_q // self.n_heads_kv
        K = K.repeat_interleave(G, dim=1)
        V = V.repeat_interleave(G, dim=1)

        # Step 8: scaled dot-product attention
        d_total = self.d_head + self.d_head_rope
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_total)
        if attn_mask is not None:
            scores = scores.masked_fill(~attn_mask[:seq_len, :full_seq], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        attn_out = torch.matmul(weights, V)   # [B, n_q, S, d_head]

        # Step 9: merge heads → project
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        return self.W_O(attn_out), new_kv_cache  # [B, S, d_model]
```

---

## 8. Shape Summary

| Tensor | Shape |
|---|---|
| Input `x` | `[B, S, d_model]` |
| KV latent `c_kv` | `[B, S+prev, d_kv_compressed]` |
| K content | `[B, n_kv, S+prev, d_head]` |
| K rope | `[B, n_kv, S+prev, d_head_rope]` |
| Full K | `[B, n_q, S+prev, d_head+d_rope]` |
| Attn output | `[B, S, n_q * d_head]` |
| Output | `[B, S, d_model]` |

---

**Next:** [08 — GQA + Sliding Window →](08-attention-gqa.md)
