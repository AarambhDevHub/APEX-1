"""
Attention modules for APEX-1.

Implements two attention mechanisms:
1. Multi-Head Latent Attention (MLA) — used on global layers
2. Grouped Query Attention + Sliding Window (GQA+SW) — used on local layers

Fix BUG-01: The MLA KV cache is now a tuple ``(c_kv, K_rope_cache)``
instead of a bare ``c_kv`` tensor.  Previously, cached K_rope positions
were always filled with zeros, corrupting all autoregressive inference
steps after the first.  K_rope is now projected from the input, rotated,
cached alongside c_kv, and concatenated correctly on each step.

Fix BUG-02: ``W_O`` is initialised with ``n_heads_q * d_head`` input
features (not ``n_heads_q * (d_head + d_head_rope)``).  The rope component
lives in Q and K only; attention output is ``weights @ V`` which has head
dim ``d_head``, so after merging heads the projection input is correctly
``n_heads_q * d_head``.  A note is added to make this explicit.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from apex.model.rope import apply_rope, rotate_half

# Type alias for the MLA KV cache tuple:
#   (c_kv, K_rope_cache)
#   c_kv:       [batch, seq_so_far, d_kv_compressed]
#   K_rope_cache: [batch, n_heads_kv, seq_so_far, d_head_rope]
MLACache = tuple[torch.Tensor, torch.Tensor]


class MLAAttention(nn.Module):
    """Multi-Head Latent Attention (MLA) — DeepSeek-V3 style.

    Args:
        config: APEXConfig with model dimensions and attention settings.
    """

    def __init__(self, config) -> None:
        super().__init__()
        m = config.model

        self.n_heads_q = m.n_heads_q
        self.n_heads_kv = m.n_heads_kv
        self.d_head = m.d_head
        self.d_head_rope = m.d_head_rope
        self.d_model = m.d_model
        self.d_kv_compressed = m.d_kv_compressed
        self.d_q_compressed = m.d_q_compressed
        self.use_flash = config.attention.flash

        # KV compression / decompression
        self.W_DKV = nn.Linear(m.d_model, m.d_kv_compressed, bias=False)
        self.W_UK = nn.Linear(m.d_kv_compressed, m.n_heads_kv * m.d_head, bias=False)
        self.W_UV = nn.Linear(m.d_kv_compressed, m.n_heads_kv * m.d_head, bias=False)

        # Q compression / decompression
        self.W_DQ = nn.Linear(m.d_model, m.d_q_compressed, bias=False)
        self.W_UQ = nn.Linear(m.d_q_compressed, m.n_heads_q * m.d_head, bias=False)

        # Decoupled RoPE projections
        self.W_KR = nn.Linear(m.d_model, m.n_heads_kv * m.d_head_rope, bias=False)
        self.W_QR = nn.Linear(m.d_model, m.n_heads_q * m.d_head_rope, bias=False)

        # BUG-02 FIX: W_O input dim is n_heads_q * d_head, NOT
        # n_heads_q * (d_head + d_head_rope).  The rope component exists
        # only in Q and K (for score computation).  The attention output is
        # ``attn_weights @ V`` where V has d_head → merged dim is
        # n_heads_q * d_head.
        self.W_O = nn.Linear(m.n_heads_q * m.d_head, m.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[MLACache] = None,
    ) -> tuple[torch.Tensor, MLACache]:
        """Forward pass for Multi-Head Latent Attention.

        BUG-01 FIX: ``kv_cache`` is now a tuple ``(c_kv, K_rope_cache)``
        so that cached K_rope rotations are preserved across autoregressive
        steps instead of being re-initialised to zeros.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            cos_cache: RoPE cosine cache ``[max_seq, d_head_rope]``.
            sin_cache: RoPE sine cache ``[max_seq, d_head_rope]``.
            positions: Position indices ``[seq_len]``.
            attn_mask: Boolean attention mask (True = attend).
            kv_cache: Tuple ``(c_kv, K_rope_cache)`` from previous steps.

        Returns:
            Tuple of (output ``[batch, seq_len, d_model]``, new_kv_cache).
        """
        batch, seq_len, _ = x.shape

        # ── Step 1: Compress input to KV latent ──────────────────────────
        c_kv_new = self.W_DKV(x)  # [batch, seq, d_kv_compressed]

        # ── Step 2: Decoupled RoPE keys for NEW tokens ───────────────────
        # BUG-01 FIX: compute K_rope for the current input tokens and store
        # alongside c_kv.  Previous steps' K_rope is retrieved from cache.
        K_rope_new = (
            self.W_KR(x).view(batch, seq_len, self.n_heads_kv, self.d_head_rope).transpose(1, 2)
        )  # [batch, n_heads_kv, seq, d_head_rope]

        # Rotate K_rope for the current positions
        cos_k = cos_cache[positions].unsqueeze(0).unsqueeze(0)[..., : self.d_head_rope]
        sin_k = sin_cache[positions].unsqueeze(0).unsqueeze(0)[..., : self.d_head_rope]
        K_rope_new = K_rope_new * cos_k + rotate_half(K_rope_new) * sin_k

        # ── Step 3: Concatenate with cache ───────────────────────────────
        if kv_cache is not None:
            c_kv_prev, K_rope_prev = kv_cache
            c_kv_full = torch.cat([c_kv_prev, c_kv_new], dim=1)
            # BUG-01 FIX: use stored K_rope (correct rotations) instead of zeros
            K_rope_full = torch.cat([K_rope_prev, K_rope_new], dim=2)
        else:
            c_kv_full = c_kv_new
            K_rope_full = K_rope_new

        # Store for next step
        new_kv_cache: MLACache = (c_kv_full, K_rope_full)
        full_seq = c_kv_full.shape[1]

        # ── Step 4: Reconstruct K and V from latent ───────────────────────
        K_content = (
            self.W_UK(c_kv_full).view(batch, full_seq, self.n_heads_kv, self.d_head).transpose(1, 2)
        )  # [batch, n_heads_kv, full_seq, d_head]
        V = (
            self.W_UV(c_kv_full).view(batch, full_seq, self.n_heads_kv, self.d_head).transpose(1, 2)
        )  # [batch, n_heads_kv, full_seq, d_head]

        # ── Step 5: Reconstruct Q ────────────────────────────────────────
        c_q = self.W_DQ(x)
        Q_content = (
            self.W_UQ(c_q).view(batch, seq_len, self.n_heads_q, self.d_head).transpose(1, 2)
        )  # [batch, n_heads_q, seq, d_head]

        # ── Step 6: Decoupled RoPE queries for current tokens ────────────
        Q_rope = self.W_QR(x).view(batch, seq_len, self.n_heads_q, self.d_head_rope).transpose(1, 2)
        cos_q = cos_cache[positions].unsqueeze(0).unsqueeze(0)[..., : self.d_head_rope]
        sin_q = sin_cache[positions].unsqueeze(0).unsqueeze(0)[..., : self.d_head_rope]
        Q_rope = Q_rope * cos_q + rotate_half(Q_rope) * sin_q

        # ── Step 7: Concatenate content + rope components ────────────────
        Q = torch.cat([Q_content, Q_rope], dim=-1)  # [b, n_q, seq, d_head+d_rope]
        K = torch.cat([K_content, K_rope_full], dim=-1)  # [b, n_kv, full, d_head+d_rope]

        # ── Step 8: GQA-style head expansion ────────────────────────────
        G = self.n_heads_q // self.n_heads_kv
        K = K.repeat_interleave(G, dim=1)
        V = V.repeat_interleave(G, dim=1)

        # ── Step 9: Scaled dot-product attention ─────────────────────────
        d_total = self.d_head + self.d_head_rope

        if self.use_flash and x.is_cuda:
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    bool_mask = attn_mask[:seq_len, :full_seq]
                    float_mask = torch.zeros(
                        seq_len, full_seq, device=x.device, dtype=x.dtype
                    ).masked_fill(~bool_mask, float("-inf"))
                    float_mask = float_mask.unsqueeze(0).unsqueeze(0)
                else:
                    float_mask = torch.zeros_like(
                        attn_mask[..., :seq_len, :full_seq], dtype=x.dtype
                    ).masked_fill(~attn_mask[..., :seq_len, :full_seq], float("-inf"))
            else:
                float_mask = None

            attn_out = F.scaled_dot_product_attention(
                Q, K, V, attn_mask=float_mask, dropout_p=0.0, is_causal=False
            )
        else:
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_total)

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    m_slice = attn_mask[:seq_len, :full_seq]
                    scores = scores.masked_fill(~m_slice.unsqueeze(0).unsqueeze(0), float("-inf"))
                else:
                    scores = scores.masked_fill(~attn_mask[..., :seq_len, :full_seq], float("-inf"))

            weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(weights, V)  # [batch, n_heads_q, seq, d_head]

        # ── Step 10: Merge heads and project ─────────────────────────────
        # attn_out: [batch, n_heads_q, seq, d_head]  (V has d_head, not d_total)
        # After merge: [batch, seq, n_heads_q * d_head]
        # BUG-02: W_O is (n_heads_q * d_head → d_model) which is correct.
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.W_O(attn_out)

        return output, new_kv_cache


class GQASlidingWindowAttention(nn.Module):
    """Grouped Query Attention with Sliding Window.

    Args:
        config: APEXConfig with model dimensions and attention settings.
    """

    def __init__(self, config) -> None:
        super().__init__()
        m = config.model

        self.n_heads_q = m.n_heads_q
        self.n_heads_kv = m.n_heads_kv
        self.d_head = m.d_head
        self.d_model = m.d_model
        self.local_window = config.attention.local_window
        self.use_flash = config.attention.flash

        self.W_Q = nn.Linear(m.d_model, m.n_heads_q * m.d_head, bias=False)
        self.W_K = nn.Linear(m.d_model, m.n_heads_kv * m.d_head, bias=False)
        self.W_V = nn.Linear(m.d_model, m.n_heads_kv * m.d_head, bias=False)
        self.W_O = nn.Linear(m.n_heads_q * m.d_head, m.d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cos_cache: torch.Tensor,
        sin_cache: torch.Tensor,
        positions: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        kv_cache: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass for GQA + Sliding Window attention.

        Args:
            x: Input tensor ``[batch, seq_len, d_model]``.
            cos_cache: RoPE cosine cache.
            sin_cache: RoPE sine cache.
            positions: Position indices ``[seq_len]``.
            attn_mask: Attention mask (optional).
            kv_cache: Tuple of (K_cache, V_cache) from previous steps.

        Returns:
            Tuple of (output ``[batch, seq_len, d_model]``, new_kv_cache).
        """
        batch, seq_len, _ = x.shape

        Q = self.W_Q(x).view(batch, seq_len, self.n_heads_q, self.d_head).transpose(1, 2)
        K = self.W_K(x).view(batch, seq_len, self.n_heads_kv, self.d_head).transpose(1, 2)
        V = self.W_V(x).view(batch, seq_len, self.n_heads_kv, self.d_head).transpose(1, 2)

        Q, K = apply_rope(Q, K, cos_cache, sin_cache, positions)

        if kv_cache is not None:
            K_prev, V_prev = kv_cache
            K = torch.cat([K_prev, K], dim=2)
            V = torch.cat([V_prev, V], dim=2)

        if K.shape[2] > self.local_window:
            K = K[:, :, -self.local_window :, :]
            V = V[:, :, -self.local_window :, :]

        new_kv_cache = (K.detach(), V.detach())

        G = self.n_heads_q // self.n_heads_kv
        K_exp = K.repeat_interleave(G, dim=1)
        V_exp = V.repeat_interleave(G, dim=1)

        kv_len = K_exp.shape[2]

        if self.use_flash and x.is_cuda:
            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    m_slice = attn_mask[:seq_len, :kv_len]
                    float_mask = torch.zeros(
                        seq_len, kv_len, device=x.device, dtype=x.dtype
                    ).masked_fill(~m_slice, float("-inf"))
                    float_mask = float_mask.unsqueeze(0).unsqueeze(0)
                else:
                    float_mask = torch.zeros_like(
                        attn_mask[..., :seq_len, :kv_len], dtype=x.dtype
                    ).masked_fill(~attn_mask[..., :seq_len, :kv_len], float("-inf"))
            else:
                float_mask = None

            attn_out = F.scaled_dot_product_attention(
                Q, K_exp, V_exp, attn_mask=float_mask, dropout_p=0.0, is_causal=False
            )
        else:
            scores = torch.matmul(Q, K_exp.transpose(-2, -1)) / math.sqrt(self.d_head)

            if attn_mask is not None:
                if attn_mask.dim() == 2:
                    mask_2d = attn_mask[:seq_len, :kv_len]
                    scores = scores.masked_fill(~mask_2d.unsqueeze(0).unsqueeze(0), float("-inf"))
                else:
                    scores = scores.masked_fill(~attn_mask[..., :seq_len, :kv_len], float("-inf"))
            else:
                sw_mask = torch.zeros(seq_len, kv_len, dtype=torch.bool, device=x.device)
                for i in range(seq_len):
                    q_pos = positions[i].item() if positions.dim() == 1 else i
                    for j in range(kv_len):
                        k_pos = kv_len - seq_len + j if kv_cache is not None else j
                        if k_pos <= q_pos and q_pos - k_pos < self.local_window:
                            sw_mask[i, j] = True
                scores = scores.masked_fill(~sw_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            weights = torch.softmax(scores, dim=-1)
            attn_out = torch.matmul(weights, V_exp)

        attn_out = attn_out.transpose(1, 2).contiguous().view(batch, seq_len, -1)
        output = self.W_O(attn_out)

        return output, new_kv_cache
