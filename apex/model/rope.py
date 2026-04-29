"""
Rotary Positional Encoding (RoPE) with YaRN Extension.

Implements RoPE for encoding position information into Q and K vectors
via rotation, and YaRN for extending context length without retraining.

Fix BUG-22: ``apply_yarn_scaling`` is now fully vectorised using
``torch.where`` instead of a Python ``for`` loop over dimensions.
"""

from __future__ import annotations

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

    Args:
        d_head: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        rope_base: Base frequency for the rotation.
        device: Device to place tensors on.
        dtype: Data type for the computation.

    Returns:
        Tuple of (cos_cache, sin_cache) each of shape
        ``[max_seq_len, d_head]``.
    """
    i = torch.arange(0, d_head, 2, dtype=dtype, device=device)
    theta = 1.0 / (rope_base ** (i / d_head))

    positions = torch.arange(max_seq_len, dtype=dtype, device=device)
    angles = torch.outer(positions, theta)

    cos_cache = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin_cache = torch.sin(angles).repeat_interleave(2, dim=-1)

    return cos_cache, sin_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate alternate dimension pairs: [-x1, x0, -x3, x2, ...].

    Args:
        x: Input tensor of shape ``[..., d_head]`` where d_head is even.

    Returns:
        Rotated tensor of the same shape.
    """
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack([-x2, x1], dim=-1).flatten(-2)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE rotation to Query and Key tensors.

    Args:
        q: Query tensor ``[batch, n_heads, seq_len, d_head]``.
        k: Key tensor ``[batch, n_heads, seq_len, d_head]``.
        cos_cache: Precomputed cosine cache ``[max_seq_len, d_head]``.
        sin_cache: Precomputed sine cache ``[max_seq_len, d_head]``.
        positions: Position indices ``[seq_len]`` or ``[batch, seq_len]``.

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes as inputs.
    """
    if positions.dim() == 1:
        cos = cos_cache[positions].unsqueeze(0).unsqueeze(0)
        sin = sin_cache[positions].unsqueeze(0).unsqueeze(0)
    else:
        cos = cos_cache[positions].unsqueeze(2)
        sin = sin_cache[positions].unsqueeze(2)
        cos = cos.transpose(1, 2)
        sin = sin.transpose(1, 2)

    d = q.shape[-1]
    cos = cos[..., :d]
    sin = sin[..., :d]

    q_rot = q * cos + rotate_half(q) * sin
    k_rot = k * cos + rotate_half(k) * sin

    return q_rot, k_rot


def apply_yarn_scaling(
    theta: torch.Tensor,
    scale_factor: float,
    d_head: int,
    beta_fast: float = 32.0,
    beta_slow: float = 1.0,
) -> tuple[torch.Tensor, float]:
    """Apply YaRN frequency scaling for context length extension.

    BUG-22 FIX: The original implementation used a Python ``for`` loop
    over all dimensions, which is O(d_head) Python calls.  This version
    is fully vectorised with ``torch.where``, executing in a single fused
    kernel regardless of d_head.

    Args:
        theta: Original frequency vector of shape ``[d_head // 2]``.
        scale_factor: Target context / training context ratio.
        d_head: Head dimension size (unused directly but kept for API compat).
        beta_fast: High-frequency cutoff wavelength.
        beta_slow: Low-frequency cutoff multiplier.

    Returns:
        Tuple of (scaled_theta, attn_factor).
    """
    if scale_factor <= 1.0:
        return theta.clone(), 1.0

    # Wavelength for each dimension pair: λ_i = 2π / θ_i
    wavelength = 2.0 * math.pi / theta.clamp(min=1e-30)  # [d_head // 2]

    # --- BUG-22 FIX: vectorised three-regime scaling via torch.where ---
    # Regime 1 — high-frequency (short wavelength): no scaling
    # Regime 2 — low-frequency (long wavelength): full /= scale_factor
    # Regime 3 — mid-frequency: smooth interpolation
    t = (wavelength / beta_slow - 1.0) / (scale_factor - 1.0)
    mid_divisor = t * scale_factor + (1.0 - t)

    # Start with mid-frequency result
    scaled_theta = theta / mid_divisor

    # Override low-freq dimensions (full scaling) FIRST
    low_freq_mask = wavelength > beta_slow * scale_factor
    scaled_theta = torch.where(low_freq_mask, theta / scale_factor, scaled_theta)

    # Override high-freq dimensions (no scaling) LAST — highest priority,
    # matching the reference loop's if/elif where high-freq is checked first.
    high_freq_mask = wavelength < beta_fast
    scaled_theta = torch.where(high_freq_mask, theta, scaled_theta)

    # Temperature correction to prevent attention entropy collapse
    attn_factor = 0.1 * math.log(scale_factor) + 1.0

    return scaled_theta, attn_factor


def precompute_rope_cache_with_yarn(
    d_head: int,
    max_seq_len: int,
    rope_base: float = 10000.0,
    scale_factor: float = 1.0,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """Precompute RoPE cache with optional YaRN scaling.

    Args:
        d_head: Dimension of each attention head.
        max_seq_len: Maximum sequence length to precompute for.
        rope_base: Base frequency.
        scale_factor: YaRN scaling factor (1.0 = no scaling).
        device: Device to place tensors on.
        dtype: Data type for the computation.

    Returns:
        Tuple of (cos_cache, sin_cache, attn_factor).
    """
    i = torch.arange(0, d_head, 2, dtype=dtype, device=device)
    theta = 1.0 / (rope_base ** (i / d_head))

    attn_factor = 1.0
    if scale_factor > 1.0:
        theta, attn_factor = apply_yarn_scaling(theta, scale_factor, d_head)
        if device is not None:
            theta = theta.to(device)

    positions = torch.arange(max_seq_len, dtype=dtype, device=device)
    angles = torch.outer(positions, theta)

    cos_cache = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin_cache = torch.sin(angles).repeat_interleave(2, dim=-1)

    return cos_cache, sin_cache, attn_factor