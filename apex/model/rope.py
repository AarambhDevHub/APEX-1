"""
Rotary Positional Encoding (RoPE) with YaRN Extension.

Implements RoPE for encoding position information into Q and K vectors
via rotation, and YaRN for extending context length without retraining.

RoPE rotates pairs of dimensions by position-dependent angles:
    x₂ᵢ'   =  x₂ᵢ · cos(m·θᵢ) - x₂ᵢ₊₁ · sin(m·θᵢ)
    x₂ᵢ₊₁' =  x₂ᵢ · sin(m·θᵢ) + x₂ᵢ₊₁ · cos(m·θᵢ)

YaRN selectively scales frequencies so low-frequency (long-range)
dimensions get compressed while high-frequency (local) dimensions
are left unchanged. This enables models trained at short context
to run at much longer context without quality collapse.

Used by: KIMI (1M ctx), DeepSeek-V3 (128k), Qwen3 (128k extended)
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

    These tables are computed once and reused every forward pass,
    avoiding redundant computation.

    Args:
        d_head: Dimension of each attention head (must be even).
        max_seq_len: Maximum sequence length to precompute for.
        rope_base: Base frequency for the rotation (10000 standard,
                   500k-1M for long context models).
        device: Device to place tensors on.
        dtype: Data type for the computation.

    Returns:
        Tuple of (cos_cache, sin_cache) each of shape
        ``[max_seq_len, d_head]``.
    """
    # Frequencies: shape [d_head // 2]
    i = torch.arange(0, d_head, 2, dtype=dtype, device=device)
    theta = 1.0 / (rope_base ** (i / d_head))

    # Positions: shape [max_seq_len]
    positions = torch.arange(max_seq_len, dtype=dtype, device=device)

    # Outer product: shape [max_seq_len, d_head // 2]
    angles = torch.outer(positions, theta)

    # Repeat-interleave to match full d_head: shape [max_seq_len, d_head]
    cos_cache = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin_cache = torch.sin(angles).repeat_interleave(2, dim=-1)

    return cos_cache, sin_cache


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate alternate dimension pairs: [-x1, x0, -x3, x2, ...].

    This implements the rotation needed for RoPE by negating and
    swapping adjacent pairs of dimensions.

    Args:
        x: Input tensor of shape ``[..., d_head]`` where d_head is even.

    Returns:
        Rotated tensor of the same shape.
    """
    x1 = x[..., ::2]  # even indices
    x2 = x[..., 1::2]  # odd indices
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
        q: Query tensor of shape ``[batch, n_heads, seq_len, d_head]``.
        k: Key tensor of shape ``[batch, n_heads, seq_len, d_head]``.
        cos_cache: Precomputed cosine cache ``[max_seq_len, d_head]``.
        sin_cache: Precomputed sine cache ``[max_seq_len, d_head]``.
        positions: Position indices ``[seq_len]`` or ``[batch, seq_len]``.

    Returns:
        Tuple of (q_rotated, k_rotated) with same shapes as inputs.
    """
    # Select cos/sin for the given positions
    if positions.dim() == 1:
        cos = cos_cache[positions].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, d_head]
        sin = sin_cache[positions].unsqueeze(0).unsqueeze(0)  # [1, 1, seq, d_head]
    else:
        # positions: [batch, seq_len]
        cos = cos_cache[positions].unsqueeze(2)  # [batch, seq, 1, d_head]
        sin = sin_cache[positions].unsqueeze(2)
        cos = cos.transpose(1, 2)  # [batch, 1, seq, d_head]
        sin = sin.transpose(1, 2)

    # Handle case where q/k d_head might differ from cos/sin d_head
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

    YaRN selectively scales frequencies:
    - High-frequency dimensions (short wavelength < beta_fast): unchanged
    - Low-frequency dimensions (long wavelength > beta_slow * scale): full scaling
    - Middle dimensions: smooth linear interpolation

    Args:
        theta: Original frequency vector of shape ``[d_head // 2]``.
        scale_factor: Target context / training context ratio.
        d_head: Head dimension size.
        beta_fast: High-frequency cutoff (dimensions below this: no scaling).
        beta_slow: Low-frequency cutoff (dimensions above this × scale: full scaling).

    Returns:
        Tuple of (scaled_theta, attn_factor) where attn_factor is a
        temperature correction to prevent attention entropy collapse.
    """
    if scale_factor <= 1.0:
        return theta.clone(), 1.0

    scaled_theta = theta.clone()

    for i in range(len(theta)):
        # Convert dimension index to wavelength
        wavelength = 2.0 * math.pi / theta[i].item()

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
            t = (wavelength / beta_slow - 1.0) / (scale_factor - 1.0)
            scaled_theta[i] = theta[i] / (t * scale_factor + (1.0 - t))

    # Temperature correction: prevents attention entropy collapse at long context
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

    Combines rope precomputation with YaRN frequency scaling for
    models that need to operate at extended context lengths.

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
    # Base frequencies
    i = torch.arange(0, d_head, 2, dtype=dtype, device=device)
    theta = 1.0 / (rope_base ** (i / d_head))

    # Apply YaRN scaling if needed
    attn_factor = 1.0
    if scale_factor > 1.0:
        theta, attn_factor = apply_yarn_scaling(theta, scale_factor, d_head)
        if device is not None:
            theta = theta.to(device)

    # Positions
    positions = torch.arange(max_seq_len, dtype=dtype, device=device)

    # Outer product
    angles = torch.outer(positions, theta)

    # Repeat-interleave to full d_head
    cos_cache = torch.cos(angles).repeat_interleave(2, dim=-1)
    sin_cache = torch.sin(angles).repeat_interleave(2, dim=-1)

    return cos_cache, sin_cache, attn_factor
