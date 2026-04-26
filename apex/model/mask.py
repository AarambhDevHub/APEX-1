"""
Attention Mask Builder for APEX-1.

Builds the combined attention mask used across the model:
- Prefix bidirectional attention (GLM-4 style) for system+user tokens
- Full causal attention for global MLA layers
- Sliding window causal attention for local GQA layers

The mask is a boolean tensor where True = can attend, False = masked.
"""

from __future__ import annotations

import torch


def build_apex_attention_mask(
    prefix_len: int,
    total_len: int,
    local_window: int,
    is_global_layer: bool,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bool,
) -> torch.Tensor:
    """Build the APEX-1 attention mask combining prefix-bidir + causal + window.

    The mask encodes three attention regimes:
    1. Prefix block (positions 0..prefix_len-1): full bidirectional attention
       among all prefix tokens. This lets the model read the entire system
       prompt before generating any output.
    2. Global layers: full causal attention over the entire sequence (each
       generation token can attend to all past tokens).
    3. Local layers: causal + sliding window (each generation token can
       attend only to the most recent ``local_window`` tokens).

    Args:
        prefix_len: Number of system prompt + user turn tokens that get
                    bidirectional attention.
        total_len: Total sequence length (prefix_len + generated tokens).
        local_window: Sliding window size for local layers.
        is_global_layer: If True, use full causal for generation tokens.
                        If False, use sliding window causal.
        device: Device to place the mask on.
        dtype: Data type (default: torch.bool).

    Returns:
        Boolean mask of shape ``[total_len, total_len]`` where True means
        the query position (row) can attend to the key position (column).
    """
    mask = torch.zeros(total_len, total_len, dtype=dtype, device=device)

    # Prefix block: bidirectional (GLM-4 style)
    # All prefix tokens attend to all other prefix tokens in both directions
    if prefix_len > 0:
        mask[:prefix_len, :prefix_len] = True

    if is_global_layer:
        # Global layer: full causal attention over entire sequence
        for i in range(prefix_len, total_len):
            mask[i, : i + 1] = True
    else:
        # Local layer: causal + sliding window
        for i in range(prefix_len, total_len):
            start = max(0, i - local_window + 1)
            mask[i, start: i + 1] = True

    return mask


def build_apex_attention_mask_batched(
    prefix_len: int,
    total_len: int,
    local_window: int,
    is_global_layer: bool,
    batch_size: int = 1,
    n_heads: int = 1,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build attention mask expanded for batched multi-head attention.

    Creates the base mask via ``build_apex_attention_mask`` and expands it
    to shape ``[batch_size, n_heads, total_len, total_len]`` for direct
    use with ``scaled_dot_product_attention``.

    Args:
        prefix_len: Number of prefix tokens with bidirectional attention.
        total_len: Total sequence length.
        local_window: Sliding window size for local layers.
        is_global_layer: Whether this is a global or local attention layer.
        batch_size: Batch size.
        n_heads: Number of attention heads.
        device: Device to place the mask on.

    Returns:
        Boolean mask of shape ``[batch, n_heads, total_len, total_len]``.
    """
    base_mask = build_apex_attention_mask(
        prefix_len, total_len, local_window, is_global_layer, device=device
    )
    # Expand: [total_len, total_len] -> [1, 1, total_len, total_len] -> [B, H, T, T]
    return base_mask.unsqueeze(0).unsqueeze(0).expand(batch_size, n_heads, -1, -1)


def build_causal_mask(
    seq_len: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a simple causal (lower-triangular) attention mask.

    Args:
        seq_len: Sequence length.
        device: Device to place the mask on.

    Returns:
        Boolean mask of shape ``[seq_len, seq_len]``.
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def is_global_layer(layer_idx: int, global_layer_freq: int) -> bool:
    """Determine if a given layer index is a global (MLA) or local (GQA+SW) layer.

    Global layers use Multi-Head Latent Attention with full causal attention.
    Local layers use Grouped Query Attention with sliding window.

    The rule: layer_idx % global_layer_freq == (global_layer_freq - 1)

    Args:
        layer_idx: Zero-based index of the transformer layer.
        global_layer_freq: Frequency of global layers (e.g., 6 means every 6th layer).

    Returns:
        True if the layer is a global MLA layer, False if local GQA+SW.

    Examples:
        >>> is_global_layer(5, 6)   # Layer 5 is global
        True
        >>> is_global_layer(0, 6)   # Layer 0 is local
        False
        >>> is_global_layer(11, 6)  # Layer 11 is global
        True
    """
    return (layer_idx % global_layer_freq) == (global_layer_freq - 1)
