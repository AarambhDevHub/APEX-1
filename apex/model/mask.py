"""
Attention Mask Builder for APEX-1.

Builds the combined attention mask used across the model:
- Prefix bidirectional attention (GLM-4 style) for system+user tokens
- Full causal attention for global MLA layers
- Sliding window causal attention for local GQA layers

The mask is a boolean tensor where True = can attend, False = masked.

Fix BUG-10: The sliding-window loop is now fully vectorised with
``torch.arange`` broadcasting instead of a Python ``for`` loop.
For a 128 K sequence that loop executed 128 000 Python iterations per
local layer per forward pass, which dominated training wall-clock time.
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
       among all prefix tokens.
    2. Global layers: full causal attention over the entire sequence.
    3. Local layers: causal + sliding window.

    BUG-10 FIX: The generation-token rows of the local-layer mask are now
    filled with a single vectorised operation instead of a Python ``for``
    loop.  This is O(1) Python calls regardless of sequence length.

    Args:
        prefix_len: Number of system prompt + user turn tokens.
        total_len: Total sequence length (prefix_len + generated tokens).
        local_window: Sliding window size for local layers.
        is_global_layer: Full causal (True) or windowed causal (False).
        device: Device to place the mask on.
        dtype: Data type (default: torch.bool).

    Returns:
        Boolean mask ``[total_len, total_len]``.
    """
    mask = torch.zeros(total_len, total_len, dtype=dtype, device=device)

    # Prefix block: bidirectional (GLM-4 style)
    if prefix_len > 0:
        mask[:prefix_len, :prefix_len] = True

    if is_global_layer:
        # BUG-10 FIX (global path): vectorised lower-triangular fill
        # Equivalent to the original per-row loop but a single op.
        if prefix_len < total_len:
            # BUG-10 FIX (global path): vectorised lower-triangular fill.
            # Row (prefix_len + k) can attend to cols 0..(prefix_len + k).
            gen_len = total_len - prefix_len
            row_idx = torch.arange(gen_len, device=device).unsqueeze(1)  # [gen, 1]
            col_idx = torch.arange(total_len, device=device).unsqueeze(0)  # [1, total]
            gen_causal = col_idx <= (row_idx + prefix_len)  # [gen, total]
            mask[prefix_len:, :] = gen_causal
    else:
        # BUG-10 FIX (local / sliding-window path): fully vectorised.
        # Original code: Python loop over range(prefix_len, total_len) —
        # 128 000 iterations for a 128 K sequence.
        # New code: two arange broadcasts, one boolean op, one masked_fill.
        gen_len = total_len - prefix_len
        if gen_len > 0:
            row_idx = torch.arange(gen_len, device=device).unsqueeze(1)  # [gen, 1]
            col_idx = torch.arange(total_len, device=device).unsqueeze(0)  # [1, total]
            abs_row = row_idx + prefix_len  # absolute position of each generated row
            # causal: col <= abs_row
            # window: abs_row - col < local_window
            window_mask = (col_idx <= abs_row) & ((abs_row - col_idx) < local_window)
            mask[prefix_len:, :] = window_mask  # [gen, total]

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

    Args:
        layer_idx: Zero-based index of the transformer layer.
        global_layer_freq: Frequency of global layers.

    Returns:
        True if the layer is a global MLA layer, False if local GQA+SW.

    Examples:
        >>> is_global_layer(5, 6)
        True
        >>> is_global_layer(0, 6)
        False
    """
    return (layer_idx % global_layer_freq) == (global_layer_freq - 1)