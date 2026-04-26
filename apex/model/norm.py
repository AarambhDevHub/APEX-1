"""
RMSNorm — Root Mean Square Layer Normalization.

Implements the RMSNorm normalization layer used throughout APEX-1.
RMSNorm normalizes using the root mean square of activations without
mean-centering, making it 20-40% faster than standard LayerNorm while
delivering equal or better quality.

Formula:
    RMSNorm(x) = x / RMS(x) * γ
    where RMS(x) = sqrt(mean(x²) + ε)
    γ is a learned per-dimension scale parameter initialized to 1.0

Used by: Llama 3, DeepSeek-V3, Qwen3, Gemma 4, Claude
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Normalizes the input tensor by its root mean square value and applies
    a learned scale parameter. No mean-centering is performed (unlike LayerNorm).

    Args:
        d_model: The dimension of the input features.
        eps: Small constant for numerical stability. Default: 1e-6.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMSNorm to the input tensor.

        Args:
            x: Input tensor of shape ``[batch, seq_len, d_model]``.

        Returns:
            Normalized tensor of the same shape as input.
        """
        # x: [batch, seq_len, d_model]
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return (x / rms) * self.weight

    def extra_repr(self) -> str:
        """Return a string representation of extra module info."""
        return f"d_model={self.d_model}, eps={self.eps}"
