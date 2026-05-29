"""Vision-to-language projector for APEX-1.

The projector maps image patch features into APEX language-model hidden size.
The default Perceiver-style resampler compresses many image patches into a
fixed number of visual tokens, which keeps long-context cost predictable.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class PerceiverResampler(nn.Module):
    """Compress variable patch features into a fixed visual token count."""

    def __init__(
        self,
        d_vision: int,
        d_model: int,
        n_visual_tokens: int,
        n_heads: int,
        n_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads for the resampler")

        self.n_visual_tokens = n_visual_tokens
        self.latents = nn.Parameter(torch.randn(n_visual_tokens, d_model) * 0.02)
        self.input_proj = nn.Linear(d_vision, d_model)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm_latents": nn.LayerNorm(d_model),
                        "norm_features": nn.LayerNorm(d_model),
                        "cross_attn": nn.MultiheadAttention(
                            embed_dim=d_model,
                            num_heads=n_heads,
                            dropout=dropout,
                            batch_first=True,
                        ),
                        "norm_mlp": nn.LayerNorm(d_model),
                        "mlp": nn.Sequential(
                            nn.Linear(d_model, d_model * 4),
                            nn.GELU(),
                            nn.Dropout(dropout),
                            nn.Linear(d_model * 4, d_model),
                            nn.Dropout(dropout),
                        ),
                    }
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        """Return visual tokens with shape ``[B, n_visual_tokens, d_model]``."""
        if image_features.ndim != 3:
            raise ValueError(
                f"image_features must be [B,N,d_vision], got {tuple(image_features.shape)}"
            )
        batch = image_features.shape[0]
        features = self.input_proj(image_features)
        latents = self.latents.unsqueeze(0).expand(batch, -1, -1)

        for layer in self.layers:
            q = layer["norm_latents"](latents)
            kv = layer["norm_features"](features)
            attn_out, _ = layer["cross_attn"](q, kv, kv, need_weights=False)
            latents = latents + attn_out
            latents = latents + layer["mlp"](layer["norm_mlp"](latents))

        return self.final_norm(latents)


class MLPProjector(nn.Module):
    """Simple MLP projector with optional sequence compression."""

    def __init__(
        self,
        d_vision: int,
        d_model: int,
        hidden_dim: int,
        n_visual_tokens: int,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        if n_layers < 1:
            raise ValueError("n_layers must be >= 1")
        layers: list[nn.Module] = []
        in_dim = d_vision
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden_dim), nn.GELU()])
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, d_model))
        self.net = nn.Sequential(*layers)
        self.n_visual_tokens = n_visual_tokens
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        x = self.net(image_features)
        if x.shape[1] != self.n_visual_tokens:
            # Adaptive pool across sequence length: [B, N, D] -> [B, T, D]
            x = x.transpose(1, 2)
            x = F.adaptive_avg_pool1d(x, self.n_visual_tokens)
            x = x.transpose(1, 2).contiguous()
        return self.final_norm(x)


class VisionToTextProjector(nn.Module):
    """Build the configured visual-token projector."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        v = config.vision
        m = config.model
        if v.projector_type == "perceiver":
            n_heads = max(1, min(v.n_heads, m.n_heads_q))
            while m.d_model % n_heads != 0 and n_heads > 1:
                n_heads -= 1
            self.projector = PerceiverResampler(
                d_vision=v.d_vision,
                d_model=m.d_model,
                n_visual_tokens=v.n_visual_tokens,
                n_heads=n_heads,
                n_layers=v.projector_layers,
                dropout=v.dropout,
            )
        elif v.projector_type == "mlp":
            self.projector = MLPProjector(
                d_vision=v.d_vision,
                d_model=m.d_model,
                hidden_dim=v.projector_hidden_dim,
                n_visual_tokens=v.n_visual_tokens,
                n_layers=v.projector_layers,
            )
        else:
            raise ValueError(f"Unknown projector_type: {v.projector_type}")

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        return self.projector(image_features)
