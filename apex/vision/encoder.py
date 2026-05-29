"""Native vision encoder for APEX-1.

This is a small ViT-style encoder used for the open-source/course version.
For production-quality visual understanding, later versions can add a frozen
CLIP/SigLIP/DINOv2 adapter while keeping the same output contract:

    pixel_values -> patch features [batch, n_patches, d_vision]
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


class PatchEmbedding(nn.Module):
    """Convert image pixels into patch tokens."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        in_channels: int,
        d_vision: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=d_vision,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, d_vision))
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Return patch tokens with shape ``[batch, n_patches, d_vision]``."""
        if pixel_values.ndim != 4:
            raise ValueError(f"pixel_values must be [B,C,H,W], got {tuple(pixel_values.shape)}")
        if pixel_values.shape[-2:] != (self.image_size, self.image_size):
            raise ValueError(
                f"Expected image size {(self.image_size, self.image_size)}, "
                f"got {tuple(pixel_values.shape[-2:])}"
            )
        x = self.proj(pixel_values)
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = x + self.pos_embed.to(dtype=x.dtype, device=x.device)
        return self.dropout(x)


class VisionTransformerBlock(nn.Module):
    """A small pre-norm Transformer encoder block for image patch features."""

    def __init__(self, d_vision: int, n_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_vision)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_vision,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_vision)
        hidden = int(d_vision * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_vision, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_vision),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class NativeVisionEncoder(nn.Module):
    """Pure PyTorch ViT encoder for APEX-1 vision preview."""

    def __init__(self, config: Any) -> None:
        super().__init__()
        v = config.vision
        self.config = config
        self.patch_embed = PatchEmbedding(
            image_size=v.image_size,
            patch_size=v.patch_size,
            in_channels=v.in_channels,
            d_vision=v.d_vision,
            dropout=v.dropout,
        )
        self.blocks = nn.ModuleList(
            [
                VisionTransformerBlock(
                    d_vision=v.d_vision,
                    n_heads=v.n_heads,
                    mlp_ratio=v.mlp_ratio,
                    dropout=v.dropout,
                )
                for _ in range(v.n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(v.d_vision)

        if v.freeze_vision_encoder:
            self.freeze()

    @property
    def n_patches(self) -> int:
        return self.patch_embed.n_patches

    def freeze(self) -> None:
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Encode images into patch features."""
        x = self.patch_embed(pixel_values)
        for block in self.blocks:
            x = block(x)
        return self.final_norm(x)
