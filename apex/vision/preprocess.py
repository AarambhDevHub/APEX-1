"""Image preprocessing for APEX-1 vision.

This file intentionally avoids torchvision so the core repo stays lightweight.
It supports PIL images, file paths, and torch tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class ImagePreprocessor:
    """Convert an image into normalized ``[C, H, W]`` or ``[B, C, H, W]`` tensors.

    Args:
        image_size: Square output size.
        mean: Channel mean in RGB order.
        std: Channel standard deviation in RGB order.
    """

    image_size: int = 224
    mean: Sequence[float] = (0.48145466, 0.4578275, 0.40821073)
    std: Sequence[float] = (0.26862954, 0.26130258, 0.27577711)

    def __call__(self, image: object) -> torch.Tensor:
        """Preprocess one image.

        Args:
            image: PIL image, path, or torch tensor.

        Returns:
            Tensor with shape ``[3, image_size, image_size]``.
        """
        tensor = self.to_tensor(image)
        if tensor.ndim != 3:
            raise ValueError(f"Expected one image with shape [C,H,W], got {tuple(tensor.shape)}")
        return self.normalize(self.resize(tensor))

    def batch(self, images: Sequence[object]) -> torch.Tensor:
        """Preprocess a batch of images into ``[B, 3, H, W]``."""
        if len(images) == 0:
            raise ValueError("images must not be empty")
        return torch.stack([self(image) for image in images], dim=0)

    def to_tensor(self, image: object) -> torch.Tensor:
        """Convert PIL/path/tensor input to float tensor in RGB format."""
        if isinstance(image, torch.Tensor):
            x = image.detach().clone().float()
            if x.ndim == 4:
                if x.shape[0] != 1:
                    raise ValueError("Use ImagePreprocessor.batch for multi-image tensors")
                x = x[0]
            if x.ndim != 3:
                raise ValueError(f"Expected tensor image rank 3 or 4, got rank {x.ndim}")
            # Accept HWC or CHW.
            if x.shape[0] not in {1, 3} and x.shape[-1] in {1, 3}:
                x = x.permute(2, 0, 1).contiguous()
            if x.shape[0] == 1:
                x = x.repeat(3, 1, 1)
            if x.shape[0] != 3:
                raise ValueError(f"Expected 1 or 3 channels, got shape {tuple(x.shape)}")
            if x.max() > 2.0:
                x = x / 255.0
            return x.clamp(0.0, 1.0)

        if isinstance(image, (str, Path)):
            try:
                from PIL import Image
            except ImportError as exc:
                raise ImportError("Pillow required for image file loading: pip install pillow") from exc
            image = Image.open(image).convert("RGB")

        # PIL image support without importing PIL at module import time.
        if hasattr(image, "convert") and hasattr(image, "size"):
            import numpy as np

            image = image.convert("RGB")
            arr = np.asarray(image, dtype="float32") / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1).contiguous()

        raise TypeError(f"Unsupported image type: {type(image)!r}")

    def resize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Resize CHW image tensor to square image_size."""
        x = tensor.unsqueeze(0)
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode="bicubic",
            align_corners=False,
        )
        return x.squeeze(0)

    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """Normalize image using CLIP-style mean/std defaults."""
        mean = torch.tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        std = torch.tensor(self.std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        return (tensor - mean) / std
