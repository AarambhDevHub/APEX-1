"""APEX-1 Vision-Language Model.

APEX1VisionModel keeps the existing APEX1Model unchanged and adds a vision
front-end that converts images into visual tokens. Visual tokens are inserted
into the text embedding stream at the ``<|img|>`` placeholder, then processed by
the same decoder-only APEX transformer blocks.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import torch
import torch.nn as nn

from apex.config import APEXConfig
from apex.model.apex_model import APEX1Model
from apex.model.mask import build_apex_attention_mask, is_global_layer
from apex.vision.encoder import NativeVisionEncoder
from apex.vision.projector import VisionToTextProjector

logger = logging.getLogger(__name__)


class APEX1VisionModel(nn.Module):
    """APEX-1 with image understanding through visual tokens.

    Args:
        config: APEXConfig with ``config.vision.enabled=True``.

    Inputs:
        token_ids: Text token IDs. Use one ``<|img|>`` token where visual tokens
            should be inserted.
        pixel_values: Optional image tensor ``[B, 3, H, W]``.
        image_features: Optional precomputed patch features ``[B, N, d_vision]``.
        image_embeds: Optional precomputed visual tokens ``[B, T, d_model]``.

    The model returns logits over the original language vocabulary. Visual token
    positions are context only; training labels for those positions should be
    set to ``ignore_index=-100``.
    """

    def __init__(self, config: APEXConfig) -> None:
        super().__init__()
        if not config.vision.enabled:
            logger.warning(
                "APEX1VisionModel created with vision.enabled=False. "
                "Text-only forward still works, but image inputs are disabled."
            )
        config.validate()
        self.config = config
        self.language_model = APEX1Model(config)
        self.vision_encoder = NativeVisionEncoder(config)
        self.vision_projector = VisionToTextProjector(config)

        if config.vision.freeze_language_model:
            self.freeze_language_model()

    @property
    def image_token_id(self) -> int:
        return self.config.vision.image_token_id

    def freeze_language_model(self) -> None:
        """Freeze the APEX text backbone for projector-only alignment."""
        for param in self.language_model.parameters():
            param.requires_grad = False

    def encode_images(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Return visual tokens ``[B, n_visual_tokens, d_model]`` or None."""
        provided = [x is not None for x in (pixel_values, image_features, image_embeds)]
        if sum(provided) == 0:
            return None
        if sum(provided) > 1:
            raise ValueError("Provide only one of pixel_values, image_features, or image_embeds")
        if image_embeds is not None:
            if image_embeds.ndim != 3:
                raise ValueError("image_embeds must have shape [B,T,d_model]")
            if image_embeds.shape[-1] != self.config.model.d_model:
                raise ValueError(
                    f"image_embeds last dim must be d_model={self.config.model.d_model}, "
                    f"got {image_embeds.shape[-1]}"
                )
            return image_embeds
        if image_features is None:
            if pixel_values is None:
                raise AssertionError("unreachable")
            image_features = self.vision_encoder(pixel_values)
        return self.vision_projector(image_features)

    def _insert_visual_tokens(
        self,
        token_ids: torch.Tensor,
        text_embeds: torch.Tensor,
        visual_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """Replace the first ``<|img|>`` token with visual tokens.

        If ``<|img|>`` is missing, visual tokens are inserted after the first
        token, which is usually BOS. All batch items must produce the same final
        sequence length because the current APEX mask builder is dense and does
        not carry a separate padding mask.
        """
        batch, _, d_model = text_embeds.shape
        if visual_tokens.shape[0] != batch:
            raise ValueError(
                f"visual batch ({visual_tokens.shape[0]}) must match token batch ({batch})"
            )
        if visual_tokens.shape[-1] != d_model:
            raise ValueError(
                f"visual token dim ({visual_tokens.shape[-1]}) must equal d_model ({d_model})"
            )

        fused_rows: list[torch.Tensor] = []
        expected_len: Optional[int] = None
        for b in range(batch):
            matches = (token_ids[b] == self.image_token_id).nonzero(as_tuple=False).flatten()
            if matches.numel() > 0:
                idx = int(matches[0].item())
                row = torch.cat(
                    [text_embeds[b, :idx], visual_tokens[b], text_embeds[b, idx + 1 :]],
                    dim=0,
                )
            else:
                # No placeholder: insert after BOS/first token.
                idx = 1 if text_embeds.shape[1] > 0 else 0
                row = torch.cat([text_embeds[b, :idx], visual_tokens[b], text_embeds[b, idx:]], dim=0)

            if expected_len is None:
                expected_len = row.shape[0]
            elif row.shape[0] != expected_len:
                raise ValueError(
                    "All batch items must have the same number of <|img|> placeholders "
                    "for the current dense attention mask implementation"
                )
            fused_rows.append(row)

        return torch.stack(fused_rows, dim=0)

    def forward(
        self,
        token_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_features: Optional[torch.Tensor] = None,
        image_embeds: Optional[torch.Tensor] = None,
        positions: Optional[torch.Tensor] = None,
        prefix_len: int = 0,
        kv_caches: Optional[list[Any]] = None,
        return_hidden: bool = False,
    ) -> dict[str, Any]:
        """Forward pass for image+text or text-only input."""
        if token_ids.ndim != 2:
            raise ValueError(f"token_ids must be [B,S], got {tuple(token_ids.shape)}")

        visual_tokens = self.encode_images(pixel_values, image_features, image_embeds)
        if visual_tokens is not None and kv_caches is not None:
            raise ValueError(
                "Pass image inputs only on the first multimodal prefill call. "
                "For subsequent generation steps, reuse kv_caches and pass only new text tokens."
            )

        lm = self.language_model
        text_embeds = lm.embedding(token_ids) * lm.embed_scale
        if visual_tokens is not None:
            x = self._insert_visual_tokens(token_ids, text_embeds, visual_tokens)
            visual_token_count = visual_tokens.shape[1]
        else:
            x = text_embeds
            visual_token_count = 0

        batch, seq_len, _ = x.shape
        device = x.device

        if positions is None:
            if kv_caches is not None and kv_caches[0] is not None:
                layer_0_is_global = is_global_layer(0, self.config.attention.global_layer_freq)
                cache_0 = kv_caches[0]
                if layer_0_is_global:
                    prev_len = cache_0[0].shape[1]
                else:
                    prev_len = cache_0[0].shape[2]
                positions = torch.arange(prev_len, prev_len + seq_len, device=device)
            else:
                positions = torch.arange(seq_len, device=device)

        new_kv_caches: list[Any] = []
        for i, block in enumerate(lm.blocks):
            layer_kv = kv_caches[i] if kv_caches is not None else None
            layer_is_global = is_global_layer(i, self.config.attention.global_layer_freq)
            attn_mask = build_apex_attention_mask(
                prefix_len=prefix_len if kv_caches is None else 0,
                total_len=seq_len,
                local_window=self.config.attention.local_window,
                is_global_layer=layer_is_global,
                device=device,
            )
            if layer_is_global:
                cos = lm.cos_cache_rope
                sin = lm.sin_cache_rope
            else:
                cos = lm.cos_cache
                sin = lm.sin_cache
            x, new_kv = block(x, cos, sin, positions, attn_mask, layer_kv)
            new_kv_caches.append(new_kv)

        x = lm.final_norm(x)
        logits = torch.matmul(x, lm.embedding.weight.T)
        spec_logits = lm.multi_token_head(x) if lm.multi_token_head is not None else None

        result: dict[str, Any] = {
            "logits": logits,
            "spec_logits": spec_logits,
            "kv_caches": new_kv_caches,
            "visual_token_count": visual_token_count,
        }
        if return_hidden:
            result["hidden_states"] = x
        return result

    def total_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
