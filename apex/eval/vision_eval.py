"""Sanity checks for APEX-1 vision-language forward outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class VisionEvalReport:
    batch_size: int
    sequence_length: int
    vocab_size: int
    visual_token_count: int
    has_hidden_states: bool
    kv_cache_layers: int

    def as_dict(self) -> dict[str, int | bool]:
        return {
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length,
            "vocab_size": self.vocab_size,
            "visual_token_count": self.visual_token_count,
            "has_hidden_states": self.has_hidden_states,
            "kv_cache_layers": self.kv_cache_layers,
        }


def validate_vision_forward_output(
    output: dict[str, Any],
    expected_batch: int | None = None,
    expected_visual_tokens: int | None = None,
    expected_vocab_size: int | None = None,
) -> VisionEvalReport:
    """Validate a vision model output dictionary and return a compact report."""
    if "logits" not in output:
        raise KeyError("Vision output must contain 'logits'")
    logits = output["logits"]
    if not isinstance(logits, torch.Tensor):
        raise TypeError("output['logits'] must be a torch.Tensor")
    if logits.ndim != 3:
        raise ValueError(f"logits must be [B,S,V], got {tuple(logits.shape)}")

    batch_size, seq_len, vocab_size = logits.shape
    visual_count = int(output.get("visual_token_count", 0))
    kv_caches = output.get("kv_caches") or []

    if expected_batch is not None and batch_size != expected_batch:
        raise ValueError(f"Expected batch {expected_batch}, got {batch_size}")
    if expected_visual_tokens is not None and visual_count != expected_visual_tokens:
        raise ValueError(f"Expected {expected_visual_tokens} visual tokens, got {visual_count}")
    if expected_vocab_size is not None and vocab_size != expected_vocab_size:
        raise ValueError(f"Expected vocab size {expected_vocab_size}, got {vocab_size}")

    return VisionEvalReport(
        batch_size=batch_size,
        sequence_length=seq_len,
        vocab_size=vocab_size,
        visual_token_count=visual_count,
        has_hidden_states="hidden_states" in output,
        kv_cache_layers=len(kv_caches),
    )
