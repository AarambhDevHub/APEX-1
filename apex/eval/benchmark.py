"""Tiny forward-pass benchmarking helpers for APEX-1."""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class BenchmarkResult:
    device: str
    batch_size: int
    seq_len: int
    repeats: int
    mean_ms: float
    min_ms: float
    max_ms: float
    tokens_per_second: float
    logits_shape: tuple[int, ...]
    cuda_memory_mb: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "seq_len": self.seq_len,
            "repeats": self.repeats,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "tokens_per_second": self.tokens_per_second,
            "logits_shape": self.logits_shape,
            "cuda_memory_mb": self.cuda_memory_mb,
        }

    def to_markdown(self) -> str:
        memory = "n/a" if self.cuda_memory_mb is None else f"{self.cuda_memory_mb:.2f} MB"
        return "\n".join(
            [
                "| Metric | Value |",
                "|---|---:|",
                f"| Device | {self.device} |",
                f"| Batch size | {self.batch_size} |",
                f"| Sequence length | {self.seq_len} |",
                f"| Repeats | {self.repeats} |",
                f"| Mean forward time | {self.mean_ms:.3f} ms |",
                f"| Min forward time | {self.min_ms:.3f} ms |",
                f"| Max forward time | {self.max_ms:.3f} ms |",
                f"| Tokens / second | {self.tokens_per_second:.2f} |",
                f"| Logits shape | {self.logits_shape} |",
                f"| CUDA memory | {memory} |",
            ]
        )


def _sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def run_forward_benchmark(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    pixel_values: torch.Tensor | None = None,
    warmup: int = 1,
    repeats: int = 5,
    device: torch.device | str | None = None,
) -> BenchmarkResult:
    """Benchmark model forward pass on a small batch.

    This intentionally benchmarks the full Python model path, not only kernels.
    It is designed for course-level comparison between configs and features.
    """
    if repeats <= 0:
        raise ValueError("repeats must be positive")
    if warmup < 0:
        raise ValueError("warmup cannot be negative")
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must be [B,S], got {tuple(input_ids.shape)}")

    was_training = model.training
    model.eval()

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    model.to(device)
    input_ids = input_ids.to(device)
    if pixel_values is not None:
        pixel_values = pixel_values.to(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    kwargs = {"pixel_values": pixel_values} if pixel_values is not None else {}

    output: dict[str, Any] = {}
    for _ in range(warmup):
        output = model(input_ids, **kwargs)
    _sync_if_needed(device)

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        output = model(input_ids, **kwargs)
        _sync_if_needed(device)
        timings.append((time.perf_counter() - start) * 1000.0)

    if was_training:
        model.train()

    mean_ms = statistics.mean(timings)
    tokens = int(input_ids.numel())
    tokens_per_second = tokens / (mean_ms / 1000.0) if mean_ms > 0 else float("inf")
    cuda_memory = None
    if device.type == "cuda":
        cuda_memory = torch.cuda.max_memory_allocated(device) / (1024**2)

    return BenchmarkResult(
        device=str(device),
        batch_size=int(input_ids.shape[0]),
        seq_len=int(input_ids.shape[1]),
        repeats=repeats,
        mean_ms=mean_ms,
        min_ms=min(timings),
        max_ms=max(timings),
        tokens_per_second=tokens_per_second,
        logits_shape=tuple(output["logits"].shape),
        cuda_memory_mb=cuda_memory,
    )
