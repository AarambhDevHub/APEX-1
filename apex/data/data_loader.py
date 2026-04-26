"""
Data Loaders for APEX-1.

Factory functions for creating DataLoaders with proper shuffling,
batching, and prefetching for pretraining and SFT.
"""

from __future__ import annotations

from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset


def create_pretrain_loader(
    dataset: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
    drop_last: bool = True,
    prefetch_factor: int = 2,
) -> DataLoader:
    """Create a DataLoader for pretraining.

    Args:
        dataset: PretrainDataset or StreamingPretrainDataset.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Pin memory for faster GPU transfer.
        shuffle: Whether to shuffle (disabled for IterableDataset).
        drop_last: Drop last incomplete batch.
        prefetch_factor: Number of batches to prefetch per worker.

    Returns:
        Configured DataLoader.
    """
    is_iterable = isinstance(dataset, IterableDataset)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and not is_iterable,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=drop_last,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )


def create_sft_loader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 2,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for SFT training.

    Args:
        dataset: SFTDataset instance.
        batch_size: Batch size.
        num_workers: Number of workers.
        pin_memory: Pin memory for GPU.
        shuffle: Whether to shuffle.

    Returns:
        Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,
    )


def create_preference_loader(
    dataset: Dataset,
    batch_size: int = 8,
    num_workers: int = 2,
    shuffle: bool = True,
) -> DataLoader:
    """Create a DataLoader for preference/DPO training.

    Args:
        dataset: PreferenceDataset instance.
        batch_size: Batch size.
        num_workers: Number of workers.
        shuffle: Whether to shuffle.

    Returns:
        Configured DataLoader.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
