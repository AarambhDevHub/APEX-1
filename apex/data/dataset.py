"""
APEX-1 Dataset Classes.

Provides PyTorch datasets for:
1. PretrainDataset — packed sequences for next-token prediction
2. SFTDataset — instruction/response pairs with token type labels
3. PreferenceDataset — (prompt, chosen, rejected) triples for DPO/GRPO
4. StreamingPretrainDataset — memory-efficient streaming for large corpora

Fix BUG-24: ``StreamingPretrainDataset`` now emits an ``attention_mask``
alongside ``input_ids`` so the training loss can exclude padding tokens.
Previously, when the final buffer was shorter than ``seq_len``, padding
tokens were added but treated as real training data — polluting the loss
signal with meaningless pad-token predictions.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Optional

import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


class PretrainDataset(Dataset):
    """Dataset for pretraining on packed token sequences.

    Takes tokenized text and packs it into fixed-length sequences
    for efficient batch training.

    Args:
        token_ids: Flat tensor of all token IDs.
        seq_len: Length of each training sequence.
        stride: Stride between consecutive sequences (default: seq_len).
    """

    def __init__(
        self,
        token_ids: torch.Tensor,
        seq_len: int = 2048,
        stride: Optional[int] = None,
    ) -> None:
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = stride or seq_len

        n_tokens = len(token_ids)
        self.n_samples = max(1, (n_tokens - seq_len) // self.stride)

        logger.info(
            "PretrainDataset: %d tokens, seq_len=%d, %d samples",
            n_tokens,
            seq_len,
            self.n_samples,
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample.

        Args:
            idx: Sample index.

        Returns:
            Dict with 'input_ids' tensor of shape ``[seq_len]``.
        """
        start = idx * self.stride
        end = start + self.seq_len
        return {"input_ids": self.token_ids[start:end].clone()}


class SFTDataset(Dataset):
    """Dataset for supervised fine-tuning.

    Each sample contains token IDs and token type labels (0=system,
    1=user, 2=assistant) for computing SFT loss on assistant tokens only.

    Args:
        samples: List of dicts with 'input_ids' and 'token_types' keys.
        max_seq_len: Maximum sequence length (truncate if longer).
        pad_token_id: Padding token ID.
    """

    def __init__(
        self,
        samples: list[dict[str, list[int]]],
        max_seq_len: int = 2048,
        pad_token_id: int = 0,
    ) -> None:
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

        logger.info(
            "SFTDataset: %d samples, max_seq_len=%d",
            len(samples),
            max_seq_len,
        )

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a training sample with padding.

        Args:
            idx: Sample index.

        Returns:
            Dict with 'input_ids' and 'token_types' tensors.
        """
        sample = self.samples[idx]
        input_ids = sample["input_ids"][: self.max_seq_len]
        token_types = sample["token_types"][: self.max_seq_len]

        # Pad to max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            token_types = token_types + [0] * pad_len  # pad tokens are type 0

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_types": torch.tensor(token_types, dtype=torch.long),
        }

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        tokenizer: Any,
        max_seq_len: int = 2048,
    ) -> SFTDataset:
        """Load SFT dataset from a JSONL file.

        Each line should be a JSON object with a 'messages' key containing
        a list of {role, content} dicts.

        Args:
            path: Path to JSONL file.
            tokenizer: APEX1Tokenizer instance.
            max_seq_len: Maximum sequence length.

        Returns:
            SFTDataset instance.
        """
        samples: list[dict[str, list[int]]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                messages = data["messages"]

                input_ids = tokenizer.encode_chat(messages, add_generation_prompt=False)
                token_types = tokenizer.get_token_types(input_ids)

                samples.append(
                    {
                        "input_ids": input_ids,
                        "token_types": token_types,
                    }
                )

        return cls(samples, max_seq_len, tokenizer.pad_token_id)


class PreferenceDataset(Dataset):
    """Dataset for preference-based training (DPO/GRPO).

    Each sample contains a prompt and two responses (chosen, rejected).

    Args:
        samples: List of dicts with 'prompt_ids', 'chosen_ids', 'rejected_ids'.
        max_seq_len: Maximum sequence length.
        pad_token_id: Padding token ID.
    """

    def __init__(
        self,
        samples: list[dict[str, list[int]]],
        max_seq_len: int = 2048,
        pad_token_id: int = 0,
    ) -> None:
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        """Return number of samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a preference pair.

        Args:
            idx: Sample index.

        Returns:
            Dict with 'prompt_ids', 'chosen_ids', 'rejected_ids' tensors
            and 'prompt_len' integer.
        """
        sample = self.samples[idx]

        prompt = sample["prompt_ids"][: self.max_seq_len]
        chosen = sample["chosen_ids"][: self.max_seq_len]
        rejected = sample["rejected_ids"][: self.max_seq_len]

        return {
            "prompt_ids": torch.tensor(prompt, dtype=torch.long),
            "chosen_ids": torch.tensor(chosen, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected, dtype=torch.long),
            "prompt_len": len(sample["prompt_ids"]),
        }

    @classmethod
    def from_jsonl(
        cls,
        path: str | Path,
        tokenizer: Any,
        max_seq_len: int = 2048,
    ) -> PreferenceDataset:
        """Load preference dataset from JSONL.

        Each line: {"prompt": str, "chosen": str, "rejected": str}

        Args:
            path: Path to JSONL file.
            tokenizer: APEX1Tokenizer instance.
            max_seq_len: Maximum sequence length.

        Returns:
            PreferenceDataset instance.
        """
        samples: list[dict[str, list[int]]] = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())

                prompt_ids = tokenizer.encode(data["prompt"], add_special_tokens=False)
                chosen_ids = tokenizer.encode(
                    data["prompt"] + data["chosen"], add_special_tokens=False
                )
                rejected_ids = tokenizer.encode(
                    data["prompt"] + data["rejected"], add_special_tokens=False
                )

                samples.append(
                    {
                        "prompt_ids": prompt_ids,
                        "chosen_ids": chosen_ids,
                        "rejected_ids": rejected_ids,
                    }
                )

        return cls(samples, max_seq_len, tokenizer.pad_token_id)


class StreamingPretrainDataset(IterableDataset):
    """Memory-efficient streaming dataset for large pretraining corpora.

    Reads data file-by-file, tokenizes on the fly, and packs into
    fixed-length sequences without loading everything into memory.

    Args:
        file_paths: List of paths to text files.
        tokenizer: Tokenizer with encode() method.
        seq_len: Length of each sequence.
        shuffle_files: Whether to shuffle file order each epoch.
        seed: Random seed for shuffling.
    """

    def __init__(
        self,
        file_paths: list[str | Path],
        tokenizer: Any,
        seq_len: int = 2048,
        shuffle_files: bool = True,
        seed: int = 42,
    ) -> None:
        self.file_paths = [Path(p) for p in file_paths]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_files = shuffle_files
        self.seed = seed

    def __iter__(self):
        """Yield packed sequences by streaming through files.

        Yields:
            Dict with 'input_ids' tensor of shape ``[seq_len]``.
        """
        files = list(self.file_paths)
        if self.shuffle_files:
            rng = random.Random(self.seed)
            rng.shuffle(files)

        buffer: list[int] = []

        for file_path in files:
            if not file_path.exists():
                logger.warning("Skipping missing file: %s", file_path)
                continue

            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    tokens = self.tokenizer.encode(line.strip(), add_special_tokens=False)
                    buffer.extend(tokens)

                    while len(buffer) >= self.seq_len:
                        chunk = buffer[: self.seq_len]
                        yield {
                            "input_ids": torch.tensor(chunk, dtype=torch.long),
                            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                        }
                        buffer = buffer[self.seq_len :]

        # Yield remaining if enough tokens
        if len(buffer) >= self.seq_len // 2:
            # BUG-24 FIX: emit an attention_mask that marks real tokens
            # as 1 and padding tokens as 0, so the training loss can
            # exclude pad positions instead of training on them.
            real_len = len(buffer)
            pad_len = self.seq_len - real_len
            buffer.extend([self.tokenizer.pad_token_id] * pad_len)
            yield {
                "input_ids": torch.tensor(buffer[: self.seq_len], dtype=torch.long),
                "attention_mask": torch.cat(
                    [
                        torch.ones(real_len, dtype=torch.long),
                        torch.zeros(pad_len, dtype=torch.long),
                    ]
                ),
            }
