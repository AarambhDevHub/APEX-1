# 20 — Datasets: Feeding the Model

> **Difficulty:** ⭐⭐☆☆☆ Beginner-Intermediate  
> **Source file:** `apex/data/dataset.py`, `apex/data/data_loader.py`  
> **You will learn:** The four dataset types, how packing works, the BUG-24 padding mask fix, and DataLoader settings.

---

## 1. What Is a PyTorch Dataset?

A PyTorch Dataset is any Python class with two methods:
- `__len__()` — returns how many samples exist
- `__getitem__(idx)` — returns sample at index `idx`

The `DataLoader` wraps a Dataset and provides batches automatically, with shuffling, parallel loading, and memory pinning.

---

## 2. PretrainDataset — Packing for Efficiency

During pretraining, we tokenise entire books, articles, and web pages into one long flat sequence of token IDs, then **pack** them into fixed-length chunks.

**Why packing?**
- Avoids padding (waste) — every position in every batch is a real token
- Maximum compute utilisation
- Simple: no variable-length collation needed

```
Flat token stream: [t₀, t₁, t₂, ..., t₁₀₀₀₀₀₀]
                    ↓ chunk into seq_len=2048
Sample 0: [t₀   ... t₂₀₄₇]
Sample 1: [t₂₀₄₈... t₄₀₉₅]
Sample 2: [t₄₀₉₆... t₆₁₄₃]
...
```

---

## 3. SFTDataset — Chat Format with Token Types

SFT samples are formatted conversations. Each sample has:
- `input_ids`: token IDs of the full conversation
- `token_types`: type label for each token (0=system, 1=user, 2=assistant)

Short conversations are padded to `max_seq_len` with `pad_token_id = 0`.

---

## 4. PreferenceDataset — For DPO/GRPO

Each sample contains three texts encoded as token IDs:
- `prompt_ids`: the user's question
- `chosen_ids`: a good response (human-preferred)
- `rejected_ids`: a bad response (human-dispreferred)

Used to train DPO (comparing chosen vs rejected) and GRPO (computing relative rewards).

---

## 5. StreamingPretrainDataset — Memory-Efficient Large Corpora

For corpora too large to fit in memory (terabytes of text), we stream file-by-file:
1. Open one file
2. Tokenise line-by-line
3. Fill a buffer with tokens
4. When buffer reaches `seq_len`, yield a sample and clear the buffer
5. Move to next file

**BUG-24 Fix:** The last partial buffer (shorter than `seq_len`) was padded to `seq_len` but the padding tokens were treated as real training data. This pollutes the loss — the model tries to learn to predict `pad_token_id` after real text ends, which is meaningless.

The fix: emit an `attention_mask` (1=real, 0=padding) alongside each sample. The training loop uses this mask to exclude padded positions from the loss.

---

## 6. Full Annotated Source: `apex/data/dataset.py`

```python
"""
APEX-1 Dataset Classes.

BUG-24 FIX: StreamingPretrainDataset now emits an attention_mask
so padding tokens are excluded from training loss.
"""

import json, logging, random
from pathlib import Path
from typing import Any, Optional
import torch
from torch.utils.data import Dataset, IterableDataset

logger = logging.getLogger(__name__)


class PretrainDataset(Dataset):
    """Packs a flat token tensor into fixed-length sequences."""

    def __init__(self, token_ids: torch.Tensor, seq_len: int = 2048, stride: Optional[int] = None):
        self.token_ids = token_ids
        self.seq_len = seq_len
        self.stride = stride or seq_len   # Non-overlapping by default
        n = len(token_ids)
        # Number of complete samples (need seq_len+1 tokens for input+target)
        self.n_samples = max(1, (n - seq_len) // self.stride)
        logger.info("PretrainDataset: %d tokens → %d samples", n, self.n_samples)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        # Slice a chunk of seq_len tokens starting at stride*idx
        start = idx * self.stride
        end = start + self.seq_len
        return {"input_ids": self.token_ids[start:end].clone()}


class SFTDataset(Dataset):
    """Chat conversation dataset for Supervised Fine-Tuning."""

    def __init__(self, samples: list[dict], max_seq_len: int = 2048, pad_token_id: int = 0):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        sample = self.samples[idx]

        # Truncate to max_seq_len
        input_ids = sample["input_ids"][: self.max_seq_len]
        token_types = sample["token_types"][: self.max_seq_len]

        # Pad if shorter than max_seq_len
        pad_len = self.max_seq_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.pad_token_id] * pad_len
            # Padding tokens are type 0 (treated as system → ignored in SFT loss)
            token_types = token_types + [0] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "token_types": torch.tensor(token_types, dtype=torch.long),
        }

    @classmethod
    def from_jsonl(cls, path: str | Path, tokenizer: Any, max_seq_len: int = 2048):
        """Load SFT dataset from JSONL where each line is {"messages": [...]}."""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                messages = data["messages"]
                # Tokenizer encodes full conversation with role markers
                input_ids = tokenizer.encode_chat(messages, add_generation_prompt=False)
                # Get token type (0/1/2) for each token
                token_types = tokenizer.get_token_types(input_ids)
                samples.append({"input_ids": input_ids, "token_types": token_types})
        return cls(samples, max_seq_len, tokenizer.pad_token_id)


class PreferenceDataset(Dataset):
    """(prompt, chosen, rejected) triples for DPO/GRPO training."""

    def __init__(self, samples: list[dict], max_seq_len: int = 2048, pad_token_id: int = 0):
        self.samples = samples
        self.max_seq_len = max_seq_len
        self.pad_token_id = pad_token_id

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        sample = self.samples[idx]
        # Truncate all three sequences
        prompt = sample["prompt_ids"][: self.max_seq_len]
        chosen = sample["chosen_ids"][: self.max_seq_len]
        rejected = sample["rejected_ids"][: self.max_seq_len]
        return {
            "prompt_ids": torch.tensor(prompt, dtype=torch.long),
            "chosen_ids": torch.tensor(chosen, dtype=torch.long),
            "rejected_ids": torch.tensor(rejected, dtype=torch.long),
            "prompt_len": len(sample["prompt_ids"]),  # Needed for DPO loss masking
        }

    @classmethod
    def from_jsonl(cls, path: str | Path, tokenizer: Any, max_seq_len: int = 2048):
        """Load from JSONL: each line = {"prompt": str, "chosen": str, "rejected": str}"""
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                data = json.loads(line.strip())
                # Encode each part independently
                prompt_ids = tokenizer.encode(data["prompt"], add_special_tokens=False)
                # Chosen and rejected include the full prompt + response
                chosen_ids = tokenizer.encode(data["prompt"] + data["chosen"], add_special_tokens=False)
                rejected_ids = tokenizer.encode(data["prompt"] + data["rejected"], add_special_tokens=False)
                samples.append({"prompt_ids": prompt_ids, "chosen_ids": chosen_ids, "rejected_ids": rejected_ids})
        return cls(samples, max_seq_len, tokenizer.pad_token_id)


class StreamingPretrainDataset(IterableDataset):
    """Memory-efficient streaming dataset for large text corpora.
    
    Reads files one by one, tokenises on the fly, and yields
    fixed-length sequences without loading everything into memory.
    
    BUG-24 FIX: Emits attention_mask so padding in the last partial
    buffer is excluded from training loss.
    """

    def __init__(self, file_paths: list, tokenizer: Any, seq_len: int = 2048,
                 shuffle_files: bool = True, seed: int = 42):
        self.file_paths = [Path(p) for p in file_paths]
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.shuffle_files = shuffle_files
        self.seed = seed

    def __iter__(self):
        """Yield samples as {'input_ids': ..., 'attention_mask': ...}."""
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
                    # Tokenise one line at a time
                    tokens = self.tokenizer.encode(line.strip(), add_special_tokens=False)
                    buffer.extend(tokens)

                    # Yield complete chunks (every full seq_len tokens)
                    while len(buffer) >= self.seq_len:
                        chunk = buffer[: self.seq_len]
                        # All tokens are real → attention_mask all 1s
                        yield {
                            "input_ids": torch.tensor(chunk, dtype=torch.long),
                            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
                        }
                        buffer = buffer[self.seq_len :]

        # Handle the remaining partial buffer
        if len(buffer) >= self.seq_len // 2:
            real_len = len(buffer)
            pad_len = self.seq_len - real_len

            # Pad to full seq_len
            buffer.extend([self.tokenizer.pad_token_id] * pad_len)

            # BUG-24 FIX: emit attention_mask
            # 1 for real tokens, 0 for padding → trainer excludes pad positions
            yield {
                "input_ids": torch.tensor(buffer[: self.seq_len], dtype=torch.long),
                "attention_mask": torch.cat([
                    torch.ones(real_len, dtype=torch.long),   # real tokens
                    torch.zeros(pad_len, dtype=torch.long),   # padding
                ]),
            }
```

---

## 7. DataLoader Settings

```python
# apex/data/data_loader.py

def create_pretrain_loader(dataset, batch_size=32, num_workers=4, ...):
    """Create optimised DataLoader for pretraining."""
    is_iterable = isinstance(dataset, IterableDataset)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        # IterableDatasets handle their own ordering — don't shuffle externally
        shuffle=shuffle and not is_iterable,
        num_workers=num_workers,          # Parallel data loading (CPUs)
        # pin_memory=True: allocate pinned (non-pageable) RAM for faster GPU transfer
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True,                   # Drop incomplete last batch
        # prefetch_factor=2: each worker pre-loads 2 batches ahead of time
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,  # Keep worker processes alive between epochs
    )
```

---

## 8. Summary: Which Dataset for Which Stage?

| Training Stage | Dataset | Loss |
|---|---|---|
| Pretraining (offline) | `PretrainDataset` | `compute_pretrain_loss` |
| Pretraining (streaming) | `StreamingPretrainDataset` | `compute_pretrain_loss` |
| SFT | `SFTDataset` | `compute_sft_loss` |
| DPO / GRPO | `PreferenceDataset` | `dpo_loss` / `grpo_training_step` |

---

**Next:** [21 — Generation & Sampling →](21-generation-sampling.md)
