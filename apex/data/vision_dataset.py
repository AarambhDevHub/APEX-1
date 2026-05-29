"""Vision-language instruction dataset for APEX-1.

JSONL format:

{"image":"data/images/cat.jpg","prompt":"What is in this image?","response":"A cat sitting on a chair."}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset

from apex.tokenizer.tokenizer import APEX1Tokenizer, SPECIAL_TOKENS
from apex.vision.preprocess import ImagePreprocessor

IGNORE_INDEX = -100


@dataclass
class VisionInstructionExample:
    image: str
    prompt: str
    response: str


class VisionInstructionDataset(Dataset):
    """Dataset for image-question-answer fine-tuning."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: APEX1Tokenizer,
        image_root: str | Path | None = None,
        image_size: int = 224,
        max_length: int = 2048,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.image_root = Path(image_root) if image_root is not None else self.jsonl_path.parent
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.preprocess = ImagePreprocessor(image_size=image_size)
        self.examples = self._load_examples(self.jsonl_path)

    def _load_examples(self, path: Path) -> list[VisionInstructionExample]:
        examples: list[VisionInstructionExample] = []
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                try:
                    examples.append(
                        VisionInstructionExample(
                            image=str(raw["image"]),
                            prompt=str(raw["prompt"]),
                            response=str(raw["response"]),
                        )
                    )
                except KeyError as exc:
                    raise ValueError(f"Missing field {exc!s} on line {line_no} of {path}") from exc
        if not examples:
            raise ValueError(f"No examples loaded from {path}")
        return examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        ex = self.examples[idx]
        image_path = self.image_root / ex.image
        pixel_values = self.preprocess(image_path)

        prompt_text = (
            f"{SPECIAL_TOKENS['bos']}"
            f"{SPECIAL_TOKENS['user']}\n"
            f"{SPECIAL_TOKENS['img']}\n{ex.prompt}\n"
            f"{SPECIAL_TOKENS['assistant']}\n"
        )
        answer_text = f"{ex.response}{SPECIAL_TOKENS['eos']}"

        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)
        token_ids = (prompt_ids + answer_ids)[: self.max_length]

        labels = [IGNORE_INDEX] * min(len(prompt_ids), len(token_ids))
        remaining = len(token_ids) - len(labels)
        if remaining > 0:
            labels.extend(answer_ids[:remaining])

        return {
            "token_ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "pixel_values": pixel_values,
            "image_path": str(image_path),
            "prompt": ex.prompt,
            "response": ex.response,
        }


def collate_vision_batch(
    batch: Sequence[dict[str, Any]],
    pad_token_id: int = 0,
    ignore_index: int = IGNORE_INDEX,
) -> dict[str, Any]:
    """Pad a list of dataset items into a batch."""
    max_len = max(item["token_ids"].numel() for item in batch)
    token_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), ignore_index, dtype=torch.long)
    pixel_values = torch.stack([item["pixel_values"] for item in batch], dim=0)

    for i, item in enumerate(batch):
        n = item["token_ids"].numel()
        token_ids[i, :n] = item["token_ids"]
        labels[i, :n] = item["labels"]

    return {
        "token_ids": token_ids,
        "labels": labels,
        "pixel_values": pixel_values,
        "image_paths": [item["image_path"] for item in batch],
        "prompts": [item["prompt"] for item in batch],
        "responses": [item["response"] for item in batch],
    }
