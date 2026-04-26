"""
Tokenizer Training Script for APEX-1.

Trains a BPE tokenizer from raw text files using the HuggingFace
tokenizers library. The resulting tokenizer has 151,643 tokens including
all APEX-1 special tokens.

Usage:
    python -m apex.tokenizer.train_tokenizer \
        --input data/raw_text/*.txt \
        --output tokenizer/apex1_tokenizer.json \
        --vocab-size 151643
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def train_tokenizer(
    input_files: list[str],
    output_path: str,
    vocab_size: int = 151643,
    min_frequency: int = 2,
) -> None:
    """Train a BPE tokenizer from raw text files.

    Args:
        input_files: List of paths to raw text files.
        output_path: Path to save the trained tokenizer JSON.
        vocab_size: Target vocabulary size.
        min_frequency: Minimum frequency for a merge to be applied.
    """
    from tokenizers import Tokenizer, pre_tokenizers, trainers
    from tokenizers.models import BPE
    from tokenizers.normalizers import NFC, Sequence
    from tokenizers.pre_tokenizers import ByteLevel

    logger.info("Training BPE tokenizer with vocab_size=%d", vocab_size)
    logger.info("Input files: %s", input_files)

    # Create tokenizer with BPE model
    tokenizer = Tokenizer(BPE(unk_token=None))

    # Normalizer: NFC unicode normalization
    tokenizer.normalizer = Sequence([NFC()])

    # Pre-tokenizer: byte-level (ensures zero unknown tokens)
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)

    # Define special tokens
    special_tokens = [
        "<|pad|>",
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|system|>",
        "<|user|>",
        "<|assistant|>",
        "<|thinking|>",
        "<|/thinking|>",
        "<|img|>",
    ]

    # Trainer configuration
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
    )

    # Validate input files
    valid_files = []
    for f in input_files:
        path = Path(f)
        if path.exists() and path.is_file():
            valid_files.append(str(path))
        else:
            logger.warning("Skipping non-existent file: %s", f)

    if not valid_files:
        logger.error("No valid input files found. Creating a demo tokenizer instead.")
        # Create demo tokenizer with minimal training data
        demo_text = (
            "The quick brown fox jumps over the lazy dog. "
            "This is a test of the APEX-1 tokenizer training pipeline. "
            "def hello_world():\n    print('Hello, World!')\n    return 42\n"
            "Machine learning is a subset of artificial intelligence. "
            "import torch\nimport numpy as np\n"
        ) * 100

        # Write temp file
        temp_path = Path(output_path).parent / "_temp_train.txt"
        temp_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path.write_text(demo_text)
        valid_files = [str(temp_path)]

    # Train
    logger.info("Training on %d files...", len(valid_files))
    tokenizer.train(valid_files, trainer)

    # Add special tokens post-training
    tokenizer.add_special_tokens(special_tokens)

    # Save
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(output))

    logger.info(
        "Tokenizer trained and saved to %s (vocab_size=%d)",
        output,
        tokenizer.get_vocab_size(),
    )

    # Clean up temp file if created
    temp_path = Path(output_path).parent / "_temp_train.txt"
    if temp_path.exists():
        temp_path.unlink()


def main() -> None:
    """CLI entry point for tokenizer training."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    parser = argparse.ArgumentParser(
        description="Train APEX-1 BPE tokenizer from raw text",
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=[],
        help="Path(s) to raw text files for training",
    )
    parser.add_argument(
        "--output",
        default="tokenizer/apex1_tokenizer.json",
        help="Output path for tokenizer JSON",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=151643,
        help="Target vocabulary size (default: 151643)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for BPE merges (default: 2)",
    )

    args = parser.parse_args()

    train_tokenizer(
        input_files=args.input,
        output_path=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )


if __name__ == "__main__":
    main()
