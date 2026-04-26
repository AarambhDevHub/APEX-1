"""
APEX-1 Tokenizer.

BPE tokenizer with 151,643 tokens including special tokens for chat,
thinking mode, and multimodal placeholders. Built on the HuggingFace
tokenizers library.
"""

from apex.tokenizer.tokenizer import APEX1Tokenizer

__all__ = ["APEX1Tokenizer"]
