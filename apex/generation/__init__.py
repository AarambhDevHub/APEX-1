"""
APEX-1 Generation & Inference package.

Provides sampling, generation loop, thinking mode, and speculative decoding.
"""

from apex.generation.generator import APEX1Generator
from apex.generation.sampler import sample_next_token

__all__ = ["APEX1Generator", "sample_next_token"]
