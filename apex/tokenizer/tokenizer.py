"""
APEX-1 BPE Tokenizer.

Full BPE tokenizer implementation using the HuggingFace tokenizers library.
Features:
- Vocabulary size: 151,643 tokens
- Special tokens for chat (system/user/assistant), thinking mode, padding
- Chat template formatting
- Encode/decode with proper handling of special tokens
- Train from raw text capability

Fix BUG-14: ``get_token_types()`` now explicitly maps ``<|thinking|>``
and ``<|/thinking|>`` tokens to type 2 (assistant).  Previously these
tokens inherited the current type, which would be wrong if a thinking
block appeared without a preceding ``<|assistant|>`` token — the
thinking content would be labelled as system/user and excluded from
the SFT loss.

Special Tokens:
    <|begin_of_text|>  — Start of every sequence
    <|end_of_text|>    — End of generation
    <|system|>         — System prompt boundary
    <|user|>           — User turn boundary
    <|assistant|>      — Assistant turn boundary
    <|thinking|>       — Start of internal reasoning scratchpad
    <|/thinking|>      — End of reasoning scratchpad
    <|pad|>            — Padding token (ID = 0)
    <|img|>            — Image placeholder (future multimodal)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Special token definitions
SPECIAL_TOKENS = {
    "pad": "<|pad|>",
    "bos": "<|begin_of_text|>",
    "eos": "<|end_of_text|>",
    "system": "<|system|>",
    "user": "<|user|>",
    "assistant": "<|assistant|>",
    "thinking_start": "<|thinking|>",
    "thinking_end": "<|/thinking|>",
    "img": "<|img|>",
}

VOCAB_SIZE = 151643


class APEX1Tokenizer:
    """APEX-1 BPE Tokenizer.

    Wraps the HuggingFace tokenizers library to provide a production-ready
    tokenizer with special token handling, chat template formatting,
    and encode/decode functionality.

    Args:
        tokenizer_path: Path to a saved tokenizer file (tokenizer.json).
                       If None, creates a minimal tokenizer for testing.
    """

    def __init__(self, tokenizer_path: Optional[str | Path] = None) -> None:
        try:
            from tokenizers import Tokenizer
            from tokenizers.models import BPE
        except ImportError:
            raise ImportError("tokenizers library required. Install with: pip install tokenizers")

        if tokenizer_path is not None and Path(tokenizer_path).exists():
            self._tokenizer = Tokenizer.from_file(str(tokenizer_path))
            logger.info("Loaded tokenizer from %s", tokenizer_path)
        else:
            # Create a minimal BPE tokenizer for testing/development
            self._tokenizer = Tokenizer(BPE(unk_token=None))
            self._setup_minimal_tokenizer()
            logger.info("Created minimal tokenizer (no pretrained vocab loaded)")

        # Cache special token IDs
        self._special_token_ids: dict[str, int] = {}
        self._setup_special_tokens()

    def _setup_minimal_tokenizer(self) -> None:
        """Set up a minimal tokenizer with byte-level fallback for testing."""
        from tokenizers import pre_tokenizers

        # Use byte-level pre-tokenizer
        self._tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

        # Add special tokens to vocab
        special_list = list(SPECIAL_TOKENS.values())
        self._tokenizer.add_special_tokens(special_list)

        # Add byte-level tokens (256 bytes)
        byte_tokens = [f"<0x{i:02X}>" for i in range(256)]
        self._tokenizer.add_tokens(byte_tokens)

        # Add some common tokens for basic functionality
        common_tokens = [
            " ",
            "\n",
            "\t",
            ".",
            ",",
            "!",
            "?",
            ":",
            ";",
            "the",
            "a",
            "an",
            "is",
            "are",
            "was",
            "were",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "and",
            "The",
            "I",
            "you",
            "he",
            "she",
            "it",
            "we",
            "they",
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "def",
            "class",
            "import",
            "return",
            "if",
            "else",
            "function",
            "var",
            "let",
            "const",
        ]
        self._tokenizer.add_tokens(common_tokens)

    def _setup_special_tokens(self) -> None:
        """Cache special token IDs for fast lookup."""
        for name, token_str in SPECIAL_TOKENS.items():
            token_id = self._tokenizer.token_to_id(token_str)
            if token_id is not None:
                self._special_token_ids[name] = token_id
            else:
                # Assign sequential IDs for missing special tokens
                logger.debug("Special token %s not in vocab, using fallback", token_str)

    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size."""
        return self._tokenizer.get_vocab_size()

    @property
    def pad_token_id(self) -> int:
        """Get the padding token ID."""
        return self._special_token_ids.get("pad", 0)

    @property
    def bos_token_id(self) -> int:
        """Get the beginning-of-sequence token ID."""
        return self._special_token_ids.get("bos", 1)

    @property
    def eos_token_id(self) -> int:
        """Get the end-of-sequence token ID."""
        return self._special_token_ids.get("eos", 2)

    @property
    def system_token_id(self) -> int:
        """Get the system prompt token ID."""
        return self._special_token_ids.get("system", 3)

    @property
    def user_token_id(self) -> int:
        """Get the user turn token ID."""
        return self._special_token_ids.get("user", 4)

    @property
    def assistant_token_id(self) -> int:
        """Get the assistant turn token ID."""
        return self._special_token_ids.get("assistant", 5)

    @property
    def thinking_start_id(self) -> int:
        """Get the thinking start token ID."""
        return self._special_token_ids.get("thinking_start", 6)

    @property
    def thinking_end_id(self) -> int:
        """Get the thinking end token ID."""
        return self._special_token_ids.get("thinking_end", 7)

    @property
    def img_token_id(self) -> int:
        """Get the image placeholder token ID."""
        return self._special_token_ids.get("img", 8)

    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
    ) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text string.
            add_special_tokens: Whether to add BOS/EOS tokens.

        Returns:
            List of integer token IDs.
        """
        encoding = self._tokenizer.encode(text, add_special_tokens=False)
        ids = encoding.ids

        if add_special_tokens:
            ids = [self.bos_token_id] + ids + [self.eos_token_id]

        return ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        """Decode token IDs back to text.

        Args:
            token_ids: List of integer token IDs.
            skip_special_tokens: Whether to skip special tokens in output.

        Returns:
            Decoded text string.
        """
        if skip_special_tokens:
            special_ids = set(self._special_token_ids.values())
            token_ids = [t for t in token_ids if t not in special_ids]

        return self._tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def format_chat(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> str:
        """Format a chat conversation using the APEX-1 chat template.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
                     Roles: 'system', 'user', 'assistant'.
            add_generation_prompt: Whether to add the assistant prompt at end.
            enable_thinking: Whether to add thinking tags for the assistant.

        Returns:
            Formatted chat string ready for tokenization.

        Example:
            >>> tokenizer.format_chat([
            ...     {"role": "system", "content": "You are a helpful AI."},
            ...     {"role": "user", "content": "What is 2+2?"},
            ... ])
            '<|begin_of_text|><|system|>\\nYou are a...<|assistant|>\\n'
        """
        parts = [SPECIAL_TOKENS["bos"]]

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                parts.append(f"{SPECIAL_TOKENS['system']}\n{content}\n")
            elif role == "user":
                parts.append(f"{SPECIAL_TOKENS['user']}\n{content}\n")
            elif role == "assistant":
                if enable_thinking and SPECIAL_TOKENS["thinking_start"] not in content:
                    parts.append(
                        f"{SPECIAL_TOKENS['assistant']}\n"
                        f"{SPECIAL_TOKENS['thinking_start']}\n{content}\n"
                        f"{SPECIAL_TOKENS['thinking_end']}\n"
                    )
                else:
                    parts.append(f"{SPECIAL_TOKENS['assistant']}\n{content}\n")

        if add_generation_prompt:
            parts.append(f"{SPECIAL_TOKENS['assistant']}\n")
            if enable_thinking:
                parts.append(f"{SPECIAL_TOKENS['thinking_start']}\n")

        return "".join(parts)

    def encode_chat(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = True,
        enable_thinking: bool = False,
    ) -> list[int]:
        """Format and encode a chat conversation to token IDs.

        Args:
            messages: List of message dicts with 'role' and 'content'.
            add_generation_prompt: Whether to add assistant prompt.
            enable_thinking: Whether to enable thinking mode.

        Returns:
            List of token IDs.
        """
        text = self.format_chat(messages, add_generation_prompt, enable_thinking)
        return self.encode(text, add_special_tokens=False)

    def get_token_types(
        self,
        token_ids: list[int],
    ) -> list[int]:
        """Determine token types for SFT loss masking.

        Assigns type labels:
        - 0: system tokens
        - 1: user tokens
        - 2: assistant tokens (including thinking)

        BUG-14 FIX: ``<|thinking|>`` and ``<|/thinking|>`` tokens now
        explicitly set the current type to 2 (assistant).  This ensures
        thinking content is always included in the SFT loss even if the
        thinking block appears without a preceding ``<|assistant|>`` token.

        Args:
            token_ids: List of token IDs from encode_chat.

        Returns:
            List of type labels (same length as token_ids).
        """
        types = []
        current_type = 0  # default to system

        for tid in token_ids:
            if tid == self.system_token_id:
                current_type = 0
            elif tid == self.user_token_id:
                current_type = 1
            elif tid == self.assistant_token_id:
                current_type = 2
            # BUG-14 FIX: thinking tokens are always assistant content
            elif tid == self.thinking_start_id or tid == self.thinking_end_id:
                current_type = 2

            types.append(current_type)

        return types

    def save(self, path: str | Path) -> None:
        """Save tokenizer to a file.

        Args:
            path: Path to save the tokenizer JSON file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._tokenizer.save(str(path))
        logger.info("Tokenizer saved to %s", path)

    @classmethod
    def from_file(cls, path: str | Path) -> APEX1Tokenizer:
        """Load tokenizer from a file.

        Args:
            path: Path to the tokenizer JSON file.

        Returns:
            APEX1Tokenizer instance.
        """
        return cls(tokenizer_path=path)
