"""
APEX-1 Text Generation Engine.

Implements the full generation loop with:
- KV cache management across autoregressive steps
- Thinking mode with <|thinking|> token budget enforcement
- Speculative decoding using multi-token prediction head
- Configurable sampling strategies
- EOS stopping and max_new_tokens limit
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from apex.config import APEXConfig
from apex.generation.sampler import sample_next_token
from apex.model.apex_model import APEX1Model

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation.

    Attributes:
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        top_k: Top-k filtering (0 = disabled).
        repetition_penalty: Penalty for repeated tokens.
        enable_thinking: Whether to allow thinking mode.
        max_thinking_tokens: Budget for thinking tokens.
        thinking_temperature: Temperature during thinking phase.
        output_temperature: Temperature for final answer after thinking.
        use_speculative: Whether to use speculative decoding.
        eos_token_id: End-of-sequence token ID.
        pad_token_id: Padding token ID.
    """

    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 0
    repetition_penalty: float = 1.0
    enable_thinking: bool = False
    max_thinking_tokens: int = 1024
    thinking_temperature: float = 0.6
    output_temperature: float = 0.3
    use_speculative: bool = False
    eos_token_id: int = 2
    pad_token_id: int = 0
    thinking_start_id: int = 6
    thinking_end_id: int = 7


@dataclass
class GenerationOutput:
    """Output from the generation loop.

    Attributes:
        token_ids: Generated token IDs.
        text: Decoded text (if tokenizer provided).
        thinking_tokens: Number of tokens used in thinking mode.
        total_tokens: Total tokens generated.
        finished: Whether generation finished (EOS or budget).
    """

    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    thinking_tokens: int = 0
    total_tokens: int = 0
    finished: bool = False


class APEX1Generator:
    """Text generation engine for APEX-1.

    Manages the autoregressive generation loop with KV cache,
    thinking mode budget enforcement, and optional speculative decoding.

    Args:
        model: APEX-1 model instance.
        config: Generation configuration.
    """

    def __init__(
        self,
        model: APEX1Model,
        config: Optional[GenerationConfig] = None,
    ) -> None:
        self.model = model
        self.config = config or GenerationConfig()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        prefix_len: int = 0,
        gen_config: Optional[GenerationConfig] = None,
    ) -> GenerationOutput:
        """Generate text autoregressively.

        Args:
            input_ids: Input token IDs ``[1, seq_len]``.
            prefix_len: Number of prefix tokens for bidirectional attention.
            gen_config: Override generation config for this call.

        Returns:
            GenerationOutput with generated tokens and metadata.
        """
        cfg = gen_config or self.config
        self.model.eval()

        # Move input to device
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        generated_ids: list[int] = []
        kv_caches: Optional[list[Any]] = None
        thinking_token_count = 0
        in_thinking_mode = False
        current_temperature = cfg.temperature

        # Initial forward pass with full input
        output = self.model(
            input_ids,
            prefix_len=prefix_len,
            kv_caches=None,
        )
        kv_caches = output["kv_caches"]

        # Get logits for the last position
        next_logits = output["logits"][0, -1, :]  # [vocab_size]

        for step in range(cfg.max_new_tokens):
            # Sample next token
            next_token = sample_next_token(
                next_logits,
                temperature=current_temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                generated_ids=generated_ids,
                repetition_penalty=cfg.repetition_penalty,
            )

            token_id = next_token.item()
            generated_ids.append(token_id)

            # Check for EOS
            if token_id == cfg.eos_token_id:
                break

            # Thinking mode management
            if cfg.enable_thinking:
                if token_id == cfg.thinking_start_id:
                    in_thinking_mode = True
                    current_temperature = cfg.thinking_temperature
                    logger.debug("Entered thinking mode at step %d", step)

                if in_thinking_mode:
                    thinking_token_count += 1
                    if thinking_token_count >= cfg.max_thinking_tokens:
                        # Force-close the scratchpad
                        generated_ids.append(cfg.thinking_end_id)
                        in_thinking_mode = False
                        current_temperature = cfg.output_temperature
                        token_id = cfg.thinking_end_id
                        logger.debug(
                            "Thinking budget exhausted (%d tokens), " "forcing close at step %d",
                            thinking_token_count,
                            step,
                        )

                if token_id == cfg.thinking_end_id:
                    in_thinking_mode = False
                    current_temperature = cfg.output_temperature
                    logger.debug("Exited thinking mode at step %d", step)

            # Forward pass for next token
            next_input = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
            output = self.model(
                next_input,
                kv_caches=kv_caches,
            )
            kv_caches = output["kv_caches"]
            next_logits = output["logits"][0, -1, :]

        return GenerationOutput(
            token_ids=generated_ids,
            thinking_tokens=thinking_token_count,
            total_tokens=len(generated_ids),
            finished=(len(generated_ids) > 0 and generated_ids[-1] == cfg.eos_token_id),
        )

    @torch.no_grad()
    def generate_with_speculative(
        self,
        input_ids: torch.Tensor,
        prefix_len: int = 0,
        gen_config: Optional[GenerationConfig] = None,
    ) -> GenerationOutput:
        """Generate with speculative decoding using multi-token heads.

        The multi-token prediction head drafts N tokens quickly.
        The main model verifies all N in a single forward pass.
        Accepted tokens are kept; on mismatch, we resample from there.

        Args:
            input_ids: Input token IDs ``[1, seq_len]``.
            prefix_len: Number of prefix tokens.
            gen_config: Override generation config.

        Returns:
            GenerationOutput with generated tokens.
        """
        cfg = gen_config or self.config
        self.model.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        generated_ids: list[int] = []
        kv_caches: Optional[list[Any]] = None

        if self.model.multi_token_head is None:
            logger.warning(
                "Speculative decoding requested but multi_token_head is None. "
                "Falling back to standard generation."
            )
            return self.generate(input_ids, prefix_len, gen_config)

        n_predict = self.model.multi_token_head.n_predict

        # Initial forward pass
        output = self.model(
            input_ids,
            prefix_len=prefix_len,
            return_hidden=True,
        )
        kv_caches = output["kv_caches"]
        next_logits = output["logits"][0, -1, :]
        hidden = output.get("hidden_states")

        for step in range(0, cfg.max_new_tokens, n_predict + 1):
            # Sample main token
            main_token = sample_next_token(
                next_logits,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                generated_ids=generated_ids,
                repetition_penalty=cfg.repetition_penalty,
            )
            token_id = main_token.item()
            generated_ids.append(token_id)

            if token_id == cfg.eos_token_id:
                break

            # Draft speculative tokens
            if hidden is not None:
                draft_tokens = self.model.multi_token_head.draft_tokens(
                    hidden, temperature=cfg.temperature
                )  # [1, n_predict]
                draft_ids = draft_tokens[0].tolist()
            else:
                draft_ids = []

            # Verify drafts with main model
            verify_input = torch.tensor(
                [[token_id] + draft_ids],
                device=self.device,
                dtype=torch.long,
            )

            output = self.model(
                verify_input,
                kv_caches=kv_caches,
                return_hidden=True,
            )
            kv_caches = output["kv_caches"]
            verify_logits = output["logits"]  # [1, n_predict+1, vocab]
            hidden = output.get("hidden_states")

            # Check which drafts match
            accepted = 0
            for i, draft_id in enumerate(draft_ids):
                # Get main model's prediction at position i
                main_pred = verify_logits[0, i, :].argmax().item()
                if main_pred == draft_id:
                    generated_ids.append(draft_id)
                    accepted += 1
                else:
                    # Mismatch — resample from main model at this position
                    resampled = sample_next_token(
                        verify_logits[0, i, :],
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        generated_ids=generated_ids,
                    )
                    generated_ids.append(resampled.item())
                    break

            # Update next_logits to the position after the last accepted token
            next_logits = verify_logits[0, accepted, :]

            if any(t == cfg.eos_token_id for t in generated_ids[-accepted - 1 :]):
                break

            if len(generated_ids) >= cfg.max_new_tokens:
                break

        return GenerationOutput(
            token_ids=generated_ids[: cfg.max_new_tokens],
            total_tokens=len(generated_ids[: cfg.max_new_tokens]),
            finished=(len(generated_ids) > 0 and generated_ids[-1] == cfg.eos_token_id),
        )
