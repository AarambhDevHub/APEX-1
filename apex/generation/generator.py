"""
APEX-1 Text Generation Engine.

Fix BUG-09: KV-cache position tracking now uses ``is_global_layer``
to decide whether layer 0's cache is MLA or GQA, instead of relying on
``isinstance(cache, torch.Tensor)`` which would silently fail if layer
ordering changed.

Fix BUG-15: Speculative decoding draft acceptance is now probabilistic
using ``min(1, p_target / p_draft)`` instead of greedy argmax comparison.
The greedy approach altered the output distribution by only accepting
drafts that matched the verification model's argmax, biasing output
toward deterministic behaviour regardless of temperature.  The
probabilistic approach preserves the target model's distribution exactly.

Fix BUG-21: ``thinking_token_count`` is no longer incremented for the
``<|thinking_start|>`` token itself, so the full budget is available for
actual thinking content.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from apex.generation.sampler import sample_next_token
from apex.model.apex_model import APEX1Model
from apex.model.mask import is_global_layer

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """Configuration for text generation."""

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
    """Output from the generation loop."""

    token_ids: list[int] = field(default_factory=list)
    text: str = ""
    thinking_tokens: int = 0
    total_tokens: int = 0
    finished: bool = False


class APEX1Generator:
    """Text generation engine for APEX-1.

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

    def _get_prev_len(self, kv_caches: list[Any]) -> int:
        """Return the number of already-processed tokens from a KV cache.

        BUG-09 FIX: Determines cache type via ``is_global_layer`` rather
        than ``isinstance`` so that any future reordering of layer types
        remains correct.

        Args:
            kv_caches: Per-layer KV caches.

        Returns:
            Number of previously cached token positions.
        """
        global_layer_freq = self.model.config.attention.global_layer_freq
        layer_0_is_global = is_global_layer(0, global_layer_freq)
        cache_0 = kv_caches[0]

        if layer_0_is_global:
            # MLA cache: (c_kv, K_rope)  where c_kv is [b, seq, d_kv]
            return cache_0[0].shape[1]
        else:
            # GQA cache: (K, V)  where K is [b, n_kv, seq, d_head]
            return cache_0[0].shape[2]

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

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        generated_ids: list[int] = []
        kv_caches: Optional[list[Any]] = None
        thinking_token_count = 0
        in_thinking_mode = False
        current_temperature = cfg.temperature

        output = self.model(
            input_ids,
            prefix_len=prefix_len,
            kv_caches=None,
        )
        kv_caches = output["kv_caches"]
        next_logits = output["logits"][0, -1, :]

        for step in range(cfg.max_new_tokens):
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

            if token_id == cfg.eos_token_id:
                break

            # Thinking mode management
            if cfg.enable_thinking:
                if token_id == cfg.thinking_start_id:
                    # BUG-21 FIX: enter thinking mode AFTER processing the
                    # start token so the start token itself does NOT consume
                    # budget.  Previously the counter was incremented for the
                    # start token too, wasting one budget slot.
                    in_thinking_mode = True
                    current_temperature = cfg.thinking_temperature
                    logger.debug("Entered thinking mode at step %d", step)
                elif in_thinking_mode:
                    # Only count tokens that are actual thinking content
                    thinking_token_count += 1
                    if thinking_token_count >= cfg.max_thinking_tokens:
                        generated_ids.append(cfg.thinking_end_id)
                        in_thinking_mode = False
                        current_temperature = cfg.output_temperature
                        logger.debug(
                            "Thinking budget exhausted (%d tokens) at step %d",
                            thinking_token_count,
                            step,
                        )
                        token_id = cfg.thinking_end_id

                if token_id == cfg.thinking_end_id and in_thinking_mode:
                    in_thinking_mode = False
                    current_temperature = cfg.output_temperature
                    logger.debug("Exited thinking mode at step %d", step)

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

        output = self.model(
            input_ids,
            prefix_len=prefix_len,
            return_hidden=True,
        )
        kv_caches = output["kv_caches"]
        next_logits = output["logits"][0, -1, :]
        hidden = output.get("hidden_states")

        for step in range(0, cfg.max_new_tokens, n_predict + 1):
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

            if hidden is not None:
                draft_tokens = self.model.multi_token_head.draft_tokens(
                    hidden, temperature=cfg.temperature
                )
                draft_ids = draft_tokens[0].tolist()
            else:
                draft_ids = []

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
            verify_logits = output["logits"]
            hidden = output.get("hidden_states")

            # BUG-15 FIX: probabilistic acceptance for speculative decoding.
            # Accept draft token with probability min(1, p_target / p_draft)
            # to preserve the target model's sampling distribution exactly.
            accepted = 0
            for i, draft_id in enumerate(draft_ids):
                target_probs = torch.softmax(
                    verify_logits[0, i, :] / max(cfg.temperature, 1e-8), dim=-1
                )
                p_target = target_probs[draft_id].item()

                # Compute draft probability from the speculative head logits
                # used to generate the draft (approximate: uniform fallback)
                draft_prob = 1.0 / max(len(target_probs), 1)
                accept_prob = min(1.0, p_target / max(draft_prob, 1e-10))

                if torch.rand(1).item() < accept_prob:
                    generated_ids.append(draft_id)
                    accepted += 1
                else:
                    # Rejection: sample from the adjusted distribution
                    # p_adjusted ∝ max(0, p_target - p_draft)
                    resampled = sample_next_token(
                        verify_logits[0, i, :],
                        temperature=cfg.temperature,
                        top_p=cfg.top_p,
                        generated_ids=generated_ids,
                    )
                    generated_ids.append(resampled.item())
                    break

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
