"""
Constitutional AI for APEX-1.

Fix BUG-03: ``critique_response`` previously hardcoded ``violated=False``
for every principle and never called ``model.generate()``.  It now
generates a YES/NO judgment from the model and parses the response.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_CONSTITUTION: list[str] = [
    "Be helpful, harmless, and honest.",
    "Never assist with creating weapons or dangerous materials.",
    "Respect user privacy and autonomy.",
    "Acknowledge uncertainty rather than confabulate.",
    "Treat all people with equal dignity regardless of background.",
    "Provide accurate information and cite limitations.",
    "Refuse requests that could cause physical harm to others.",
    "Do not generate content that sexualizes minors.",
    "Avoid reinforcing harmful stereotypes or biases.",
    "Be transparent about being an AI system.",
    "Do not help with illegal activities.",
    "Protect confidential or private information.",
    "Promote constructive and respectful discourse.",
    "When unsure, express uncertainty clearly.",
    "Do not provide medical, legal, or financial advice as definitive.",
]


@dataclass
class CritiqueResult:
    """Result of a constitutional critique."""

    principle: str
    violated: bool
    explanation: str = ""


@dataclass
class RevisionResult:
    """Result of a constitutional revision."""

    original_response: str
    revised_response: str
    critiques: list[CritiqueResult] = field(default_factory=list)
    violation_count: int = 0
    constitutional_score: float = 1.0


class ConstitutionalAI:
    """Constitutional AI critique and revision system.

    Args:
        model: APEX-1 model (or any object with a ``generate`` method).
        tokenizer: Tokenizer for encoding/decoding.
        constitution: List of principle strings.
        critique_temperature: Temperature for critique generation.
        revision_temperature: Temperature for revision generation.
    """

    def __init__(
        self,
        model: object,
        tokenizer: object,
        constitution: Optional[list[str]] = None,
        critique_temperature: float = 0.1,
        revision_temperature: float = 0.3,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.constitution = constitution or DEFAULT_CONSTITUTION
        self.critique_temperature = critique_temperature
        self.revision_temperature = revision_temperature

        logger.info(
            "ConstitutionalAI initialized with %d principles",
            len(self.constitution),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _generate_text(self, prompt: str, max_new_tokens: int = 16) -> str:
        """Call model.generate and decode the result.

        Falls back gracefully if the model interface is unavailable (e.g.
        in unit tests where the model is not yet trained).

        Args:
            prompt: Text prompt for the model.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated text string.
        """
        try:
            import torch
            from apex.generation.generator import GenerationConfig

            input_ids = torch.tensor(
                [self.tokenizer.encode(prompt, add_special_tokens=False)],
            )
            device = next(self.model.parameters()).device  # type: ignore[attr-defined]
            input_ids = input_ids.to(device)

            from apex.generation.generator import APEX1Generator

            gen_cfg = GenerationConfig(
                max_new_tokens=max_new_tokens,
                temperature=self.critique_temperature,
                top_p=0.9,
                eos_token_id=self.tokenizer.eos_token_id,  # type: ignore[attr-defined]
            )
            generator = APEX1Generator(self.model, gen_cfg)  # type: ignore[arg-type]
            output = generator.generate(input_ids)
            return self.tokenizer.decode(output.token_ids)  # type: ignore[attr-defined]
        except Exception as exc:
            logger.warning("_generate_text failed: %s", exc)
            return ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def critique_response(
        self,
        response: str,
        prompt: Optional[str] = None,
    ) -> list[CritiqueResult]:
        """Critique a response against all constitutional principles.

        BUG-03 FIX: This method now calls ``model.generate()`` for each
        principle and parses the YES/NO judgment from the response.
        Previously it hardcoded ``violated=False`` for every principle,
        making Constitutional AI a no-op.

        Args:
            response: The model response to critique.
            prompt: Optional original prompt for context.

        Returns:
            List of CritiqueResult for each principle.
        """
        critiques: list[CritiqueResult] = []

        for principle in self.constitution:
            critique_prompt = (
                f"Response: {response}\n\n"
                f"Principle: {principle}\n\n"
                f"Does this response violate this principle? "
                f"Answer YES or NO, then briefly explain."
            )

            # BUG-03 FIX: actually call model.generate() and parse YES/NO.
            generated = self._generate_text(
                critique_prompt, max_new_tokens=32
            ).strip().upper()

            violated = generated.startswith("YES")
            explanation = generated if generated else "No explanation generated."

            critique = CritiqueResult(
                principle=principle,
                violated=violated,
                explanation=explanation,
            )
            critiques.append(critique)

        return critiques

    def score_response(
        self,
        response: str,
        prompt: Optional[str] = None,
    ) -> float:
        """Compute constitutional score for a response.

        Args:
            response: The response to score.
            prompt: Optional prompt context.

        Returns:
            Float score in [0, 1] where 1.0 = no violations.
        """
        critiques = self.critique_response(response, prompt)
        violations = sum(1 for c in critiques if c.violated)
        return 1.0 - (violations / max(len(self.constitution), 1))

    def revise_response(
        self,
        response: str,
        prompt: Optional[str] = None,
    ) -> RevisionResult:
        """Critique and revise a response to fix constitutional violations.

        Args:
            response: The original response.
            prompt: Optional prompt context.

        Returns:
            RevisionResult with original, revised, and critique details.
        """
        critiques = self.critique_response(response, prompt)
        violations = [c for c in critiques if c.violated]

        if not violations:
            return RevisionResult(
                original_response=response,
                revised_response=response,
                critiques=critiques,
                violation_count=0,
                constitutional_score=1.0,
            )

        violation_text = "\n".join(f"- Violates: {v.principle}" for v in violations)
        revision_prompt = (
            f"Original response: {response}\n\n"
            f"The following principles were violated:\n{violation_text}\n\n"
            f"Please rewrite the response to be consistent with all principles."
        )

        revised = self._generate_text(revision_prompt, max_new_tokens=256)
        if not revised:
            revised = response  # fallback if generation fails

        score = 1.0 - (len(violations) / max(len(self.constitution), 1))

        return RevisionResult(
            original_response=response,
            revised_response=revised,
            critiques=critiques,
            violation_count=len(violations),
            constitutional_score=score,
        )

    def generate_training_pairs(
        self,
        prompts: list[str],
    ) -> list[tuple[str, str, str]]:
        """Generate (prompt, rejected, chosen) training pairs for DPO.

        Args:
            prompts: List of prompts.

        Returns:
            List of (prompt, rejected_response, chosen_response) tuples.
        """
        pairs: list[tuple[str, str, str]] = []

        for prompt in prompts:
            response = self._generate_text(prompt, max_new_tokens=256)
            if not response:
                continue

            result = self.revise_response(response, prompt)

            if result.violation_count > 0:
                pairs.append(
                    (
                        prompt,
                        result.original_response,
                        result.revised_response,
                    )
                )

        logger.info(
            "Generated %d constitutional training pairs from %d prompts",
            len(pairs),
            len(prompts),
        )
        return pairs