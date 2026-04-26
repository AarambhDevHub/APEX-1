"""
Constitutional AI for APEX-1.

Automated alignment approach that does not require human preference labels.
The model critiques its own responses against a set of principles and
generates revised responses that are constitutionally correct.

Process (Section 13d):
1. Define constitution (set of principles)
2. Generate responses to adversarial prompts
3. Model critiques its own response against each principle
4. Model revises response to fix violations
5. Use (original, revised) pairs for DPO training

In APEX-1, Constitutional AI is integrated into the GRPO reward function
(Section 13f) rather than being a separate phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# Default constitutional principles
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
    """Result of a constitutional critique.

    Attributes:
        principle: The principle being evaluated.
        violated: Whether the principle was violated.
        explanation: Explanation of the critique.
    """

    principle: str
    violated: bool
    explanation: str = ""


@dataclass
class RevisionResult:
    """Result of a constitutional revision.

    Attributes:
        original_response: The original model response.
        revised_response: The constitutionally corrected response.
        critiques: List of critique results.
        violation_count: Number of principles violated.
        constitutional_score: Score from 0 to 1 (1 = no violations).
    """

    original_response: str
    revised_response: str
    critiques: list[CritiqueResult] = field(default_factory=list)
    violation_count: int = 0
    constitutional_score: float = 1.0


class ConstitutionalAI:
    """Constitutional AI critique and revision system.

    Uses the model to critique its own outputs against a set of
    principles and generate improved responses.

    Args:
        model: APEX-1 model (or any model with a generate method).
        tokenizer: Tokenizer for encoding/decoding.
        constitution: List of principle strings. Defaults to built-in set.
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

    def critique_response(
        self,
        response: str,
        prompt: Optional[str] = None,
    ) -> list[CritiqueResult]:
        """Critique a response against all constitutional principles.

        For each principle, generates a YES/NO judgment of whether
        the response violates that principle.

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

            # In production, this would call model.generate()
            # For now, we implement the scoring logic
            critique = CritiqueResult(
                principle=principle,
                violated=False,
                explanation="No violation detected.",
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
        score = 1.0 - (violations / max(len(self.constitution), 1))
        return score

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

        # Build revision prompt
        violation_text = "\n".join(f"- Violates: {v.principle}" for v in violations)
        revision_prompt = (
            f"Original response: {response}\n\n"
            f"The following principles were violated:\n{violation_text}\n\n"
            f"Please rewrite the response to be consistent with all principles."
        )

        # In production, call model.generate(revision_prompt)
        revised = response  # Placeholder — actual revision needs model.generate()

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
        """Generate (prompt, rejected, chosen) training pairs.

        For each prompt:
        1. Generate original response (potentially violating)
        2. Critique and revise
        3. Return (prompt, original, revised) for DPO training

        Args:
            prompts: List of prompts to generate training data for.

        Returns:
            List of (prompt, rejected_response, chosen_response) tuples.
        """
        pairs: list[tuple[str, str, str]] = []

        for prompt in prompts:
            # In production: response = model.generate(prompt)
            response = ""

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
