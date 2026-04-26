"""
Combined Reward Function for APEX-1 Alignment.

Combines three reward signals into a single score for GRPO training:
1. Outcome reward (0.5 weight) — is the final answer correct?
2. Process reward (0.2 weight) — are the reasoning steps valid?
3. Constitutional score (0.3 weight) — does the response follow principles?

This is APEX-1's dual-signal alignment (Section 13f): all three signals
shape every GRPO gradient step simultaneously, rather than being applied
in separate sequential phases.
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def combined_reward(
    outcome_reward: float,
    process_reward: float,
    constitutional_score: float,
    w_outcome: float = 0.5,
    w_process: float = 0.2,
    w_constitutional: float = 0.3,
) -> float:
    """Compute combined reward from three alignment signals.

    Args:
        outcome_reward: Is the final answer correct? Float in [0, 1].
        process_reward: Are the reasoning steps valid? Float in [0, 1].
        constitutional_score: Does the response follow principles? Float in [0, 1].
        w_outcome: Weight for outcome reward (default: 0.5).
        w_process: Weight for process reward (default: 0.2).
        w_constitutional: Weight for constitutional score (default: 0.3).

    Returns:
        Combined reward as a float in [0, 1].
    """
    total = (
        w_outcome * outcome_reward
        + w_process * process_reward
        + w_constitutional * constitutional_score
    )
    return total


def build_reward_function(
    outcome_checker: Optional[Callable] = None,
    prm: Optional[object] = None,
    constitutional_ai: Optional[object] = None,
    w_outcome: float = 0.5,
    w_process: float = 0.2,
    w_constitutional: float = 0.3,
) -> Callable:
    """Build a combined reward function for GRPO training.

    Creates a callable that takes (prompt, response) and returns
    a combined reward score integrating all three signals.

    Args:
        outcome_checker: Function(prompt, response) -> float reward.
        prm: ProcessRewardModel with score_steps method.
        constitutional_ai: ConstitutionalAI with score_response method.
        w_outcome: Weight for outcome reward.
        w_process: Weight for process reward.
        w_constitutional: Weight for constitutional score.

    Returns:
        Callable that computes combined reward.
    """

    def reward_fn(prompt: str, response: str) -> float:
        """Compute combined reward for a prompt-response pair.

        Args:
            prompt: The input prompt.
            response: The model's response.

        Returns:
            Combined reward float.
        """
        # Signal 1: Outcome reward
        if outcome_checker is not None:
            outcome_r = float(outcome_checker(prompt, response))
        else:
            outcome_r = 0.5  # neutral default

        # Signal 2: Process reward
        if prm is not None:
            try:
                steps = _extract_thinking_text(response)
                if steps:
                    step_scores = prm.score_steps_from_text(prompt, steps, None)
                    process_r = sum(step_scores) / len(step_scores)
                else:
                    process_r = 0.5
            except Exception as e:
                logger.warning("PRM scoring failed: %s", e)
                process_r = 0.5
        else:
            process_r = 0.5

        # Signal 3: Constitutional score
        if constitutional_ai is not None:
            try:
                constitutional_s = constitutional_ai.score_response(response, prompt)
            except Exception as e:
                logger.warning("Constitutional scoring failed: %s", e)
                constitutional_s = 0.5
        else:
            constitutional_s = 1.0  # assume safe if no checker

        return combined_reward(
            outcome_r, process_r, constitutional_s,
            w_outcome, w_process, w_constitutional,
        )

    return reward_fn


def _extract_thinking_text(response: str) -> list[str]:
    """Extract thinking steps from response text.

    Args:
        response: Full response text possibly containing thinking tags.

    Returns:
        List of reasoning step strings.
    """
    steps: list[str] = []
    start_tag = "<|thinking|>"
    end_tag = "<|/thinking|>"

    start_idx = response.find(start_tag)
    end_idx = response.find(end_tag)

    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        thinking_text = response[start_idx + len(start_tag):end_idx].strip()
        # Split by newlines into steps
        for line in thinking_text.split("\n"):
            line = line.strip()
            if line:
                steps.append(line)

    return steps
