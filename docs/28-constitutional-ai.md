# 28 — Constitutional AI: Safety Built In

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/alignment/constitutional.py`  
> **You will learn:** The critique-revision loop, how APEX-1 bakes safety into training, and BUG-03 (hardcoded non-violation).

---

## 1. What Is Constitutional AI?

**Constitutional AI (CAI)** (Anthropic, 2022) teaches the model to critique its own responses against a set of safety principles (the "constitution") and then revise the response to be safer.

A "constitution" is a list of principles like:
- "The response should not help plan violence."
- "The response should not produce discriminatory content."
- "The response should be honest and not deceptive."

---

## 2. The Three-Stage Process

**Stage 1: Generate initial response**

```
Prompt: "How do I pick a lock?"
Response A: "Here's a step-by-step guide to lock picking..."
```

**Stage 2: Self-critique**

The model is asked: "Does your response violate this principle: *Do not assist with activities that could endanger others*? Explain your reasoning."

```
Critique: "Yes, this response could enable someone to break into homes or
           steal. I should not have provided detailed lock-picking instructions."
```

**Stage 3: Revision**

```
Revised: "Lock picking is a legitimate skill for locksmiths. However, for 
          safety reasons, I'd recommend contacting a licensed locksmith 
          rather than attempting this yourself."
```

The revised response is then scored by the constitutional judge.

---

## 3. CAI in Training

The constitutional reward is used during GRPO training. Instead of only rewarding good answers and penalising bad ones based on human preference, we also reward responses that:
1. Pass all constitutional checks (no violations detected)
2. Are coherent revisions when violations are detected

This bakes safety directly into the policy.

---

## 4. BUG-03: Hardcoded Non-Violation

The original critique function was:

```python
def generate_critique(self, text: str, principle: str) -> dict:
    # BUG-03: ALWAYS returns no_violation=True without running the model!
    # This means the model never detects safety violations during training.
    return {
        "is_violation": False,    # hardcoded
        "critique": "",           # empty
        "severity": 0.0,          # always 0
    }
```

This was clearly a placeholder that was never replaced. The constitutional reward was always 1.0 (no violations found), providing zero safety training signal.

**Fix:** Actually run the model to generate a critique:

```python
def generate_critique(self, text: str, principle: str) -> dict:
    prompt = self._build_critique_prompt(text, principle)
    token_ids = self.tokenizer.encode(prompt)
    # BUG-03 FIX: call model.generate() to actually generate a critique
    critique_output = self.generator.generate(
        torch.tensor([token_ids]), max_new_tokens=256
    )
    critique_text = self.tokenizer.decode(critique_output.token_ids)
    return self._parse_critique(critique_text)
```

---

## 5. Full Annotated Source: `apex/alignment/constitutional.py`

```python
"""
Constitutional AI for APEX-1.

Implements the CAI critique-revision loop:
1. Detect constitutional violations via model self-critique
2. Generate revisions for violated principles
3. Score the revised response

BUG-03 FIX: generate_critique() now calls model.generate() to produce
real critiques, instead of hardcoding is_violation=False.
"""

import re, logging
from dataclasses import dataclass
from typing import Optional
import torch
from apex.generation.generator import APEX1Generator, GenerationConfig

logger = logging.getLogger(__name__)


CRITIQUE_TEMPLATE = """You are reviewing a response for safety and ethics.

Response to review:
{text}

Constitutional principle to check:
{principle}

Does the response violate this principle? Explain concisely.
Format your answer as:
VIOLATION: YES or NO
SEVERITY: 0.0 to 1.0 (0=no violation, 1=severe)
EXPLANATION: (brief explanation)
"""

REVISION_TEMPLATE = """The following response has a problem:

Original response:
{original}

Problem identified:
{critique}

Please provide a revised response that addresses this problem while still 
being helpful and informative:
"""


@dataclass
class CritiqueResult:
    """Result from constitutional critique."""
    principle: str
    is_violation: bool
    severity: float      # 0.0 = no problem, 1.0 = serious violation
    critique_text: str
    raw_output: str


class ConstitutionalAI:
    """Constitutional AI safety filter for APEX-1.
    
    Args:
        model:         APEX-1 model (used for critique and revision).
        tokenizer:     Tokenizer for text encoding/decoding.
        principles:    List of safety principle strings.
        max_revisions: Maximum number of revision attempts.
    """

    def __init__(self, model, tokenizer, principles: list[str], max_revisions: int = 2):
        self.model = model
        self.tokenizer = tokenizer
        self.principles = principles
        self.max_revisions = max_revisions

        # Use a low-temperature config for critique (factual assessment)
        self.critique_config = GenerationConfig(
            max_new_tokens=256,
            temperature=0.3,     # Low temperature for consistent critiques
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
        # Medium temperature for revision (needs to be helpful)
        self.revision_config = GenerationConfig(
            max_new_tokens=512,
            temperature=0.5,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
        self.generator = APEX1Generator(model, self.critique_config)

    def generate_critique(self, text: str, principle: str) -> CritiqueResult:
        """Evaluate text against a constitutional principle.
        
        BUG-03 FIX: This now actually calls model.generate() to produce
        a real critique. Previously returned hardcoded no-violation.
        """
        # Build the critique prompt
        prompt = CRITIQUE_TEMPLATE.format(text=text, principle=principle)
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        # BUG-03 FIX: Run the model to generate a real critique
        output = self.generator.generate(
            input_ids=torch.tensor([token_ids], dtype=torch.long),
            gen_config=self.critique_config,
        )
        critique_text = self.tokenizer.decode(output.token_ids)

        # Parse the structured output
        return self._parse_critique(critique_text, principle)

    def _parse_critique(self, text: str, principle: str) -> CritiqueResult:
        """Extract structured fields from generated critique text."""
        is_violation = False
        severity = 0.0

        # Look for "VIOLATION: YES" in the generated text
        violation_match = re.search(r"VIOLATION:\s*(YES|NO)", text, re.IGNORECASE)
        if violation_match:
            is_violation = violation_match.group(1).strip().upper() == "YES"

        # Look for "SEVERITY: 0.75" etc.
        severity_match = re.search(r"SEVERITY:\s*([\d.]+)", text, re.IGNORECASE)
        if severity_match:
            try:
                severity = float(severity_match.group(1).strip())
                severity = max(0.0, min(1.0, severity))   # Clamp to [0, 1]
            except ValueError:
                severity = 0.5 if is_violation else 0.0

        return CritiqueResult(
            principle=principle,
            is_violation=is_violation,
            severity=severity,
            critique_text=text,
            raw_output=text,
        )

    def generate_revision(self, original: str, critique: CritiqueResult) -> str:
        """Generate a safer version of the response."""
        prompt = REVISION_TEMPLATE.format(
            original=original,
            critique=critique.critique_text[:500],   # Truncate long critiques
        )
        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True)

        output = self.generator.generate(
            input_ids=torch.tensor([token_ids], dtype=torch.long),
            gen_config=self.revision_config,
        )
        return self.tokenizer.decode(output.token_ids)

    def evaluate(self, text: str) -> dict:
        """Check text against all constitutional principles.
        
        Returns:
            dict with 'violations', 'constitutional_score', 'all_passed'.
        """
        violations = []

        for principle in self.principles:
            result = self.generate_critique(text, principle)
            if result.is_violation:
                violations.append(result)
                logger.debug(
                    "Violation detected: principle='%s...', severity=%.2f",
                    principle[:50], result.severity
                )

        # Constitutional score: 1.0 = fully compliant, 0.0 = many violations
        if not violations:
            constitutional_score = 1.0
        else:
            # Weighted by severity: severe violations hurt more
            max_severity = max(v.severity for v in violations)
            constitutional_score = max(0.0, 1.0 - max_severity)

        return {
            "violations": violations,
            "n_violations": len(violations),
            "constitutional_score": constitutional_score,
            "all_passed": len(violations) == 0,
        }

    def critique_and_revise(self, prompt: str, response: str) -> dict:
        """Full CAI pipeline: evaluate and optionally revise.
        
        Returns:
            dict with 'final_response', 'n_revisions', 'constitutional_score'.
        """
        current_response = response
        n_revisions = 0

        for revision_round in range(self.max_revisions):
            eval_result = self.evaluate(current_response)

            if eval_result["all_passed"]:
                logger.debug(
                    "Response passes constitutional check after %d revisions", n_revisions
                )
                break

            # Revise for the most severe violation
            worst_violation = max(eval_result["violations"], key=lambda v: v.severity)
            current_response = self.generate_revision(current_response, worst_violation)
            n_revisions += 1

        final_eval = self.evaluate(current_response)

        return {
            "final_response": current_response,
            "n_revisions": n_revisions,
            "constitutional_score": final_eval["constitutional_score"],
            "all_passed": final_eval["all_passed"],
        }
```

---

## 6. Example APEX-1 Constitution (Simplified)

```python
APEX_CONSTITUTION = [
    "The response must not provide detailed instructions for creating weapons.",
    "The response must not include sexually explicit content.",
    "The response must not promote or celebrate violence against people.",
    "The response must be honest and not intentionally misleading.",
    "The response must not help with illegal activities that harm others.",
    "The response should respect privacy and not doxx individuals.",
]
```

---

**Next:** [29 — Combined Reward →](29-combined-reward.md)
