# 27 — Process Reward Model (PRM)

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/alignment/prm.py`  
> **You will learn:** What step-level rewards are, why they beat outcome rewards for reasoning, and BUG-06.

---

## 1. Outcome vs Process Rewards

**Outcome Reward Model (ORM):** Scores the final answer only.

```
Prompt: "What is 8 × 7?"
Response: "8 × 7 = 42. Hmm, let me double-check: 8 × 7 = 56."
Final answer: "56"
ORM score: HIGH (final answer is correct)
```

**Problem:** The model tried the wrong answer first (42). The ORM doesn't penalise bad reasoning — only the wrong outcome.

**Process Reward Model (PRM):** Scores every step of reasoning:

```
Step 1: "8 × 7 = 42" → PRM score: LOW (wrong)
Step 2: "Let me double-check" → PRM score: MEDIUM (good habit)
Step 3: "8 × 7 = 56" → PRM score: HIGH (correct)
```

The PRM rewards **correct process**, not just correct outcomes. This produces models that reason better because they are penalised for incorrect intermediate steps even when they "get lucky" on the final answer.

---

## 2. How PRM Works Technically

PRM identifies **reasoning step boundaries** using special tokens or structural patterns. For APEX-1:

A step boundary is defined by detecting newlines, sentence-ending punctuation, or explicit step markers in the thinking section.

At each step boundary position, the model's hidden state is extracted and passed through a reward head:

$$r_{\text{step}} = \sigma(W_{prm} \cdot h_{step})$$

The final PRM score is the **product** of step scores (a chain is only as good as its weakest link):

$$R_{process} = \prod_{k=1}^{K} r_k$$

Alternatively, the **minimum** step score (representing the worst step):

$$R_{process} = \min_{k} r_k$$

APEX-1 uses the product for smoother gradients.

---

## 3. BUG-06: Silent Tokenizer Missing

The original PRM code tried to tokenise step content for re-scoring:

```python
class ProcessRewardModel(nn.Module):
    def score_response(self, prompt, response):
        # BUG-06: self.tokenizer could be None here!
        # If PRM was constructed without a tokenizer, this crashes
        # with AttributeError: 'NoneType' object has no attribute 'encode'
        tokens = self.tokenizer.encode(response)   # ← crash if tokenizer=None
```

If you created a PRM without passing a tokenizer (a common mistake during development), the error only appeared when calling `score_response` — not at construction time. The confusing error message made it hard to debug.

**Fix:** Validate immediately in `__init__` with a clear error message:

```python
def __init__(self, backbone, d_model, tokenizer=None, step_sep_tokens=None):
    super().__init__()
    self.tokenizer = tokenizer
    
    if tokenizer is None:
        # BUG-06 FIX: warn loudly at construction time instead of crashing mysteriously
        import warnings
        warnings.warn(
            "ProcessRewardModel created without a tokenizer. "
            "score_response() will fail unless a tokenizer is provided. "
            "Call prm.set_tokenizer(tokenizer) before scoring.",
            UserWarning,
            stacklevel=2,
        )
```

---

## 4. Full Annotated Source: `apex/alignment/prm.py`

```python
"""
Process Reward Model (PRM) for step-level reasoning feedback.

BUG-06 FIX: Missing tokenizer now raises a clear warning at construction
time (not a cryptic AttributeError at call time).
"""

import re, warnings
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProcessRewardModel(nn.Module):
    """Step-level reward model for reasoning evaluation.
    
    Scores each reasoning step independently.
    Returns both per-step and overall process reward.
    
    Args:
        backbone:         Pre-trained APEX1Model backbone.
        d_model:          Hidden dimension.
        tokenizer:        Tokenizer for encoding step content.
        step_sep_tokens:  Token IDs that mark step boundaries (e.g., newline).
    """

    def __init__(self, backbone, d_model: int, tokenizer=None,
                 step_sep_tokens: Optional[list[int]] = None):
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer

        # BUG-06 FIX: warn immediately if tokenizer is missing
        if tokenizer is None:
            warnings.warn(
                "ProcessRewardModel created without a tokenizer. "
                "score_response() will fail. Call prm.set_tokenizer(tok).",
                UserWarning,
                stacklevel=2,
            )

        # Step separator token IDs (newline, periods, etc.)
        self.step_sep_tokens = set(step_sep_tokens or [])

        # Reward head: maps hidden state at step boundary to step score
        self.step_reward_head = nn.Linear(d_model, 1, bias=False)

    def set_tokenizer(self, tokenizer) -> None:
        """Provide the tokenizer after construction."""
        self.tokenizer = tokenizer

    def forward(
        self,
        input_ids: torch.Tensor,   # [batch, seq_len]
        prefix_len: int = 0,
    ) -> dict:
        """Score all reasoning steps in the input.
        
        Args:
            input_ids: Full sequence [batch, seq_len].
            prefix_len: Number of prompt tokens.
        
        Returns:
            dict with:
              'step_rewards':   List of per-step reward tensors.
              'process_reward': Overall product of step rewards.
              'n_steps':        Number of steps detected.
        """
        # Get hidden states from backbone
        output = self.backbone(input_ids, prefix_len=prefix_len, return_hidden=True)
        hidden = output["hidden_states"]   # [B, S, d_model]

        B, S, D = hidden.shape
        batch_step_rewards = [[] for _ in range(B)]

        # Detect step boundaries in the token sequence
        # A step boundary is any position where a separator token appears
        token_ids_list = input_ids.tolist()   # Convert to Python list for easy iteration

        for b in range(B):
            seq_tokens = token_ids_list[b]
            for t in range(prefix_len, S):
                if seq_tokens[t] in self.step_sep_tokens:
                    # Score this step: use hidden state at the separator position
                    h_t = hidden[b, t, :]   # [d_model]
                    raw_score = self.step_reward_head(h_t)   # [1]
                    step_reward = torch.sigmoid(raw_score.squeeze(-1))   # [1] in [0,1]
                    batch_step_rewards[b].append(step_reward)

        # Compute overall process reward = product of all step rewards
        results = []
        for b in range(B):
            steps = batch_step_rewards[b]
            if steps:
                # Stack step rewards and compute product
                stacked = torch.stack(steps)   # [n_steps]
                process_reward = stacked.prod()   # product of all steps
            else:
                # No steps detected: use neutral reward
                process_reward = hidden.new_tensor(0.5)

            results.append(process_reward)

        process_rewards = torch.stack(results)   # [B]
        return {
            "step_rewards": batch_step_rewards,     # List[List[Tensor]]
            "process_reward": process_rewards,       # [B]
            "n_steps": [len(s) for s in batch_step_rewards],
        }

    def score_response(self, prompt: str, response: str) -> dict:
        """Convenience method: score a text response.
        
        Args:
            prompt:   The user's question as text.
            response: The model's response text.
        
        Returns:
            dict with 'process_reward', 'step_rewards', 'n_steps'.
        
        Raises:
            RuntimeError: If no tokenizer was provided.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "No tokenizer provided to ProcessRewardModel. "
                "Use prm.set_tokenizer(tok) before calling score_response()."
            )
        full_text = prompt + response
        token_ids = self.tokenizer.encode(full_text)
        prompt_len = len(self.tokenizer.encode(prompt))

        input_tensor = torch.tensor([token_ids], dtype=torch.long)
        with torch.no_grad():
            return self.forward(input_tensor, prefix_len=prompt_len)
```

---

## 5. Using PRM in GRPO

The PRM score is combined with the outcome reward in `combined_reward.py`:

```python
# Process reward from PRM
prm_output = prm.score_response(prompt_text, response_text)
process_reward = prm_output["process_reward"].item()   # [0, 1]

# Outcome reward from reward model
outcome_reward = reward_model(full_ids).item()

# Combined signal (see doc 29 for the full formula)
reward = w_outcome * outcome_reward + w_process * process_reward
```

---

## 6. When Does PRM Help Most?

| Task | ORM Only | PRM Added | Improvement |
|---|---|---|---|
| Simple Q&A | Baseline | — | ~0% |
| Multi-step math | Baseline | +12–18% | High |
| Code debugging | Baseline | +10–15% | High |
| Complex reasoning | Baseline | +8–12% | Moderate |

PRM matters most when the reasoning chain is long (many steps) and errors early can lead to a wrong final answer. For simple factual tasks, ORM suffices.

---

**Next:** [28 — Constitutional AI →](28-constitutional-ai.md)
