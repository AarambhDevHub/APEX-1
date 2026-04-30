# 21 — Generation & Sampling: How Text Is Generated

> **Difficulty:** ⭐⭐☆☆☆ Intermediate  
> **Source files:** `apex/generation/sampler.py`, `apex/generation/generator.py`  
> **You will learn:** Autoregressive generation, the KV cache, temperature/top-p/top-k, and the full generation loop.

---

## 1. Autoregressive Generation

Language model inference works one token at a time:

```
Input: "The capital of France is"
Step 1: Model sees all 6 tokens → predicts next → "Paris" (prob 0.91)
Step 2: Model sees 7 tokens → predicts next → "." (prob 0.82)
Step 3: Model sees 8 tokens → predicts next → EOS (prob 0.73) → stop
Output: "Paris."
```

Each step the model sees the **entire history** (input + everything generated so far). Without optimisation, each step requires a full forward pass over all previous tokens — cost grows as $O(n^2)$.

---

## 2. The KV Cache: Avoiding Redundant Computation

Once a token has been processed, its Key and Value representations do not change (causal attention — past never changes). We can cache them:

**Step 1 (prefill):** Run full forward pass over input → compute and cache K, V for every input token.

**Step 2–N (decode):** Only the **new token** needs to be processed. Retrieve cached K, V for past tokens, compute K, V for new token, run attention.

Cost: $O(n)$ per step (instead of $O(n^2)$).

---

## 3. Sampling Strategies

The model outputs **logits** — raw scores for every token in the vocabulary. We need to convert these to a probability distribution and then sample from it.

### Temperature

$$P(w) = \text{softmax}(z / T)$$

where $z$ is the logit vector and $T$ is temperature.

| Temperature | Effect |
|---|---|
| $T \to 0$ | Greedy: always pick the highest-probability token |
| $T = 1.0$ | Raw model distribution |
| $T > 1.0$ | Flatter distribution: more random, creative |
| $T < 1.0$ | Sharper distribution: more focused, deterministic |

**Recommended settings:**
- Code generation: $T = 0.1$ (near-deterministic)
- Factual Q&A: $T = 0.3$
- Creative writing: $T = 0.9$

### Top-p (Nucleus) Sampling

Only sample from the smallest set of tokens whose cumulative probability exceeds $p$:

```
Sorted tokens by probability: [0.40, 0.25, 0.15, 0.10, 0.06, 0.02, ...]
Cumulative prob:               [0.40, 0.65, 0.80, 0.90, 0.96, 0.98, ...]

With top_p=0.90: keep first 4 tokens (cumsum reaches 0.90)
Set all others to -infinity, resample from the filtered 4.
```

This adapts automatically — when the model is confident (one token has 0.95 probability), only that one is kept. When uncertain, many tokens remain.

### Top-k Sampling

Simply keep the top-k highest-probability tokens, discard all others:

```
top_k=50: keep the 50 most likely tokens, zero out the rest
```

Simpler than top-p but less adaptive.

### Repetition Penalty

Discourages the model from repeating itself. For any token $w$ already in the generated sequence:

$$z_w \leftarrow \begin{cases} z_w / \rho & \text{if } z_w > 0 \\ z_w \times \rho & \text{if } z_w \leq 0 \end{cases}$$

where $\rho > 1$ is the penalty factor. This makes already-generated tokens less likely.

---

## 4. Full Annotated Source: `apex/generation/sampler.py`

```python
"""Sampling strategies for APEX-1."""

def apply_temperature(logits, temperature):
    """Divide logits by temperature — sharpens or flattens distribution."""
    if temperature <= 0:
        return logits   # Will use argmax (greedy)
    return logits / temperature


def apply_top_p(logits, top_p):
    """Keep only tokens whose cumulative probability reaches top_p."""
    if top_p >= 1.0:
        return logits   # Disabled — keep all

    # Sort from highest to lowest probability
    sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
    # Compute running cumulative probability
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    # Mark tokens to remove: those whose cumulative prob already exceeds top_p
    remove_mask = cumulative_probs > top_p
    # Shift right by 1: keep the FIRST token that pushes over the threshold
    # (otherwise we'd remove the token that just hit the threshold)
    remove_mask[..., 1:] = remove_mask[..., :-1].clone()
    remove_mask[..., 0] = False   # Always keep at least the best token

    # Set removed tokens to -infinity (→ probability 0 after softmax)
    sorted_logits[remove_mask] = float("-inf")

    # Scatter back to original token order
    result = torch.zeros_like(logits)
    result.scatter_(-1, sorted_idx, sorted_logits)
    return result


def apply_top_k(logits, top_k):
    """Keep only the top-k highest-probability tokens."""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits   # Disabled

    top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)
    filtered = torch.full_like(logits, float("-inf"))
    filtered.scatter_(-1, top_k_indices, top_k_logits)
    return filtered


def apply_repetition_penalty(logits, generated_ids, penalty):
    """Penalise previously generated tokens."""
    if penalty == 1.0 or not generated_ids:
        return logits

    logits = logits.clone()
    for token_id in set(generated_ids):
        if 0 <= token_id < logits.shape[-1]:
            # Divide positive logits (makes them less likely)
            # Multiply negative logits (makes them even more negative = less likely)
            if logits[token_id] > 0:
                logits[token_id] /= penalty
            else:
                logits[token_id] *= penalty
    return logits


def sample_next_token(logits, temperature=1.0, top_p=1.0, top_k=0,
                      generated_ids=None, repetition_penalty=1.0):
    """Apply all sampling strategies in order, then sample.
    
    Correct order: repetition penalty → temperature → top-k → top-p → sample
    
    Returns: token ID tensor [1].
    """
    # 1. Repetition penalty (before temperature — keeps scale consistent)
    if generated_ids and repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, generated_ids, repetition_penalty)

    # 2. Temperature scaling
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)   # Greedy decoding
    logits = apply_temperature(logits, temperature)

    # 3. Top-k filter
    logits = apply_top_k(logits, top_k)

    # 4. Top-p filter
    logits = apply_top_p(logits, top_p)

    # 5. Sample from filtered distribution
    probs = torch.softmax(logits, dim=-1)
    if probs.sum() == 0:
        # Edge case: all filtered out — fall back to uniform
        probs = torch.ones_like(probs) / probs.shape[-1]

    return torch.multinomial(probs, num_samples=1)
```

---

## 5. Full Annotated Source: `APEX1Generator.generate()`

```python
class APEX1Generator:
    def __init__(self, model, config=None):
        self.model = model
        self.config = config or GenerationConfig()
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def generate(self, input_ids, prefix_len=0, gen_config=None):
        """Generate text autoregressively.
        
        Phase 1 (Prefill): One forward pass over the full input.
                           Build KV cache for all input tokens.
        Phase 2 (Decode):  One token at a time, reusing KV cache.
        """
        cfg = gen_config or self.config
        self.model.eval()

        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        input_ids = input_ids.to(self.device)

        generated_ids = []
        kv_caches = None
        thinking_token_count = 0
        in_thinking_mode = False
        current_temperature = cfg.temperature

        # ── Phase 1: Prefill ─────────────────────────────────────────────
        # Process all input tokens at once (fast — no per-token cost)
        output = self.model(input_ids, prefix_len=prefix_len, kv_caches=None)
        kv_caches = output["kv_caches"]   # Save K, V for all input positions
        next_logits = output["logits"][0, -1, :]   # Logits for next token prediction

        # ── Phase 2: Autoregressive Decode ───────────────────────────────
        for step in range(cfg.max_new_tokens):
            # Sample the next token using our strategies
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

            # Stop condition: EOS token
            if token_id == cfg.eos_token_id:
                break

            # Thinking mode management
            if cfg.enable_thinking:
                if token_id == cfg.thinking_start_id:
                    # BUG-21 FIX: set flag AFTER appending, so the start token
                    # itself does NOT consume thinking budget
                    in_thinking_mode = True
                    current_temperature = cfg.thinking_temperature
                elif in_thinking_mode:
                    thinking_token_count += 1
                    if thinking_token_count >= cfg.max_thinking_tokens:
                        # Budget exhausted → force end thinking
                        generated_ids.append(cfg.thinking_end_id)
                        in_thinking_mode = False
                        current_temperature = cfg.output_temperature
                        token_id = cfg.thinking_end_id

                if token_id == cfg.thinking_end_id and in_thinking_mode:
                    in_thinking_mode = False
                    current_temperature = cfg.output_temperature

            # ── Step the KV cache forward by one token ───────────────────
            # Feed ONLY the new token (not the whole sequence!)
            # The model uses kv_caches for all previous context
            next_input = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
            output = self.model(next_input, kv_caches=kv_caches)
            kv_caches = output["kv_caches"]   # Updated with new token's K, V
            next_logits = output["logits"][0, -1, :]

        return GenerationOutput(
            token_ids=generated_ids,
            thinking_tokens=thinking_token_count,
            total_tokens=len(generated_ids),
            finished=len(generated_ids) > 0 and generated_ids[-1] == cfg.eos_token_id,
        )
```

---

## 6. Recommended Sampling Configs by Task

```python
# Factual Q&A
GenerationConfig(temperature=0.3, top_p=0.9, repetition_penalty=1.1)

# Code generation
GenerationConfig(temperature=0.1, top_p=1.0, repetition_penalty=1.0)

# Creative writing
GenerationConfig(temperature=0.9, top_p=0.95, top_k=50, repetition_penalty=1.2)

# Reasoning with thinking mode
GenerationConfig(
    temperature=0.3, top_p=0.9,
    enable_thinking=True,
    thinking_temperature=0.6,
    output_temperature=0.3,
    max_thinking_tokens=1024,
)
```

---

**Next:** [22 — Speculative Decoding →](22-speculative-decoding.md)
