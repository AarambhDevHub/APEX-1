# 23 — Thinking Mode: Built-In Reasoning Scratchpad

> **Difficulty:** ⭐⭐☆☆☆ Intermediate  
> **Source file:** `apex/generation/generator.py`  
> **You will learn:** What chain-of-thought is, how APEX-1 implements a thinking budget, BUG-21 (start token consuming budget), and BUG-14 (thinking token types).

---

## 1. What Is Chain-of-Thought?

When humans solve hard problems, we think out loud: "Let me see... first calculate 17×23... that's 17×20=340 plus 17×3=51... so 391." This intermediate reasoning is called **chain-of-thought**.

Models that generate this reasoning before giving a final answer are significantly more accurate on complex tasks — math, coding, multi-step logic.

**APEX-1's approach:** Instead of producing reasoning as visible output, it has a dedicated **thinking scratchpad** enclosed in special tokens:

```
<|thinking|>
Let me work through this step by step.
17 × 23: 17 × 20 = 340, 17 × 3 = 51, total = 391
<|/thinking|>
The answer is 391.
```

The thinking section can be hidden from the user if desired.

---

## 2. The Thinking Budget

Unlimited thinking would be wasteful for simple questions. APEX-1 enforces a **budget**: `max_thinking_tokens`. Once the thinking section reaches this limit, a `<|/thinking|>` token is automatically injected, ending the scratchpad.

---

## 3. Temperature Switching

Different temperatures work better for different phases:

| Phase | Temperature | Why |
|---|---|---|
| Standard output | 0.3 | Focused, factual |
| Thinking | 0.6 | More exploratory — try different approaches |
| Final answer | 0.3 | Focused conclusion |

Temperature is automatically switched when thinking tokens appear.

---

## 4. BUG-21: The Start Token Budget Bug

The original code:

```python
if token_id == cfg.thinking_start_id:
    in_thinking_mode = True
    thinking_token_count += 1   # BUG: start token consumes 1 budget slot
    current_temperature = cfg.thinking_temperature
```

This wasted 1 budget slot on the `<|thinking|>` token itself — which should not count as thinking content.

**Fix:** Set the flag and temperature first, then only count tokens that appear **while already in thinking mode**:

```python
if token_id == cfg.thinking_start_id:
    # BUG-21 FIX: set mode BEFORE incrementing counter
    in_thinking_mode = True
    current_temperature = cfg.thinking_temperature
    # Do NOT increment thinking_token_count here

elif in_thinking_mode:
    # Only count actual thinking content tokens
    thinking_token_count += 1
```

---

## 5. BUG-14: Thinking Tokens Need Type=2 for SFT

During SFT training, the `get_token_types()` function assigns:
- 0 = system
- 1 = user  
- 2 = assistant

The `<|thinking|>` and `<|/thinking|>` special tokens mark the assistant's reasoning. They should always be labelled as type 2 (assistant), so the SFT loss trains on them.

The original code tracked `current_type` by role markers. If thinking tokens appeared before an explicit `<|assistant|>` token in some edge cases, they inherited the wrong type and were excluded from the loss.

**Fix:** Explicitly force thinking delimiter tokens to type 2:

```python
elif tid in (THINKING_START_ID, THINKING_END_ID):
    types.append(2)   # Always type 2, regardless of current_type
    continue
```

---

## 6. Thinking Mode in the Generation Loop

```python
# From apex/generation/generator.py — thinking mode section

for step in range(cfg.max_new_tokens):
    # ... sample next_token ...
    token_id = next_token.item()
    generated_ids.append(token_id)

    if token_id == cfg.eos_token_id:
        break

    if cfg.enable_thinking:
        if token_id == cfg.thinking_start_id:
            # BUG-21 FIX: enter thinking mode AFTER processing the start token
            # so the start token itself does NOT consume budget
            in_thinking_mode = True
            current_temperature = cfg.thinking_temperature
            logger.debug("Entered thinking mode at step %d", step)

        elif in_thinking_mode:
            # Only count tokens that are actual thinking content
            thinking_token_count += 1

            if thinking_token_count >= cfg.max_thinking_tokens:
                # Budget exhausted → force close the thinking section
                generated_ids.append(cfg.thinking_end_id)
                in_thinking_mode = False
                current_temperature = cfg.output_temperature
                logger.debug(
                    "Thinking budget exhausted (%d tokens) at step %d",
                    thinking_token_count, step,
                )
                token_id = cfg.thinking_end_id

        # Natural end of thinking
        if token_id == cfg.thinking_end_id and in_thinking_mode:
            in_thinking_mode = False
            current_temperature = cfg.output_temperature
            logger.debug("Exited thinking mode at step %d", step)

    # Feed new token to model (with KV cache)
    next_input = torch.tensor([[token_id]], device=self.device, dtype=torch.long)
    output = self.model(next_input, kv_caches=kv_caches)
    kv_caches = output["kv_caches"]
    next_logits = output["logits"][0, -1, :]
```

---

## 7. Using Thinking Mode

```python
from apex.generation.generator import APEX1Generator, GenerationConfig

gen_config = GenerationConfig(
    max_new_tokens=512,
    temperature=0.3,             # Final answer temperature
    top_p=0.9,
    enable_thinking=True,        # Turn on thinking mode
    max_thinking_tokens=512,     # Thinking budget
    thinking_temperature=0.6,    # More exploratory during thinking
    output_temperature=0.3,      # Focused during final answer
    thinking_start_id=6,         # Token ID for <|thinking|>
    thinking_end_id=7,           # Token ID for <|/thinking|>
    eos_token_id=0,
)

generator = APEX1Generator(model, gen_config)
output = generator.generate(input_ids)

# Separate thinking from final answer
all_tokens = output.token_ids
print(f"Thinking tokens used: {output.thinking_tokens}")
```

---

## 8. Impact on Model Quality

Thinking mode significantly improves performance on:

| Task Type | Improvement |
|---|---|
| Multi-step math | +15–25% accuracy |
| Code debugging | +10–20% accuracy |
| Complex reasoning | +10–15% accuracy |
| Simple factual Q&A | ~0% (overhead not worth it) |

For simple questions, disable thinking mode (`enable_thinking=False`) to avoid unnecessary compute.

---

**Next:** [24 — Reward Model →](24-reward-model.md)
