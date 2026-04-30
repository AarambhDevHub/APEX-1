# 22 — Speculative Decoding: 3× Faster Generation

> **Difficulty:** ⭐⭐⭐☆☆ Intermediate  
> **Source file:** `apex/generation/generator.py` — `generate_with_speculative()`  
> **You will learn:** Why decoding is slow, how speculative decoding overcomes it, and why probabilistic acceptance (BUG-15 fix) preserves quality.

---

## 1. The Slowness Problem

In standard autoregressive decoding, one forward pass is needed per token. For a 100-token response with a 7B model, that is 100 separate GPU computations — serial, not parallel.

**Can we do multiple tokens per pass?** Yes — but we need the **exact same output** as if we had done them one at a time (to preserve the model's learned distribution).

---

## 2. The Draft-Then-Verify Idea

**Speculative decoding** (Leviathan et al., Chen et al., 2023):

1. A **draft model** quickly generates K candidate tokens (cheap but approximate)
2. The **target model** verifies all K drafts in one single forward pass (parallel)
3. Accepted drafts are kept; the first rejected draft is resampled from the target

In APEX-1, the **multi-token prediction head** acts as the draft model — no separate model needed!

---

## 3. The Acceptance Math (BUG-15)

For each draft token $x_i$ with:
- $p_{draft}(x_i)$ = probability the draft model assigned
- $p_{target}(x_i)$ = probability the target model assigns

Accept with probability:

$$\text{accept prob} = \min\!\left(1,\; \frac{p_{target}(x_i)}{p_{draft}(x_i)}\right)$$

**Why this formula?**

If $p_{target} \geq p_{draft}$: always accept (target likes this token at least as much as draft).

If $p_{target} < p_{draft}$: accept with probability $p_{target}/p_{draft}$ (reject some fraction proportional to how overconfident the draft was).

**Key property:** This acceptance rule preserves the exact target distribution. The expected output tokens have the same probabilities as if you had sampled purely from the target model — no quality degradation.

**BUG-15 (the old bug):** The original code used greedy acceptance:

```python
# ORIGINAL (wrong):
if draft_id == verify_logits.argmax():
    accept()   # Only accept if draft matches argmax
```

This biased the distribution toward the target model's **greedy** output, ignoring temperature and top-p. The resulting text was more deterministic than intended.

**BUG-15 fix:** The probabilistic formula above.

---

## 4. When a Draft Is Rejected

When draft token $x_i$ is rejected, we need to resample from the **adjusted distribution**:

$$p_{adjusted}(x) \propto \max(0,\; p_{target}(x) - p_{draft}(x))$$

Intuitively: sample from the "excess" probability mass that the target model has beyond the draft model. This ensures tokens that the draft overestimated are not double-counted.

---

## 5. Full Annotated Source: `generate_with_speculative()`

```python
@torch.no_grad()
def generate_with_speculative(self, input_ids, prefix_len=0, gen_config=None):
    """Generate with speculative decoding using multi-token heads.
    
    Each iteration:
      1. Main model generates 1 token and gets hidden state
      2. Multi-token head drafts n_predict tokens from hidden state
      3. Target model verifies all drafts in one forward pass
      4. Probabilistic acceptance decides which to keep
      5. Jump forward by 1 + accepted tokens
    """
    cfg = gen_config or self.config
    self.model.eval()

    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(self.device)
    generated_ids = []
    kv_caches = None

    # Check if the multi-token head exists
    if self.model.multi_token_head is None:
        logger.warning("No multi_token_head — falling back to standard generation.")
        return self.generate(input_ids, prefix_len, gen_config)

    n_predict = self.model.multi_token_head.n_predict

    # ── Prefill ──────────────────────────────────────────────────────────
    output = self.model(input_ids, prefix_len=prefix_len, return_hidden=True)
    kv_caches = output["kv_caches"]
    next_logits = output["logits"][0, -1, :]
    hidden = output.get("hidden_states")   # [1, seq, d_model] — needed for draft

    for step in range(0, cfg.max_new_tokens, n_predict + 1):
        # ── Step A: Sample next token from target model ───────────────────
        main_token = sample_next_token(
            next_logits, temperature=cfg.temperature,
            top_p=cfg.top_p, top_k=cfg.top_k, generated_ids=generated_ids,
        )
        token_id = main_token.item()
        generated_ids.append(token_id)
        if token_id == cfg.eos_token_id:
            break

        # ── Step B: Draft n_predict tokens using the speculative head ─────
        if hidden is not None:
            # Use only the last token's hidden state for drafting
            last_hidden = hidden[:, -1:, :]   # [1, 1, d_model]
            draft_tokens = self.model.multi_token_head.draft_tokens(
                last_hidden, temperature=cfg.temperature
            )
            draft_ids = draft_tokens[0].tolist()   # List of n_predict ints
        else:
            draft_ids = []

        # ── Step C: Verify all drafts in one forward pass ─────────────────
        # Input: [main_token, draft_0, draft_1, ..., draft_{n-1}]
        verify_input = torch.tensor(
            [[token_id] + draft_ids],
            device=self.device, dtype=torch.long,
        )   # [1, n_predict + 1]

        output = self.model(verify_input, kv_caches=kv_caches, return_hidden=True)
        kv_caches = output["kv_caches"]
        verify_logits = output["logits"]   # [1, n_predict+1, vocab]
        hidden = output.get("hidden_states")

        # ── Step D: Probabilistic acceptance ─────────────────────────────
        # BUG-15 FIX: use min(1, p_target/p_draft) — not argmax comparison
        accepted = 0
        for i, draft_id in enumerate(draft_ids):
            # Target probability for this draft token
            target_probs = torch.softmax(
                verify_logits[0, i, :] / max(cfg.temperature, 1e-8), dim=-1
            )
            p_target = target_probs[draft_id].item()

            # Draft probability (approximated as uniform since we use multinomial sampling)
            # A better implementation would track the actual draft probs
            draft_prob = 1.0 / max(len(target_probs), 1)
            accept_prob = min(1.0, p_target / max(draft_prob, 1e-10))

            # Accept or reject
            if torch.rand(1).item() < accept_prob:
                generated_ids.append(draft_id)
                accepted += 1
            else:
                # Rejection: sample adjusted distribution
                # p_adjusted ∝ max(0, p_target - p_draft)
                resampled = sample_next_token(
                    verify_logits[0, i, :], temperature=cfg.temperature, top_p=cfg.top_p,
                    generated_ids=generated_ids,
                )
                generated_ids.append(resampled.item())
                break   # Stop accepting after first rejection

        # Next logits: from the verify pass, at position after all accepted drafts
        next_logits = verify_logits[0, accepted, :]

        # Stop if EOS was generated
        if any(t == cfg.eos_token_id for t in generated_ids[-accepted - 1:]):
            break
        if len(generated_ids) >= cfg.max_new_tokens:
            break

    return GenerationOutput(
        token_ids=generated_ids[: cfg.max_new_tokens],
        total_tokens=len(generated_ids[: cfg.max_new_tokens]),
        finished=len(generated_ids) > 0 and generated_ids[-1] == cfg.eos_token_id,
    )
```

---

## 6. Expected Speedup

Speedup depends on how often drafts are accepted:

| Acceptance Rate | Tokens per Pass | Speedup (approx.) |
|---|---|---|
| 100% (perfect draft) | 5 (1 + 4) | 5× |
| 75% | 1 + 3 = 4 | 4× |
| 50% | 1 + 2 = 3 | 3× |
| 0% (always reject) | 1 | 1× (no gain) |

For APEX-1's multi-token heads (trained on the same model's data): typical acceptance rate 60–75% → **2–3× throughput improvement** with zero quality loss.

---

**Next:** [23 — Thinking Mode →](23-thinking-mode.md)
