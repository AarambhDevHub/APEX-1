# 26 — GRPO: Group Relative Policy Optimization

> **Difficulty:** ⭐⭐⭐⭐☆ Advanced  
> **Source file:** `apex/alignment/grpo.py`  
> **You will learn:** What RL training is, why GRPO simplifies PPO, the advantage calculation, PPO-clip loss, and BUG-04.

---

## 1. What Is Reinforcement Learning for LLMs?

In RL, the model is a **policy** $\pi$ that takes a state (the prompt) and produces an action (a response). A reward signal tells the model how good its action was.

The goal: adjust the policy to maximise expected reward while staying close to the reference (SFT) model.

$$\max_\pi \mathbb{E}_{x \sim D,\, y \sim \pi(\cdot|x)}\!\left[r(x, y)\right] - \beta \cdot D_{KL}(\pi \| \pi_{ref})$$

The KL term prevents the model from "hacking" the reward by drifting arbitrarily far from the reference.

---

## 2. Why GRPO Instead of PPO?

**PPO** (Proximal Policy Optimization) is the standard RL algorithm for LLMs. It requires:
- A **value function** $V(x)$ — a separate neural network that estimates how good a state is
- Actor-critic architecture (two models running simultaneously)

This doubles training complexity and memory usage.

**GRPO** (DeepSeek-R1) eliminates the value function by using **group-relative advantages**: instead of estimating absolute value, compare responses within a group.

---

## 3. GRPO Algorithm

**Step 1: For each prompt, sample G responses**

$$y_1, y_2, \ldots, y_G \sim \pi(\cdot | x), \quad G = 8$$

**Step 2: Score each response**

$$r_i = r(x, y_i) \in [0, 1]$$

**Step 3: Normalise rewards to get advantages**

$$\hat{A}_i = \frac{r_i - \text{mean}(r_1, \ldots, r_G)}{\text{std}(r_1, \ldots, r_G) + \varepsilon}$$

This is group-relative: an advantage of +1 means this response was 1 standard deviation better than the group average.

**Step 4: Compute PPO-clip loss for each response**

$$\rho_i = \frac{\pi(y_i|x)}{\pi_{ref}(y_i|x)} \quad \text{(ratio of new to reference log-probs)}$$

$$L_i = -\min\!\left(\rho_i \hat{A}_i,\; \text{clip}(\rho_i, 1-\varepsilon, 1+\varepsilon) \hat{A}_i\right) + \beta \cdot (\log\pi - \log\pi_{ref})$$

**Step 5: Mean over the group**

$$L = \frac{1}{G} \sum_{i=1}^G L_i$$

---

## 4. Understanding PPO-Clip

The clip objective is the core of PPO:

$$\min\!\left(\rho \hat{A},\; \text{clip}(\rho, 1-\varepsilon, 1+\varepsilon)\hat{A}\right)$$

**Case 1: Positive advantage ($\hat{A} > 0$)**
- We want to increase $\rho$ (make this response more likely)
- But clip at $1 + \varepsilon$ (don't increase by more than 20%)
- Prevents overfitting to a single "lucky" response

**Case 2: Negative advantage ($\hat{A} < 0$)**
- We want to decrease $\rho$ (make this response less likely)
- Clip at $1 - \varepsilon$
- Prevents completely suppressing a response in one step

---

## 5. BUG-04: Broken Generation Loop

The original GRPO rollout used a manual token-by-token loop:

```python
# ORIGINAL (broken):
for _ in range(G):
    ids = prompt_ids
    response = []
    for step in range(max_new_tokens):
        logits = model(ids)["logits"][:, -1, :]   # BUG: always last position
        ids = torch.cat([ids, torch.argmax(logits, -1, keepdim=True)], dim=1)
        # BUT: model was called without KV cache → O(n²) and slow
        # AND: logits were always from position -1, but position tracking was wrong
        response.append(...)
```

Problems:
1. No KV cache → exponentially slow
2. Position tracking was broken → wrong output
3. Greedy decoding → collapsed diversity (all G responses are identical)

**Fix:** Use `APEX1Generator` for all rollouts:

```python
# FIXED: uses proper KV cache + sampling
generator = APEX1Generator(model, rollout_cfg)
for _ in range(G):
    output = generator.generate(prompt_ids.to(device))
    response_ids_list.append(torch.tensor([output.token_ids], device=device))
```

---

## 6. Full Annotated Source: `apex/alignment/grpo.py`

```python
"""
GRPO — Group Relative Policy Optimization.

BUG-04 FIX: Uses APEX1Generator for rollout generation.
"""

def grpo_training_step(
    model, reference_model, optimizer,
    prompt_ids, response_ids_list, rewards,
    prompt_len, beta=0.04, clip_eps=0.2, max_grad_norm=1.0,
):
    """Execute one GRPO training step.
    
    Args:
        model:             Policy model being trained.
        reference_model:   Frozen SFT model (reference).
        optimizer:         Policy optimizer.
        prompt_ids:        Prompt token IDs [1, prompt_len].
        response_ids_list: List of G response tensors.
        rewards:           Reward for each response [G].
        prompt_len:        Length of the prompt.
        beta:              KL penalty coefficient.
        clip_eps:          PPO clipping epsilon.
    
    Returns:
        (loss_value, metrics_dict)
    """
    device = next(model.parameters()).device

    # ── Step 1: Compute group-relative advantages ──────────────────────
    group_mean = rewards.mean()
    group_std = rewards.std().clamp(min=1e-6)   # Prevent div by zero
    # Advantages: normalised so mean=0, std=1 within the group
    advantages = (rewards - group_mean) / group_std

    all_losses = []
    all_kl = []
    all_ratios = []

    for i, response_ids in enumerate(response_ids_list):
        advantage = advantages[i]   # Scalar advantage for this response

        if response_ids.dim() == 1:
            response_ids = response_ids.unsqueeze(0)

        # Concatenate prompt + response for log-prob computation
        full_ids = torch.cat([prompt_ids, response_ids], dim=1).to(device)

        # ── Log-prob from policy (gradients flow through) ────────────────
        log_pi = compute_sequence_log_prob(model, full_ids, prompt_len)

        # ── Log-prob from reference (no gradient) ────────────────────────
        with torch.no_grad():
            log_ref = compute_sequence_log_prob(reference_model, full_ids, prompt_len)

        # ── KL divergence term ────────────────────────────────────────────
        # Approximate KL: log π - log π_ref (sequence-level)
        kl_div = log_pi - log_ref
        all_kl.append(kl_div.item())

        # ── PPO ratio: how much has policy changed from reference? ────────
        # ratio = π(y|x) / π_ref(y|x) = exp(log π - log π_ref)
        ratio = torch.exp(log_pi - log_ref.detach())
        all_ratios.append(ratio.item())

        # ── PPO-clip objective ────────────────────────────────────────────
        # Normal term: ratio × advantage
        l_normal = ratio * advantage
        # Clipped term: clip(ratio, 1-ε, 1+ε) × advantage
        l_clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        # Take the minimum (conservative update)
        l_clip = torch.min(l_normal, l_clipped)

        # Final loss = -(policy improvement) + (KL penalty)
        # Negative because we minimise loss but want to MAXIMISE reward
        loss = -(l_clip - beta * kl_div)
        all_losses.append(loss)

    # ── Backprop over all G responses ────────────────────────────────────
    total_loss = torch.stack(all_losses).mean()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    optimizer.zero_grad()

    metrics = {
        "grpo_loss": total_loss.item(),
        "mean_reward": rewards.mean().item(),
        "reward_std": rewards.std().item(),
        "mean_kl": sum(all_kl) / len(all_kl),
        "mean_ratio": sum(all_ratios) / len(all_ratios),
        "advantage_max": advantages.max().item(),
        "advantage_min": advantages.min().item(),
    }
    return total_loss.item(), metrics


def grpo_full_loop(model, reference_model, optimizer, prompts, reward_fn, G=8, ...):
    """Full GRPO loop over a batch of prompts.
    
    BUG-04 FIX: Uses APEX1Generator for rollout generation
    instead of the broken manual loop.
    """
    from apex.generation.generator import APEX1Generator, GenerationConfig

    device = next(model.parameters()).device
    rollout_cfg = GenerationConfig(max_new_tokens=128, temperature=0.7, top_p=0.95)
    all_metrics = []

    for prompt_ids in prompts:
        if prompt_ids.dim() == 1:
            prompt_ids = prompt_ids.unsqueeze(0)
        prompt_len = prompt_ids.shape[1]

        # BUG-04 FIX: Generate G responses with proper generator
        generator = APEX1Generator(model, rollout_cfg)
        response_ids_list = []

        model.eval()
        with torch.no_grad():
            for _ in range(G):
                output = generator.generate(prompt_ids.to(device))
                resp = torch.tensor([output.token_ids], device=device, dtype=torch.long)
                response_ids_list.append(resp)
        model.train()

        # Score each response
        rewards_list = [float(reward_fn(prompt_ids, resp)) for resp in response_ids_list]
        rewards_tensor = torch.tensor(rewards_list, device=device)

        # GRPO training step
        loss, metrics = grpo_training_step(
            model, reference_model, optimizer,
            prompt_ids.to(device), response_ids_list, rewards_tensor,
            prompt_len,
        )
        all_metrics.append(metrics)

    # Aggregate metrics
    if all_metrics:
        agg = {}
        for key in all_metrics[0]:
            agg[key] = sum(m[key] for m in all_metrics) / len(all_metrics)
        return agg
    return {"grpo_loss": 0.0}
```

---

## 7. GRPO vs PPO Summary

| Aspect | PPO | GRPO |
|---|---|---|
| Value function | Yes (extra neural network) | No (group-relative) |
| Memory | 3× (policy + reference + value) | 2× (policy + reference) |
| Stability | Good with tuning | Very stable |
| Advantage estimation | GAE (complex) | Group mean subtraction (simple) |
| DeepSeek-R1 uses | ✓ | |

---

**Next:** [27 — Process Reward Model →](27-process-reward-model.md)
