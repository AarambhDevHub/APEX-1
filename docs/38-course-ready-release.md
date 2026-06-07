
# 38 — Course-Ready Stable Release

> **Difficulty:** ⭐☆☆☆☆ Beginner  
> **Purpose:** Explain what changed in APEX-1 v3.0.0 and how to verify the repo before using it for the full course.

---

## What Is v3.0.0?

APEX-1 v3.0.0 is the **course-ready stable release**.

This release does not try to add a new model architecture. Instead, it makes the repository easier to teach, test, verify, and maintain.

The goal is simple:

```txt
clone repo -> install -> run demos -> run tests -> follow lessons
```

---

## What v3.0.0 Adds

APEX-1 v3.0.0 adds course-stability infrastructure:

- `MODEL_CARD.md`
- `COURSE_READY_CHECKLIST.md`
- `scripts/course_ready_check.py`
- `.github/workflows/ci.yml`
- `.github/pull_request_template.md`
- updated `Makefile` commands
- fixed Dockerfile setup
- clearer README version history
- clearer limitation notes

---

## Why This Matters

A course repository must be more stable than an experimental repo.

Students will run the commands from the README. If demos, tests, or version references are broken, learners lose trust.

v3.0.0 is designed to be the stable base for teaching:

- language model basics,
- tokenizer,
- embeddings,
- RoPE,
- attention,
- FFN / MoE,
- full model assembly,
- training,
- generation,
- vision,
- LoRA,
- QLoRA,
- DoRA,
- adapter-DPO,
- and real ML engineering bugs.

---

## Course-Ready Checker

Run:

```bash
python scripts/course_ready_check.py --mode quick
```

This verifies:

- required files,
- required docs,
- version consistency,
- README course references,
- import sanity.

For examples:

```bash
python scripts/course_ready_check.py --mode examples
```

For tests:

```bash
python scripts/course_ready_check.py --mode tests
```

For everything:

```bash
python scripts/course_ready_check.py --mode full
```

The report is saved to:

```txt
outputs/course_ready_report.json
```

---

## Important Learning Note

APEX-1 is an educational from-scratch LLM + VLM architecture.

It does not ship with a large pretrained checkpoint. Tiny CPU demos prove the mechanics and architecture, but real high-quality generation requires real training data, large compute, trained checkpoints, and evaluation.

That honesty is part of the course.

---

## Recommended Release Flow

```bash
python scripts/course_ready_check.py --mode quick
python scripts/course_ready_check.py --mode examples
pytest tests/ -v
git diff
```

Then open a PR.

Only tag v3.0.0 after the PR passes CI and local tests.
