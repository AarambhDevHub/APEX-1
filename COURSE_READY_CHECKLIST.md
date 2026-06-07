
# APEX-1 v3.0.0 Course-Ready Checklist

Use this checklist before tagging `v3.0.0`.

## Version

- [ ] `pyproject.toml` version is `3.0.0`
- [ ] `apex/__init__.py` version is `3.0.0`
- [ ] README title says `v3.0.0`
- [ ] CHANGELOG has `v3.0.0 — Course-Ready Stable Release`

## Documentation

- [ ] README curriculum says 37 lessons
- [ ] README version history is consistent
- [ ] DoRA / QDoRA is marked as v2.8.0
- [ ] Adapter-DPO is marked as v2.9.0
- [ ] v3.0.0 is marked as course-ready stable
- [ ] README explains that APEX-1 does not ship with a large pretrained checkpoint
- [ ] `MODEL_CARD.md` is present
- [ ] `docs/38-course-ready-release.md` is present
- [ ] No `private planning files` is added to the repo

## Stability Infrastructure

- [ ] `.github/workflows/ci.yml` is present
- [ ] `.github/pull_request_template.md` is present
- [ ] `scripts/course_ready_check.py` is present
- [ ] Makefile has `course-check`
- [ ] Makefile has `demo-all`
- [ ] Dockerfile does not copy missing `requirements.txt`

## Examples

- [ ] `python examples/forward_pass_demo.py`
- [ ] `python examples/generation_demo.py`
- [ ] `python examples/thinking_mode_demo.py`
- [ ] `python examples/mask_visualization.py`
- [ ] `python examples/vision_forward_demo.py`
- [ ] `python examples/lora_finetune_demo.py`
- [ ] `python examples/lora_generation_demo.py`
- [ ] `python examples/qlora_finetune_demo.py`
- [ ] `python examples/dora_finetune_demo.py`
- [ ] `python examples/adapter_dpo_demo.py`

## Tests

- [ ] `pytest tests/ -v`
- [ ] LoRA tests pass
- [ ] LoRA inference tests pass
- [ ] QLoRA tests pass
- [ ] DoRA tests pass
- [ ] Adapter-DPO tests pass
- [ ] Vision tests pass

## Course Ready Commands

- [ ] `python scripts/course_ready_check.py --mode quick`
- [ ] `python scripts/course_ready_check.py --mode examples`
- [ ] `python scripts/course_ready_check.py --mode tests`
- [ ] `python scripts/course_ready_check.py --mode full`

## Release

- [ ] `git diff` reviewed
- [ ] No generated outputs committed
- [ ] CHANGELOG updated
- [ ] GitHub release notes prepared
- [ ] PR opened and CI passing
- [ ] Tag created only after tests pass: `v3.0.0`
