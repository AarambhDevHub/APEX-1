
## Summary

<!-- Explain what this PR changes and why. -->

## Type of change

- [ ] Bug fix
- [ ] Documentation update
- [ ] Course-ready stability update
- [ ] Model / training feature
- [ ] Test / CI update

## Local verification

- [ ] `python scripts/course_ready_check.py --mode quick`
- [ ] `python scripts/course_ready_check.py --mode examples`
- [ ] `pytest tests/ -v`

## APEX-1 course stability checklist

- [ ] README still explains that APEX-1 is educational and does not ship with a large pretrained checkpoint.
- [ ] New commands are CPU-friendly where possible.
- [ ] No large checkpoints, datasets, or generated outputs are committed.
- [ ] Version references are consistent.
- [ ] Changelog is updated.
