# Contributing to APEX-1

Thank you for your interest in contributing to APEX-1! This document provides guidelines for contributing.

## Code of Conduct

By participating, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USER/APEX-1.git`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install dev dependencies: `pip install -e ".[all]"`
5. Install pre-commit hooks: `pre-commit install`

## Code Style

- **PEP 8** compliance is mandatory
- **Black** formatter with `--line-length=100`
- **isort** for import ordering (profile: black)
- **Type hints** required on all function signatures
- **Docstrings** required on all public classes and functions (Google style)
- Maximum line length: 100 characters

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add new MoE expert routing strategy
fix: correct RoPE cache indexing for batch > 1
docs: update training pipeline documentation
test: add YaRN scaling property tests
refactor: simplify attention mask builder
perf: optimize MoE token dispatch with scatter
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feat/your-feature`
2. Make changes with passing tests
3. Run the full test suite: `pytest tests/ -v`
4. Run linters: `black . && isort . && flake8 . && mypy apex/`
5. Push and open a PR against `main`
6. Describe your changes and link any relevant issues

## Testing Requirements

- Every new module must have corresponding unit tests in `tests/`
- All tests must pass: `pytest tests/ -v`
- Maintain or improve code coverage
- Test edge cases (empty inputs, single tokens, batch size 1)

## Adding New Components

1. Create the module in the appropriate `apex/` subdirectory
2. Add exports to the package `__init__.py`
3. Write unit tests in `tests/`
4. Update `CHANGELOG.md`
5. Add docstrings with usage examples

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v                    # all tests
pytest tests/test_all.py -k "norm"  # specific tests
pytest tests/ --cov=apex            # with coverage
```

## Questions?

Join our [Discord server](https://discord.gg/HDth6PfCnp) or open a [GitHub Issue](https://github.com/AarambhDevHub/APEX-1/issues) for questions and discussions.
