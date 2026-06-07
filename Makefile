
.PHONY: help install install-core test test-cov test-peft test-vision test-quick lint format clean demo demo-all course-check course-check-examples course-check-tests course-check-full train-tokenizer docker-build docker-run

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-24s\033[0m %s\n", $$1, $$2}'

install-core: ## Install package in editable mode with core + vision + dev dependencies
	pip install -e ".[dev,vision]"

install: ## Install package in editable mode with all dependencies
	pip install -e ".[all]"

test: ## Run all tests
	pytest tests/ -v --tb=short

test-quick: ## Run a lightweight course-ready static check
	python scripts/course_ready_check.py --mode quick

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=apex --cov-report=term-missing --cov-report=html

test-peft: ## Run PEFT-related tests
	pytest tests/test_lora_peft.py tests/test_lora_inference.py tests/test_qlora.py tests/test_dora.py tests/test_adapter_dpo.py -v --tb=short

test-vision: ## Run vision tests
	pytest tests/test_vision.py -v --tb=short

lint: ## Run all linters
	black --check --line-length=100 apex/ tests/ examples/ scripts/
	isort --check --profile=black --line-length=100 apex/ tests/ examples/ scripts/
	flake8 apex/ tests/ examples/ scripts/ --max-line-length=100 --extend-ignore=E203,W503
	mypy apex/ --ignore-missing-imports

format: ## Auto-format code
	black --line-length=100 apex/ tests/ examples/ scripts/
	isort --profile=black --line-length=100 apex/ tests/ examples/ scripts/

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
	rm -rf outputs/course_ready_report.json
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo: ## Run core text example scripts
	python examples/forward_pass_demo.py
	python examples/generation_demo.py
	python examples/thinking_mode_demo.py
	python examples/mask_visualization.py

demo-all: ## Run all course demo scripts
	python examples/forward_pass_demo.py
	python examples/generation_demo.py
	python examples/thinking_mode_demo.py
	python examples/mask_visualization.py
	python examples/vision_forward_demo.py
	python examples/lora_finetune_demo.py
	python examples/lora_generation_demo.py
	python examples/qlora_finetune_demo.py
	python examples/dora_finetune_demo.py
	python examples/adapter_dpo_demo.py

course-check: ## Run quick course-ready static checks
	python scripts/course_ready_check.py --mode quick

course-check-examples: ## Run quick checks and all examples
	python scripts/course_ready_check.py --mode examples

course-check-tests: ## Run quick checks and pytest
	python scripts/course_ready_check.py --mode tests

course-check-full: ## Run quick checks, all examples, and pytest
	python scripts/course_ready_check.py --mode full

train-tokenizer: ## Train tokenizer from raw text
	python -m apex.tokenizer.train_tokenizer --output tokenizer/apex1_tokenizer.json

docker-build: ## Build Docker image
	docker build -t apex1:latest .

docker-run: ## Run Docker container
	docker run --rm -it apex1:latest
