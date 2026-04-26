.PHONY: install test lint format clean docs help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install package in dev mode with all dependencies
	pip install -e ".[all]"

test: ## Run all tests
	pytest tests/ -v --tb=short

test-cov: ## Run tests with coverage report
	pytest tests/ -v --cov=apex --cov-report=term-missing --cov-report=html

lint: ## Run all linters
	black --check --line-length=100 apex/ tests/
	isort --check --profile=black --line-length=100 apex/ tests/
	flake8 apex/ tests/ --max-line-length=100 --extend-ignore=E203,W503
	mypy apex/ --ignore-missing-imports

format: ## Auto-format code
	black --line-length=100 apex/ tests/ examples/
	isort --profile=black --line-length=100 apex/ tests/ examples/

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ htmlcov/ .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

demo: ## Run all example scripts
	python examples/forward_pass_demo.py
	python examples/generation_demo.py
	python examples/thinking_mode_demo.py
	python examples/mask_visualization.py

train-tokenizer: ## Train tokenizer from raw text
	python -m apex.tokenizer.train_tokenizer --output tokenizer/apex1_tokenizer.json

docker-build: ## Build Docker image
	docker build -t apex1:latest .

docker-run: ## Run Docker container
	docker run --gpus all -it apex1:latest
