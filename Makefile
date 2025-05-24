.PHONY: help test lint format type-check install clean all check

# Default target
help:
	@echo "Available commands:"
	@echo "  install     - Install dependencies"
	@echo "  format      - Format code with ruff"
	@echo "  lint        - Lint code with ruff"
	@echo "  type-check  - Type check with mypy"
	@echo "  test        - Run tests with pytest"
	@echo "  check       - Run all checks (lint + type-check + test)"
	@echo "  all         - Format + check"
	@echo "  clean       - Clean cache files"

install:
	uv pip install -e ".[dev]"

format:
	@echo "🎨 Formatting code..."
	ruff format .

lint:
	@echo "🔍 Linting code..."
	ruff check . --fix

type-check:
	@echo "🏷️  Type checking..."
	mypy .

test:
	@echo "🧪 Running tests..."
	pytest -v

# Run all checks without formatting
check: lint type-check test
	@echo "✅ All checks passed!"

# Format first, then run all checks
all: format check
	@echo "🚀 All tasks completed!"

clean:
	@echo "🧹 Cleaning up..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
