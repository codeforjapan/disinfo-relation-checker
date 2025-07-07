# disinfo-relation-checker

A Python package that checks if a given text is related to a topic. This is intended to be used to judge if we should include the text into our disinformation analysis.

## Installation

```bash
pip install git+https://github.com/codeforjapan/disinfo-relation-checker
```

For development:

```bash
git clone https://github.com/codeforjapan/disinfo-relation-checker.git
cd disinfo-relation-checker
uv sync --group dev
```

## Usage

```python
import disinfo_relation_checker as drc

# Check if text is related to a topic
# TODO: Add usage examples once the API is implemented
```

## Development

This project uses modern Python tooling:

- **uv** for dependency management
- **nox** for automation
- **ruff** for linting and formatting
- **mypy** for type checking
- **pytest** for testing with coverage

### Running Tests

```bash
# Run all tests with coverage
uv run nox -s full_test

# Run specific test file
uv run pytest tests/test_specific.py

# Run tests with coverage directly
uv run pytest --cov=disinfo_relation_checker --cov-report=term-missing
```

### Code Quality

```bash
# Run linting and format checking
uv run nox -s lint_format

# Format code
uv run ruff format .

# Lint code
uv run ruff check .
```

### Type Checking

```bash
# Run type checking
uv run nox -s type_check

# Run mypy directly
uv run mypy --strict src/disinfo_relation_checker tests
```

### Run All Checks

```bash
# Run all nox sessions (tests, linting, type checking)
uv run nox
```

## Requirements

- Python 3.11+

## License

MIT
