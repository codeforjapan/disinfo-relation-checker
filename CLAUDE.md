# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python package called `disinfo-relation-checker` that checks if a given text is related to a topic. This is intended to be used to judge if we should include the text into our disinformation analysis. Uses modern Python tooling with uv for dependency management, nox for automation, ruff for linting/formatting, mypy for type checking, and pytest for testing with coverage.

## Architecture Overview

The application follows a layered architecture with dependency injection:

### Core Components:
- **CLI Layer** (`cli.py`): Command-line interface with subcommands for classify, validate, optimize, evaluate, register-model, list-models, ab-test-setup, ab-test-results, and monitor-performance
- **Settings Layer** (`settings.py`): Pydantic-based configuration with discriminated unions for LLM provider configs
- **LLM Abstraction** (`llm_factory.py`, `llm_providers.py`): Factory pattern for pluggable LLM providers (Mock, Ollama, extensible for others)
- **Classification Engine** (`classifier.py`): Core text classification with configurable prompt templates
- **Data Processing** (`csv_processor.py`): CSV input/output handling with structured data models
- **Optimization Engine** (`prompt_optimizer.py`, `optimization_strategies.py`): Genetic algorithm-based prompt optimization
- **Model Management** (`model_registry.py`): Model versioning and metadata storage
- **Monitoring** (`performance_monitoring.py`): Performance tracking and alerting
- **A/B Testing** (`ab_testing.py`): Model comparison framework
- **Training Data** (`training_data.py`): Dataset management and validation

### Key Design Patterns:
- **Factory Pattern**: LLM provider creation based on configuration
- **Strategy Pattern**: Pluggable optimization strategies
- **Dependency Injection**: All components accept dependencies through constructors
- **Pydantic Models**: Type-safe data validation throughout the system

## Development Commands

Primary development workflow uses nox:
- `uv run nox -s full_test` - Run tests with coverage reporting
- `uv run nox -s lint_format` - Run linting and format checking with ruff
- `uv run nox -s type_check` - Run strict type checking with mypy
- `uv run nox` - Run all sessions

Direct tool usage:
- `uv run pytest --cov=disinfo_relation_checker --cov-report=term-missing` - Run tests with coverage
- `uv run pytest tests/test_specific.py` - Run specific test file
- `uv run pytest -m e2e` - Run end-to-end tests only
- `uv run pytest -m "not e2e"` - Run unit tests only
- `uv run ruff check .` - Lint code
- `uv run ruff format .` - Format code
- `uv run mypy --strict src/disinfo_relation_checker tests` - Type check with strict settings

## Test Strategy

### Test Structure:
- **Unit Tests**: Individual component testing with mocked dependencies
- **E2E Tests**: End-to-end CLI testing (marked with `@pytest.mark.e2e`)
- **Integration Tests**: Component interaction testing

### Test Configuration:
- Tests are run with strict warning treatment (`filterwarnings = ["error"]`)
- Coverage reporting excludes test files and focuses on source code
- Pytest markers distinguish between unit and e2e tests

## Development Philosophy

Follow Test-Driven Development (TDD) practices with Outside-In approach:

### TDD Workflow:
1. **Start with E2E tests**: Write failing end-to-end tests first (marked with `@pytest.mark.e2e`)
2. **Run E2E tests**: `uv run pytest -m e2e` to see failure messages
3. **Identify next module**: Use failure messages to determine which module/component needs implementation
4. **Write unit tests with SOLID design**: Create failing unit tests that enforce SOLID principles from the start
   - Design interfaces that follow single responsibility
   - Use dependency injection in test setup
   - Mock unimplemented dependencies
5. **Run unit tests only**: `uv run pytest -m "not e2e"` to focus on unit test failures
6. **Implement module**: Write minimal code to make unit tests pass, following the SOLID design enforced by tests. After each implementation, run the nox command `uv run nox` to check if the tests, linting, formatting and type checking pass.
7. **Refactor**: Clean up the code while keeping all tests passing
   - Remove duplication
   - Improve naming and structure
   - Extract common patterns
   - Run `uv run nox` after each refactoring to ensure tests still pass
8. **Repeat**: Continue until E2E tests pass
9. **Final refactor**: Once E2E tests pass, perform final refactoring across the entire feature
   - Review the complete implementation for code quality
   - Extract shared utilities and patterns
   - Ensure consistent architecture across modules
   - Run `uv run nox` to verify all tests, linting, formatting and type checking still pass

### Design Principles:
- **SOLID principles**: Design tests to enforce Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion from the beginning
- **Test-driven design**: Use unit tests to drive interface design and enforce single responsibility
- **Dependency Injection**: Structure tests to require dependency injection for better testability
- **Mocking**: Mock external dependencies and unimplemented modules using pytest-mock during development
- **Pydantic for Data Models**: Use Pydantic BaseModel instead of dataclasses for all data structures to ensure runtime validation, automatic serialization/deserialization, and better type safety
- **Code Quality Standards**: Follow ruff linting rules strictly without ignoring them unless absolutely necessary
  - Prefer refactoring code to meet linting standards rather than adding ignore directives
  - Only ignore linting rules when there is a legitimate technical reason that cannot be resolved through refactoring
  - When ignoring rules, add detailed comments explaining why the ignore is necessary
  - Regularly review and remove unnecessary ignore directives during refactoring phases

## Development Plan

### Phase 1: CSV Processing and Classification (Current Status: CLI --version implemented)
**Goal**: Process CSV files and classify each record for disinformation analysis relevance

#### Features to implement:
1. **CSV Input Processing**
   - CSV file reader with configurable column mapping
   - Data validation and preprocessing
   - Batch processing capabilities

2. **Text Classification Engine**
   - LLM-based text classifier interface
   - Prompt template system for classification
   - Confidence scoring for predictions
   - Support for multiple classification models

3. **CLI Commands**
   - `disinfo-relation-checker classify --input file.csv --output results.csv`
   - `disinfo-relation-checker validate --labeled-data labeled.csv`
   - Configuration file support for model settings

### Phase 2: Automated Prompt Engineering
**Goal**: Automatically optimize prompts to achieve target accuracy on labeled datasets

#### Features to implement:
1. **Prompt Optimization Engine**
   - Genetic algorithm or iterative refinement for prompt optimization
   - Performance metrics tracking (accuracy, precision, recall, F1)
   - Prompt template versioning and management

2. **Training Data Management**
   - Labeled dataset loader and validator
   - Cross-validation support for robust evaluation
   - Data splitting utilities (train/validation/test)

3. **Optimization Strategies**
   - Few-shot learning prompt generation
   - Chain-of-thought reasoning integration
   - Multi-objective optimization (accuracy vs. cost)

4. **CLI Commands**
   - `disinfo-relation-checker optimize --training-data labeled.csv --target-accuracy 0.9`
   - `disinfo-relation-checker evaluate --model-config config.json --test-data test.csv`

### Phase 3: Advanced Features
**Goal**: Production-ready system with monitoring and scalability

#### Features to implement:
1. **Model Management**
   - Model registry and versioning
   - A/B testing framework for prompt variants
   - Performance monitoring and alerting

2. **Scalability and Performance**
   - Async processing for large datasets
   - Caching layer for repeated classifications
   - Rate limiting and error handling

3. **Integration and Deployment**
   - API server for real-time classification
   - Integration with existing disinformation analysis workflows
   - Docker containerization and deployment scripts

### Development Priority Order:
1. CSV processing and basic classification (Phase 1.1-1.2)
2. CLI commands for classification (Phase 1.3)
3. Prompt optimization core engine (Phase 2.1)
4. Training data management (Phase 2.2)
5. Optimization strategies (Phase 2.3)
6. Advanced CLI commands (Phase 2.4)
7. Production features (Phase 3)

### Architecture Principles:
- **Modular Design**: Each feature should be independently testable and replaceable
- **Plugin Architecture**: Support for different LLM providers and optimization strategies
- **Configuration-Driven**: All behavior configurable through files and CLI arguments
- **Observability**: Comprehensive logging and metrics for all operations

## LLM Provider Configuration

### Factory Pattern Implementation
Use Factory Pattern with discriminated union for type-safe LLM provider configuration using pydantic-settings.

### Supported Providers:
1. **MockLLMProvider** (default): Keyword-based heuristic classification
2. **OllamaProvider**: Local Ollama server integration
3. **Future providers**: OpenAI, Anthropic, Azure OpenAI, etc.

### Configuration Structure:
```python
# Settings using pydantic-settings with discriminated union
class MockLLMConfig(BaseModel):
    provider_type: Literal["mock"] = "mock"

class OllamaConfig(BaseModel):
    provider_type: Literal["ollama"] = "ollama"
    base_url: str = "http://localhost:11434"
    model: str = "gemma3n:e4b"
    timeout: int = 30

LLMConfig = Annotated[Union[MockLLMConfig, OllamaConfig], Field(discriminator="provider_type")]
```

### Configuration File Example:
```yaml
# config.yaml
llm:
  provider_type: "ollama"
  base_url: "http://localhost:11434"
  model: "gemma3n:e4b"
  timeout: 60
```

### Environment Variables:
```bash
# For Ollama
LLM_PROVIDER_TYPE=ollama
LLM_BASE_URL=http://localhost:11434
LLM_MODEL=gemma3n:e4b
LLM_TIMEOUT=60

# For Mock (default)
LLM_PROVIDER_TYPE=mock
```

### CLI Integration:
```bash
# Use config file
disinfo-relation-checker classify --config config.yaml --input file.csv --output results.csv

# Use environment variables
LLM_PROVIDER_TYPE=ollama disinfo-relation-checker validate --labeled-data labeled.csv
```

## CLI Usage Examples

### Basic Classification:
```bash
# Classify texts in CSV file
disinfo-relation-checker classify --input data.csv --output results.csv

# Validate model on labeled data
disinfo-relation-checker validate --labeled-data labeled.csv --config config.yaml
```

### Advanced Features:
```bash
# Optimize prompts with genetic algorithm
disinfo-relation-checker optimize --training-data train.csv --target-accuracy 0.9 --max-iterations 20 --output optimized.json

# Evaluate specific model configuration
disinfo-relation-checker evaluate --model-config model.json --test-data test.csv --output evaluation.json

# Model registry operations
disinfo-relation-checker register-model --model-config model.json --model-name "classifier-v1" --version "1.0.0" --description "Initial classifier"
disinfo-relation-checker list-models

# A/B testing
disinfo-relation-checker ab-test-setup --model-a "classifier-v1:1.0.0" --model-b "classifier-v2:1.0.0" --test-data test.csv --traffic-split 50 --test-name "v1-vs-v2"
disinfo-relation-checker ab-test-results --test-name "v1-vs-v2"

# Performance monitoring
disinfo-relation-checker monitor-performance --model-name "classifier-v1" --time-range "7d"
```

### Implementation Requirements:
1. **LLMProviderFactory**: Create providers based on discriminated config
2. **Type Safety**: Full type checking for all provider configurations
3. **Validation**: pydantic validation for all settings
4. **Error Handling**: Graceful degradation if provider unavailable
5. **Testing**: Mock all external dependencies in unit tests
6. **Documentation**: Clear examples for each provider type

### Development Dependencies:
Add to pyproject.toml dev dependencies:
- `pydantic-settings>=2.0.0`
- `httpx>=0.24.0` (for Ollama HTTP client)
- `pyyaml>=6.0` (for YAML config files)
