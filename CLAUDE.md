# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A Python package called `disinfo-relation-checker` that checks if a given text is related to a topic. This is intended to be used to judge if we should include the text into our disinformation analysis.Uses modern Python tooling with uv for dependency management, nox for automation, ruff for linting/formatting, mypy for type checking, and pytest for testing with coverage.

## Development Commands

Primary development workflow uses nox:
- `uv run nox -s full_test` - Run tests with coverage reporting
- `uv run nox -s lint_format` - Run linting and format checking with ruff
- `uv run nox -s type_check` - Run strict type checking with mypy
- `uv run nox` - Run all sessions

Direct tool usage:
- `uv run pytest --cov=disinfo_relation_checker --cov-report=term-missing` - Run tests with coverage
- `uv run pytest tests/test_specific.py` - Run specific test file
- `uv run ruff check .` - Lint code
- `uv run ruff format .` - Format code
- `uv run mypy --strict src/disinfo_relation_checker tests` - Type check with strict settings

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
7. **Repeat**: Continue until E2E tests pass

### Design Principles:
- **SOLID principles**: Design tests to enforce Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion from the beginning
- **Test-driven design**: Use unit tests to drive interface design and enforce single responsibility
- **Dependency Injection**: Structure tests to require dependency injection for better testability
- **Mocking**: Mock external dependencies and unimplemented modules using pytest-mock during development

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
