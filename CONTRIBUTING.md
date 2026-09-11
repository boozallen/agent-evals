# Contributing to agent-evals

Thank you for your interest in contributing to agent-evals! This document provides guidelines and information for contributors working on the library codebase.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Project Structure](#project-structure)
- [Architecture Overview](#architecture-overview)
- [Extending the Framework](#extending-the-framework)
- [Creating Example Files](#creating-example-files)
- [Testing & Quality](#testing--quality)
- [Architectural Governance](#architectural-governance)
- [Pull Request Process](#pull-request-process)

---

## Getting Started

### Prerequisites

- Python 3.14+
- [uv](https://docs.astral.sh/uv/) package manager
- Git

### Developer Setup

Fork the repository on GitHub, then clone your fork and install all dependencies:

```bash
# Fork via GitHub UI, then clone your fork
# (copy the clone URL from your fork's "Code" button)
git clone <your-fork-clone-url>
cd agent-evals
git checkout develop  # feature work branches off develop, not main

# Add upstream remote for syncing
git remote add upstream https://github.com/boozallen/agent-evals.git

# Install all dependencies (including the dev group and all extras)
uv sync --all-extras

# Install pre-commit hooks
uv run pre-commit install

# Run tests to verify setup
uv run pytest
```

### Installation for Users

```bash
# Standard installation (includes braintrust, autoevals, agentevals)
# Pinned release (recommended for production)
pip install "agent-evals @ git+https://github.com/boozallen/agent-evals@v1.0.0"

# Or with uv
uv add "agent-evals @ git+https://github.com/boozallen/agent-evals@v1.0.0"

# Straight from source, to track an unreleased change
uv add "agent-evals @ git+https://github.com/boozallen/agent-evals@develop"
```

**Available extras**: `langchain` and `strands` (each pulls in that framework so its message converter can be used)

**Note:** Core dependencies (braintrust, autoevals, agentevals) are always included. Dev dependencies (pytest, ruff, etc.) are NOT needed for library users - only for contributors.

---

## Project Structure

```
src/agent_evals/
├── core/              # Evaluation orchestration (run_eval, Score, EvalRunner)
│   └── ports/         # Protocol definitions (Platform, Scorer)
├── adapters/          # All adapters (hexagonal secondary ports)
│   ├── platforms/     # Platform adapters (braintrust, mlflow, langfuse, local)
│   └── scorers/       # Scorer adapters (agentevals, autoevals)
└── utils/             # Shared helpers

tests/
├── unit/          # Fast unit tests (no external dependencies)
├── integration/   # Integration tests (may require services)
└── validation/    # Contract validation tests

docs/
├── architecture.md       # Complete architectural design
├── new-adapter-dev-guide.md
├── platforms.md
├── scorers.md
└── specs/                # Historical feature specifications and designs
    └── 00X-feature-name/
```

---

## Architecture Overview

This library uses **hexagonal architecture** (also known as ports and adapters):

- **Core** defines port interfaces, validates inputs, and dispatches to adapters (`run_eval`, `EvalRunner`)
- **Adapters** implement Core's Protocol interfaces and connect to external systems (two types):
  - **Platform Adapters:** Connect to evaluation platforms (Braintrust, MLflow, LangFuse, Local)
  - **Scorer Adapters:** Connect to scoring libraries (AgentEvals, Autoevals, Custom)

### Key Principles

1. **Core Independence**: Core never imports from adapters or scorers
2. **Protocol-Driven**: Use `typing.Protocol` for interfaces, not ABC
3. **Context Extensibility**: Pass data through `**context` parameter
4. **Adapter Interface**: Protocol-based
5. **Strategy-Based Execution**: Adapters control evaluation strategy; core validates and dispatches
6. **Async-First**: `run_eval_async` is primary, `run_eval` is wrapper
7. **Pydantic Types**: All core types use Pydantic v2 for validation (see [types.py](src/agent_evals/core/types.py))
8. **Zero Breaking Changes**: Public APIs are sacred

**Read More**: [Architecture Document](docs/foundry/guides/architecture.md) - Complete design, diagrams, and architectural decisions

---

## Extending the Framework

The agent-evals library is designed for easy extensibility through two types of adapters:

- **Platform Adapters**: Control WHERE and HOW evaluations run (Braintrust, MLflow, LangFuse, Local)
- **Scorer Adapters**: Define WHAT metrics to evaluate (AgentEvals, Autoevals, Custom scorers)

Both adapter types use Protocol interfaces for seamless integration without inheritance requirements.

### Quick Start

**Platform Adapters:**
- Implement the `Platform` Protocol with `evaluate()`, `aevaluate()`, and `pull_traces()` methods
- Register with `@register_adapter("platform_name")`
- Handle data conversion between unified types and platform-specific formats

**Scorer Adapters:**
- Create callables that accept `(TaskResult, ExpectedResult)` and return `Score` objects
- No registration needed - duck typing with automatic protocol detection
- Support both sync and async scoring functions

### Reference Implementations

Study these complete examples:

**Platform Adapters:**
- [`braintrust.py`](src/agent_evals/adapters/platforms/braintrust.py) - SDK delegation pattern
- [`mlflow.py`](src/agent_evals/adapters/platforms/mlflow.py) - SDK integration with trace export
- [`langfuse.py`](src/agent_evals/adapters/platforms/langfuse.py) - Async evaluation with historical data
- [`local.py`](src/agent_evals/adapters/platforms/local.py) - Internal orchestration pattern

**Scorer Adapters:**
- [`autoevals.py`](src/agent_evals/adapters/scorers/autoevals.py) - External library wrappers (25+ scorers)
- [`agentevals.py`](src/agent_evals/adapters/scorers/agentevals.py) - Trajectory evaluation patterns

### Detailed Guide

For comprehensive implementation instructions, including:
- Complete Protocol interface documentation
- Step-by-step implementation guides
- Error handling and best practices
- Testing strategies
- Integration patterns

**See: [New Adapter Development Guide](docs/foundry/guides/new-adapter-dev-guide.md)**

This guide provides detailed instructions for implementing each method, handling edge cases, and ensuring proper integration with the evaluation pipeline.

---

## Examples Guidelines

When contributing new features to agent-evals, consider adding corresponding examples. Examples demonstrate functionality, not teach concepts. They should be scannable, focused, and practical.

---

## Testing & Quality

### Running Tests

```bash
# Run all tests
uv run pytest

# Run specific test suites
uv run pytest tests/unit/                        # Fast unit tests only
uv run pytest tests/integration/ --no-cov        # Integration tests
uv run pytest tests/validation/ --no-cov         # Contract validation tests

# Coverage is always on: `addopts` in pyproject.toml adds the terminal report
# and writes HTML into htmlcov/, so no --cov flag is needed. Its coverage gate
# is a whole-tree number, so pass --no-cov to any run narrower than the whole
# tree -- a single tier, a single file, or a -k selection. Only tests/unit/ is
# broad enough to clear the gate on its own. The just recipes already do this:
# `just test-integration` and `just test-validation` pass it for you.
```

### Code Quality

```bash
# Lint and format
uv run ruff check .      # Check for issues
uv run ruff format .     # Auto-format code

# Type check
uv run ty check          # Fast type checker

# Run all quality checks
uv run pre-commit run --all-files
```

### Writing Tests

**Test Organization**:
- `tests/unit/` - Fast, isolated unit tests (no external dependencies)
- `tests/integration/` - Integration tests (may require Braintrust, etc.)
- `tests/validation/` - Contract validation tests (ensure backward compatibility)

**Test Naming**:
- Files: `test_*.py`
- Classes: `Test*`
- Functions: `test_*`

**Dataset Format**:
All tests must use `ExampleData` instances (Pydantic models), not plain dicts:

```python
from agent_evals import ExampleData

# ✅ Correct - ExampleData instance
dataset = [
    ExampleData(
        input="What's 2+2?",
        expected="4",
        context={"user_id": "test123"}  # Optional context for scorers
    )
]

# ❌ Wrong - plain dict (will fail validation)
dataset = [{"input": "What's 2+2?", "expected": "4"}]
```

**Best Practices**:
- Unit tests should be fast (< 1s each)
- Use fixtures for common setup
- Mock external services in unit tests
- Integration tests can use real services
- Always test both sync and async paths
- Test error cases and edge cases
- When mocking Score objects, explicitly set all fields (including `reasoning=None`, `metadata={}`) to avoid Pydantic validation issues

---

## Architectural Governance

This project follows strict architectural principles (hexagonal architecture, Protocol-driven interfaces, async-first I/O, zero breaking changes to public API). See [docs/foundry/guides/architecture.md](docs/foundry/guides/architecture.md) for the full rationale.

### Architectural Compliance

All pull requests **MUST** include an architectural compliance check:

- **Core Independence**: Does this PR respect dependency boundaries?
- **Protocol-Driven**: Are new interfaces using Protocol not ABC?
- **Context Extensibility**: Are new fields added via **context?
- **Minimal Interface**: Does this expand adapter interface unnecessarily?
- **Control Flow**: Does core still own orchestration?
- **Async-First**: Are I/O operations using async/await?
- **Zero Breaking**: Is this change backward compatible?

### Breaking Changes

Breaking changes require extraordinary justification and a major version bump. Before proposing a breaking change:

1. Open an RFC (Request for Comments) issue
2. Present alternatives and their trade-offs
3. Get explicit approval from all core maintainers

---

## Pull Request Process

### Contribution Workflow

This project uses a **fork-based contribution workflow**:

1. Fork the repository on GitHub
2. Create a feature branch off `develop` in your fork
3. Make your changes, commit, and push to your fork
4. Open a pull request from your fork to the upstream `develop` branch

**Open all pull requests against `develop`, not `main`.** Development happens on
`develop`; `main` holds released code and is updated only through the release
process. GitHub may default a new PR's base to `main` — change it to `develop`
before submitting.

### Keeping Your Fork in Sync

```bash
git fetch upstream
git checkout develop
git merge upstream/develop
```

### Branch Naming

- `feature/<short-description>` for new features
- `fix/<short-description>` for bug fixes

### Before Submitting

1. **Review the Architecture**: [docs/foundry/guides/architecture.md](docs/foundry/guides/architecture.md)
2. **Create an Issue**: Discuss the change before starting work (for non-trivial changes)
3. **Follow Conventions**: Match existing code style and patterns
4. **Write Tests**: All new features require tests
5. **Update Docs**: Update README, examples, or documentation as needed

### PR Checklist

- [ ] All tests pass (`uv run pytest`)
- [ ] Code is formatted (`uv run ruff format .`)
- [ ] No linter errors (`uv run ruff check .`)
- [ ] Type checks pass (`uv run ty check`)
- [ ] Architectural compliance verified
- [ ] Tests added for new features
- [ ] Documentation updated
- [ ] CHANGELOG updated (if applicable)

### Adding to `__all__` (the public API)

Names in `agent_evals.__all__` are contract. Each name must pass the test "a user, on the documented path, has to write this name in their own source code." Per issue #198, drift between aspirational and actual public surface is the failure mode this checklist prevents.

If your PR adds a name to `__all__`:

- [ ] **Documented user-need exists.** The name is referenced in `README.md`, `docs/foundry/guides/scorers.md`, `docs/foundry/guides/benchmarks.md`, `docs/foundry/guides/converters.md`, `docs/foundry/guides/platforms.md`, or appears in example code in a known consumer repo or `.scratch/` project (the authoritative consumer-repo list lives in CLAUDE.md's public-API section — keep it consistent there).
- [ ] **No alternative public path serves the same need.** A user can't accomplish the same task through a different documented entry point.
- [ ] **Rationale stated in PR description.** One-line justification that updates CLAUDE.md's per-name rationale table.

### Review and Merge Process

This project uses a maintainer-controlled merge model:

- Only designated maintainers can merge pull requests; contributors cannot
  self-merge
- At least one approving review is required before merge
- Code Owners review required — files covered by
  [`.github/CODEOWNERS`](.github/CODEOWNERS) must be approved by a designated
  owner
- Stale reviews dismissed — new pushes invalidate previous approvals
- All CI checks must pass before merge

External contributors do not have direct push or merge access. You may be asked
to make changes; push additional commits to your branch and a maintainer will
merge once approved.

### Commit Messages

Follow conventional commits format:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style (formatting, no logic change)
- `refactor`: Code restructuring (no feature change)
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

**Examples**:
```
feat(adapters): add Weights & Biases platform adapter

Implements the 3-method Platform interface for W&B.
Includes both sync and async support.

Closes #123

---

fix(scorers): handle None expected values in autoevals wrappers

Some scorers don't require expected values. Updated wrappers
to gracefully handle None.

Fixes #456
```

---

## Getting Help

- **Questions**: Open a discussion on GitHub
- **Bugs**: Open an issue with reproduction steps
- **Features**: Open an RFC issue with proposal
- **Security**: See [SECURITY.md](SECURITY.md) for reporting vulnerabilities
- **Architecture**: See [docs/foundry/guides/architecture.md](docs/foundry/guides/architecture.md)

---

## Code of Conduct

All participants are expected to treat others with respect and
professionalism. Harassment or abusive behavior will not be tolerated.

## License

By contributing, you agree that your contributions will be licensed under the
same license as this project. See [LICENSE](LICENSE) for details.
