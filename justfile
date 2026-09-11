# Agent Evals - Development Commands
# Usage: just <command> [args]

# Show all available recipes
default:
    @just --list

# ============================================================================
# Setup & Dependencies
# ============================================================================

# Sync the virtual environment with the lockfile (dev group + all extras)
sync:
    uv sync --all-extras

# Set up the local development environment (dependencies + pre-commit hooks)
setup: sync
    uv run pre-commit install
    @echo "Development environment ready"

# Add a package to the project
add-package package:
    uv add "{{package}}"
    uv lock

# Add a development package to the project
add-dev-package package:
    uv add --dev "{{package}}"
    uv lock

# ============================================================================
# Code Quality
# ============================================================================

# Run ruff lint + the bandit command-injection sink set
lint:
    uv run ruff check src/ tests/
    uv run bandit -c pyproject.toml -r src/agent_evals -t B602,B603,B605,B607

# Auto-fix ruff lint findings
lint-fix:
    uv run ruff check src/ tests/ --fix

# Check code formatting
format:
    uv run ruff format --check src/ tests/

# Auto-format code
format-fix:
    uv run ruff format src/ tests/

# Run ty type checking
type-check:
    uv run ty check src/

# Check architecture import boundaries with import-linter
lint-imports:
    uv run lint-imports

# Run every quality gate: lint, format, type-check, test
check: lint format type-check test
    @echo "All quality checks passed"

# ============================================================================
# Testing
# ============================================================================
#
# The tier split is by DIRECTORY, not by marker: `tests/unit/`, `tests/integration/`
# and `tests/validation/` are the three tiers, so a test's tier is decided by where
# its file lives and cannot be forgotten. `test` runs the whole tree, and each
# `test-<tier>` recipe runs one directory. No recipe passes `-m`; markers stay
# declared as cross-cutting trait tags but nothing selects on them.
#
# Coverage is always on via `[tool.pytest.ini_options] addopts` in pyproject.toml
# (terminal + HTML into `htmlcov/`), never from a recipe, so there is no separate
# coverage recipe. The gate in `addopts` is a whole-tree number, so any run
# narrower than the whole tree needs `--no-cov --cov-fail-under=0`: only
# `tests/unit/` is broad enough to clear it alone.

# Run all tests (unit + integration + validation)
test:
    uv run pytest tests/

# Run unit tests only
test-unit:
    uv run pytest tests/unit/

# Coverage is disabled for this isolated slice (`--no-cov --cov-fail-under=0`): on
# its own it does not exercise enough of the source tree to meet the configured
# threshold, and instrumenting it would overwrite `htmlcov/` with integration-only
# data. pytest exit code 5 (no tests collected) is treated as success so the recipe
# stays valid if this tier is ever empty.
# Run integration tests only
test-integration:
    #!/usr/bin/env bash
    set -uo pipefail
    uv run pytest tests/integration/ --no-cov --cov-fail-under=0
    status=$?
    if [ "$status" -eq 5 ]; then
        echo "No integration tests collected - not a failure."
        exit 0
    fi
    exit $status

# Run contract validation tests only
test-validation:
    uv run pytest tests/validation/ --no-cov --cov-fail-under=0

# ============================================================================
# Maintenance
# ============================================================================

# Not part of `check` — expect false positives from Protocol method definitions,
# registry-decorated adapter factories, and Pydantic models.
# Advisory dead-code scan with vulture
dead-code:
    uv run vulture

# Clean build artifacts and tooling caches
clean:
    rm -rf build/
    rm -rf dist/
    rm -rf htmlcov/
    rm -rf *.egg-info/
    rm -f .coverage
    find . -path ./.venv -prune -o -type d \( -name .pytest_cache -o -name .ruff_cache -o -name .import_linter_cache -o -name __pycache__ \) -prune -exec rm -rf {} +
    find . -path ./.venv -prune -o -type f -name "*.pyc" -exec rm -f {} +
    @echo "Cleaned build artifacts"
