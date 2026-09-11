# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for AsyncEvalRunner initialization and validation.

Tests AsyncEvalRunner.__init__ validates inputs properly:
- Non-empty dataset
- Non-empty scorers
- Valid platform
- Callable task
"""

import pytest
from helpers.scorers import passing_scorer

from agent_evals.core.runner import AsyncEvalRunner
from agent_evals.core.types import ExampleData


# Mock platform registry for testing (dependency injection pattern).
# The runner calls platform_registry.get(name)(); the fake registry's get()
# returns a class whose __init__ produces a permissive MagicMock.
class _MockPlatformRegistry:
    """Minimal AdapterRegistry stub returning MagicMock-backed classes."""

    def get(self, key):
        from unittest.mock import MagicMock

        return lambda: MagicMock()

    def list(self):
        return []

    def __contains__(self, key):
        return True


mock_platform_registry = _MockPlatformRegistry()


def test_async_eval_runner_init_validates_empty_dataset():
    """Test AsyncEvalRunner rejects empty dataset."""

    def dummy_task(x):
        return x

    from agent_evals.adapters.platforms.local import LocalConfig

    with pytest.raises(ValueError, match="Dataset cannot be empty"):
        AsyncEvalRunner(
            task=dummy_task,
            dataset=[],  # Empty dataset should fail
            scorers=[passing_scorer],
            platform=LocalConfig(),
            platform_registry=mock_platform_registry,
        )


def test_async_eval_runner_init_validates_empty_scorers():
    """Test AsyncEvalRunner rejects empty scorers list."""

    def dummy_task(x):
        return x

    from agent_evals.adapters.platforms.local import LocalConfig

    with pytest.raises(ValueError, match="At least one scorer is required"):
        AsyncEvalRunner(
            task=dummy_task,
            dataset=[ExampleData(input="test")],
            scorers=[],  # Empty scorers should fail
            platform=LocalConfig(),
            platform_registry=mock_platform_registry,
        )


def test_async_eval_runner_init_accepts_valid_inputs():
    """Test AsyncEvalRunner accepts valid inputs."""

    def dummy_task(x):
        return x

    from agent_evals.adapters.platforms.local import LocalConfig

    # Should not raise
    runner = AsyncEvalRunner(
        task=dummy_task,
        dataset=[ExampleData(input="test")],
        scorers=[passing_scorer],
        platform=LocalConfig(experiment="test"),
        platform_registry=mock_platform_registry,
    )

    assert runner.task == dummy_task
    assert len(runner.dataset) == 1
    assert len(runner.scorers) == 1
    assert runner.platform.name == "local"


def test_async_eval_runner_init_with_execution_config():
    """Test AsyncEvalRunner accepts execution config (now the 'config' param)."""

    def dummy_task(x):
        return x

    from agent_evals.adapters.platforms.local import LocalConfig
    from agent_evals.core.types import EvalConfig

    exec_config = EvalConfig()  # Default: max_concurrent_tests=1

    runner = AsyncEvalRunner(
        task=dummy_task,
        dataset=[ExampleData(input="test")],
        scorers=[passing_scorer],
        platform=LocalConfig(experiment="test"),
        config=exec_config,
        platform_registry=mock_platform_registry,
    )

    assert runner.config == exec_config
