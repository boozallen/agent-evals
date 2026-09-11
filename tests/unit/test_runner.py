# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for evaluation runner."""

import pytest
from helpers.scorers import passing_scorer

from agent_evals import ExpectedResult, Score, TaskResult
from agent_evals.adapters.platforms.local import LocalConfig
from agent_evals.core.runner import EvalRunner
from agent_evals.core.types import EvalConfig, ExampleData


# Mock platform registry for testing (dependency injection pattern).
# The runner calls platform_registry.get(name)(); the fake registry's get()
# returns a class whose __init__ produces a permissive MagicMock.
class _MockPlatformRegistry:
    """Minimal AdapterRegistry stub returning MagicMock-backed classes."""

    def __init__(self, adapter_class=None):
        from unittest.mock import MagicMock

        self._adapter_class = adapter_class or (lambda: MagicMock())

    def get(self, key):
        return self._adapter_class

    def list(self):
        return []

    def __contains__(self, key):
        return True


mock_platform_registry = _MockPlatformRegistry()


class TestEvalRunnerValidation:
    """Test EvalRunner input validation."""

    def test_empty_dataset_raises_error(self):
        """Test that empty dataset raises ValueError."""

        def task(x):
            return TaskResult(output=x)

        with pytest.raises(ValueError, match="Dataset cannot be empty"):
            EvalRunner(
                task=task,
                dataset=[],
                scorers=[passing_scorer],
                platform=LocalConfig(),
                platform_registry=mock_platform_registry,
            )

    def test_no_scorers_raises_error(self):
        """Test that empty scorers list raises ValueError."""

        def task(x):
            return TaskResult(output=x)

        dataset = [ExampleData(input="test")]

        with pytest.raises(ValueError, match="At least one scorer is required"):
            EvalRunner(
                task=task,
                dataset=dataset,
                scorers=[],
                platform=LocalConfig(),
                platform_registry=mock_platform_registry,
            )

    def test_valid_initialization(self):
        """Test that valid inputs create runner successfully."""

        def task(x):
            return TaskResult(output=x)

        dataset = [ExampleData(input="test")]

        runner = EvalRunner(
            task=task,
            dataset=dataset,
            scorers=[passing_scorer],
            platform=LocalConfig(),
            platform_registry=mock_platform_registry,
        )

        assert runner.task == task
        assert len(runner.dataset) == 1
        assert runner.scorers == [passing_scorer]

    def test_validates_dataset_structure(self):
        """Test runner rejects dict datasets."""
        with pytest.raises(ValueError) as exc:
            EvalRunner(
                task=lambda x: TaskResult(output=x),
                dataset=[
                    {"expected": "y"}
                ],  # intentional bad input - verifies runtime rejection
                scorers=[passing_scorer],
                platform=LocalConfig(),
                platform_registry=mock_platform_registry,
            )

        error_msg = str(exc.value).lower()
        assert "exampledata" in error_msg
        assert "dict" in error_msg

    def test_accepts_explicit_context_dict(self):
        """Test proper context dict usage works."""
        runner = EvalRunner(
            task=lambda x: TaskResult(output=x),
            dataset=[
                ExampleData(
                    input="x",
                    expected=ExpectedResult(expected="x"),
                    output=TaskResult(output="", context={"outputs": []}),
                )
            ],
            scorers=[passing_scorer],
            platform=LocalConfig(),
            platform_registry=mock_platform_registry,
        )

        assert len(runner.dataset) == 1
        assert runner.dataset[0].output is not None
        assert runner.dataset[0].output.context == {"outputs": []}


class TestEvalRunnerOrchestration:
    """Test EvalRunner orchestration logic (User Story 3)."""

    def test_runner_controls_initialization(self):
        """Test US3: Runner controls adapter initialization timing."""
        # Track method calls
        calls = []

        class MockAdapter:
            def evaluate(self, task, dataset, evaluators, platform=None, config=None):
                calls.append("evaluate")
                # Mock a simple result
                return type(
                    "EvalResult",
                    (),
                    {
                        "experiment_id": "test-id",
                        "experiment_url": "test-url",
                        "platform": "local",
                        "scores": {"test": 1.0},
                        "pass_rates": {"test": 1.0},
                        "examples": [],
                        "summary": {
                            "total_examples": 1,
                            "successful_examples": 1,
                            "failed_examples": 0,
                        },
                        "duration": 0.1,
                    },
                )()

        def task(x):
            return TaskResult(output=f"output_{x}")

        dataset = [ExampleData(input="test")]

        runner = EvalRunner(
            task=task,
            dataset=dataset,
            scorers=[passing_scorer],
            platform=LocalConfig(),
            platform_registry=_MockPlatformRegistry(MockAdapter),
        )
        runner.run()

        # Verify evaluate was called
        assert "evaluate" in calls

    def test_runner_controls_logging_timing(self):
        """Test US3: Runner controls adapter evaluation flow."""
        evaluate_called = []

        class MockAdapter:
            def evaluate(self, task, dataset, evaluators, platform=None, config=None):
                evaluate_called.append(True)
                # Track the dataset inputs
                inputs = [ex.input for ex in dataset]
                return type(
                    "EvalResult",
                    (),
                    {
                        "experiment_id": "test-id",
                        "experiment_url": "test-url",
                        "platform": "local",
                        "scores": {"test": 1.0},
                        "pass_rates": {"test": 1.0},
                        "examples": [],
                        "summary": {
                            "total_examples": len(dataset),
                            "successful_examples": len(dataset),
                            "failed_examples": 0,
                        },
                        "duration": 0.1,
                        "inputs": inputs,  # Add for verification
                    },
                )()

        def task(x):
            return TaskResult(output=f"output_{x}")

        dataset = [
            ExampleData(input="a", expected=ExpectedResult(expected="a")),
            ExampleData(input="b", expected=ExpectedResult(expected="b")),
        ]

        runner = EvalRunner(
            task=task,
            dataset=dataset,
            scorers=[passing_scorer],
            platform=LocalConfig(),
            platform_registry=_MockPlatformRegistry(MockAdapter),
        )
        result = runner.run()

        # Verify evaluate was called and processed all examples
        assert len(evaluate_called) == 1
        assert hasattr(result, "inputs")
        inputs = getattr(result, "inputs", [])
        assert "a" in inputs
        assert "b" in inputs

    def test_runner_controls_finalization(self):
        """Test US3: Runner controls evaluation completion."""
        finalize_called = []

        class MockAdapter:
            def evaluate(self, task, dataset, evaluators, platform=None, config=None):
                # Track how many examples were processed
                finalize_called.append(len(dataset))
                return type(
                    "EvalResult",
                    (),
                    {
                        "experiment_id": "test-id",
                        "experiment_url": "test-url",
                        "platform": "local",
                        "scores": {"test": 1.0},
                        "pass_rates": {"test": 1.0},
                        "examples": [],
                        "summary": {
                            "total_examples": len(dataset),
                            "successful_examples": len(dataset),
                            "failed_examples": 0,
                        },
                        "duration": 0.1,
                    },
                )()

        def task(x):
            return TaskResult(output=f"output_{x}")

        dataset = [
            ExampleData(input="a", expected=ExpectedResult(expected="a")),
            ExampleData(input="b", expected=ExpectedResult(expected="b")),
            ExampleData(input="c", expected=ExpectedResult(expected="c")),
        ]

        runner = EvalRunner(
            task=task,
            dataset=dataset,
            scorers=[passing_scorer],
            platform=LocalConfig(),
            platform_registry=_MockPlatformRegistry(MockAdapter),
        )
        runner.run()

        # Verify evaluate was called with all examples
        assert finalize_called == [3]


class TestEvalConfigValidation:
    """Test EvalConfig validation."""

    def test_valid_config(self):
        config = EvalConfig(max_concurrent_tests=10)
        assert config.max_concurrent_tests == 10

    def test_invalid_max_concurrent_tests(self):
        with pytest.raises(ValueError, match="max_concurrent_tests"):
            EvalConfig(max_concurrent_tests=0)


class TestEdgeCases:
    """Test edge cases and validation."""

    def test_invalid_score_value_raises_error(self):
        """Test that Score with invalid value raises ValueError."""
        # Test value > 1.0
        with pytest.raises(ValueError, match="Score value must be between 0.0 and 1.0"):
            Score(name="test", value=1.5, passed=True)

        # Test value < 0.0
        with pytest.raises(ValueError, match="Score value must be between 0.0 and 1.0"):
            Score(name="test", value=-0.5, passed=True)
