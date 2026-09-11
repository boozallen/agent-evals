# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test output field functionality for ExampleData and precedence logic.

This test file covers Phase 2 Foundational tasks:
- T001: Add output field to ExampleData (✅ IMPLEMENTED)
- T002: Implement output precedence logic in LocalPlatform._process_example() (✅ IMPLEMENTED)
- T003: Add warning logic for task+output conflict (✅ IMPLEMENTED)
- T004: Add error handling for missing both task and output (✅ IMPLEMENTED)
- T005-T006: Make task parameter optional in run_eval/run_eval_async (✅ IMPLEMENTED)

Tests updated to work with current architecture:
- Uses run_eval/run_eval_async API instead of deprecated internal methods
- Tests actual functionality through the public API using LocalPlatform
- All task functions return TaskResult objects
- Scorers receive the output string from TaskResult.output
"""

from unittest.mock import patch

import pytest
from helpers.scorers import passing_scorer

from agent_evals import ExpectedResult, TaskResult, run_eval, run_eval_async
from agent_evals.core.types import ExampleData, Score


def _comparison_scorer(result, expected=None, **context):
    """Canonical-shape scorer that reads output/expected values for comparison."""
    reference = expected.expected if expected is not None else None
    matches = result.output == reference
    return Score(name="comparison", value=1.0 if matches else 0.0, passed=matches)


class TestExampleDataOutputField:
    """Test T001: Add output field to ExampleData."""

    def test_example_data_accepts_output_field(self):
        """ExampleData should accept optional output field."""
        example = ExampleData(
            input="What is 2+2?",
            expected=ExpectedResult(expected="4"),
            output=TaskResult(output="4"),
        )
        assert example.output is not None
        assert example.output.output == "4"

    def test_example_data_output_defaults_to_none(self):
        """ExampleData output field should default to None."""
        example = ExampleData(
            input="What is 2+2?", expected=ExpectedResult(expected="4")
        )
        assert example.output is None

    def test_example_data_accepts_various_output_types(self):
        """Output field should accept TaskResult or None."""
        # TaskResult output
        ex1 = ExampleData(
            input="q1",
            expected=ExpectedResult(expected=""),
            output=TaskResult(output="answer"),
        )
        assert ex1.output is not None
        assert ex1.output.output == "answer"

        # Empty string (valid output, not None)
        ex2 = ExampleData(
            input="q2",
            expected=ExpectedResult(expected=""),
            output=TaskResult(output=""),
        )
        assert ex2.output is not None
        assert ex2.output.output == ""

        # None (default)
        ex3 = ExampleData(input="q3", expected=ExpectedResult(expected=""), output=None)
        assert ex3.output is None

    def test_example_data_backward_compatible(self):
        """Existing code without output field should work unchanged."""
        # Old code pattern
        example = ExampleData(
            input="test",
            expected=ExpectedResult(expected="result"),
            metadata={"source": "test"},
        )
        assert example.input == "test"
        assert example.expected is not None
        assert example.expected.expected == "result"
        assert example.output is None  # Defaults to None


class TestOutputPrecedenceLogic:
    """Test T002: Implement output precedence logic in LocalPlatform._process_example()."""

    def test_uses_pre_populated_output_when_present(self):
        """When output is provided, use it instead of executing task."""

        def task(x):
            return TaskResult(output="TASK OUTPUT - SHOULD NOT BE CALLED")

        # Run evaluation with pre-populated output
        result = run_eval(
            task=task,
            dataset=[
                ExampleData(
                    input="test",
                    output=TaskResult(output="PRE-POPULATED"),
                    expected=ExpectedResult(expected="PRE-POPULATED"),
                )
            ],
            scorers=[passing_scorer],
        )

        # Should use pre-populated output, not task output
        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.output == "PRE-POPULATED"
        # Duration should be very small since no task execution
        assert example.duration < 0.01  # Less than 10ms

    def test_executes_task_when_output_is_none(self):
        """When output is None, execute task normally."""

        def task(x):
            return TaskResult(output=f"TASK OUTPUT: {x}")

        # Run evaluation without pre-populated output
        result = run_eval(
            task=task,
            dataset=[
                ExampleData(
                    input="test", expected=ExpectedResult(expected="TASK OUTPUT: test")
                )
            ],
            scorers=[passing_scorer],
        )

        # Should execute task
        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.output == "TASK OUTPUT: test"
        assert example.duration > 0  # Task execution took time

    def test_empty_string_output_is_valid(self):
        """Empty string is a valid output, task should NOT execute."""

        def task(x):
            return TaskResult(output="TASK OUTPUT - SHOULD NOT BE CALLED")

        # Run evaluation with empty string output
        result = run_eval(
            task=task,
            dataset=[
                ExampleData(
                    input="test",
                    output=TaskResult(output=""),
                    expected=ExpectedResult(expected=""),
                )
            ],
            scorers=[passing_scorer],
        )

        # Empty string is valid - should use it, not execute task
        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.output == ""
        # Duration should be very small since no task execution
        assert example.duration < 0.01  # Less than 10ms


class TestTaskOutputConflictWarning:
    """Test T003: Add warning logic for task+output conflict."""

    @patch("agent_evals.adapters.platforms.local.logger")
    def test_warns_when_both_task_and_output_provided(self, mock_logger):
        """Should emit warning when both task and pre-populated output exist."""

        def task(x):
            return TaskResult(output="TASK OUTPUT")

        # Run evaluation with both task and pre-populated output
        result = run_eval(
            task=task,
            dataset=[
                ExampleData(
                    input="test",
                    output=TaskResult(output="PRE-POPULATED"),  # Output provided
                    expected=ExpectedResult(expected="PRE-POPULATED"),
                )
            ],
            scorers=[passing_scorer],
        )

        # Should use output (precedence)
        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.output == "PRE-POPULATED"

        # Should have logged a warning
        mock_logger.warning.assert_called_once()
        warning_message = mock_logger.warning.call_args[0][0]
        assert "Pre-populated output found" in warning_message
        assert "Skipping task execution" in warning_message

    @patch("agent_evals.adapters.platforms.local.logger")
    def test_no_warning_when_output_without_task(self, mock_logger):
        """Should NOT warn when output is provided but task is None."""

        # Run evaluation without task but with pre-populated output
        result = run_eval(
            task=None,
            dataset=[
                ExampleData(
                    input="test",
                    output=TaskResult(output="PRE-POPULATED"),
                    expected=ExpectedResult(expected="PRE-POPULATED"),
                )
            ],
            scorers=[passing_scorer],
        )

        # Should use output without warning
        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.output == "PRE-POPULATED"
        mock_logger.warning.assert_not_called()


class TestMissingTaskAndOutputError:
    """Test T004: Add error handling for missing both task and output."""

    def test_raises_error_when_both_missing(self):
        """Should create error example when neither task nor output provided."""

        # This should create an error example instead of raising an exception
        result = run_eval(
            task=None,  # No task
            dataset=[
                ExampleData(
                    input="test",
                    # No output field
                    expected=ExpectedResult(expected="result"),
                )
            ],
            scorers=[passing_scorer],
        )

        # Should have created an error example
        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.error is not None
        error_message = example.error
        assert "task" in error_message.lower()
        assert "output" in error_message.lower()


class TestOptionalTaskParameter:
    """Test T005-T006: Make task parameter optional in run_eval/run_eval_async."""

    def test_run_eval_accepts_none_task(self):
        """run_eval should accept task=None when examples have outputs."""

        # This should NOT raise an error
        result = run_eval(
            task=None,  # No task!
            dataset=[
                ExampleData(
                    input="test",
                    output=TaskResult(output="result"),
                    expected=ExpectedResult(expected="result"),
                )
            ],
            scorers=[_comparison_scorer],
        )

        assert result.scores["comparison"] == 1.0
        assert len(result.examples) == 1
        assert result.examples[0].output == "result"

    @pytest.mark.asyncio
    async def test_run_eval_async_accepts_none_task(self):
        """run_eval_async should accept task=None when examples have outputs."""

        result = await run_eval_async(
            task=None,  # No task!
            dataset=[
                ExampleData(
                    input="test",
                    output=TaskResult(output="result"),
                    expected=ExpectedResult(expected="result"),
                )
            ],
            scorers=[_comparison_scorer],
        )

        assert result.scores["comparison"] == 1.0
        assert len(result.examples) == 1
        assert result.examples[0].output == "result"

    def test_task_parameter_still_works_when_provided(self):
        """Existing code with task parameter should continue working (backward compat)."""

        def task(x):
            return TaskResult(output=f"OUTPUT: {x}")

        result = run_eval(
            task=task,  # Task provided (old pattern)
            dataset=[
                ExampleData(
                    input="test", expected=ExpectedResult(expected="OUTPUT: test")
                )
            ],
            scorers=[passing_scorer],
        )

        assert result.scores["passing"] == 1.0
        assert len(result.examples) == 1
        assert result.examples[0].output == "OUTPUT: test"


class TestAsyncOutputPrecedenceLogic:
    """Test output precedence logic for run_eval_async."""

    @pytest.mark.asyncio
    async def test_async_runner_uses_pre_populated_output(self):
        """run_eval_async should respect output precedence."""

        def task(x):
            return TaskResult(output="TASK OUTPUT - SHOULD NOT BE CALLED")

        result = await run_eval_async(
            task=task,
            dataset=[
                ExampleData(
                    input="test",
                    output=TaskResult(output="PRE-POPULATED"),  # Output provided
                    expected=ExpectedResult(expected="PRE-POPULATED"),
                )
            ],
            scorers=[passing_scorer],
        )

        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.output == "PRE-POPULATED"
        # Duration should be very small since no task execution
        assert example.duration < 0.01  # Less than 10ms

    @pytest.mark.asyncio
    async def test_async_runner_executes_task_when_no_output(self):
        """run_eval_async should execute task when output is None."""

        def task(x):
            return TaskResult(output=f"TASK OUTPUT: {x}")

        result = await run_eval_async(
            task=task,
            dataset=[
                ExampleData(
                    input="test", expected=ExpectedResult(expected="TASK OUTPUT: test")
                )
            ],
            scorers=[passing_scorer],
        )

        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.output == "TASK OUTPUT: test"
        assert example.duration > 0

    @pytest.mark.asyncio
    @patch("agent_evals.adapters.platforms.local.logger")
    async def test_async_runner_warns_on_conflict(self, mock_logger):
        """run_eval_async should warn when both task and output exist."""

        def task(x):
            return TaskResult(output="TASK OUTPUT")

        result = await run_eval_async(
            task=task,
            dataset=[
                ExampleData(
                    input="test",
                    output=TaskResult(output="PRE-POPULATED"),
                    expected=ExpectedResult(expected="PRE-POPULATED"),
                )
            ],
            scorers=[passing_scorer],
        )

        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.output == "PRE-POPULATED"
        mock_logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_runner_raises_error_when_both_missing(self):
        """run_eval_async should create error example when neither task nor output."""

        result = await run_eval_async(
            task=None,
            dataset=[
                ExampleData(input="test", expected=ExpectedResult(expected="result"))
            ],
            scorers=[passing_scorer],
        )

        # Should have created an error example
        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.error is not None
        error_message = example.error
        assert "task" in error_message.lower()
        assert "output" in error_message.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
