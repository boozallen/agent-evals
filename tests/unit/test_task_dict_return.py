# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for task TaskResult return and context merge behavior.

Tests that tasks return TaskResult objects with output and optional context,
enabling trajectory evaluation and context passing to scorers.
"""

from helpers.scorers import make_context_check_scorer, passing_scorer

from agent_evals import ExampleData, ExpectedResult, TaskResult, run_eval
from agent_evals.core.types import Score


class TestTaskDictReturn:
    """Test task dict return merges into scorer context."""

    def test_dict_return_with_output_key(self):
        """Task dict with 'output' key extracts it as output string."""

        def task_with_output(input_val):
            return TaskResult(output="answer", context={"custom_field": "data"})

        # Elevated: scorer outcome encodes the propagation assertions
        # (output + context). Asserting on the post-run Score is robust
        # even if the scorer were never invoked.
        scorer = make_context_check_scorer(
            lambda r, e: (
                r.output == "answer" and (r.context or {}).get("custom_field") == "data"
            ),
            name="dict_check",
        )

        result = run_eval(
            task=task_with_output,
            dataset=[
                ExampleData(input="query", expected=ExpectedResult(expected="query"))
            ],
            scorers=[scorer],
        )

        assert result.scores["dict_check"] == 1.0

    def test_dict_return_without_output_key(self):
        """Task dict without 'output' key uses empty string as output."""

        def task_without_output(input_val):
            return TaskResult(
                output="", context={"trajectory": [1, 2, 3], "metadata": "info"}
            )

        # Elevated: empty output + trajectory/metadata propagation encoded
        # in the Score outcome.
        scorer = make_context_check_scorer(
            lambda r, e: (
                r.output == ""
                and (r.context or {}).get("trajectory") == [1, 2, 3]
                and (r.context or {}).get("metadata") == "info"
            ),
            name="dict_check",
        )

        result = run_eval(
            task=task_without_output,
            dataset=[
                ExampleData(input="query", expected=ExpectedResult(expected="query"))
            ],
            scorers=[scorer],
        )

        assert result.scores["dict_check"] == 1.0

    def test_string_return_still_works(self):
        """Task returning string works as before (backward compatibility)."""

        def task_returns_string(input_val):
            return TaskResult(output="simple answer")

        # Elevated: output value + absence of context encoded in the outcome.
        scorer = make_context_check_scorer(
            lambda r, e: r.output == "simple answer" and r.context is None,
            name="dict_check",
        )

        result = run_eval(
            task=task_returns_string,
            dataset=[
                ExampleData(input="query", expected=ExpectedResult(expected="query"))
            ],
            scorers=[scorer],
        )

        assert result.scores["dict_check"] == 1.0


class TestMixedScoring:
    """Test mixed string + trajectory scoring (the key use case)."""

    def test_string_and_trajectory_scorers_together(self):
        """Both string and trajectory scorers can run on same data."""

        def agent_task(input_val):
            return TaskResult(
                output="42",  # For string scorers
                context={
                    "outputs": [  # For trajectory scorers
                        {"role": "user", "content": input_val},
                        {"role": "assistant", "content": "42"},
                    ]
                },
            )

        def string_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            # Uses output from TaskResult
            expected_str = expected.expected if expected else ""
            match = result.output == expected_str
            return Score(name="StringMatch", value=1.0 if match else 0.0, passed=match)

        def trajectory_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ):
            # Uses context from TaskResult
            context = result.context or {}
            trajectory = context.get("outputs", [])
            has_trajectory = len(trajectory) > 0
            return Score(
                name="TrajectoryCheck",
                value=1.0 if has_trajectory else 0.0,
                passed=has_trajectory,
            )

        result = run_eval(
            task=agent_task,
            dataset=[
                ExampleData(
                    input="What is the answer?", expected=ExpectedResult(expected="42")
                )
            ],
            scorers=[string_scorer, trajectory_scorer],
        )

        # Both scorers should pass
        assert result.scores["StringMatch"] == 1.0
        assert result.scores["TrajectoryCheck"] == 1.0

    def test_trajectory_only_scoring_with_empty_output(self):
        """Trajectory scorers work with empty output string."""

        def agent_task(input_val):
            return TaskResult(
                output="",  # Empty for trajectory-only scoring
                context={
                    "outputs": [{"role": "user", "content": input_val}],
                    "reference_outputs": [{"role": "user", "content": "expected"}],
                },
            )

        def trajectory_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ):
            # Doesn't use result.output at all, only context
            context = result.context or {}
            outputs = context.get("outputs", [])
            refs = context.get("reference_outputs", [])
            return Score(
                name="TrajectoryMatch",
                value=1.0 if len(outputs) > 0 and len(refs) > 0 else 0.0,
                passed=True,
            )

        result = run_eval(
            task=agent_task,
            dataset=[ExampleData(input="query", expected=ExpectedResult(expected=""))],
            scorers=[trajectory_scorer],
        )

        assert result.scores["TrajectoryMatch"] == 1.0


class TestHistoricalDataWithContext:
    """Test historical data pattern with pre-populated output and context."""

    def test_historical_trajectory_with_empty_output(self):
        """Historical data with trajectory in context and empty output."""

        def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
            # For historical data, trajectory should be in context parameter, not result.context
            context = result.context or {}
            trajectory = context.get("outputs", [])
            # Empty output + 2-message trajectory encoded as the Score value.
            ok = result.output == "" and len(trajectory) == 2
            return Score(name="test", value=1.0 if ok else 0.0, passed=ok)

        result = run_eval(
            task=None,  # No task - using pre-populated data
            dataset=[
                ExampleData(
                    input="query",
                    expected=ExpectedResult(expected=""),
                    output=TaskResult(
                        output="",
                        context={
                            "outputs": [
                                {"role": "user", "content": "query"},
                                {"role": "assistant", "content": "answer"},
                            ]
                        },
                    ),
                )
            ],
            scorers=[scorer],
        )

        assert result.scores["test"] == 1.0

    def test_historical_mixed_output_and_trajectory(self):
        """Historical data with both output string and trajectory."""

        def string_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            # Pre-populated output == "42" encoded as the Score value.
            ok = result.output == "42"
            return Score(name="StringScorer", value=1.0 if ok else 0.0, passed=ok)

        def trajectory_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ):
            # For historical data, trajectory should be in context parameter
            context = result.context or {}
            trajectory = context.get("outputs", [])
            ok = len(trajectory) > 0
            return Score(name="TrajectoryScorer", value=1.0 if ok else 0.0, passed=ok)

        result = run_eval(
            task=None,
            dataset=[
                ExampleData(
                    input="query",
                    output=TaskResult(
                        output="42",
                        context={"outputs": [{"role": "user", "content": "query"}]},
                    ),  # Pre-populated output
                )
            ],
            scorers=[string_scorer, trajectory_scorer],
        )

        assert result.scores["StringScorer"] == 1.0
        assert result.scores["TrajectoryScorer"] == 1.0


class TestErrorCases:
    """Test error handling for dict returns."""

    def test_no_task_no_output_raises_error(self):
        """Neither task nor output creates error example."""

        # This should create an error example instead of raising an exception
        result = run_eval(
            task=None,  # No task
            dataset=[ExampleData(input="query")],  # No output
            scorers=[passing_scorer],
        )

        # Should have created an error example
        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.error is not None
        error_message = example.error
        assert "task" in error_message.lower()
        assert "output" in error_message.lower()
