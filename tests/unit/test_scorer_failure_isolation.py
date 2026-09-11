# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scorer failure isolation tests.

Verify that when one scorer fails, it doesn't block other scorers from executing.
All scorers should run in parallel, and failures should be converted to Error Scores.
"""

import pytest
from helpers.scorers import raising_scorer

from agent_evals import Score, TaskResult, run_eval_async
from agent_evals.core.types import ExampleData


@pytest.mark.asyncio
async def test_one_scorer_failure_does_not_block_others():
    """Test that one failing scorer doesn't prevent other scorers from executing.

    All scorers should execute in parallel. Failed scorer gets Error Score with
    error metadata, successful scorers return normal Scores.
    """

    def successful_scorer_1(result, expected=None, **context):
        return Score(name="Success1", value=1.0, passed=True)

    def successful_scorer_2(result, expected=None, **context):
        return Score(name="Success2", value=0.9, passed=True)

    def simple_task(x):
        return TaskResult(output=f"output_{x}")

    result = await run_eval_async(
        task=simple_task,
        dataset=[ExampleData(input="test")],
        scorers=[successful_scorer_1, raising_scorer, successful_scorer_2],
    )

    # Verify all 3 scorers executed (including the failed one)
    assert len(result.examples[0].scores) == 3

    # Successful scorers should have correct values
    assert result.examples[0].scores["Success1"].value == 1.0
    assert result.examples[0].scores["Success1"].passed is True

    assert result.examples[0].scores["Success2"].value == 0.9
    assert result.examples[0].scores["Success2"].passed is True

    # Failed scorer should have Error Score with error metadata.
    # The runner keys a raising scorer by its function __name__.
    failed_score = result.examples[0].scores["raising_scorer"]
    assert failed_score.value == 0.0
    assert failed_score.passed is False
    assert "error" in failed_score.metadata
    assert "scorer failure (intentional)" in failed_score.metadata["error"]


@pytest.mark.asyncio
async def test_all_scorers_execute_even_with_multiple_failures():
    """Test that multiple failing scorers don't stop other scorers."""

    def failing_scorer_1(result, expected=None, **context):
        raise RuntimeError("Failure 1")

    def successful_scorer(result, expected=None, **context):
        return Score(name="Success", value=0.8, passed=True)

    def failing_scorer_2(result, expected=None, **context):
        raise RuntimeError("Failure 2")

    def simple_task(x):
        return TaskResult(output=x)

    result = await run_eval_async(
        task=simple_task,
        dataset=[ExampleData(input="test")],
        scorers=[failing_scorer_1, successful_scorer, failing_scorer_2],
    )

    # All 3 scorers should have attempted execution
    assert len(result.examples[0].scores) == 3

    # Successful scorer
    assert result.examples[0].scores["Success"].value == 0.8
    assert result.examples[0].scores["Success"].passed is True

    # Both failed scorers should have Error Scores
    assert result.examples[0].scores["failing_scorer_1"].value == 0.0
    assert result.examples[0].scores["failing_scorer_1"].passed is False

    assert result.examples[0].scores["failing_scorer_2"].value == 0.0
    assert result.examples[0].scores["failing_scorer_2"].passed is False


@pytest.mark.asyncio
async def test_task_failure_skips_scoring_and_does_not_report_a_pass():
    """A crashed task is not scored, and cannot yield a passing aggregate.

    Previously the adapter substituted ``TaskResult(output="")`` for a crashed
    task and ran the scorers against it. Any scorer indifferent to the output
    then returned a genuine pass, which averaged into the headline aggregate
    as real performance — a totally broken agent could report a 1.0 pass rate.
    """

    def scorer_indifferent_to_output(result, expected=None, **context):
        """A scorer that passes regardless of what the task produced."""
        return Score(name="AlwaysPasses", value=1.0, passed=True)

    def failing_task(x):
        raise ValueError("Task intentionally fails")

    result = await run_eval_async(
        task=failing_task,
        dataset=[ExampleData(input="test")],
        scorers=[scorer_indifferent_to_output],
    )

    # The example still exists and records why the task failed.
    assert len(result.examples) == 1
    assert result.examples[0].error is not None
    assert "Task intentionally fails" in result.examples[0].error
    assert result.examples[0].output == ""

    # The crashed task was not scored against its placeholder output.
    assert result.examples[0].scores == {}

    # And the run does not report itself as passing.
    assert result.summary["successful_examples"] == 0
    assert result.summary["failed_examples"] == 1
    assert not all(v >= 0.9 for v in result.pass_rates.values())
