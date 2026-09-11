# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for async evaluation API (run_eval_async).

Tests the complete async evaluation flow:
- Single async scorer returns EvalResult
- Multiple examples processed sequentially
- Local adapter logs results correctly
"""

import asyncio

import pytest
from helpers.scorers import async_passing_scorer, make_slow_scorer

from agent_evals import ExpectedResult, Score, TaskResult, run_eval_async
from agent_evals.core.types import EvalResult, ExampleData


@pytest.mark.asyncio
async def test_run_eval_async_with_single_async_scorer():
    """Test run_eval_async with single async scorer returns EvalResult."""

    def simple_task(x):
        """Simple task function."""
        return TaskResult(output=f"Output: {x}")

    result = await run_eval_async(
        task=simple_task,
        dataset=[
            ExampleData(input="test", expected=ExpectedResult(expected="Output: test"))
        ],
        scorers=[async_passing_scorer],
    )

    # Verify result structure
    assert isinstance(result, EvalResult)
    assert result.platform == "local"
    assert "async_passing" in result.scores
    assert result.scores["async_passing"] == 1.0
    assert len(result.examples) == 1
    assert result.examples[0].scores["async_passing"].value == 1.0


@pytest.mark.asyncio
async def test_run_eval_async_processes_multiple_examples_sequentially():
    """Test async evaluation processes all examples sequentially."""

    processed_order = []

    async def tracking_scorer(
        result: TaskResult,
        expected: ExpectedResult | None = None,
    ) -> Score:
        """Scorer that tracks execution order."""
        context = result.context or {}
        processed_order.append(context.get("input"))
        await asyncio.sleep(0.01)
        return Score(name="Tracker", value=1.0, passed=True)

    def simple_task(x):
        return TaskResult(output=f"output_{x}")

    dataset = [
        ExampleData(
            input="first", output=TaskResult(output="", context={"input": "first"})
        ),
        ExampleData(
            input="second", output=TaskResult(output="", context={"input": "second"})
        ),
        ExampleData(
            input="third", output=TaskResult(output="", context={"input": "third"})
        ),
    ]

    result = await run_eval_async(
        task=simple_task,
        dataset=dataset,
        scorers=[tracking_scorer],
    )

    # Verify all examples processed
    assert len(result.examples) == 3
    assert result.examples[0].input == "first"
    assert result.examples[1].input == "second"
    assert result.examples[2].input == "third"

    # Verify sequential order
    assert processed_order == ["first", "second", "third"]


@pytest.mark.asyncio
async def test_run_eval_async_with_local_adapter_logs_correctly():
    """Test async evaluation with local adapter logs results to filesystem."""
    import pathlib
    import tempfile

    def simple_task(x):
        return TaskResult(output=f"result_{x}")

    # Use temporary directory for test
    with tempfile.TemporaryDirectory() as tmpdir:
        from agent_evals.adapters.platforms.local import LocalConfig

        result = await run_eval_async(
            task=simple_task,
            dataset=[ExampleData(input="test1"), ExampleData(input="test2")],
            scorers=[async_passing_scorer],
            platform=LocalConfig(experiment="local_logging", output_dir=tmpdir),
        )

        # Verify result
        assert len(result.examples) == 2
        assert result.scores["async_passing"] == 1.0

        # Verify files created (local adapter should create experiment directory)
        output_path = pathlib.Path(tmpdir)
        assert output_path.exists()

        # Check experiment directory exists
        experiment_dirs = list(output_path.glob("*"))
        assert len(experiment_dirs) > 0, (
            "Local adapter should create experiment directory"
        )


@pytest.mark.asyncio
async def test_run_eval_async_with_multiple_scorers():
    """Test async evaluation with multiple async scorers."""

    scorer_1 = make_slow_scorer(0.01, name="Scorer1")
    scorer_2 = make_slow_scorer(0.01, name="Scorer2")
    scorer_3 = make_slow_scorer(0.01, name="Scorer3")

    def simple_task(x):
        return TaskResult(output=x)

    result = await run_eval_async(
        task=simple_task,
        dataset=[ExampleData(input="test")],
        scorers=[scorer_1, scorer_2, scorer_3],
    )

    # All scorers should execute
    assert len(result.examples[0].scores) == 3
    assert "Scorer1" in result.scores
    assert "Scorer2" in result.scores
    assert "Scorer3" in result.scores
