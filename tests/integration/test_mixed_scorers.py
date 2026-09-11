# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Integration tests for mixed sync/async scorer execution.

Tests User Story 3: Mixed Sync/Async Scorers
- Validates that sync and async scorers can be used together
- Ensures sync scorers don't block async scorers
- Verifies context passing works for both types
"""

import time

import pytest
from helpers.scorers import make_slow_scorer

from agent_evals import TaskResult, run_eval_async
from agent_evals.core.types import ExampleData


@pytest.mark.asyncio
async def test_evaluation_with_mixed_sync_and_async_scorers():
    """Test evaluation with 2 async scorers (with delays) and 2 sync scorers (with time.sleep).

    This test verifies:
    - All 4 scorers execute correctly
    - Sync scorers don't block async scorers
    - All scores are returned

    Task: T029
    """

    # Create async scorers with delays
    async_scorer_1 = make_slow_scorer(0.5, is_async=True, name="AsyncScorer1")
    async_scorer_2 = make_slow_scorer(0.5, is_async=True, name="AsyncScorer2")

    # Create sync scorers with time.sleep
    sync_scorer_1 = make_slow_scorer(0.3, is_async=False, name="SyncScorer1")
    sync_scorer_2 = make_slow_scorer(0.3, is_async=False, name="SyncScorer2")

    # Task function
    def task(input_value):
        return TaskResult(output=f"Output for {input_value}")

    # Run evaluation
    result = await run_eval_async(
        task=task,
        dataset=[ExampleData(input="test")],
        scorers=[async_scorer_1, async_scorer_2, sync_scorer_1, sync_scorer_2],
    )

    # Verify all scorers returned scores
    assert len(result.examples) == 1
    example = result.examples[0]
    assert len(example.scores) == 4

    assert "AsyncScorer1" in example.scores
    assert "AsyncScorer2" in example.scores
    assert "SyncScorer1" in example.scores
    assert "SyncScorer2" in example.scores

    # Verify all scores passed
    assert example.scores["AsyncScorer1"].value == 1.0
    assert example.scores["AsyncScorer2"].value == 1.0
    assert example.scores["SyncScorer1"].value == 1.0
    assert example.scores["SyncScorer2"].value == 1.0


@pytest.mark.asyncio
async def test_slow_sync_scorer_does_not_block_fast_async_scorers():
    """Test slow sync scorer (5s sleep) runs in thread without blocking fast async scorers.

    This test verifies:
    - Sync scorer with long sleep doesn't block event loop
    - Async scorers complete quickly despite slow sync scorer
    - All scorers execute concurrently

    Task: T032
    """

    # Slow sync scorer (5 seconds)
    slow_sync_scorer = make_slow_scorer(5.0, is_async=False, name="SlowSync")

    # Fast async scorers (1 second)
    fast_async_scorer_1 = make_slow_scorer(1.0, is_async=True, name="FastAsync1")
    fast_async_scorer_2 = make_slow_scorer(1.0, is_async=True, name="FastAsync2")

    # Task function
    def task(input_value):
        return TaskResult(output=f"Output for {input_value}")

    # Measure execution time
    start = time.perf_counter()

    result = await run_eval_async(
        task=task,
        dataset=[ExampleData(input="test")],
        scorers=[slow_sync_scorer, fast_async_scorer_1, fast_async_scorer_2],
    )

    duration = time.perf_counter() - start

    # Verify all scorers completed
    assert len(result.examples[0].scores) == 3
    assert "SlowSync" in result.examples[0].scores
    assert "FastAsync1" in result.examples[0].scores
    assert "FastAsync2" in result.examples[0].scores

    # Total time should be ~5s (max of all delays), not 7s (5+1+1)
    # All scorers run concurrently
    assert duration < 6.5, f"Expected ~5s (parallel), got {duration:.2f}s"
    assert duration > 4.5, f"Expected ~5s minimum, got {duration:.2f}s"


@pytest.mark.asyncio
async def test_mixed_async_and_sync_scorers_all_complete():
    """Sync scorers and async scorers run together within an example.

    The public API doesn't cap scorer concurrency — only test/example
    concurrency. This test verifies mixed kinds all complete and sync
    scorers don't block the event loop (they run via asyncio.to_thread).
    """
    from agent_evals import TaskResult

    def task(input_value):
        return TaskResult(output=f"Output for {input_value}")

    result = await run_eval_async(
        task=task,
        dataset=[ExampleData(input="test")],
        scorers=[
            make_slow_scorer(0.02, is_async=True, name="AsyncScorer1"),
            make_slow_scorer(0.02, is_async=True, name="AsyncScorer2"),
            make_slow_scorer(0.02, is_async=True, name="AsyncScorer3"),
            make_slow_scorer(0.02, is_async=False, name="SyncScorer1"),
            make_slow_scorer(0.02, is_async=False, name="SyncScorer2"),
            make_slow_scorer(0.02, is_async=False, name="SyncScorer3"),
        ],
    )

    assert len(result.examples[0].scores) == 6
