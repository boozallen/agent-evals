# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Parallel execution proof tests.

Mathematical proof that scorers execute in parallel, not sequentially.
Tests use artificial delays (asyncio.sleep) to prove execution patterns.
"""

import time

import pytest
from helpers.scorers import make_slow_scorer

from agent_evals import TaskResult, run_eval_async
from agent_evals.core.types import ExampleData


@pytest.mark.asyncio
async def test_three_scorers_one_second_each_complete_in_one_second():
    """Prove parallel execution: 3 scorers @ 1s each = ~1s total (not 3s).

    This is mathematical proof of parallel execution:
    - Sequential: 3 × 1s = 3s total
    - Parallel: max(1s, 1s, 1s) = 1s total

    If test completes in ~1s, scorers ran in parallel. If ~3s, they ran sequentially.
    """

    slow_scorer_1 = make_slow_scorer(1.0, name="Slow1")
    slow_scorer_2 = make_slow_scorer(1.0, name="Slow2")
    slow_scorer_3 = make_slow_scorer(1.0, name="Slow3")

    def simple_task(x):
        return TaskResult(output=f"output_{x}")

    start = time.perf_counter()
    result = await run_eval_async(
        task=simple_task,
        dataset=[ExampleData(input="test")],
        scorers=[slow_scorer_1, slow_scorer_2, slow_scorer_3],
    )
    duration = time.perf_counter() - start

    # Verify all scorers executed
    assert len(result.examples[0].scores) == 3
    assert "Slow1" in result.examples[0].scores
    assert "Slow2" in result.examples[0].scores
    assert "Slow3" in result.examples[0].scores

    # Verify parallel execution: ~1s (not ~3s)
    # Allow 0.5s overhead for test framework, file I/O, etc.
    assert duration < 1.5, f"Expected ~1s (parallel), got {duration:.2f}s (sequential?)"

    print(
        f"✓ Parallel execution confirmed: 3 scorers @ 1s = {duration:.2f}s total (not 3s)"
    )


@pytest.mark.asyncio
async def test_five_async_scorers_execute_concurrently():
    """Test 5 async scorers execute concurrently and all return correct Scores."""

    # Create 5 scorers with delays: 0.5s, 0.6s, 0.7s, 0.8s, 0.9s
    # Max delay = 0.9s, so total should be ~0.9s (not 3.5s if sequential)
    scorers = [
        make_slow_scorer(0.5, name="S1"),
        make_slow_scorer(0.6, name="S2"),
        make_slow_scorer(0.7, name="S3"),
        make_slow_scorer(0.8, name="S4"),
        make_slow_scorer(0.9, name="S5"),
    ]

    def simple_task(x):
        return TaskResult(output=x)

    start = time.perf_counter()
    result = await run_eval_async(
        task=simple_task,
        dataset=[ExampleData(input="test")],
        scorers=scorers,
    )
    duration = time.perf_counter() - start

    # Verify all 5 scorers executed
    assert len(result.examples[0].scores) == 5
    for i in range(1, 6):
        assert f"S{i}" in result.examples[0].scores
        assert result.examples[0].scores[f"S{i}"].value == 1.0

    # Verify parallel: ~0.9s (max delay) not ~3.5s (sum of delays)
    assert duration < 1.4, f"Expected ~0.9s (parallel), got {duration:.2f}s"

    print(f"✓ 5 scorers executed in parallel: {duration:.2f}s (not 3.5s)")
