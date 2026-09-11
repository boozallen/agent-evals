# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Performance comparison tests for async vs sync execution.

These tests verify that async parallel execution provides meaningful
performance improvements over sequential execution.
"""

import time

import pytest
from helpers.scorers import make_slow_scorer

from agent_evals import ExpectedResult, TaskResult, run_eval_async
from agent_evals.core.types import ExampleData


@pytest.mark.asyncio
async def test_async_parallel_is_faster_than_sequential():
    """Compare async parallel execution vs sequential execution.

    With 5 scorers @ 0.2s each:
    - Sequential would take: 5 × 0.2s = 1.0s
    - Parallel should take: max(0.2s) = 0.2s
    - Speedup: 5x (or close to it)

    This test proves that async parallel execution is significantly faster.
    """

    # 5 scorers @ 0.2s each
    delay = 0.2
    scorers = [make_slow_scorer(delay, name=f"Scorer{i}") for i in range(1, 6)]

    def simple_task(x):
        return TaskResult(output=f"output_{x}")

    # Measure async parallel execution time
    start = time.perf_counter()
    result = await run_eval_async(
        task=simple_task,
        dataset=[ExampleData(input="test", expected=ExpectedResult(expected="test"))],
        scorers=scorers,
    )
    parallel_duration = time.perf_counter() - start

    # Verify all scorers executed
    assert len(result.examples[0].scores) == 5

    # Expected times:
    # - Sequential: 5 × 0.2s = 1.0s
    # - Parallel: max(0.2s) = 0.2s
    # - Speedup: ~5x

    # Parallel should be close to max delay (0.2s), not sum (1.0s)
    # Allow some overhead for framework, I/O, etc.
    max_expected_parallel = delay + 0.5  # 0.2s + 0.5s overhead = 0.8s

    assert parallel_duration < max_expected_parallel, (
        f"Parallel took {parallel_duration:.2f}s, expected < {max_expected_parallel:.2f}s"
    )

    # Calculate theoretical sequential time
    sequential_theoretical = len(scorers) * delay

    # Speedup should be at least 2x (conservative estimate)
    speedup = sequential_theoretical / parallel_duration
    assert speedup >= 2.0, f"Expected at least 2x speedup, got {speedup:.2f}x"

    print(
        f"✓ Performance: {len(scorers)} scorers @ {delay}s each = {parallel_duration:.2f}s (speedup: {speedup:.2f}x)"
    )


@pytest.mark.asyncio
async def test_parallel_execution_with_varying_delays():
    """Test parallel execution with scorers that have different delays.

    Total time should be approximately equal to the slowest scorer (max delay),
    not the sum of all delays.
    """

    fast_scorer = make_slow_scorer(0.1, name="Fast")
    medium_scorer = make_slow_scorer(0.3, name="Medium")
    slow_scorer = make_slow_scorer(0.5, name="Slow")

    def simple_task(x):
        return TaskResult(output=x)

    start = time.perf_counter()
    result = await run_eval_async(
        task=simple_task,
        dataset=[ExampleData(input="test", expected=ExpectedResult(expected="test"))],
        scorers=[fast_scorer, medium_scorer, slow_scorer],
    )
    duration = time.perf_counter() - start

    # Verify all scorers executed
    assert len(result.examples[0].scores) == 3

    # Total time should be ~0.5s (max delay), not ~0.9s (sum of delays)
    # Delays: 0.1s + 0.3s + 0.5s = 0.9s (sequential)
    # Expected: ~0.5s (parallel, max delay)
    assert duration < 0.8, f"Expected ~0.5s (max delay), got {duration:.2f}s"

    print(f"✓ Varying delays: max(0.1, 0.3, 0.5) = {duration:.2f}s (not 0.9s)")


@pytest.mark.asyncio
async def test_async_llm_scorers_vs_sequential_baseline():
    """Benchmark async LLM scorers vs sequential execution baseline.

    This test simulates realistic LLM scorer behavior to measure the
    performance improvement from native async execution.

    Expected:
    - With 5 LLM scorers @ 1.5s each on 10 examples
    - Sequential: 5 × 1.5s × 10 = 75s per scorer batch
    - Async (max_concurrent_tests=10): All 10 examples run concurrently
      -> 5 × 1.5s = 7.5s per example, but 10 in parallel
      -> ~7.5-10s total
    - Expected improvement: 40-60% faster (or more)

    This validates the core value proposition: native async execution
    dramatically improves performance for LLM-heavy evaluations.
    """

    # Simulate LLM scorer behavior (realistic 1.5s delay per call)
    llm_delay = 1.5

    # 5 LLM scorers
    num_scorers = 5
    async_scorers = [
        make_slow_scorer(llm_delay, is_async=True, name=f"AsyncLLM{i}")
        for i in range(num_scorers)
    ]
    sync_scorers = [
        make_slow_scorer(llm_delay, is_async=False, name=f"SyncLLM{i}")
        for i in range(num_scorers)
    ]

    # 10 test examples
    num_examples = 10
    dataset = [
        ExampleData(input=i, expected=ExpectedResult(expected=str(i)))
        for i in range(num_examples)
    ]

    def simple_task(x):
        # Simply return the input as output
        return TaskResult(output=str(x))

    # Measure ASYNC execution time with 10 concurrent tests
    print(
        f"\nBenchmarking: {num_scorers} scorers × {num_examples} examples @ {llm_delay}s/call"
    )
    print("Running async (native async scorers with max_concurrent_tests=10)...")
    start_async = time.perf_counter()

    from agent_evals import TaskResult
    from agent_evals.core.types import EvalConfig

    async_result = await run_eval_async(
        task=simple_task,
        dataset=dataset,
        scorers=async_scorers,
        config=EvalConfig(max_concurrent_tests=10),
    )
    async_duration = time.perf_counter() - start_async

    # Measure SYNC execution time (sync scorers via asyncio.to_thread)
    print("Running sync scorers (baseline)...")
    start_sync = time.perf_counter()
    sync_result = await run_eval_async(
        task=simple_task,
        dataset=dataset,
        scorers=sync_scorers,
        config=EvalConfig(max_concurrent_tests=10),
    )
    sync_duration = time.perf_counter() - start_sync

    # Verify both produced same number of scores
    assert len(async_result.examples) == num_examples
    assert len(sync_result.examples) == num_examples
    assert len(async_result.examples[0].scores) == num_scorers
    assert len(sync_result.examples[0].scores) == num_scorers

    # Calculate improvement
    improvement = ((sync_duration - async_duration) / sync_duration) * 100
    speedup = sync_duration / async_duration

    print("\n📊 Performance Results:")
    print(f"  Async execution: {async_duration:.2f}s")
    print(f"  Sync baseline:   {sync_duration:.2f}s")
    print(f"  Improvement:     {improvement:.1f}%")
    print(f"  Speedup:         {speedup:.2f}x")

    # Verify async is significantly faster
    assert async_duration < sync_duration, (
        f"Async ({async_duration:.2f}s) should be faster than sync ({sync_duration:.2f}s)"
    )

    # Verify at least 30% improvement (conservative target)
    # Note: The actual improvement depends on concurrency settings
    # With max_concurrent_tests=10, we expect 40-60% improvement
    min_improvement = 30.0
    assert improvement >= min_improvement, (
        f"Expected ≥{min_improvement}% improvement, got {improvement:.1f}%"
    )

    # Theoretical analysis:
    # - Sync baseline: num_examples × num_scorers × llm_delay
    #   = 10 × 5 × 1.5s = 75s (all sequential)
    # - Async (max_concurrent_tests=10): num_scorers × llm_delay
    #   = 5 × 1.5s = 7.5s (10 examples run in parallel)
    # - Expected improvement: (75 - 7.5) / 75 = 90%

    # In practice, with thread pool for sync scorers, the baseline
    # is faster than fully sequential, so improvement is typically 40-60%

    print(
        f"\n✓ Native async execution is {improvement:.1f}% faster ({speedup:.2f}x speedup)"
    )
    print("✓ Validated: Async LLM scorers provide significant performance improvement")
