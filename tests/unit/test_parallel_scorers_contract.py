# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for EvalConfig.max_concurrent_tests.

Defines the public-API behavior: `max_concurrent_tests` caps how many
test cases (dataset examples) are processed at once. Each test case is
one task/agent invocation plus its scorers.

Tests are RED first — against the current codebase they fail, proving the
gap. They stay as the permanent contract once implementation lands.

Behavioral (counter + lock) rather than timing-based — timing tests are
flaky in CI and reward wall-clock speed, not semantic correctness.
"""

from __future__ import annotations

import asyncio

import pytest
from helpers.scorers import make_slow_scorer, passing_scorer
from pydantic import ValidationError

from agent_evals import (
    ExampleData,
    ExpectedResult,
    TaskResult,
    run_eval_async,
)
from agent_evals.core.types import EvalConfig


class _ConcurrencyProbe:
    """Tracks max-observed-concurrent invocations via a counter."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    async def exit(self) -> None:
        async with self._lock:
            self.active -= 1


def _dataset(n: int) -> list[ExampleData]:
    return [
        ExampleData(input=i, expected=ExpectedResult(expected=str(i))) for i in range(n)
    ]


# ----------------------------- schema contract ------------------------------


class TestEvalConfigSurface:
    """EvalConfig exposes a single concurrency knob."""

    def test_exposes_max_concurrent_tests(self):
        assert "max_concurrent_tests" in EvalConfig.model_fields

    def test_default_is_serial(self):
        # Default 1 — safe for stateful agents; users opt into parallelism.
        assert EvalConfig().max_concurrent_tests == 1

    def test_custom_value(self):
        assert EvalConfig(max_concurrent_tests=5).max_concurrent_tests == 5

    def test_must_be_positive(self):
        with pytest.raises(ValidationError, match="max_concurrent_tests"):
            EvalConfig(max_concurrent_tests=0)

    def test_unknown_fields_rejected(self):
        with pytest.raises(ValidationError):
            EvalConfig(**{"not_a_real_field": 5})


# --------------------------- behavioral contract ----------------------------


@pytest.mark.asyncio
async def test_max_concurrent_tests_one_serializes_tasks():
    """max_concurrent_tests=1 → only one task invocation runs at a time.

    Records peak concurrency inside the user's task function. With a cap of
    1, no two invocations may overlap, regardless of dataset size.
    """
    probe = _ConcurrencyProbe()

    async def probed_task(x):
        await probe.enter()
        try:
            # Hold long enough to guarantee overlap if cap is not enforced.
            await asyncio.sleep(0.01)
            return TaskResult(output=str(x))
        finally:
            await probe.exit()

    await run_eval_async(
        task=probed_task,
        dataset=_dataset(8),
        scorers=[passing_scorer],
        config=EvalConfig(max_concurrent_tests=1),
    )

    assert probe.peak == 1, f"expected serial tasks, saw peak={probe.peak}"


@pytest.mark.asyncio
async def test_max_concurrent_tests_caps_task_concurrency():
    """max_concurrent_tests=N → peak task concurrency is bounded by N."""
    probe = _ConcurrencyProbe()
    cap = 3

    async def probed_task(x):
        await probe.enter()
        try:
            await asyncio.sleep(0.02)
            return TaskResult(output=str(x))
        finally:
            await probe.exit()

    # 16 examples — without a cap all would fan out at once.
    await run_eval_async(
        task=probed_task,
        dataset=_dataset(16),
        scorers=[passing_scorer],
        config=EvalConfig(max_concurrent_tests=cap),
    )

    assert probe.peak <= cap, f"peak={probe.peak} exceeded cap={cap}"
    assert probe.peak >= 2, (
        f"test is not exercising parallelism; peak={probe.peak} "
        "(expected at least 2 concurrent before the cap kicked in)"
    )


@pytest.mark.asyncio
async def test_sync_scorer_does_not_block_event_loop():
    """A blocking sync scorer must not freeze the event loop.

    Independent of max_concurrent_tests — sync scorers go through
    asyncio.to_thread so other async work can interleave.
    """
    ticks = 0
    stop = asyncio.Event()

    async def sentinel() -> None:
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            await asyncio.sleep(0.005)

    blocking_sync_scorer = make_slow_scorer(0.1, is_async=False)  # 100ms blocking

    sentinel_task = asyncio.create_task(sentinel())
    try:
        await run_eval_async(
            task=lambda x: TaskResult(output=str(x)),
            dataset=_dataset(1),
            scorers=[blocking_sync_scorer],
            config=EvalConfig(max_concurrent_tests=1),
        )
    finally:
        stop.set()
        await sentinel_task

    # 100ms sleep / 5ms tick ≈ 20 ticks if loop stays responsive. Be generous.
    assert ticks >= 5, (
        f"event loop appeared blocked during sync scorer execution "
        f"(ticks={ticks}; expected >=5)"
    )
