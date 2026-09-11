# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for async callable-class tasks and scorers on LocalPlatform.

``inspect.iscoroutinefunction`` inspects the object it is handed, not that
object's ``__call__``, so it returns False for an instance whose ``__call__``
is ``async def``. The Scorer Protocol documents callable classes as a
supported shape, so LocalPlatform must classify them as async rather than
dispatching them to a worker thread and returning the coroutine unawaited.
"""

import asyncio

import pytest

from agent_evals.adapters.platforms.local import LocalPlatform
from agent_evals.core.types import (
    ExampleData,
    ExpectedResult,
    Score,
    TaskResult,
)


class AsyncCallableAgent:
    """Agent shaped as a callable class with an async ``__call__``."""

    def __init__(self) -> None:
        self.thread_name: str | None = None

    async def __call__(self, input_value):
        import threading

        self.thread_name = threading.current_thread().name
        return TaskResult(output="real answer")


class AsyncCallableScorer:
    """Scorer shaped as a callable class with an async ``__call__``."""

    __name__ = "async_cc"

    def __init__(self) -> None:
        self.thread_name: str | None = None

    async def __call__(self, result: TaskResult, expected=None, **context) -> Score:
        import threading

        self.thread_name = threading.current_thread().name
        return Score(
            name="async_cc", value=1.0, passed=True, reasoning=None, metadata={}
        )


def _sync_scorer(result: TaskResult, expected=None, **context) -> Score:
    """Scores 1.0 only when the task output survived intact."""
    ok = result.output == "real answer"
    return Score(
        name="exact", value=1.0 if ok else 0.0, passed=ok, reasoning=None, metadata={}
    )


@pytest.mark.asyncio
async def test_invoke_evaluator_returns_score_for_async_callable_class():
    """An async callable-class scorer resolves to a Score, not a coroutine."""
    score = await LocalPlatform()._invoke_evaluator(
        AsyncCallableScorer(), TaskResult(output="x"), None
    )

    assert isinstance(score, Score)
    assert score.value == 1.0


@pytest.mark.asyncio
async def test_execute_task_returns_output_for_async_callable_class():
    """An async callable-class task resolves to its output, not a coroutine."""
    output = await LocalPlatform()._execute_task_async(AsyncCallableAgent(), "hi")

    assert not asyncio.iscoroutine(output)
    assert isinstance(output, TaskResult)
    assert output.output == "real answer"


@pytest.mark.asyncio
async def test_async_callables_run_on_event_loop_not_worker_thread():
    """Async callables are awaited on the loop, never offloaded to a thread.

    Offloading is reserved for blocking sync callables. Running an async
    callable on a worker thread would leave it without the running loop it
    expects.
    """
    loop_thread = __import__("threading").current_thread().name
    agent, scorer = AsyncCallableAgent(), AsyncCallableScorer()

    await LocalPlatform()._execute_task_async(agent, "hi")
    await LocalPlatform()._invoke_evaluator(scorer, TaskResult(output="x"), None)

    assert agent.thread_name == loop_thread
    assert scorer.thread_name == loop_thread


@pytest.mark.asyncio
async def test_aevaluate_scores_async_callable_class_task_and_scorer(tmp_path):
    """End-to-end: neither the task nor the scorer is silently dropped."""
    from agent_evals.adapters.platforms.local import LocalConfig

    result = await LocalPlatform().aevaluate(
        AsyncCallableAgent(),
        [ExampleData(input="hi", expected=ExpectedResult(expected="real answer"))],
        [AsyncCallableScorer(), _sync_scorer],
        platform=LocalConfig(output_dir=str(tmp_path)),
    )

    example = result.examples[0]
    assert example.error is None
    assert example.output == "real answer"
    # The sync scorer proves the task output arrived intact rather than as an
    # empty string standing in for a dropped coroutine.
    assert result.scores == {"async_cc": 1.0, "exact": 1.0}
