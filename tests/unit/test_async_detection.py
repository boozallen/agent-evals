# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for async/sync callable detection.

Tests the is_async_callable() utility function from tests/helpers/async_helpers.py.
"""

import asyncio
from functools import partial

import pytest

from agent_evals import ExpectedResult
from agent_evals.core.types import ExampleData


def test_is_async_callable_identifies_async_functions():
    """Test is_async_callable() correctly identifies async functions.

    Task: T031
    """
    from helpers.async_helpers import is_async_callable

    # Async function
    async def async_func():
        pass

    assert is_async_callable(async_func) is True


def test_is_async_callable_identifies_sync_functions():
    """Test is_async_callable() correctly identifies sync functions.

    Task: T031
    """
    from helpers.async_helpers import is_async_callable

    # Sync function
    def sync_func():
        pass

    assert is_async_callable(sync_func) is False


def test_is_async_callable_identifies_lambda_functions():
    """Test is_async_callable() correctly identifies lambda functions as sync.

    Task: T031
    """
    from helpers.async_helpers import is_async_callable

    # Lambda function (always sync)
    def lambda_func(x):
        return x * 2

    assert is_async_callable(lambda_func) is False


def test_is_async_callable_identifies_callable_classes():
    """Test is_async_callable() correctly identifies callable classes.

    Task: T031
    """
    from helpers.async_helpers import is_async_callable

    # Callable class with sync __call__
    class SyncCallable:
        def __call__(self, output, expected):
            return {"value": 1.0}

    # Callable class with async __call__
    class AsyncCallable:
        async def __call__(self, output, expected):
            await asyncio.sleep(0.01)
            return {"value": 1.0}

    sync_obj = SyncCallable()
    async_obj = AsyncCallable()

    assert is_async_callable(sync_obj) is False
    assert is_async_callable(async_obj) is True


def test_is_async_callable_handles_partial_functions():
    """Test is_async_callable() correctly handles partial functions.

    Task: T031
    """
    from helpers.async_helpers import is_async_callable

    # Sync function with partial application
    def sync_func(a, b, c):
        return a + b + c

    # Async function with partial application
    async def async_func(a, b, c):
        await asyncio.sleep(0.01)
        return a + b + c

    sync_partial = partial(sync_func, 1, 2)
    async_partial = partial(async_func, 1, 2)

    assert is_async_callable(sync_partial) is False
    assert is_async_callable(async_partial) is True


def test_is_async_callable_handles_coroutine_functions():
    """Test is_async_callable() correctly identifies coroutine functions.

    Task: T031
    """
    from helpers.async_helpers import is_async_callable

    # Regular async function
    async def coro_func():
        return 42

    # Verify it's detected as async
    assert is_async_callable(coro_func) is True

    # Verify calling it returns a coroutine (not the result)
    result = coro_func()
    assert asyncio.iscoroutine(result)

    # Clean up the coroutine
    result.close()


@pytest.mark.asyncio
async def test_context_passing_works_for_sync_and_async_scorers():
    """Test that **context works correctly for both sync and async scorers.

    This verifies that context parameters are properly passed through to scorers
    of both types during async evaluation.

    Task: T035
    """
    from agent_evals import Score, TaskResult, run_eval_async

    # Track context received by scorers
    received_context = {}

    # Async scorer that captures context
    async def async_context_scorer(
        result: TaskResult, expected: ExpectedResult | None = None
    ):
        await asyncio.sleep(0.01)
        context = result.context or {}
        received_context["async_scorer"] = context.copy()
        return Score(name="AsyncContextScorer", value=1.0, passed=True)

    # Sync scorer that captures context
    def sync_context_scorer(
        result: TaskResult, expected: ExpectedResult | None = None
    ) -> Score:
        context = result.context or {}
        received_context["sync_scorer"] = context.copy()
        return Score(name="SyncContextScorer", value=1.0, passed=True)

    # Task function
    def task(input_value):
        return TaskResult(output=f"Output: {input_value}")

    # Run evaluation with custom context
    result = await run_eval_async(
        task=task,
        dataset=[
            ExampleData(
                input="test",
                expected=ExpectedResult(expected="Output: test"),
                output=TaskResult(
                    output="",
                    context={
                        "custom_field": "custom_value",
                        "extra_key": "extra_value",
                    },
                ),
            )
        ],
        scorers=[async_context_scorer, sync_context_scorer],
    )

    # Verify both scorers received context
    assert "async_scorer" in received_context
    assert "sync_scorer" in received_context

    # Verify context contains expected fields
    async_ctx = received_context["async_scorer"]
    sync_ctx = received_context["sync_scorer"]

    # Both should have custom_field from dataset (pass-through)
    assert "custom_field" in async_ctx
    assert async_ctx["custom_field"] == "custom_value"
    assert "custom_field" in sync_ctx
    assert sync_ctx["custom_field"] == "custom_value"

    # Both should have extra_key from context dict
    assert "extra_key" in async_ctx
    assert async_ctx["extra_key"] == "extra_value"
    assert "extra_key" in sync_ctx
    assert sync_ctx["extra_key"] == "extra_value"

    # Verify evaluation completed successfully
    assert len(result.examples) == 1
    assert len(result.examples[0].scores) == 2
