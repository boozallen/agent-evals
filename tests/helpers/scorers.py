# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared canonical test scorers.

Every scorer here matches the canonical Scorer Protocol shape:
``(result: TaskResult, expected: ExpectedResult | None = None, **context) -> Score``
(async variants return ``Awaitable[Score]``). Tests that need an
incidental scorer import from here instead of defining their own, so
the suite is not coupled to scorer-signature spelling.

Provides:
- passing_scorer: always passes (value=1.0)
- failing_scorer: always fails (value=0.0)
- raising_scorer: raises RuntimeError (for failure-isolation tests)
- async_passing_scorer: async, always passes
- make_slow_scorer: factory for a scorer that sleeps `delay` then passes
  (sync or async) — for parallelism/timing proofs
- make_context_check_scorer: factory whose Score.passed is a predicate on
  (result, expected) — for outcome-level context/propagation assertions
"""

import asyncio
import time
from collections.abc import Awaitable, Callable

from agent_evals.core.types import ExpectedResult, Score, TaskResult

Scorer = Callable[..., Score | Awaitable[Score]]


def passing_scorer(
    result: TaskResult, expected: ExpectedResult | None = None, **context
) -> Score:
    """Canonical scorer that always passes."""
    return Score(name="passing", value=1.0, passed=True)


def failing_scorer(
    result: TaskResult, expected: ExpectedResult | None = None, **context
) -> Score:
    """Canonical scorer that always fails."""
    return Score(name="failing", value=0.0, passed=False)


def raising_scorer(
    result: TaskResult, expected: ExpectedResult | None = None, **context
) -> Score:
    """Canonical scorer that raises — for failure-isolation tests."""
    raise RuntimeError("scorer failure (intentional)")


async def async_passing_scorer(
    result: TaskResult, expected: ExpectedResult | None = None, **context
) -> Score:
    """Canonical async scorer that always passes."""
    return Score(name="async_passing", value=1.0, passed=True)


def make_slow_scorer(
    delay: float, *, is_async: bool = True, name: str = "slow"
) -> Scorer:
    """Build a scorer that waits ``delay`` seconds then passes.

    is_async=True -> async scorer using asyncio.sleep (for concurrency proofs).
    is_async=False -> sync scorer using time.sleep (for to_thread / blocking proofs).
    """
    if is_async:

        async def _async_slow(
            result: TaskResult, expected: ExpectedResult | None = None, **context
        ) -> Score:
            await asyncio.sleep(delay)
            return Score(name=name, value=1.0, passed=True)

        return _async_slow

    def _sync_slow(
        result: TaskResult, expected: ExpectedResult | None = None, **context
    ) -> Score:
        time.sleep(delay)
        return Score(name=name, value=1.0, passed=True)

    return _sync_slow


def make_context_check_scorer(
    predicate: Callable[[TaskResult, ExpectedResult | None], bool],
    *,
    is_async: bool = False,
    name: str = "context_check",
) -> Scorer:
    """Build a scorer whose ``Score.passed`` is ``predicate(result, expected)``.

    Use to assert context/output PROPAGATION at the outcome level instead of
    spying on captured scorer arguments. The test asserts
    ``result.examples[i].scores[name].passed is True`` rather than reaching
    into a ``captured[...]`` dict — so it survives scorer-wiring refactors.
    """
    if is_async:

        async def _async_check(
            result: TaskResult, expected: ExpectedResult | None = None, **context
        ) -> Score:
            ok = predicate(result, expected)
            return Score(name=name, value=1.0 if ok else 0.0, passed=ok)

        # The braintrust/langfuse/mlflow adapters derive the scores-dict key
        # from the callable's __qualname__/__name__ (mlflow prefers __name__);
        # the local adapter keys by the returned Score.name. We set both
        # __name__/__qualname__ AND Score(name=...) to `name` so every adapter
        # produces the same key. Without the __name__/__qualname__ patch, the
        # closure name ("make_context_check_scorer.<locals>._async_check")
        # would leak through on the name-inferring adapters.
        _async_check.__name__ = name
        _async_check.__qualname__ = name
        return _async_check

    def _sync_check(
        result: TaskResult, expected: ExpectedResult | None = None, **context
    ) -> Score:
        ok = predicate(result, expected)
        return Score(name=name, value=1.0 if ok else 0.0, passed=ok)

    _sync_check.__name__ = name
    _sync_check.__qualname__ = name
    return _sync_check
