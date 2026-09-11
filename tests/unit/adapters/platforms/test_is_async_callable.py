# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ``is_async_callable`` in adapters/platforms/utils.py.

The predicate decides whether a task or scorer is awaited on the event loop
or offloaded to a worker thread, so both false negatives (an async callable
treated as sync) and false positives (a blocking callable awaited on the
loop) are behavioral bugs.
"""

import functools

import pytest

from agent_evals.adapters.platforms.utils import is_async_callable


class AsyncCallable:
    async def __call__(self):  # pragma: no cover - never invoked
        return 1


class SyncCallable:
    def __call__(self):  # pragma: no cover - never invoked
        return 1


async def _async_fn():  # pragma: no cover - never invoked
    return 1


def _sync_fn():  # pragma: no cover - never invoked
    return 1


@pytest.mark.parametrize(
    ("label", "func"),
    [
        ("async function", _async_fn),
        ("async callable class instance", AsyncCallable()),
        ("partial wrapping an async function", functools.partial(_async_fn)),
        ("bound async __call__", AsyncCallable().__call__),
    ],
)
def test_detects_async_callables(label, func):
    assert is_async_callable(func) is True, label


@pytest.mark.parametrize(
    ("label", "func"),
    [
        ("sync function", _sync_fn),
        ("sync callable class instance", SyncCallable()),
        ("lambda", lambda: 1),
        ("builtin", len),
        # The class object itself is a constructor, not an async callable:
        # calling it returns an instance, never an awaitable.
        ("async callable class object", AsyncCallable),
    ],
)
def test_rejects_sync_callables(label, func):
    assert is_async_callable(func) is False, label


def test_detects_asyncmock_but_not_magicmock():
    """Mocks are callable classes, so they exercise the ``__call__`` path.

    Users mocking scorers in their own suites depend on this: an AsyncMock
    must be awaited, a MagicMock must not be.
    """
    from unittest.mock import AsyncMock, MagicMock

    assert is_async_callable(AsyncMock()) is True
    assert is_async_callable(MagicMock()) is False


def test_partial_wrapping_async_callable_class_is_a_known_blind_spot():
    """Documents the limit the docstring warns about.

    ``functools.partial`` around an *instance* hides the async ``__call__``
    from introspection, so detection reports False. Callers pair this
    predicate with an awaitable check to stay correct for such shapes.
    """
    assert is_async_callable(functools.partial(AsyncCallable())) is False
