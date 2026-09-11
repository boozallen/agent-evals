# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Async utilities for testing.

Provides:
- is_async_callable: Detect if a callable is async
"""

import inspect
from collections.abc import Callable
from typing import Any


def is_async_callable(func: Callable[..., Any]) -> bool:
    """Detect if a callable is an async coroutine function.

    Uses inspect.iscoroutinefunction() for reliable detection of async functions.
    Handles:
    - Regular async functions (async def)
    - Async methods
    - Callable classes with async __call__
    - Partial functions wrapping async callables

    Args:
        func: Callable to check

    Returns:
        True if func is an async coroutine function, False otherwise

    Examples:
        >>> async def async_fn():
        ...     pass
        >>> def sync_fn():
        ...     pass
        >>> is_async_callable(async_fn)
        True
        >>> is_async_callable(sync_fn)
        False

        Callable class with async __call__:
        >>> class AsyncCallable:
        ...     async def __call__(self):
        ...         pass
        >>> obj = AsyncCallable()
        >>> is_async_callable(obj)
        True
    """
    # Check if it's a coroutine function directly
    if inspect.iscoroutinefunction(func):
        return True

    # Handle callable classes (check __call__ method)
    if callable(func) and not inspect.isfunction(func):
        return inspect.iscoroutinefunction(func.__call__)

    return False
