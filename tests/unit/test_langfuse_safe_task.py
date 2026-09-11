# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""LangFuse safe-task wrapper contracts.

``LangFusePlatform._make_safe_task`` adapts the langfuse SDK's
``task(item={...})`` calling shape to the agent-evals contract
``task(input_value)``, captures user-task exceptions and timing into
sidecar dicts so the adapter can surface them in
``EvalResult.examples``, and defends against several flavors of
misbehaving user task. Each test class names the property it covers.
"""

import asyncio
import functools

import pytest

from agent_evals.adapters.platforms.langfuse import LangFusePlatform
from agent_evals.core.types import TaskResult


def _new_adapter():
    # Return type intentionally unannotated: ty cannot infer the exact
    # type of a class decorated by the generic registry, and the inferred
    # return type is sufficient for this internal test helper.
    return LangFusePlatform()


async def _maybe_await(value):
    return await value if asyncio.iscoroutine(value) else value


class TestSafeTaskBasics:
    def test_success_returns_user_output(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        def task(input_value):
            return f"hello {input_value}"

        wrapped = adapter._make_safe_task(task, errors, durations)
        item = {"input": "world"}
        out = wrapped(item=item)

        assert out == "hello world"
        assert errors == {}
        assert id(item) in durations

    def test_exception_captured_in_sidecar(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        def task(input_value):
            raise RuntimeError(f"boom on {input_value}")

        wrapped = adapter._make_safe_task(task, errors, durations)
        item = {"input": "alpha"}
        out = wrapped(item=item)

        assert isinstance(out, TaskResult)
        assert out.output == "Error: boom on alpha"
        assert errors[id(item)] == "boom on alpha"
        assert id(item) in durations

    def test_duration_recorded_even_when_task_raises(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        def task(input_value):
            raise RuntimeError("fail")

        wrapped = adapter._make_safe_task(task, errors, durations)
        item = {"input": "x"}
        wrapped(item=item)

        assert errors[id(item)] == "fail"
        assert id(item) in durations

    def test_user_task_receives_bare_input_not_dict(self):
        """The user task receives ``item["input"]``, not the SDK
        wrapper dict. This is the cross-platform contract: a single
        task body must work against every platform adapter.
        """
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        seen: list = []

        def task(input_value):
            seen.append(input_value)
            return input_value

        wrapped = adapter._make_safe_task(task, errors, durations)
        wrapped(item={"input": "alpha", "expected_output": "ignored"})

        assert seen == ["alpha"], (
            "user task must receive the bare input value, not the SDK item dict"
        )


class TestSafeTaskSidecarKeyingIsCollisionFree:
    """Sidecar keys are derived from ``id(item)`` (the SDK dict).

    Each ``langfuse_dataset`` entry is a freshly-constructed dict
    even when two scenarios share an input value, so dict identity
    is unique per case. Keying off the bare ``input_value`` instead
    would collide when two cases share an interned scalar (e.g. two
    scenarios both with ``input="alpha"``), losing per-item error
    attribution under concurrency.
    """

    def test_duplicate_input_values_get_distinct_sidecar_keys(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        def task(input_value):
            raise RuntimeError(f"err-{input_value}")

        wrapped = adapter._make_safe_task(task, errors, durations)

        # Two distinct dict instances with the same input value.
        item_a = {"input": "duplicate"}
        item_b = {"input": "duplicate"}

        # Precondition: the bare values share id() under string interning,
        # so keying sidecars off id(input_value) would collide. The dicts
        # do not, so id(item) is collision-free. This test fails closed if
        # CPython ever stops interning short string literals — which would
        # silently invalidate the invariant elsewhere too.
        assert id(item_a["input"]) == id(item_b["input"]), (
            "interned-scalar precondition: this test exercises the case "
            "where keying off id(input_value) would collide"
        )
        assert id(item_a) != id(item_b)

        wrapped(item=item_a)
        wrapped(item=item_b)

        assert errors[id(item_a)] == "err-duplicate"
        assert errors[id(item_b)] == "err-duplicate"
        assert id(item_a) in durations
        assert id(item_b) in durations
        assert len(errors) == 2
        assert len(durations) == 2


class TestSafeTaskConcurrency:
    @pytest.mark.asyncio
    async def test_concurrent_failures_attributed_to_correct_items(self):
        """Per-item error attribution holds under concurrent execution."""
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        async def task(input_value):
            await asyncio.sleep(0.01)
            raise RuntimeError(f"err-{input_value}")

        wrapped = adapter._make_safe_task(task, errors, durations)
        items = [{"input": f"i{i}"} for i in range(20)]
        await asyncio.gather(*(wrapped(item=it) for it in items))

        for it in items:
            assert errors[id(it)] == f"err-{it['input']}"
            assert id(it) in durations


class TestSafeTaskAsync:
    @pytest.mark.asyncio
    async def test_async_user_task_success(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        async def task(input_value):
            await asyncio.sleep(0.01)
            return f"async-{input_value}"

        wrapped = adapter._make_safe_task(task, errors, durations)
        item = {"input": "a"}
        result = await wrapped(item=item)
        assert result == "async-a"
        assert id(item) in durations

    @pytest.mark.asyncio
    async def test_async_user_task_exception_captured(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        async def task(input_value):
            raise ValueError("async fail")

        wrapped = adapter._make_safe_task(task, errors, durations)
        item = {"input": "a"}
        out = await wrapped(item=item)

        assert isinstance(out, TaskResult)
        assert out.output == "Error: async fail"
        assert errors[id(item)] == "async fail"


class TestSafeTaskTypeErrorIsNotControlFlow:
    """A ``TypeError`` raised inside the user task body is captured once.

    Catching ``TypeError`` at the call site (to detect calling-shape
    mismatches) would re-execute the user body on any in-body
    ``TypeError``, doubling side effects — a critical concern for
    tasks that drive LLM calls or other paid I/O.
    """

    def test_user_task_body_typeerror_captured_once(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}
        invocation_count = {"n": 0}

        def task(input_value):
            invocation_count["n"] += 1
            raise TypeError("user-body type error")

        wrapped = adapter._make_safe_task(task, errors, durations)
        item = {"input": "x"}
        out = wrapped(item=item)

        assert isinstance(out, TaskResult)
        assert errors[id(item)] == "user-body type error"
        assert invocation_count["n"] == 1, (
            "user task body must not be re-executed on TypeError"
        )

    @pytest.mark.asyncio
    async def test_async_user_task_body_typeerror_captured_once(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}
        invocation_count = {"n": 0}

        async def task(input_value):
            invocation_count["n"] += 1
            raise TypeError("async user-body TE")

        wrapped = adapter._make_safe_task(task, errors, durations)
        item = {"input": "x"}
        out = await wrapped(item=item)

        assert isinstance(out, TaskResult)
        assert errors[id(item)] == "async user-body TE"
        assert invocation_count["n"] == 1


class TestSafeTaskCallingConvention:
    """The wrapper invokes the user task with a single positional arg.

    The cross-platform contract is ``task(input_value)``; the wrapper
    has no signature-sniffing branch and no fallback path. Anything
    other than positional invocation is the wrapper exceeding its
    role.
    """

    def test_positional_arg_invocation(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        observed: list = []

        def task(input_value):
            observed.append(input_value)
            return input_value

        wrapped = adapter._make_safe_task(task, errors, durations)
        wrapped(item={"input": "positional"})
        assert observed == ["positional"]
        assert errors == {}


class TestSafeTaskCoroutineFromSyncCallable:
    """Callables that return coroutines without being ``async def`` are awaited.

    Covers ``functools.partial(async_fn)`` and instances of classes with
    ``async def __call__`` — both pass through ``iscoroutinefunction`` as
    False, so the wrapper must detect and await their result instead of
    handing an unawaited coroutine back to the SDK.
    """

    @pytest.mark.asyncio
    async def test_partial_of_async_function_caught(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        async def real_task(extra, input_value):
            assert extra == "ctx"
            raise RuntimeError(f"partial-boom-{input_value}")

        wrapped = adapter._make_safe_task(
            functools.partial(real_task, "ctx"), errors, durations
        )

        item = {"input": "alpha"}
        out = await _maybe_await(wrapped(item=item))

        assert isinstance(out, TaskResult)
        assert out.output == "Error: partial-boom-alpha"
        assert errors[id(item)] == "partial-boom-alpha"

    @pytest.mark.asyncio
    async def test_callable_class_with_async_call_caught(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        class _AsyncCallable:
            async def __call__(self, input_value):
                raise ValueError(f"class-boom-{input_value}")

        wrapped = adapter._make_safe_task(_AsyncCallable(), errors, durations)

        item = {"input": "beta"}
        out = await _maybe_await(wrapped(item=item))

        assert isinstance(out, TaskResult)
        assert out.output == "Error: class-boom-beta"
        assert errors[id(item)] == "class-boom-beta"


class TestSafeTaskMalformedItem:
    """Items missing the ``"input"`` key are passed through as ``None``.

    The adapter constructs every SDK item with an ``"input"`` key, so
    this path should be unreachable in production. The test pins the
    fallback so a future code path that builds items differently
    fails loudly (the user task receiving ``None`` is visible in
    captured output) rather than producing a confusing error from
    deeper in the user code.
    """

    def test_dict_without_input_key_passes_none_to_user_task(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        seen: list = []

        def task(input_value):
            seen.append(input_value)
            return f"got {input_value!r}"

        wrapped = adapter._make_safe_task(task, errors, durations)
        item = {"expected_output": "x"}  # no "input" key
        out = wrapped(item=item)

        assert seen == [None]
        assert out == "got None"
        assert errors == {}
        assert id(item) in durations


class TestSafeTaskMissingItem:
    """SDK contract violations (no item passed) get unique sidecar keys."""

    def test_missing_item_invocations_do_not_collide(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        def task(input_value):
            raise RuntimeError("no item")

        wrapped = adapter._make_safe_task(task, errors, durations)
        wrapped()
        wrapped()

        # Two distinct invocations → two distinct sidecar entries.
        assert len(errors) == 2
        assert len(durations) == 2
        assert all(msg == "no item" for msg in errors.values())


class TestSafeTaskAsyncCoroutineDefense:
    """``safe_task_async`` defends against ``iscoroutinefunction`` false positives."""

    @pytest.mark.asyncio
    async def test_async_marked_callable_returning_non_coroutine(self):
        """Some decorators set ``_is_coroutine`` to spoof async detection but
        return a plain value. The wrapper must not blindly await the result.
        """
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        def forged(input_value):
            return f"plain-{input_value}"

        # asyncio uses an internal marker to identify coroutine functions; setting
        # it on a plain function makes iscoroutinefunction return True without
        # changing what the function actually does.
        forged._is_coroutine = asyncio.coroutines._is_coroutine
        assert asyncio.iscoroutinefunction(forged)

        wrapped = adapter._make_safe_task(forged, errors, durations)
        item = {"input": "x"}
        out = await wrapped(item=item)
        assert out == "plain-x"
        assert errors == {}


class TestSafeStrFallback:
    """An exception with a broken ``__str__`` is captured as a typename placeholder."""

    def test_unrepr_able_exception_yields_typename_placeholder(self):
        adapter = _new_adapter()
        errors: dict[int, str] = {}
        durations: dict[int, float] = {}

        class _BrokenStrError(Exception):
            def __str__(self):
                raise RuntimeError("oh no")

        def task(input_value):
            raise _BrokenStrError()

        wrapped = adapter._make_safe_task(task, errors, durations)
        item = {"input": "x"}
        out = wrapped(item=item)

        assert isinstance(out, TaskResult)
        assert errors[id(item)] == "<_BrokenStrError: unrepr-able>"
        assert out.output == "Error: <_BrokenStrError: unrepr-able>"
