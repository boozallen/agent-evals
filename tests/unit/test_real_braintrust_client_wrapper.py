# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct tests for _RealBraintrustClient against SimpleNamespace stubs.

These tests pin the wrapper's defensive paths without touching the real
braintrust SDK. The wrapper is the only place in the adapter family that
constructs `braintrust.EvalCase` and calls `braintrust.EvalAsync`; these
tests verify the construction shape and the call kwargs.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from agent_evals.adapters.platforms.braintrust_client import _RealBraintrustClient

pytestmark = pytest.mark.requires_braintrust


class _StubEvalCase:
    """SimpleNamespace-style EvalCase stub: records its construction kwargs."""

    instances: list[dict] = []

    def __init__(self, **kwargs):
        type(self).instances.append(kwargs)
        self.__dict__.update(kwargs)


class _StubBraintrust:
    """Stub braintrust module exposing EvalCase + EvalAsync."""

    EvalCase = _StubEvalCase

    eval_async_calls: list[dict] = []
    eval_async_return: object = SimpleNamespace(results=[])

    @classmethod
    async def EvalAsync(cls, **kwargs):  # noqa: N802 - matches SDK symbol
        cls.eval_async_calls.append(kwargs)
        return cls.eval_async_return


@pytest.fixture(autouse=True)
def _reset_stubs():
    _StubEvalCase.instances = []
    _StubBraintrust.eval_async_calls = []
    _StubBraintrust.eval_async_return = SimpleNamespace(results=[])
    yield
    _StubEvalCase.instances = []
    _StubBraintrust.eval_async_calls = []


def test_run_eval_builds_one_evalcase_per_dict():
    """Wrapper builds an EvalCase per input dict with input/expected/metadata."""
    wrapper = _RealBraintrustClient(_StubBraintrust)

    asyncio.run(
        wrapper.run_eval(
            project="p",
            experiment="e",
            data=[
                {"input": "a", "expected": "x", "metadata": {"row": 0}},
                {"input": "b", "expected": "y", "metadata": {"row": 1}},
            ],
            task=lambda x: x,
            scorers=[],
            metadata={},
        )
    )

    assert len(_StubEvalCase.instances) == 2
    assert _StubEvalCase.instances[0] == {
        "input": "a",
        "expected": "x",
        "metadata": {"row": 0},
    }
    assert _StubEvalCase.instances[1] == {
        "input": "b",
        "expected": "y",
        "metadata": {"row": 1},
    }


def test_run_eval_handles_missing_expected_and_metadata():
    """Dict without `expected` or `metadata` -> EvalCase with None defaults."""
    wrapper = _RealBraintrustClient(_StubBraintrust)

    asyncio.run(
        wrapper.run_eval(
            project="p",
            experiment="e",
            data=[{"input": "a"}],
            task=lambda x: x,
            scorers=[],
            metadata={},
        )
    )

    assert _StubEvalCase.instances == [
        {"input": "a", "expected": None, "metadata": None}
    ]


def test_run_eval_passes_correct_kwargs_to_evalasync():
    """Wrapper maps adapter kwargs -> SDK kwargs (project=name, experiment=experiment_name)."""
    wrapper = _RealBraintrustClient(_StubBraintrust)

    def task(x):
        return x

    def scorer(_o, _e=None):
        return 1.0

    asyncio.run(
        wrapper.run_eval(
            project="proj-1",
            experiment="exp-1",
            data=[{"input": "a"}],
            task=task,
            scorers=[scorer],
            metadata={"k": "v"},
        )
    )

    assert len(_StubBraintrust.eval_async_calls) == 1
    call = _StubBraintrust.eval_async_calls[0]
    assert call["name"] == "proj-1"
    assert call["experiment_name"] == "exp-1"
    assert call["task"] is task
    assert call["scores"] == [scorer]
    assert call["metadata"] == {"k": "v"}
    # `data` is the list of EvalCase instances built by the wrapper.
    assert len(call["data"]) == 1
    assert isinstance(call["data"][0], _StubEvalCase)


def test_run_eval_returns_evalasync_result_unchanged():
    """The SDK's return value passes through verbatim."""
    sentinel = SimpleNamespace(
        results=[
            SimpleNamespace(
                input="a",
                output="o",
                scores={"S": 1.0},
                expected="x",
                metadata={},
                error=None,
                duration=0.0,
            )
        ]
    )
    _StubBraintrust.eval_async_return = sentinel

    wrapper = _RealBraintrustClient(_StubBraintrust)
    result = asyncio.run(
        wrapper.run_eval(
            project="p",
            experiment="e",
            data=[{"input": "a"}],
            task=lambda x: x,
            scorers=[],
            metadata={},
        )
    )

    assert result is sentinel


def test_run_eval_propagates_sdk_exceptions_verbatim():
    """If braintrust.EvalAsync raises, the wrapper does NOT swallow it."""

    class _BoomError(Exception):
        pass

    class _BoomBraintrust:
        EvalCase = _StubEvalCase

        @staticmethod
        async def EvalAsync(**kwargs):  # noqa: N802 - matches SDK symbol
            raise _BoomError("simulated SDK failure")

    wrapper = _RealBraintrustClient(_BoomBraintrust)

    with pytest.raises(_BoomError, match="simulated SDK failure"):
        asyncio.run(
            wrapper.run_eval(
                project="p",
                experiment="e",
                data=[{"input": "a"}],
                task=lambda x: x,
                scorers=[],
                metadata={},
            )
        )
