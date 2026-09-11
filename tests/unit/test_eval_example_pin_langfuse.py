# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression pin for LangFusePlatform EvalExample field shape.

Drives ``aevaluate`` end-to-end against ``FakeLangfuseClient``. Pins
every populated field on the resulting ``EvalExample``, including
``trajectory`` and ``tool_calls`` populated from
``TaskResult.context["outputs"]`` via ``task_result_to_example``.
"""

from __future__ import annotations

import pytest
from helpers.fake_langfuse import FakeLangfuseClient

from agent_evals.adapters.platforms.langfuse import LangfuseConfig, LangFusePlatform
from agent_evals.core.types import (
    ExampleData,
    ExpectedResult,
    Score,
    TaskResult,
)


def _task_with_trajectory(query: str) -> TaskResult:
    messages = [
        {"role": "user", "content": query},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }
            ],
        },
        {"role": "assistant", "content": f"answer: {query.upper()}"},
    ]
    return TaskResult.from_messages(messages)


def dummy(result, expected=None, **_kwargs):  # noqa: ARG001
    return Score(name="dummy", value=1.0, passed=True)


@pytest.mark.asyncio
async def test_langfuse_eval_example_field_shape() -> None:
    fake = FakeLangfuseClient()
    adapter = LangFusePlatform(client=fake)

    result = await adapter.aevaluate(
        task=_task_with_trajectory,
        dataset=[ExampleData(input="alpha", expected=ExpectedResult(expected="alpha"))],
        evaluators=[dummy],
        platform=LangfuseConfig(experiment="langfuse-pin"),
    )

    assert result.platform == "langfuse"
    assert len(result.examples) == 1
    ex = result.examples[0]

    assert ex.input == "alpha"
    assert ex.output == "answer: ALPHA"
    assert ex.expected == "alpha"
    assert "dummy" in ex.scores
    assert ex.scores["dummy"].passed is True
    assert ex.error is None
    assert isinstance(ex.duration, float)

    # Forward progress: trajectory now persists.
    assert ex.trajectory == [
        {"role": "user", "content": "alpha"},
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "noop", "arguments": "{}"},
                }
            ],
        },
        {"role": "assistant", "content": "answer: ALPHA"},
    ]
    assert ex.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "noop", "arguments": "{}"},
        }
    ]
