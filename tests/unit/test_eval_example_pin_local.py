# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression pin for LocalPlatform EvalExample field shape.

Pins every populated field on the resulting ``EvalExample``, including
``trajectory`` and ``tool_calls`` populated from
``TaskResult.context["outputs"]`` via ``task_result_to_example``. If a
future refactor drops or reshapes any of these fields, this test fires.
"""

import pytest
from helpers.scorers import passing_scorer

from agent_evals import (
    ExampleData,
    ExpectedResult,
    TaskResult,
    run_eval_async,
)


def _task_with_trajectory(query: str) -> TaskResult:
    """Returns a TaskResult with a 3-message trajectory in context."""
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


@pytest.mark.asyncio
async def test_local_eval_example_field_shape() -> None:
    result = await run_eval_async(
        task=_task_with_trajectory,
        dataset=[ExampleData(input="alpha", expected=ExpectedResult(expected="alpha"))],
        scorers=[passing_scorer],
    )

    assert result.platform == "local"
    assert len(result.examples) == 1
    ex = result.examples[0]

    assert ex.input == "alpha"
    assert ex.output == "answer: ALPHA"
    assert ex.expected == "alpha"
    assert "passing" in ex.scores
    assert ex.scores["passing"].passed is True
    assert ex.error is None
    assert isinstance(ex.duration, float)
    assert ex.metadata == {}

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
