# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression pin for BraintrustPlatform EvalExample field shape.

Drives ``aevaluate`` end-to-end against ``FakeBraintrustClient`` (no
live SDK). Pins every populated field on the resulting ``EvalExample``,
including ``trajectory`` and ``tool_calls`` populated from
``TaskResult.context["outputs"]`` via ``task_result_to_example``.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest
from helpers.fake_braintrust import FakeBraintrustClient
from helpers.scorers import passing_scorer

from agent_evals.adapters.platforms.braintrust import (
    BraintrustConfig,
    BraintrustPlatform,
)
from agent_evals.core.types import (
    ExampleData,
    ExpectedResult,
    TaskResult,
)

pytestmark = pytest.mark.requires_braintrust


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


@pytest.mark.asyncio
async def test_braintrust_eval_example_field_shape() -> None:
    with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
        fake = FakeBraintrustClient()
        adapter = BraintrustPlatform(client=fake)

        result = await adapter.aevaluate(
            task=_task_with_trajectory,
            dataset=[
                ExampleData(input="alpha", expected=ExpectedResult(expected="alpha"))
            ],
            evaluators=[passing_scorer],
            platform=BraintrustConfig(project="pin", experiment="braintrust-pin"),
        )

    assert result.platform == "braintrust"
    assert len(result.examples) == 1
    ex = result.examples[0]

    assert ex.input == "alpha"
    assert ex.output == "answer: ALPHA"
    assert ex.expected == "alpha"
    assert "passing_scorer" in ex.scores
    assert ex.scores["passing_scorer"].passed is True
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
