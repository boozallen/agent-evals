# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reverse-path: mlflow _convert_traces preserves trajectory under context['outputs']."""

from __future__ import annotations

import json

from agent_evals.adapters.platforms.mlflow import MLflowPlatform
from agent_evals.adapters.platforms.mlflow_client import TraceRecord

_MESSAGES = [
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


def test_convert_traces_places_trajectory_under_context_outputs() -> None:
    adapter = MLflowPlatform()
    record = TraceRecord(
        request=json.dumps({"messages": [_MESSAGES[0]]}),
        response=json.dumps({"messages": _MESSAGES[1:]}),
        assessments=[],
        execution_duration_ms=100,
        trace_metadata={},
    )

    [example] = adapter._convert_traces([record])

    assert example.output is not None
    assert example.output.context is not None
    assert example.output.context["outputs"] == _MESSAGES


def test_convert_traces_no_messages_yields_no_context() -> None:
    """If the trace carries no decodable messages, output stays bare TaskResult."""
    adapter = MLflowPlatform()
    record = TraceRecord(
        request="",
        response="answer",
        assessments=[],
        execution_duration_ms=100,
        trace_metadata={},
    )

    [example] = adapter._convert_traces([record])

    assert example.output is not None
    assert example.output.context is None
