# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reverse-path: langfuse _convert_traces preserves trajectory under context['outputs']."""

from __future__ import annotations

from helpers.fake_langfuse import FakeLangfuseClient

from agent_evals.adapters.platforms.langfuse import LangFusePlatform
from agent_evals.adapters.platforms.langfuse_client import DatasetItemRecord

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
    fake = FakeLangfuseClient(
        trace_outputs={"trace-1": "answer: ALPHA"},
        trace_messages={"trace-1": _MESSAGES},
    )
    adapter = LangFusePlatform(client=fake)

    items = [
        DatasetItemRecord(
            source_trace_id="trace-1",
            input="alpha",
            expected_output="alpha",
            metadata={},
        )
    ]

    [example] = adapter._convert_traces(items, fake)

    assert example.output is not None
    assert example.output.context is not None
    assert example.output.context["outputs"] == _MESSAGES


def test_convert_traces_no_messages_yields_no_context() -> None:
    """When the trace has no messages, TaskResult.context stays None."""
    fake = FakeLangfuseClient(
        trace_outputs={"trace-1": "answer"},
    )
    adapter = LangFusePlatform(client=fake)

    items = [
        DatasetItemRecord(
            source_trace_id="trace-1",
            input="alpha",
            expected_output="alpha",
            metadata={},
        )
    ]

    [example] = adapter._convert_traces(items, fake)

    assert example.output is not None
    assert example.output.context is None
