# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Reverse-path: braintrust _convert_traces preserves trajectory under context['outputs'].

The BTQL row shape is documented as ``select: *`` from ``project_logs``.
Braintrust's framework logs the task return value as the row's ``output``
field (``framework.py:1598`` — ``root_span.log(output=output, ...)``),
and serializes Pydantic v2 instances via ``model_dump(exclude_none=True)``
(``bt_json.py:78-82``). So when our task returns a ``TaskResult`` carrying
trajectory under ``context["outputs"]``, the persisted row carries it at::

    row["output"]["context"]["outputs"]

That is the canonical path. The reverse path reads from there and places
the messages back under ``TaskResult.context["outputs"]`` so re-evaluation
finds trajectory where forward-path scorers always read it.
"""

from __future__ import annotations

import pytest

from agent_evals.adapters.platforms.braintrust import BraintrustPlatform

pytestmark = pytest.mark.requires_braintrust

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


def _root_row(**overrides):
    """Minimal root-span BTQL row that ``_convert_traces`` accepts."""
    base = {
        "id": "row-1",
        "is_root": True,
        "span_id": "s-1",
        "root_span_id": "s-1",
        "input": "alpha",
        "output": "answer: ALPHA",
        "expected": "alpha",
    }
    base.update(overrides)
    return base


def test_convert_traces_reads_trajectory_from_dumped_task_result() -> None:
    """Canonical path: row['output'] is the model_dump'd TaskResult."""
    adapter = BraintrustPlatform()
    # Mirror exactly what braintrust persists: the task's TaskResult
    # serialized via model_dump(exclude_none=True).
    row = _root_row(
        output={
            "output": "answer: ALPHA",
            "context": {"outputs": _MESSAGES},
        }
    )

    [example] = adapter._convert_traces([row])

    assert example.output is not None
    assert example.output.context is not None
    assert example.output.context["outputs"] == _MESSAGES
    # And the unwrapped string output still survives for legacy scorers.
    assert example.output.output == "answer: ALPHA"


def test_convert_traces_string_output_yields_no_context() -> None:
    """String-returning tasks (no TaskResult) leave context as None."""
    adapter = BraintrustPlatform()
    row = _root_row(output="answer: ALPHA")  # plain string, no Pydantic envelope

    [example] = adapter._convert_traces([row])

    assert example.output is not None
    assert example.output.context is None
    assert example.output.output == "answer: ALPHA"


def test_convert_traces_dumped_task_result_without_context_yields_no_context() -> None:
    """A TaskResult that didn't carry context yields trajectory=None."""
    adapter = BraintrustPlatform()
    row = _root_row(output={"output": "answer: ALPHA"})  # no context key

    [example] = adapter._convert_traces([row])

    assert example.output is not None
    assert example.output.context is None
    assert example.output.output == "answer: ALPHA"


def test_convert_traces_no_output_yields_none_output() -> None:
    """Missing output field still results in ExampleData.output is None."""
    adapter = BraintrustPlatform()
    row = _root_row(output=None)

    [example] = adapter._convert_traces([row])

    assert example.output is None
