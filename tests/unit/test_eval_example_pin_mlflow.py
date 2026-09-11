# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression pin for MLflowPlatform EvalExample field shape.

mlflow's forward path differs from the others: ``mlflow.genai.evaluate``
records to traces, and ``_convert_from_platform_result`` reads them via
``session.search_traces(run_id=...)``. Trajectory comes from JSON-shaped
``TraceRecord.request`` / ``.response`` payloads — there is no
``TaskResult`` in scope to feed ``task_result_to_example``. So this test
seeds a ``TraceRecord`` directly into ``FakeMlflowSession._run_traces``
to control the JSON message shape, then calls
``_convert_from_platform_result`` against a synthesized
``platform_result`` whose ``run_id`` points at the seeded run.

This bypasses the fake's ``predict_fn`` round-trip (which would
``str(output)`` the user-task return into a one-message envelope) so the
exact trajectory shape can be pinned. The test asserts the populated
``EvalExample`` fields plus ``trajectory`` / ``tool_calls`` populated
from ``_trajectory_from_request_response``.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from helpers.fake_mlflow import FakeMlflowSession

from agent_evals.adapters.platforms.mlflow import MLflowPlatform
from agent_evals.adapters.platforms.mlflow_client import (
    SessionInfo,
    TraceRecord,
)

_REQUEST_JSON = json.dumps({"messages": [{"role": "user", "content": "alpha"}]})
_RESPONSE_JSON = json.dumps(
    {
        "messages": [
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
    }
)


def test_mlflow_eval_example_field_shape() -> None:
    fake = FakeMlflowSession()
    fake._run_traces["run-1"] = [
        TraceRecord(
            request=_REQUEST_JSON,
            response=_RESPONSE_JSON,
            assessments=[],
            execution_duration_ms=1234,
            trace_metadata={"k": "v"},
        )
    ]

    adapter = MLflowPlatform(session=fake)
    # Stand-in for what aevaluate's setup would have done.
    adapter._session_info = SessionInfo(
        experiment_id="exp-1",
        tracking_uri="https://fake",
    )
    adapter.experiment_id = "exp-1"
    adapter.run_id = "run-1"

    # The adapter only reads .run_id off the platform_result, so a
    # SimpleNamespace stand-in is sufficient for testing.
    result = adapter._convert_from_platform_result(SimpleNamespace(run_id="run-1"))

    assert result.platform == "mlflow"
    assert len(result.examples) == 1
    ex = result.examples[0]

    # Currently-populated fields (regression pin).
    assert ex.input == "alpha"
    assert ex.output == "answer: ALPHA"
    assert ex.error is None
    assert ex.metadata == {"k": "v"}
    assert ex.duration == pytest.approx(1.234)

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
