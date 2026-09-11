# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``MLflowPlatform._trajectory_from_request_response``.

The helper is a static method that converts a TraceRecord's
request/response (typed ``dict | str | None``) into a single trajectory
list (or ``None``). It backs both the forward path
(``_convert_from_platform_result``) and the reverse path
(``_convert_traces``); these tests pin the branch coverage.
"""

from __future__ import annotations

import json

from agent_evals.adapters.platforms.mlflow import MLflowPlatform

_USER_MSG = {"role": "user", "content": "alpha"}
_ASSISTANT_MSG = {"role": "assistant", "content": "answer"}


class TestTrajectoryFromRequestResponse:
    def test_concatenates_dict_payloads(self):
        result = MLflowPlatform._trajectory_from_request_response(
            {"messages": [_USER_MSG]},
            {"messages": [_ASSISTANT_MSG]},
        )
        assert result == [_USER_MSG, _ASSISTANT_MSG]

    def test_concatenates_json_string_payloads(self):
        result = MLflowPlatform._trajectory_from_request_response(
            json.dumps({"messages": [_USER_MSG]}),
            json.dumps({"messages": [_ASSISTANT_MSG]}),
        )
        assert result == [_USER_MSG, _ASSISTANT_MSG]

    def test_both_none_yields_none(self):
        assert MLflowPlatform._trajectory_from_request_response(None, None) is None

    def test_both_empty_string_yields_none(self):
        assert MLflowPlatform._trajectory_from_request_response("", "") is None

    def test_no_messages_key_yields_none(self):
        assert (
            MLflowPlatform._trajectory_from_request_response(
                {"other": "fields"},
                {"other": "fields"},
            )
            is None
        )

    def test_only_request_side_populated(self):
        result = MLflowPlatform._trajectory_from_request_response(
            {"messages": [_USER_MSG]},
            None,
        )
        assert result == [_USER_MSG]

    def test_only_response_side_populated(self):
        result = MLflowPlatform._trajectory_from_request_response(
            None,
            {"messages": [_ASSISTANT_MSG]},
        )
        assert result == [_ASSISTANT_MSG]


class TestCanonicalAgentEvalsShape:
    """Pin the canonical Pydantic-dumped TaskResult shape.

    When the user task returns a ``TaskResult`` carrying trajectory
    under ``context["outputs"]``, MLflow's auto-tracing serializes
    that Pydantic instance verbatim into ``trace.data.response``.
    The helper must recognize this canonical shape — discovered live
    via ``trajectory_e2e.py`` against a real MLflow tracking server
    where ``response`` looked like::

        {"output": "<bare>", "context": {"outputs": [...messages...]}}

    The previous fake-driven unit tests injected a literal
    ``{"messages": [...]}`` envelope that doesn't match production.
    """

    def test_response_with_context_outputs_yields_trajectory(self):
        result = MLflowPlatform._trajectory_from_request_response(
            {"_input": "x"},  # real mlflow's request shape: bare input dict
            {
                "output": "answer",
                "context": {"outputs": [_USER_MSG, _ASSISTANT_MSG]},
            },
        )
        assert result == [_USER_MSG, _ASSISTANT_MSG]

    def test_response_json_string_with_context_outputs(self):
        # The TraceRecord's response can arrive as either dict or str
        # depending on which search_traces branch produced it.
        result = MLflowPlatform._trajectory_from_request_response(
            None,
            json.dumps({"output": "answer", "context": {"outputs": [_ASSISTANT_MSG]}}),
        )
        assert result == [_ASSISTANT_MSG]

    def test_canonical_shape_takes_precedence_over_messages(self):
        # If a payload has both shapes, the canonical agent-evals path wins.
        result = MLflowPlatform._trajectory_from_request_response(
            None,
            {
                "output": "answer",
                "context": {"outputs": [_ASSISTANT_MSG]},
                "messages": [{"role": "ignored", "content": "ignored"}],
            },
        )
        assert result == [_ASSISTANT_MSG]

    def test_context_present_but_outputs_missing_falls_back_to_messages(self):
        # context exists but doesn't carry the outputs key — fall back
        # to messages if present.
        result = MLflowPlatform._trajectory_from_request_response(
            None,
            {
                "context": {"final_state": {"a": 1}},
                "messages": [_ASSISTANT_MSG],
            },
        )
        assert result == [_ASSISTANT_MSG]


class TestDefensivePayloadShapes:
    """Pin behavior for malformed/unexpected payload shapes.

    These cases shouldn't happen if the TraceRecord contract holds, but
    the helper is on the success path of every reverse-path conversion;
    a crash here would lose every example in a batch.
    """

    def test_invalid_json_string_logs_debug_and_returns_none(self, caplog):
        with caplog.at_level("DEBUG", logger="agent_evals.adapters.platforms.mlflow"):
            result = MLflowPlatform._trajectory_from_request_response(
                "not-valid-json{",
                None,
            )
        assert result is None
        assert any(
            "decode TraceRecord payload as JSON" in record.message
            for record in caplog.records
        )

    def test_non_dict_decoded_payload_yields_none(self):
        # `json.loads('[1,2,3]')` decodes to a list, not a dict.
        # The helper must reject that shape silently (no messages found).
        result = MLflowPlatform._trajectory_from_request_response(
            "[1, 2, 3]",
            None,
        )
        assert result is None

    def test_messages_present_but_not_a_list_yields_none(self):
        # `messages` is a string, not a list — discard it.
        result = MLflowPlatform._trajectory_from_request_response(
            {"messages": "not-a-list"},
            None,
        )
        assert result is None
