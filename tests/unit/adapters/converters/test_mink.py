# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the mink-sdk tool_history normalizer."""

import json
import subprocess
import sys

import pytest

from agent_evals.adapters.converters.mink import mink_to_openai


class TestMinkToOpenAI:
    def test_converts_single_call(self):
        tool_history = [
            {
                "call": {
                    "name": "fly_to",
                    "args": {"latitude": 44.1, "longitude": -93.2},
                },
                "result": {"name": "fly_to", "output": "ok"},
            }
        ]
        msgs = mink_to_openai(tool_history)

        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["tool_calls"][0]["id"] == "tc_0"
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "fly_to"
        assert json.loads(msgs[0]["tool_calls"][0]["function"]["arguments"]) == {
            "latitude": 44.1,
            "longitude": -93.2,
        }
        assert msgs[1]["role"] == "tool"
        assert msgs[1]["tool_call_id"] == "tc_0"
        assert msgs[1]["content"] == "ok"

    def test_converts_multi_call(self):
        tool_history = [
            {
                "call": {
                    "name": "fly_to",
                    "args": {"latitude": 1.0, "longitude": 2.0},
                },
                "result": {"name": "fly_to", "output": "ok1"},
            },
            {
                "call": {
                    "name": "fly_to",
                    "args": {"latitude": 3.0, "longitude": 4.0},
                },
                "result": {"name": "fly_to", "output": "ok2"},
            },
        ]
        msgs = mink_to_openai(tool_history)

        assert len(msgs) == 4
        assert [m["role"] for m in msgs] == ["assistant", "tool", "assistant", "tool"]
        assert msgs[0]["tool_calls"][0]["id"] == "tc_0"
        assert msgs[1]["tool_call_id"] == "tc_0"
        assert msgs[2]["tool_calls"][0]["id"] == "tc_1"
        assert msgs[3]["tool_call_id"] == "tc_1"

    def test_empty_list(self):
        assert mink_to_openai([]) == []

    def test_rejects_non_list(self):
        with pytest.raises(TypeError, match="expects a list"):
            mink_to_openai({"response": "hi"})  # type: ignore[arg-type]

    def test_in_flight_entry_emits_assistant_only(self):
        tool_history = [
            {
                "call": {"name": "fly_to", "args": {"latitude": 1.0}},
                "result": None,
            }
        ]
        msgs = mink_to_openai(tool_history)

        assert len(msgs) == 1
        assert msgs[0]["role"] == "assistant"

    def test_integrates_with_task_result_constructor(self):
        """The typical mink-sdk call site: non-default output from
        the `response` key, messages from `tool_history`, both passed via
        the standard TaskResult constructor."""
        from agent_evals.core.types import TaskResult

        response: str = "all clear"
        tool_history: list[dict] = [
            {
                "call": {"name": "fly_to", "args": {"latitude": 1.0}},
                "result": {"name": "fly_to", "output": "ok"},
            }
        ]

        result = TaskResult(
            output=response,
            context={
                "outputs": mink_to_openai(tool_history),
                "final_state": {"drone": {"target_lat": 1.0}},
            },
        )

        assert result.output == "all clear"
        assert result.context is not None
        assert len(result.context["outputs"]) == 2
        assert result.context["final_state"] == {"drone": {"target_lat": 1.0}}

    def test_does_not_require_sdk_installed(self):
        script = (
            "import sys\n"
            "from agent_evals.adapters.converters.mink import "
            "mink_to_openai\n"
            "msgs = mink_to_openai([])\n"
            "assert msgs == []\n"
            "assert 'mink_sdk' not in sys.modules, (\n"
            "    'mink_sdk should not be imported'\n"
            ")\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, (
            f"subprocess failed:\nstdout={completed.stdout}\nstderr={completed.stderr}"
        )
