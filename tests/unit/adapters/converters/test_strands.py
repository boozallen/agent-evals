# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Strands message normalizer."""


class TestStrandsToOpenAI:
    def test_text_only_exchange(self):
        from agent_evals.adapters.converters.strands import strands_to_openai

        messages = [
            {"role": "user", "content": [{"text": "hi"}]},
            {"role": "assistant", "content": [{"text": "hello"}]},
        ]
        outputs = strands_to_openai(messages)

        assert outputs[0]["role"] == "user"
        assert outputs[0]["content"] == "hi"
        assert outputs[1]["role"] == "assistant"
        assert outputs[1]["content"] == "hello"

    def test_tool_use_becomes_openai_tool_calls(self):
        import json

        from agent_evals.adapters.converters.strands import strands_to_openai

        messages = [
            {"role": "user", "content": [{"text": "lock front"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "abc123",
                            "name": "lock_door",
                            "input": {"door": "front", "locked": True},
                        }
                    }
                ],
            },
        ]
        outputs = strands_to_openai(messages)

        assistant = outputs[1]
        assert assistant["tool_calls"][0]["id"] == "abc123"
        assert assistant["tool_calls"][0]["function"]["name"] == "lock_door"
        assert json.loads(assistant["tool_calls"][0]["function"]["arguments"]) == {
            "door": "front",
            "locked": True,
        }

    def test_tool_result_becomes_tool_role_message(self):
        from agent_evals.adapters.converters.strands import strands_to_openai

        messages = [
            {
                "role": "user",  # Strands uses "user" role for toolResult
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "abc123",
                            "content": [{"text": "door.front=locked"}],
                            "status": "success",
                        }
                    }
                ],
            }
        ]
        outputs = strands_to_openai(messages)

        assert outputs[0]["role"] == "tool"
        assert outputs[0]["tool_call_id"] == "abc123"
        assert "door.front=locked" in outputs[0]["content"]

    def test_non_list_raises_type_error(self):
        import pytest

        from agent_evals.adapters.converters.strands import strands_to_openai

        with pytest.raises(TypeError):
            strands_to_openai("not-a-list")  # type: ignore[arg-type]

    def test_integrates_with_task_result_from_messages(self):
        from agent_evals.adapters.converters.strands import strands_to_openai
        from agent_evals.core.types import TaskResult

        result = TaskResult.from_messages(
            strands_to_openai([{"role": "assistant", "content": [{"text": "hi"}]}]),
            final_state={"door": "locked"},
        )
        assert result.output == "hi"
        assert result.context is not None
        assert result.context["final_state"] == {"door": "locked"}
