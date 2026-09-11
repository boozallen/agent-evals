# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the LangChain message normalizer."""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class TestLangChainToOpenAI:
    def test_simple_text_exchange(self):
        from agent_evals.adapters.converters.langchain import langchain_to_openai

        messages = [
            HumanMessage(content="hi"),
            AIMessage(content="hello"),
        ]
        outputs = langchain_to_openai(messages)

        assert len(outputs) == 2
        assert outputs[0]["role"] == "user"
        assert outputs[1]["role"] == "assistant"

    def test_tool_call_round_trip(self):
        from agent_evals.adapters.converters.langchain import langchain_to_openai

        messages = [
            HumanMessage(content="turn on the light"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "call_1",
                        "name": "set_light",
                        "args": {"room": "kitchen", "state": "on"},
                    }
                ],
            ),
            ToolMessage(content="light.kitchen=on", tool_call_id="call_1"),
            AIMessage(content="The kitchen light is on."),
        ]
        outputs = langchain_to_openai(messages)

        assistant_with_tools = next(
            m for m in outputs if m.get("role") == "assistant" and m.get("tool_calls")
        )
        assert assistant_with_tools["tool_calls"][0]["function"]["name"] == "set_light"

    def test_returns_plain_list_of_dicts(self):
        from agent_evals.adapters.converters.langchain import langchain_to_openai

        outputs = langchain_to_openai([AIMessage(content="ok")])
        assert isinstance(outputs, list)
        assert all(isinstance(m, dict) for m in outputs)

    def test_integrates_with_task_result_from_messages(self):
        from agent_evals.adapters.converters.langchain import langchain_to_openai
        from agent_evals.core.types import TaskResult

        result = TaskResult.from_messages(
            langchain_to_openai([AIMessage(content="hello")]),
            final_state={"lights": {"kitchen": "on"}},
        )
        assert result.output == "hello"
        assert result.context is not None
        assert result.context["final_state"] == {"lights": {"kitchen": "on"}}
