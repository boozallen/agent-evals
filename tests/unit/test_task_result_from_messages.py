# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TaskResult.from_messages classmethod.

The classmethod applies the library's two conventions for wrapping
OpenAI-format messages into a TaskResult:
  - Messages land at context["outputs"].
  - output defaults to the last assistant message's text.

Non-default-output cases are handled by the plain TaskResult(...)
constructor; this helper deliberately has no output= override.
"""

from agent_evals.core.types import TaskResult, _last_assistant_text


class TestLastAssistantText:
    def test_returns_last_assistant_content(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        assert _last_assistant_text(msgs) == "hello"

    def test_walks_in_reverse_order(self):
        msgs = [
            {"role": "assistant", "content": "first"},
            {"role": "user", "content": "follow up"},
            {"role": "assistant", "content": "second"},
        ]
        assert _last_assistant_text(msgs) == "second"

    def test_empty_list_returns_empty_string(self):
        assert _last_assistant_text([]) == ""

    def test_no_assistant_messages_returns_empty(self):
        msgs = [{"role": "user", "content": "hi"}]
        assert _last_assistant_text(msgs) == ""

    def test_assistant_with_empty_content_returns_empty(self):
        msgs = [{"role": "assistant", "content": ""}]
        assert _last_assistant_text(msgs) == ""

    def test_skips_assistant_with_non_string_content(self):
        msgs = [
            {"role": "assistant", "content": "real text"},
            {"role": "assistant", "content": [{"type": "tool_use"}]},
        ]
        assert _last_assistant_text(msgs) == "real text"


class TestFromMessages:
    def test_messages_land_in_context_outputs(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = TaskResult.from_messages(msgs)
        assert result.context is not None
        assert result.context["outputs"] == msgs

    def test_output_defaults_to_last_assistant_text(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        result = TaskResult.from_messages(msgs)
        assert result.output == "hello"

    def test_empty_messages_give_empty_output(self):
        result = TaskResult.from_messages([])
        assert result.output == ""
        assert result.context == {"outputs": []}

    def test_kwargs_passthrough_to_context(self):
        msgs = [{"role": "assistant", "content": "hi"}]
        state = {"lights": {"living": "on"}}
        result = TaskResult.from_messages(msgs, final_state=state)
        assert result.context is not None
        assert result.context["final_state"] == state
        assert result.context["outputs"] == msgs

    def test_multiple_kwargs_all_passthrough(self):
        msgs = [{"role": "assistant", "content": "hi"}]
        result = TaskResult.from_messages(
            msgs,
            final_state={"k": "v"},
            reference_outputs=[{"role": "assistant", "content": "expected"}],
            custom_scorer_input="anything",
        )
        assert result.context is not None
        assert result.context["final_state"] == {"k": "v"}
        assert result.context["reference_outputs"] == [
            {"role": "assistant", "content": "expected"}
        ]
        assert result.context["custom_scorer_input"] == "anything"

    def test_no_kwargs_still_has_outputs_key(self):
        msgs = [{"role": "assistant", "content": "hi"}]
        result = TaskResult.from_messages(msgs)
        assert result.context == {"outputs": msgs}

    def test_returns_instance_of_taskresult(self):
        result = TaskResult.from_messages([])
        assert isinstance(result, TaskResult)

    def test_equivalent_to_constructor_for_default_case(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        via_helper = TaskResult.from_messages(msgs, final_state={"k": "v"})
        via_ctor = TaskResult(
            output="hello",
            context={"outputs": msgs, "final_state": {"k": "v"}},
        )
        assert via_helper == via_ctor
