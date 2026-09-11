# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for task_result_to_example.

The helper is the consumer-side counterpart to TaskResult.from_messages:
  - from_messages: trajectory enters TaskResult under context["outputs"]
  - task_result_to_example: trajectory leaves TaskResult and lands on
    EvalExample.trajectory (with tool_calls derived from assistant turns)
"""

from agent_evals.core.types import (
    EvalExample,
    ExpectedResult,
    Score,
    TaskResult,
    task_result_to_example,
)

_TRAJECTORY = [
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


def _common_kwargs():
    return {
        "input": "alpha",
        "expected": ExpectedResult(expected="alpha"),
        "scores": {"s": Score(name="s", value=1.0, passed=True)},
        "metadata": {"k": "v"},
        "duration": 0.123,
        "error": None,
    }


class TestTaskResultToExample:
    def test_returns_eval_example(self):
        tr = TaskResult.from_messages(_TRAJECTORY)
        ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert isinstance(ex, EvalExample)

    def test_trajectory_from_context_outputs(self):
        tr = TaskResult.from_messages(_TRAJECTORY)
        ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.trajectory == _TRAJECTORY

    def test_tool_calls_derived_from_assistant_turns(self):
        tr = TaskResult.from_messages(_TRAJECTORY)
        ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.tool_calls == [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "noop", "arguments": "{}"},
            }
        ]

    def test_no_context_yields_none_trajectory_and_tool_calls(self):
        tr = TaskResult(output="answer: ALPHA")
        ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.trajectory is None
        assert ex.tool_calls is None

    def test_context_without_outputs_yields_none(self):
        tr = TaskResult(output="x", context={"final_state": {"a": 1}})
        ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.trajectory is None
        assert ex.tool_calls is None

    def test_assistant_turn_without_tool_calls_yields_empty_tool_calls_list(self):
        msgs = [{"role": "assistant", "content": "plain text"}]
        tr = TaskResult.from_messages(msgs)
        ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.trajectory == msgs
        assert ex.tool_calls == []

    def test_multiple_assistant_turns_concatenate_tool_calls_in_order(self):
        msgs = [
            {"role": "assistant", "content": "", "tool_calls": [{"id": "a"}]},
            {"role": "user", "content": "ok"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "b"}, {"id": "c"}],
            },
        ]
        tr = TaskResult.from_messages(msgs)
        ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.tool_calls == [{"id": "a"}, {"id": "b"}, {"id": "c"}]

    def test_output_passes_through_from_task_result(self):
        tr = TaskResult.from_messages(_TRAJECTORY)
        ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.output == "answer: ALPHA"

    def test_expected_extracted_from_expected_result(self):
        tr = TaskResult.from_messages(_TRAJECTORY)
        ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.expected == "alpha"

    def test_expected_none_when_expected_result_is_none(self):
        kwargs = _common_kwargs()
        kwargs["expected"] = None
        tr = TaskResult.from_messages(_TRAJECTORY)
        ex = task_result_to_example(task_result=tr, **kwargs)
        assert ex.expected is None

    def test_scores_metadata_duration_error_pass_through(self):
        tr = TaskResult.from_messages(_TRAJECTORY)
        kwargs = _common_kwargs()
        kwargs["error"] = "timeout"
        ex = task_result_to_example(task_result=tr, **kwargs)
        assert ex.scores == {"s": Score(name="s", value=1.0, passed=True)}
        assert ex.metadata == {"k": "v"}
        assert ex.duration == 0.123
        assert ex.error == "timeout"


class TestNonListOutputsBoundary:
    """Pin behavior when context['outputs'] is present but not a list.

    Public-API contract documented by ``TaskResult.from_messages`` is "a
    list of OpenAI-format message dicts". Anything else is a contract
    violation by the caller; the helper should defensively yield
    ``trajectory=None`` (and warn the user) rather than crash or store
    a malformed value.
    """

    def test_outputs_string_yields_none(self, caplog):
        tr = TaskResult(output="x", context={"outputs": "not a list"})
        with caplog.at_level("WARNING", logger="agent_evals.core.types"):
            ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.trajectory is None
        assert ex.tool_calls is None
        assert any("context['outputs']" in record.message for record in caplog.records)

    def test_outputs_dict_yields_none(self, caplog):
        tr = TaskResult(output="x", context={"outputs": {"role": "user"}})
        with caplog.at_level("WARNING", logger="agent_evals.core.types"):
            ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.trajectory is None
        assert ex.tool_calls is None
        assert any("context['outputs']" in record.message for record in caplog.records)

    def test_outputs_explicit_none_yields_none_silently(self, caplog):
        # Explicit None is the documented "no trajectory" shape and
        # should NOT log a warning — it is the success path for tasks
        # that simply don't carry trajectory.
        tr = TaskResult(output="x", context={"outputs": None})
        with caplog.at_level("WARNING", logger="agent_evals.core.types"):
            ex = task_result_to_example(task_result=tr, **_common_kwargs())
        assert ex.trajectory is None
        assert ex.tool_calls is None
        assert not any(
            "context['outputs']" in record.message for record in caplog.records
        )


class TestDeriveToolCallsTolerance:
    """Pin _derive_tool_calls' tolerance of non-dict trajectory items.

    The derivation helper used to raise ``AttributeError`` on lists
    containing non-dict elements (e.g., a stray string from a malformed
    framework normalizer). It now skips non-dicts. Pydantic's
    ``EvalExample.trajectory: list[dict[str, Any]] | None`` rejects such
    lists at construction time — so this test calls the helper directly
    rather than via ``task_result_to_example``, pinning the helper's
    own contract.
    """

    def test_derive_tool_calls_skips_non_dict_items(self):
        from agent_evals.core.types import _derive_tool_calls

        result = _derive_tool_calls(
            [
                {"role": "user", "content": "hi"},
                "string",
                42,
                {"role": "assistant", "tool_calls": [{"id": "ok"}]},
            ]
        )
        assert result == [{"id": "ok"}]
