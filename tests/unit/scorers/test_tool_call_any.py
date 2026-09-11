# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ToolCallAnyMatch native scorer.

Disjunctive existence semantics: passes when at least one actual tool
call matches at least one reference entry under the chosen
``tool_args_match_mode``. Silence (zero actual calls) fails. Extras
that don't match any reference entry are ignored.
"""

import json

import pytest

from agent_evals.core.types import ExpectedResult, Score, TaskResult


def _user(text: str) -> dict:
    return {"role": "user", "content": text}


def _assistant_tool_call(name: str, args: dict) -> dict:
    """Natural flat-shape assistant tool call (matches typical YAML)."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"name": name, "args": args}],
    }


def _assistant_tool_call_openai(name: str, args: dict) -> dict:
    """OpenAI wire-shape assistant tool call."""
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": f"tc_{name}",
                "type": "function",
                "function": {"name": name, "arguments": json.dumps(args)},
            }
        ],
    }


def _assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _ref_calls(*entries: dict) -> list[dict]:
    """Pack a flat list of {name, args} entries into the OpenAI-shaped
    reference message the trajectory scorers expect."""
    return [{"role": "assistant", "tool_calls": list(entries)}]


class TestPassPaths:
    def test_single_actual_matches_only_reference_entry(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallAnyMatch,
        )

        scorer = ToolCallAnyMatch()
        score = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("get_status", {"device": "lights"})
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": _ref_calls(
                        {"name": "get_status", "args": {"device": "lights"}}
                    )
                },
            ),
        )
        assert isinstance(score, Score)
        assert score.name == "ToolCallAnyMatch"
        assert score.passed is True
        assert score.value == 1.0
        assert score.metadata["matched_name"] == "get_status"

    def test_actual_matches_one_of_multiple_reference_entries(self):
        """Disjunction: reference lists [A, B]; actual matches B."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallAnyMatch,
        )

        scorer = ToolCallAnyMatch()
        score = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call(
                            "get_status", {"device_type": "all", "room": "bedroom"}
                        )
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": _ref_calls(
                        {
                            "name": "get_status",
                            "args": {"device_type": "lights", "room": "bedroom"},
                        },
                        {
                            "name": "get_status",
                            "args": {"device_type": "all", "room": "bedroom"},
                        },
                    )
                },
            ),
        )
        assert score.passed is True

    def test_extras_are_ignored(self):
        """Agent calls one wrong tool plus one matching call -> passes."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallAnyMatch,
        )

        scorer = ToolCallAnyMatch()
        score = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("lock_door", {"door": "front"}),
                        _assistant_tool_call("get_status", {"device": "lights"}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": _ref_calls(
                        {"name": "get_status", "args": {"device": "lights"}}
                    )
                },
            ),
        )
        assert score.passed is True

    def test_openai_wire_shape_actual_matches(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallAnyMatch,
        )

        scorer = ToolCallAnyMatch()
        score = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call_openai("ping", {"host": "example.com"})
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": _ref_calls(
                        {"name": "ping", "args": {"host": "example.com"}}
                    )
                },
            ),
        )
        assert score.passed is True


class TestFailPaths:
    def test_silence_fails(self):
        """Empty trajectory fails any_of (distinguishes from `subset`)."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallAnyMatch,
        )

        scorer = ToolCallAnyMatch()
        score = scorer(
            TaskResult(output="", context={"outputs": [_assistant_text("ok")]}),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": _ref_calls(
                        {"name": "get_status", "args": {"device": "lights"}}
                    )
                },
            ),
        )
        assert score.passed is False
        assert score.value == 0.0
        assert score.metadata["failure_mode"] == "silence"
        assert score.metadata["actual_count"] == 0

    def test_no_name_match_fails(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallAnyMatch,
        )

        scorer = ToolCallAnyMatch()
        score = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [_assistant_tool_call("lock_door", {"door": "front"})]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": _ref_calls(
                        {"name": "get_status", "args": {"device": "lights"}}
                    )
                },
            ),
        )
        assert score.passed is False
        assert score.metadata["failure_mode"] == "no_match"

    def test_name_match_but_args_mismatch_fails_under_exact(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallAnyMatch,
        )

        scorer = ToolCallAnyMatch(tool_args_match_mode="exact")
        score = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [_assistant_tool_call("get_status", {"device": "doors"})]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": _ref_calls(
                        {"name": "get_status", "args": {"device": "lights"}}
                    )
                },
            ),
        )
        assert score.passed is False


class TestArgsModes:
    """match_args ∈ {exact, subset, superset, ignore}."""

    # Reference is {"device": "lights"} (set by _ref_calls below).
    @pytest.mark.parametrize(
        "mode,actual_args,passes",
        [
            ("exact", {"device": "lights"}, True),
            ("exact", {"device": "lights", "extra": 1}, False),
            ("subset", {"device": "lights"}, True),
            ("subset", {"device": "lights", "extra": 1}, False),
            ("subset", {}, True),
            ("subset", {"device": "doors"}, False),
            ("superset", {"device": "lights"}, True),
            ("superset", {"device": "lights", "extra": 1}, True),
            ("superset", {"device": "doors"}, False),
            ("superset", {}, False),
            ("ignore", {"device": "doors"}, True),
            ("ignore", {}, True),
        ],
    )
    def test_match_args_modes(self, mode, actual_args, passes):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallAnyMatch,
        )

        scorer = ToolCallAnyMatch(tool_args_match_mode=mode)
        score = scorer(
            TaskResult(
                output="",
                context={"outputs": [_assistant_tool_call("get_status", actual_args)]},
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": _ref_calls(
                        {"name": "get_status", "args": {"device": "lights"}}
                    )
                },
            ),
        )
        assert score.passed is passes
        assert score.metadata["tool_args_match_mode"] == mode


class TestErrorPaths:
    def test_missing_outputs_returns_error_score(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallAnyMatch,
        )

        scorer = ToolCallAnyMatch()
        score = scorer(
            TaskResult(output="", context={}),
            ExpectedResult(
                expected="",
                context={"reference_outputs": _ref_calls({"name": "x", "args": {}})},
            ),
        )
        assert score.passed is False
        assert "outputs" in score.metadata.get("error", "")

    def test_missing_reference_returns_error_score(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallAnyMatch,
        )

        scorer = ToolCallAnyMatch()
        score = scorer(
            TaskResult(
                output="",
                context={"outputs": [_assistant_tool_call("x", {})]},
            ),
            ExpectedResult(expected="", context={}),
        )
        assert score.passed is False
        assert "reference_outputs" in score.metadata.get("error", "")


class TestRegistry:
    def test_registered_under_canonical_name(self):
        from agent_evals.core._registries import scorer_registry

        factory = scorer_registry.get("ToolCallAnyMatch")
        scorer = factory()
        assert callable(scorer)
