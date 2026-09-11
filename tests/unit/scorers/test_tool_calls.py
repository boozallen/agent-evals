# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ToolCallExactMatch native scorer."""

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


def _tool_response(content: str) -> dict:
    return {"role": "tool", "content": content}


def _assistant_text(text: str) -> dict:
    return {"role": "assistant", "content": text}


class TestToolCallExactMatchCore:
    """Happy-path sequence matching."""

    def test_single_call_exact_match_passes(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [_assistant_tool_call("fly_to", {"lat": 1, "lon": 2})]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2})
                    ]
                },
            ),
        )
        assert isinstance(result, Score)
        assert result.name == "ToolCallExactMatch"
        assert result.passed is True
        assert result.value == 1.0
        assert result.metadata["call_count"] == 1

    def test_multi_call_in_order_passes(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                        _assistant_tool_call("land", {"soft": True}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                        _assistant_tool_call("land", {"soft": True}),
                    ]
                },
            ),
        )
        assert result.passed is True
        assert result.metadata["call_count"] == 2


class TestToolCallExactMatchMismatches:
    """Mismatch paths."""

    def test_wrong_order_fails(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("land", {"soft": True}),
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                        _assistant_tool_call("land", {"soft": True}),
                    ]
                },
            ),
        )
        assert result.passed is False
        assert result.metadata["mismatch_index"] == 0
        assert "fly_to" in result.reasoning
        assert "land" in result.reasoning

    def test_extra_call_fails(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                        _assistant_tool_call("land", {"soft": True}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                    ]
                },
            ),
        )
        assert result.passed is False
        assert result.metadata["actual_count"] == 2
        assert result.metadata["reference_count"] == 1
        assert "Expected 1" in result.reasoning
        assert "got 2" in result.reasoning

    def test_missing_call_fails(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                        _assistant_tool_call("land", {"soft": True}),
                    ]
                },
            ),
        )
        assert result.passed is False
        assert result.metadata["actual_count"] == 1
        assert result.metadata["reference_count"] == 2
        assert "Expected 2" in result.reasoning
        assert "got 1" in result.reasoning

    def test_wrong_tool_name_fails(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                        _assistant_tool_call("crash", {"soft": True}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                        _assistant_tool_call("land", {"soft": True}),
                    ]
                },
            ),
        )
        assert result.passed is False
        assert result.metadata["mismatch_index"] == 1
        assert result.metadata["expected_name"] == "land"
        assert result.metadata["actual_name"] == "crash"


class TestToolArgsMatchMode:
    """Argument-matching modes."""

    def test_exact_mode_rejects_extra_args(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch(tool_args_match_mode="exact")
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call(
                            "fly_to", {"lat": 1, "lon": 2, "extra": 3}
                        ),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                    ]
                },
            ),
        )
        assert result.passed is False
        assert result.metadata["mismatch_index"] == 0
        assert result.metadata["tool_args_match_mode"] == "exact"

    def test_superset_mode_allows_extra_args(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch(tool_args_match_mode="superset")
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call(
                            "fly_to", {"lat": 1, "lon": 2, "extra": 3}
                        ),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                    ]
                },
            ),
        )
        assert result.passed is True
        assert result.metadata["tool_args_match_mode"] == "superset"

    def test_subset_mode_rejects_extra_args(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch(tool_args_match_mode="subset")
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call(
                            "fly_to", {"lat": 1, "lon": 2, "extra": 3}
                        ),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                    ]
                },
            ),
        )
        assert result.passed is False
        assert result.metadata["tool_args_match_mode"] == "subset"

    def test_subset_mode_allows_omitted_args(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch(tool_args_match_mode="subset")
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                    ]
                },
            ),
        )
        assert result.passed is True
        assert result.metadata["tool_args_match_mode"] == "subset"

    def test_ignore_mode_ignores_args_entirely(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch(tool_args_match_mode="ignore")
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("fly_to", {"completely": "different"}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                    ]
                },
            ),
        )
        assert result.passed is True
        assert result.metadata["tool_args_match_mode"] == "ignore"


class TestShapeNormalization:
    """Cross-shape tool-call extraction."""

    def test_accepts_openai_wire_shape(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call_openai("fly_to", {"lat": 1, "lon": 2}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                    ]
                },
            ),
        )
        assert result.passed is True

    def test_filters_non_tool_call_messages(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _user("turn on the kitchen light"),
                        _assistant_tool_call_openai(
                            "set_light", {"room": "kitchen", "state": "on"}
                        ),
                        _tool_response("ok"),
                        _assistant_text("Done."),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call(
                            "set_light", {"room": "kitchen", "state": "on"}
                        ),
                    ]
                },
            ),
        )
        assert result.passed is True
        assert result.metadata["call_count"] == 1


class TestToolCallExactMatchErrors:
    """Error-path behavior."""

    def test_missing_outputs_context_fails_cleanly(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch()
        result = scorer(
            TaskResult(output="", context={}),
            ExpectedResult(
                expected="",
                context={"reference_outputs": []},
            ),
        )
        assert result.passed is False
        assert result.value == 0.0
        assert "outputs" in result.metadata["error"]

    def test_missing_reference_outputs_fails_cleanly(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallExactMatch,
        )

        scorer = ToolCallExactMatch()
        result = scorer(
            TaskResult(output="", context={"outputs": []}),
            ExpectedResult(expected="", context={}),
        )
        assert result.passed is False
        assert result.value == 0.0
        assert "reference_outputs" in result.metadata["error"]


class TestToolCallExactMatchRegistered:
    """Registry discoverability (for YAML benchmarks)."""

    def test_registered_in_registry(self):
        from agent_evals.core._registries import scorer_registry

        factory = scorer_registry.get("ToolCallExactMatch")
        scorer = factory()
        assert callable(scorer)

    def test_registered_with_kwargs(self):
        from agent_evals.core._registries import scorer_registry

        factory = scorer_registry.get("ToolCallExactMatch")
        scorer = factory(tool_args_match_mode="subset")
        assert callable(scorer)


# =============================================================================
# ToolCallSupersetMatch — reference ⊆ actual (every reference call appears in
# actual, in order; extras allowed; ignores prose/tool-response frames).
# =============================================================================


class TestToolCallSupersetMatchPasses:
    """Happy paths: actual contains every reference call (in order)."""

    def test_actual_equals_reference_passes(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSupersetMatch,
        )

        scorer = ToolCallSupersetMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                        _assistant_tool_call("land", {"soft": True}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1, "lon": 2}),
                        _assistant_tool_call("land", {"soft": True}),
                    ]
                },
            ),
        )
        assert isinstance(result, Score)
        assert result.name == "ToolCallSupersetMatch"
        assert result.passed is True
        assert result.value == 1.0

    def test_actual_has_extras_passes(self):
        """Reference calls all present, in order; extras between/after are fine."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSupersetMatch,
        )

        scorer = ToolCallSupersetMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("authenticate", {}),
                        _assistant_tool_call("check_cache", {}),
                        _assistant_tool_call("query_db", {"q": "data"}),
                        _assistant_tool_call("log_metrics", {}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("authenticate", {}),
                        _assistant_tool_call("query_db", {"q": "data"}),
                    ]
                },
            ),
        )
        assert result.passed is True

    def test_empty_reference_against_non_empty_actual_passes(self):
        """Vacuous superset: every reference call (zero of them) is present."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSupersetMatch,
        )

        scorer = ToolCallSupersetMatch()
        result = scorer(
            TaskResult(
                output="",
                context={"outputs": [_assistant_tool_call("fly_to", {"lat": 1})]},
            ),
            ExpectedResult(expected="", context={"reference_outputs": []}),
        )
        assert result.passed is True


class TestToolCallSupersetMatchFails:
    """Failure paths."""

    def test_missing_required_call_fails(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSupersetMatch,
        )

        scorer = ToolCallSupersetMatch()
        result = scorer(
            TaskResult(
                output="",
                context={"outputs": [_assistant_tool_call("query_db", {"q": "data"})]},
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("authenticate", {}),
                        _assistant_tool_call("query_db", {"q": "data"}),
                    ]
                },
            ),
        )
        assert result.passed is False

    def test_wrong_order_fails(self):
        """Reference is [A, B]; actual is [B, A] — superset is order-preserving."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSupersetMatch,
        )

        scorer = ToolCallSupersetMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("land", {"soft": True}),
                        _assistant_tool_call("fly_to", {"lat": 1}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {"lat": 1}),
                        _assistant_tool_call("land", {"soft": True}),
                    ]
                },
            ),
        )
        assert result.passed is False


class TestToolCallSupersetMatchArgsModes:
    """Per-call args modes — uniform direction (independent of list-level mode).

    Reference call: get_status({device: lights}). Actual varied per row.
    """

    @pytest.mark.parametrize(
        "mode,actual_args,passes",
        [
            ("exact", {"device": "lights"}, True),
            ("exact", {"device": "lights", "extra": 1}, False),
            ("subset", {"device": "lights"}, True),
            ("subset", {}, True),  # actual ⊆ reference
            ("subset", {"device": "lights", "extra": 1}, False),  # extras break subset
            ("superset", {"device": "lights"}, True),
            ("superset", {"device": "lights", "extra": 1}, True),  # ref ⊆ actual
            ("superset", {}, False),  # ref not ⊆ actual
            ("ignore", {"device": "doors"}, True),
            ("ignore", {}, True),
        ],
    )
    def test_args_modes(self, mode, actual_args, passes):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSupersetMatch,
        )

        scorer = ToolCallSupersetMatch(tool_args_match_mode=mode)
        result = scorer(
            TaskResult(
                output="",
                context={"outputs": [_assistant_tool_call("get_status", actual_args)]},
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("get_status", {"device": "lights"})
                    ]
                },
            ),
        )
        assert result.passed is passes
        assert result.metadata["tool_args_match_mode"] == mode


class TestToolCallSupersetMatchRegistry:
    def test_registered(self):
        from agent_evals.core._registries import scorer_registry

        factory = scorer_registry.get("ToolCallSupersetMatch")
        assert callable(factory())


# =============================================================================
# ToolCallSubsetMatch — actual ⊆ reference (every actual call appears in
# reference, in order; agent may skip; extras forbidden).
# =============================================================================


class TestToolCallSubsetMatchPasses:
    def test_actual_equals_reference_passes(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSubsetMatch,
        )

        scorer = ToolCallSubsetMatch()
        result = scorer(
            TaskResult(
                output="",
                context={"outputs": [_assistant_tool_call("fly_to", {"lat": 1})]},
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [_assistant_tool_call("fly_to", {"lat": 1})]
                },
            ),
        )
        assert result.name == "ToolCallSubsetMatch"
        assert result.passed is True

    def test_actual_skips_optional_calls_passes(self):
        """Reference [A, B, C]; actual [A, C] — subsequence preserved."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSubsetMatch,
        )

        scorer = ToolCallSubsetMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("authenticate", {}),
                        _assistant_tool_call("query_db", {"q": "data"}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("authenticate", {}),
                        _assistant_tool_call("check_cache", {}),
                        _assistant_tool_call("query_db", {"q": "data"}),
                    ]
                },
            ),
        )
        assert result.passed is True

    def test_empty_actual_against_any_reference_passes(self):
        """Vacuous subset: empty actual ⊆ anything."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSubsetMatch,
        )

        scorer = ToolCallSubsetMatch()
        result = scorer(
            TaskResult(output="", context={"outputs": []}),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [_assistant_tool_call("fly_to", {"lat": 1})]
                },
            ),
        )
        assert result.passed is True


class TestToolCallSubsetMatchFails:
    def test_extra_call_fails(self):
        """Actual has a call not in reference."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSubsetMatch,
        )

        scorer = ToolCallSubsetMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("authenticate", {}),
                        _assistant_tool_call("unauthorized", {}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("authenticate", {}),
                        _assistant_tool_call("query_db", {}),
                    ]
                },
            ),
        )
        assert result.passed is False

    def test_wrong_order_fails(self):
        """Reference [A, B]; actual [B, A] — subset is order-preserving."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSubsetMatch,
        )

        scorer = ToolCallSubsetMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("land", {}),
                        _assistant_tool_call("fly_to", {}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("fly_to", {}),
                        _assistant_tool_call("land", {}),
                    ]
                },
            ),
        )
        assert result.passed is False


class TestToolCallSubsetMatchArgsModes:
    """Per-call args modes — uniform direction.

    Reference: get_status({device: lights}). Actual varies; subset list-mode
    means "actual is a (sub)sequence of reference," so a 1-elem actual
    against 1-elem reference still requires per-call args to satisfy the
    chosen mode in its uniform direction.
    """

    @pytest.mark.parametrize(
        "mode,actual_args,passes",
        [
            ("exact", {"device": "lights"}, True),
            ("exact", {"device": "lights", "extra": 1}, False),
            ("subset", {"device": "lights"}, True),
            ("subset", {}, True),
            ("subset", {"device": "lights", "extra": 1}, False),
            ("superset", {"device": "lights"}, True),
            ("superset", {"device": "lights", "extra": 1}, True),
            ("superset", {}, False),
            ("ignore", {"device": "doors"}, True),
            ("ignore", {}, True),
        ],
    )
    def test_args_modes(self, mode, actual_args, passes):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallSubsetMatch,
        )

        scorer = ToolCallSubsetMatch(tool_args_match_mode=mode)
        result = scorer(
            TaskResult(
                output="",
                context={"outputs": [_assistant_tool_call("get_status", actual_args)]},
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("get_status", {"device": "lights"})
                    ]
                },
            ),
        )
        assert result.passed is passes
        assert result.metadata["tool_args_match_mode"] == mode


class TestToolCallSubsetMatchRegistry:
    def test_registered(self):
        from agent_evals.core._registries import scorer_registry

        factory = scorer_registry.get("ToolCallSubsetMatch")
        assert callable(factory())


# =============================================================================
# ToolCallUnorderedMatch — same multiset of calls, any order; counts must
# match. Distinct from set-equality: if reference has [A, A, B] then actual
# must also have [A, A, B] (any order). Args matched per uniform direction.
# =============================================================================


class TestToolCallUnorderedMatchPasses:
    def test_same_calls_reverse_order_passes(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallUnorderedMatch,
        )

        scorer = ToolCallUnorderedMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("a", {}),
                        _assistant_tool_call("b", {}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("b", {}),
                        _assistant_tool_call("a", {}),
                    ]
                },
            ),
        )
        assert result.name == "ToolCallUnorderedMatch"
        assert result.passed is True

    def test_empty_against_empty_passes(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallUnorderedMatch,
        )

        scorer = ToolCallUnorderedMatch()
        result = scorer(
            TaskResult(output="", context={"outputs": []}),
            ExpectedResult(expected="", context={"reference_outputs": []}),
        )
        assert result.passed is True


class TestToolCallUnorderedMatchFails:
    def test_count_mismatch_fails(self):
        """[A, A, B] vs [A, B] — multiset, not set."""
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallUnorderedMatch,
        )

        scorer = ToolCallUnorderedMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("a", {}),
                        _assistant_tool_call("a", {}),
                        _assistant_tool_call("b", {}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("a", {}),
                        _assistant_tool_call("b", {}),
                    ]
                },
            ),
        )
        assert result.passed is False

    def test_extra_call_fails(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallUnorderedMatch,
        )

        scorer = ToolCallUnorderedMatch()
        result = scorer(
            TaskResult(
                output="",
                context={
                    "outputs": [
                        _assistant_tool_call("a", {}),
                        _assistant_tool_call("c", {}),
                    ]
                },
            ),
            ExpectedResult(
                expected="",
                context={"reference_outputs": [_assistant_tool_call("a", {})]},
            ),
        )
        assert result.passed is False

    def test_missing_call_fails(self):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallUnorderedMatch,
        )

        scorer = ToolCallUnorderedMatch()
        result = scorer(
            TaskResult(
                output="",
                context={"outputs": [_assistant_tool_call("a", {})]},
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("a", {}),
                        _assistant_tool_call("b", {}),
                    ]
                },
            ),
        )
        assert result.passed is False


class TestToolCallUnorderedMatchArgsModes:
    """Per-call args modes pair each actual call with a reference call by name.

    With a single name on each side, args must satisfy the chosen mode in its
    uniform direction. The reference is get_status({device: lights}); actual
    is one call to get_status with varying args.
    """

    @pytest.mark.parametrize(
        "mode,actual_args,passes",
        [
            ("exact", {"device": "lights"}, True),
            ("exact", {"device": "lights", "extra": 1}, False),
            ("subset", {"device": "lights"}, True),
            ("subset", {}, True),
            ("subset", {"device": "lights", "extra": 1}, False),
            ("superset", {"device": "lights"}, True),
            ("superset", {"device": "lights", "extra": 1}, True),
            ("superset", {}, False),
            ("ignore", {"device": "doors"}, True),
            ("ignore", {}, True),
        ],
    )
    def test_args_modes(self, mode, actual_args, passes):
        from agent_evals.adapters.scorers.tool_calls import (
            ToolCallUnorderedMatch,
        )

        scorer = ToolCallUnorderedMatch(tool_args_match_mode=mode)
        result = scorer(
            TaskResult(
                output="",
                context={"outputs": [_assistant_tool_call("get_status", actual_args)]},
            ),
            ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [
                        _assistant_tool_call("get_status", {"device": "lights"})
                    ]
                },
            ),
        )
        assert result.passed is passes
        assert result.metadata["tool_args_match_mode"] == mode


class TestToolCallUnorderedMatchRegistry:
    def test_registered(self):
        from agent_evals.core._registries import scorer_registry

        factory = scorer_registry.get("ToolCallUnorderedMatch")
        assert callable(factory())


# =============================================================================
# Shared error-path coverage for the three new natives.
# =============================================================================


class TestNewNativesErrorPaths:
    @pytest.mark.parametrize(
        "factory_name",
        ["ToolCallSupersetMatch", "ToolCallSubsetMatch", "ToolCallUnorderedMatch"],
    )
    def test_missing_outputs_returns_error_score(self, factory_name):
        from agent_evals.core._registries import scorer_registry

        scorer = scorer_registry.get(factory_name)()
        result = scorer(
            TaskResult(output="", context={}),
            ExpectedResult(expected="", context={"reference_outputs": []}),
        )
        assert result.passed is False
        assert "outputs" in result.metadata.get("error", "")

    @pytest.mark.parametrize(
        "factory_name",
        ["ToolCallSupersetMatch", "ToolCallSubsetMatch", "ToolCallUnorderedMatch"],
    )
    def test_missing_reference_returns_error_score(self, factory_name):
        from agent_evals.core._registries import scorer_registry

        scorer = scorer_registry.get(factory_name)()
        result = scorer(
            TaskResult(output="", context={"outputs": []}),
            ExpectedResult(expected="", context={}),
        )
        assert result.passed is False
        assert "reference_outputs" in result.metadata.get("error", "")
