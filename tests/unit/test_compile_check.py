# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cross-boundary integrity: DTO Literal vocabulary ↔ scorer factory kwargs.

``ty`` cannot bridge ``CheckSpec``'s Literal values with the scorer
factories' ``tool_args_match_mode`` kwarg shapes. This test exercises
every (match_calls × match_args) combination via runtime evidence
(score.name, score.metadata) plus the ``assert_never`` fallback.
"""

from __future__ import annotations

from typing import cast

import pytest

from agent_evals._composition import CompiledCheck, compile_check
from agent_evals.core.checks import CheckSpec, StateCheck, ToolCall, ToolUseCheck
from agent_evals.core.types import ExpectedResult, Score, TaskResult

_TOOL_USE_SCORER_NAME: dict[str, str] = {
    "exact": "ToolCallExactMatch",
    "subset": "ToolCallSubsetMatch",
    "superset": "ToolCallSupersetMatch",
    "unordered": "ToolCallUnorderedMatch",
    "any_of": "ToolCallAnyMatch",
}


def test_compile_check_returns_compiled_check_dataclass() -> None:
    """compile_check returns a CompiledCheck record exposing .scorer / .expected_context / .scorer_config."""
    result = compile_check(StateCheck(expected_state={}))
    assert isinstance(result, CompiledCheck)
    assert callable(result.scorer)
    assert isinstance(result.expected_context, dict)
    assert isinstance(result.scorer_config, dict)


def test_state_check_dispatches_to_state_match() -> None:
    spec = StateCheck(expected_state={"x": 1})
    compiled = compile_check(spec)
    # Run the scorer against a matching trace; score.name reveals dispatch.
    expected = ExpectedResult(expected="", context=compiled.expected_context)
    result = TaskResult(output="", context={"final_state": {"x": 1}})
    score = compiled.scorer(result, expected)
    # compile_check dispatches to synchronous scorers; narrow the
    # Score | Awaitable[Score] return so attribute access type-checks.
    assert isinstance(score, Score)
    assert score.name == "StateMatch"
    assert compiled.expected_context == {"expected_state": {"x": 1}}
    assert compiled.scorer_config == {}


@pytest.mark.parametrize(
    "match_calls", ["exact", "subset", "superset", "unordered", "any_of"]
)
@pytest.mark.parametrize("match_args", ["exact", "subset", "superset", "ignore"])
def test_tool_use_check_dispatches_to_correct_scorer(
    match_calls: str, match_args: str
) -> None:
    """Every (match_calls × match_args) combination dispatches correctly.

    20 cases. Catches: missing `case` arm, wrong scorer-class assignment
    per match_calls, missing `tool_args_match_mode` forwarding to the
    factory.
    """
    calls = (ToolCall(name="lock", args={"door": "front"}),)
    spec = ToolUseCheck(
        calls=calls,
        match_calls=match_calls,
        match_args=match_args,
    )
    compiled = compile_check(spec)

    # Run the scorer against a matching trace. score.name reveals which
    # factory was dispatched; score.metadata reveals which match_args
    # mode was forwarded into tool_args_match_mode.
    expected = ExpectedResult(expected="", context=compiled.expected_context)
    actual = TaskResult(
        output="",
        context={
            "outputs": [
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "lock", "args": {"door": "front"}}],
                }
            ]
        },
    )
    score = compiled.scorer(actual, expected)
    # compile_check dispatches to synchronous scorers; narrow the
    # Score | Awaitable[Score] return so attribute access type-checks.
    assert isinstance(score, Score)

    # (a) Scorer factory dispatch: score.name reflects which class fired.
    expected_scorer_name = _TOOL_USE_SCORER_NAME[match_calls]
    assert score.name == expected_scorer_name, (
        f"match_calls={match_calls!r} dispatched to {score.name}, "
        f"expected {expected_scorer_name}"
    )

    # (b) match_args forwarded to the scorer's tool_args_match_mode kwarg.
    assert score.metadata.get("tool_args_match_mode") == match_args, (
        f"match_args={match_args!r} did not reach the scorer's "
        f"tool_args_match_mode metadata (got {score.metadata!r})"
    )

    # (c) scorer_config exposes both knobs (loader uses this for grouping).
    assert compiled.scorer_config == {
        "match_calls": match_calls,
        "match_args": match_args,
    }


def test_tool_use_expected_context_uses_openai_message_envelope() -> None:
    """expected_context wraps calls in {role: assistant, tool_calls: [...]} so the scorer's _extract_tool_calls finds them."""
    calls = (
        ToolCall(name="fly_to", args={"lat": 1, "lon": 2}),
        ToolCall(name="land", args={}),
    )
    spec = ToolUseCheck(calls=calls, match_calls="exact", match_args="exact")
    compiled = compile_check(spec)

    assert compiled.expected_context == {
        "reference_outputs": [
            {
                "role": "assistant",
                "tool_calls": [
                    {"name": "fly_to", "args": {"lat": 1, "lon": 2}},
                    {"name": "land", "args": {}},
                ],
            }
        ]
    }


def test_compile_check_assert_never_on_synthetic_novel_spec() -> None:
    """Synthetic spec that bypasses the union forces assert_never to raise.

    `ty` proves exhaustiveness over `CheckSpec` at type-check time;
    this test exercises the runtime fallback for the case where a
    caller `cast`s a non-union value into the dispatch (or a future
    refactor adds a union member without a `case` arm — same failure
    mode at runtime, surfaced before the bad code can ship).
    """

    class _NotACheck:
        """Structurally not a CheckSpec member."""

    bogus = cast(CheckSpec, _NotACheck())
    with pytest.raises(AssertionError):
        compile_check(bogus)
