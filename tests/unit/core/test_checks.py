# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""DTO contract: frozen, equatable, hashable.

`compile_check` will dispatch on these types; `_scenario_group_key`
groups on hash equality of derived configs. Tests pin the invariants
those consumers rely on.
"""

from __future__ import annotations

import dataclasses

import pytest

from agent_evals.core.checks import (
    CheckSpec,
    StateCheck,
    ToolCall,
    ToolUseCheck,
)


def test_state_check_is_frozen() -> None:
    s = StateCheck(expected_state={"x": 1})
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.expected_state = {"x": 2}  # ty: ignore[invalid-assignment]


def test_tool_use_check_is_frozen() -> None:
    t = ToolUseCheck(calls=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        t.match_calls = "any_of"  # ty: ignore[invalid-assignment]


def test_tool_call_is_frozen() -> None:
    c = ToolCall(name="x", args={})
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.name = "y"  # ty: ignore[invalid-assignment]


def test_state_check_equality_by_value() -> None:
    a = StateCheck(expected_state={"x": 1})
    b = StateCheck(expected_state={"x": 1})
    c = StateCheck(expected_state={"x": 2})
    assert a == b
    assert a != c


def test_tool_use_check_equality_by_value() -> None:
    cs = (ToolCall(name="lock", args={"door": "front"}),)
    a = ToolUseCheck(calls=cs, match_calls="exact", match_args="exact")
    b = ToolUseCheck(calls=cs, match_calls="exact", match_args="exact")
    c = ToolUseCheck(calls=cs, match_calls="any_of", match_args="exact")
    assert a == b
    assert a != c


def test_grouping_keys_are_stable_for_equal_payload_dtos() -> None:
    """Two separately-constructed DTOs with equal payloads produce
    equal group keys.

    The DTOs themselves are NOT hashable (their `dict[str, Any]` leaf
    fields are mutable). Group-key construction in the loader walks
    each spec through `_freeze`, which canonicalizes dicts to
    frozensets. This test pins the contract that callers can rely on
    `_freeze`-derived equality across separately-constructed DTOs.
    """
    from agent_evals.benchmark.loader import _freeze

    state_a = StateCheck(expected_state={"x": 1, "y": 2})
    state_b = StateCheck(expected_state={"y": 2, "x": 1})  # different insertion order
    assert _freeze(state_a.expected_state) == _freeze(state_b.expected_state)

    calls_a = (ToolCall(name="x", args={"a": 1, "b": 2}),)
    calls_b = (ToolCall(name="x", args={"b": 2, "a": 1}),)  # different insertion order
    tool_a = ToolUseCheck(calls=calls_a, match_calls="exact", match_args="exact")
    tool_b = ToolUseCheck(calls=calls_b, match_calls="exact", match_args="exact")
    # Tuple of frozen ToolCall is comparable directly only if `args` matches by-value.
    # The args are different dict instances; equality is by value.
    assert tool_a == tool_b


def test_check_spec_is_a_union_of_state_and_tool_use() -> None:
    """`CheckSpec` is a closed union; instances of either member satisfy it."""
    s: CheckSpec = StateCheck(expected_state={})
    t: CheckSpec = ToolUseCheck(calls=())
    # Type-level test mostly; runtime sanity:
    assert isinstance(s, StateCheck)
    assert isinstance(t, ToolUseCheck)
