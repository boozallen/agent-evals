# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Invariant: ``scorer_config`` equality must imply group-key equality.

Walks the CheckSpec union members and asserts the contract:

    Two specs of the same type with equal ``compile_check(spec).scorer_config``
    produce equal group keys; with unequal ``scorer_config`` they produce
    unequal group keys.

A new CheckSpec member added with a configurable field that affects the
scorer factory must surface that field through ``scorer_config``.
Forgetting to fails this test.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from agent_evals import _compile_check
from agent_evals.benchmark.config import ScenarioSpec
from agent_evals.benchmark.loader import _scenario_group_key
from agent_evals.core.checks import StateCheck, ToolCall, ToolUseCheck


class _ScenarioStub:
    """Stand-in for ScenarioSpec exposing only ``parsed_check_specs``.

    ``_scenario_group_key`` reads only ``parsed_check_specs``; constructing
    a real ``ScenarioSpec`` here would require full YAML validation
    unrelated to what this test exercises. Callsites cast to
    ``ScenarioSpec`` to satisfy the type checker.
    """

    def __init__(self, parsed_check_specs: list) -> None:
        self.parsed_check_specs = parsed_check_specs


# Each tuple: (spec_a, spec_b, expected_same_key).
# expected_same_key=True asserts the two instances share a group;
# False asserts they split.
_INVARIANT_CASES: list[tuple[Any, Any, bool]] = [
    # StateCheck has no factory config — different state payloads
    # still share a group (state flows through expected_context).
    (
        StateCheck(expected_state={"a": 1}),
        StateCheck(expected_state={"a": 2}),
        True,
    ),
    # ToolUseCheck: same (match_calls, match_args) tuple → same
    # group regardless of the call list itself; different on either axis
    # → split.
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="exact",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="y", args={}),),
            match_calls="exact",
            match_args="exact",
        ),
        True,
    ),
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="subset",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="exact",
        ),
        False,
    ),
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="ignore",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="exact",
        ),
        False,
    ),
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="superset",
            match_args="exact",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="exact",
        ),
        False,
    ),
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="subset",
            match_args="exact",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="exact",
        ),
        False,
    ),
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="unordered",
            match_args="exact",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="exact",
        ),
        False,
    ),
    # any_of must split from every other match_calls value, since each
    # dispatches to a distinct scorer factory.
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="any_of",
            match_args="exact",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="exact",
        ),
        False,
    ),
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="any_of",
            match_args="exact",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="any_of",
            match_args="subset",
        ),
        False,
    ),
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="any_of",
            match_args="ignore",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="y", args={}),),
            match_calls="any_of",
            match_args="ignore",
        ),
        True,
    ),
    # match_args=superset must split from every other match_args value
    # since each one means a different containment direction.
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="superset",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="subset",
        ),
        False,
    ),
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="any_of",
            match_args="superset",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="any_of",
            match_args="subset",
        ),
        False,
    ),
    (
        ToolUseCheck(
            calls=(ToolCall(name="x", args={}),),
            match_calls="exact",
            match_args="superset",
        ),
        ToolUseCheck(
            calls=(ToolCall(name="y", args={}),),
            match_calls="exact",
            match_args="superset",
        ),
        True,
    ),
]


@pytest.mark.parametrize(
    ("spec_a", "spec_b", "expected_same_key"),
    _INVARIANT_CASES,
)
def test_grouping_invariant_for_built_in_check_specs(
    spec_a: Any, spec_b: Any, expected_same_key: bool
) -> None:
    """Same scorer_config ⇔ same group key, for every CheckSpec member."""
    config_a = _compile_check(spec_a).scorer_config
    config_b = _compile_check(spec_b).scorer_config

    if expected_same_key:
        assert config_a == config_b
    else:
        assert config_a != config_b

    key_a = _scenario_group_key(cast(ScenarioSpec, _ScenarioStub([spec_a])))
    key_b = _scenario_group_key(cast(ScenarioSpec, _ScenarioStub([spec_b])))
    if expected_same_key:
        assert key_a == key_b
    else:
        assert key_a != key_b


def test_every_check_spec_member_has_invariant_coverage() -> None:
    """Every CheckSpec union member must appear in _INVARIANT_CASES.

    Adding a new member without a coverage entry lands without grouping
    tests. Guards against silent under-coverage when CheckSpec grows.
    """
    import typing

    from agent_evals.core.checks import CheckSpec

    members = set(typing.get_args(CheckSpec))
    covered = {type(spec_a) for spec_a, _, _ in _INVARIANT_CASES}
    missing = sorted(cls.__name__ for cls in members if cls not in covered)
    assert not missing, (
        f"CheckSpec members not represented in _INVARIANT_CASES: "
        f"{missing}. Add at least one entry per member."
    )
