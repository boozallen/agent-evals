# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for ToolUseCheckParser.

Validates payload shape, rejects bad inputs, and confirms `to_spec()`
produces a `ToolUseCheck` DTO whose fields mirror the input. Behavioral
dispatch (which scorer fires per match_calls, args forwarding, etc.) is
covered by `tests/unit/test_compile_check.py`.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from agent_evals.adapters.success_checkers.tool_use import ToolUseCheckParser
from agent_evals.core._registries import success_checker_registry
from agent_evals.core.checks import ToolCall, ToolUseCheck


def _calls_one(name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
    return [{"name": name, "args": args}]


def test_parses_calls_payload() -> None:
    calls = _calls_one("fly_to", {"lat": 1, "lon": 2})
    parser = ToolUseCheckParser(calls=calls)
    # parser.calls is list[_ToolCallPayload]; compare via typed attributes
    assert len(parser.calls) == 1
    assert parser.calls[0].name == "fly_to"
    assert parser.calls[0].args == {"lat": 1, "lon": 2}

    with pytest.raises(ValidationError):
        ToolUseCheckParser(calls="bad")

    with pytest.raises(ValidationError):
        ToolUseCheckParser(calls=["not a dict"])

    with pytest.raises(ValidationError):
        ToolUseCheckParser(calls=[], extra_field=42)


def test_to_spec_produces_tool_use_check_with_typed_calls() -> None:
    calls = _calls_one("fly_to", {"lat": 1})
    parser = ToolUseCheckParser(calls=calls, match_calls="any_of", match_args="ignore")
    spec = parser.to_spec()
    assert isinstance(spec, ToolUseCheck)
    assert spec.match_calls == "any_of"
    assert spec.match_args == "ignore"
    # `calls` is converted to a tuple of frozen ToolCall records.
    assert spec.calls == (ToolCall(name="fly_to", args={"lat": 1}),)


def test_to_spec_defaults_match_calls_and_args_to_exact() -> None:
    parser = ToolUseCheckParser(calls=_calls_one("x", {}))
    spec = parser.to_spec()
    assert spec.match_calls == "exact"
    assert spec.match_args == "exact"


def test_invalid_match_calls_rejected() -> None:
    with pytest.raises(ValidationError):
        ToolUseCheckParser(
            calls=_calls_one("x", {}),
            match_calls="bogus",
        )


@pytest.mark.parametrize("match_args", ["bogus", "", "SUPERSET", "super_set", None, 42])
def test_invalid_match_args_rejected(match_args: object) -> None:
    with pytest.raises(ValidationError):
        ToolUseCheckParser(
            calls=_calls_one("x", {}),
            match_args=match_args,
        )


def test_legacy_match_field_is_rejected() -> None:
    """Old `match:` field no longer accepted (extra='forbid')."""
    with pytest.raises(ValidationError):
        ToolUseCheckParser(calls=_calls_one("x", {}), match="subset")


def test_registered_under_tool_use_key() -> None:
    cls = success_checker_registry.get("tool_use")
    assert cls is ToolUseCheckParser
    instance = cls(calls=_calls_one("fly_to", {"lat": 1}))
    spec = instance.to_spec()
    assert isinstance(spec, ToolUseCheck)
    assert spec.calls == (ToolCall(name="fly_to", args={"lat": 1}),)


def test_typo_in_call_field_name_surfaces_as_validation_error_not_keyerror() -> None:
    """A YAML typo like `nme: lock_door` must fail at the Pydantic boundary
    with a structured field-path error, not as a bare KeyError inside to_spec().

    This pins the parser-tightening contract: missing `name` is a
    validation error with field-path data preserved.
    """
    with pytest.raises(ValidationError) as excinfo:
        ToolUseCheckParser(calls=[{"nme": "lock", "args": {}}])

    # The error must mention the missing `name` field (Pydantic surfaces
    # this with field-path data via .errors()).
    errors = excinfo.value.errors()
    assert any("name" in str(err.get("loc", [])) for err in errors), (
        f"ValidationError did not name the `name` field: {errors!r}"
    )


def test_extra_field_inside_call_payload_is_rejected() -> None:
    """`extra='forbid'` on _ToolCallPayload prevents accidental fields."""
    with pytest.raises(ValidationError):
        ToolUseCheckParser(calls=[{"name": "lock", "args": {}, "wrong": 1}])
