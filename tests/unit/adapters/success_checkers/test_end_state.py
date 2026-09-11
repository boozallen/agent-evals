# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for StateCheckParser.

Validates that the parser accepts dict payloads, rejects bad shapes,
and produces a StateCheck DTO whose expected_state matches the input.
The DTO → scorer dispatch is covered by `tests/unit/test_compile_check.py`.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evals.adapters.success_checkers.end_state import StateCheckParser
from agent_evals.core._registries import success_checker_registry
from agent_evals.core.checks import StateCheck


def test_payload_is_a_dict() -> None:
    parser = StateCheckParser({"k": 1})
    assert parser.root == {"k": 1}

    with pytest.raises(ValidationError):
        StateCheckParser("bad")


def test_to_spec_produces_state_check() -> None:
    parser = StateCheckParser({"lights.kitchen": "on"})
    spec = parser.to_spec()
    assert isinstance(spec, StateCheck)
    assert spec.expected_state == {"lights.kitchen": "on"}


def test_registered_under_state_key() -> None:
    cls = success_checker_registry.get("state")
    assert cls is StateCheckParser
    instance = cls(**{"k": 1})
    assert isinstance(instance, StateCheckParser)
    assert instance.root == {"k": 1}
    assert instance.to_spec() == StateCheck(expected_state={"k": 1})
