# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for StatePreconditionApplier.

YAML key is ``state``; the applier deep-merges its payload (with
dotted-path expansion) into the agent-supplied target dict before the
agent runs. Mirrors the StateCheckParser tests stylistically.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evals.adapters.preconditions.state import StatePreconditionApplier
from agent_evals.core._registries import precondition_registry
from agent_evals.core.ports import PreconditionApplier


def test_parses_state_payload() -> None:
    """The applier's payload is the seed-state dict, directly."""
    applier = StatePreconditionApplier({"k": 1})
    assert applier.root == {"k": 1}

    # Non-dict payload fails validation.
    with pytest.raises(ValidationError):
        # intentional: testing pydantic-side validation of a bad payload.
        StatePreconditionApplier("bad")


def test_apply_deep_merges_into_target() -> None:
    """apply() deep-merges the payload into target (override wins)."""
    target: dict = {
        "lights": {"kitchen": {"state": "off"}, "office": {"state": "on"}},
        "doors": {"front": {"locked": True}},
    }
    applier = StatePreconditionApplier(
        {
            "lights": {"kitchen": {"state": "on"}},
            "windows": {"north": {"open": False}},
        }
    )
    applier.apply(target)

    # Overlapping nested key overwritten.
    assert target["lights"]["kitchen"] == {"state": "on"}
    # Non-overlapping sibling preserved.
    assert target["lights"]["office"] == {"state": "on"}
    # Untouched top-level branch preserved.
    assert target["doors"] == {"front": {"locked": True}}
    # New top-level branch added.
    assert target["windows"] == {"north": {"open": False}}


def test_apply_expands_dotted_paths() -> None:
    """Dotted-path keys expand into nested dicts before merging."""
    target: dict = {}
    applier = StatePreconditionApplier({"a.b.c": 1})
    applier.apply(target)
    assert target == {"a": {"b": {"c": 1}}}

    # Mixed dotted + plain keys.
    target2: dict = {}
    applier2 = StatePreconditionApplier({"a.b": 1, "c": {"d": 2}})
    applier2.apply(target2)
    assert target2 == {"a": {"b": 1}, "c": {"d": 2}}


def test_apply_mutates_in_place() -> None:
    """apply() mutates the passed-in dict and returns None."""
    target: dict = {"existing": "value"}
    applier = StatePreconditionApplier({"new": "value"})
    result = applier.apply(target)

    assert result is None
    assert target == {"existing": "value", "new": "value"}


def test_satisfies_protocol() -> None:
    """An instance satisfies the PreconditionApplier Protocol."""
    applier = StatePreconditionApplier({"k": 1})
    assert isinstance(applier, PreconditionApplier)


def test_registered_under_state_key() -> None:
    """precondition_registry.get('state') returns StatePreconditionApplier."""
    cls = precondition_registry.get("state")
    instance = cls(**{"k": 1})
    assert isinstance(instance, StatePreconditionApplier)
    assert instance.root == {"k": 1}


def test_apply_dotted_path_clobbers_scalar_at_same_branch() -> None:
    """When a scalar and a dotted path share a top-level prefix, dotted wins.

    Within a single payload, a scalar at ``"a"`` followed by ``"a.b"``
    has the dotted form replace the scalar at the branch root (the
    expansion grows nested dicts where it needs them). This is a
    consequence of dotted-path expansion plus deep-merge semantics, and
    is the documented behavior — pathological inputs like
    ``{"a": 1, "a.b": 2}`` are caller errors but produce predictable
    output rather than partial state.
    """
    target: dict = {}
    applier = StatePreconditionApplier({"a": 1, "a.b": 2})
    applier.apply(target)
    # The dotted-path form expanded "a" into a dict and placed "b"
    # inside; the prior scalar at "a" is replaced by the nested dict.
    assert target == {"a": {"b": 2}}


def test_apply_scalar_after_dotted_path_clobbers_branch() -> None:
    """Reverse order of the test above: a later scalar clobbers a built branch."""
    target: dict = {}
    applier = StatePreconditionApplier({"a.b": 2, "a": 1})
    applier.apply(target)
    # The dotted-path form first built {"a": {"b": 2}}, then the
    # plain "a": 1 overwrote the branch wholesale.
    assert target == {"a": 1}
