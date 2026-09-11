# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the SuccessChecker registry class."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from agent_evals.core._registries import (
    _SuccessCheckerRegistry,
    success_checker_registry,
)
from agent_evals.core.ports import AdapterRegistry


class _FakeChecker(BaseModel):
    """Conforms to the parser contract: payload + to_spec()."""

    model_config = ConfigDict(extra="forbid")

    payload: dict[str, Any]

    def to_spec(self):
        from agent_evals.core.checks import StateCheck

        return StateCheck(expected_state=self.payload)


@pytest.fixture
def fresh_registry() -> _SuccessCheckerRegistry:
    """Return a fresh, empty registry for the test."""
    return _SuccessCheckerRegistry()


def test_default_registry_implements_adapter_registry_protocol() -> None:
    assert isinstance(success_checker_registry, AdapterRegistry)


def test_default_registry_has_state_and_tool_use_registered() -> None:
    assert "state" in success_checker_registry
    assert "tool_use" in success_checker_registry
    names = success_checker_registry.list()
    assert "state" in names
    assert "tool_use" in names


def test_register_decorator_adds_class(fresh_registry: _SuccessCheckerRegistry) -> None:
    @fresh_registry.register("fake")
    class FakeChecker(BaseModel):
        model_config = ConfigDict(extra="forbid")
        payload: dict[str, Any]

        def to_spec(self):
            from agent_evals.core.checks import StateCheck

            return StateCheck(expected_state=self.payload)

    assert "fake" in fresh_registry
    cls = fresh_registry.get("fake")
    assert cls is FakeChecker


def test_get_returns_class_not_instance(
    fresh_registry: _SuccessCheckerRegistry,
) -> None:
    fresh_registry.register("fake")(_FakeChecker)

    cls = fresh_registry.get("fake")
    # Caller is responsible for invocation
    assert cls is _FakeChecker
    assert callable(cls)


def test_caller_invokes_with_payload(
    fresh_registry: _SuccessCheckerRegistry,
) -> None:
    fresh_registry.register("fake")(_FakeChecker)

    cls = fresh_registry.get("fake")
    instance = cls(payload={"k": 1})
    assert isinstance(instance, _FakeChecker)
    assert instance.payload == {"k": 1}


def test_get_unknown_name_raises_with_available(
    fresh_registry: _SuccessCheckerRegistry,
) -> None:
    fresh_registry.register("fake")(_FakeChecker)

    with pytest.raises(ValueError, match="Unknown success checker: 'missing'"):
        fresh_registry.get("missing")


def test_register_rejects_class_missing_protocol_method(
    fresh_registry: _SuccessCheckerRegistry,
) -> None:
    class Bad:
        # Missing to_spec().
        pass

    with pytest.raises(TypeError, match="missing method"):
        fresh_registry.register("bad")(Bad)


def test_register_rejects_duplicate_with_different_class(
    fresh_registry: _SuccessCheckerRegistry,
) -> None:
    fresh_registry.register("fake")(_FakeChecker)

    class OtherChecker(BaseModel):
        model_config = ConfigDict(extra="forbid")
        payload: dict[str, Any]

        def to_spec(self):
            from agent_evals.core.checks import StateCheck

            return StateCheck(expected_state=self.payload)

    with pytest.raises(ValueError, match="already registered"):
        fresh_registry.register("fake")(OtherChecker)


def test_register_idempotent_with_same_class(
    fresh_registry: _SuccessCheckerRegistry,
) -> None:
    fresh_registry.register("fake")(_FakeChecker)
    # Re-registering the same class is a no-op (decorator-on-import safety).
    fresh_registry.register("fake")(_FakeChecker)
    assert fresh_registry.get("fake") is _FakeChecker


def test_list_returns_sorted_names(fresh_registry: _SuccessCheckerRegistry) -> None:
    fresh_registry.register("zeta")(_FakeChecker)
    fresh_registry.register("alpha")(_FakeChecker)
    fresh_registry.register("mu")(_FakeChecker)

    assert fresh_registry.list() == ["alpha", "mu", "zeta"]


def test_contains_returns_bool(fresh_registry: _SuccessCheckerRegistry) -> None:
    fresh_registry.register("fake")(_FakeChecker)
    assert "fake" in fresh_registry
    assert "not_registered" not in fresh_registry


def test_fresh_registry_is_independent_from_default() -> None:
    fresh = _SuccessCheckerRegistry()
    # The default has state + tool_use registered (via family-init eager imports);
    # the fresh registry is empty.
    assert "state" not in fresh
    assert "tool_use" not in fresh
    assert "state" in success_checker_registry
    assert "tool_use" in success_checker_registry


def test_tool_calls_yaml_key_is_not_a_registered_success_checker() -> None:
    """``tool_calls`` (the OpenAI message-format field name) must not be a
    YAML registry key.

    The success-checker layer's vocabulary boundary depends on this
    separation: the YAML user surface uses ``tool_use:`` while the
    OpenAI message-format field inside ``reference_outputs:`` and
    ``outputs:`` stays ``tool_calls``. If a future contributor
    re-registers ``tool_calls:`` as an alias for back-compat, the user
    surface and the message-field layer collide — exactly the
    accidental-collision the rename was designed to prevent.

    Locks ``tool_use`` as the canonical YAML key so the diagnostic on a
    stale benchmark YAML stays clear (``Unknown success checker:
    'tool_calls'. Available success checkers: state, tool_use``).
    """
    names = success_checker_registry.list()
    assert "tool_calls" not in names, (
        "`tool_calls` reappeared as a registered success-checker key. "
        "If this is a deliberate back-compat alias, document it in "
        "CHANGELOG and remove this test; otherwise unregister it."
    )
    assert "tool_use" in names, (
        "`tool_use` is the canonical YAML key; if it disappeared the "
        "rename was reverted without updating this test."
    )
