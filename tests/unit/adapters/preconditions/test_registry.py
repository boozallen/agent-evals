# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the PreconditionApplier registry class."""

import pytest

from agent_evals.core._registries import (
    _PreconditionRegistry,
    precondition_registry,
)
from agent_evals.core.ports import AdapterRegistry


class _FakeApplier:
    """Conforms to PreconditionApplier Protocol."""

    def __init__(self, **payload):
        self.payload = payload

    def apply(self, state):
        return state


@pytest.fixture
def fresh_registry():
    """Return a fresh, empty registry for the test."""
    return _PreconditionRegistry()


def test_default_registry_implements_adapter_registry_protocol():
    assert isinstance(precondition_registry, AdapterRegistry)


def test_default_registry_has_state_registered():
    assert "state" in precondition_registry
    assert "state" in precondition_registry.list()


def test_register_decorator_adds_class(fresh_registry):
    @fresh_registry.register("fake")
    class FakeApplier:
        def __init__(self, **payload):
            self.payload = payload

        def apply(self, state):
            return state

    assert "fake" in fresh_registry
    cls = fresh_registry.get("fake")
    assert cls is FakeApplier


def test_get_returns_class_not_instance(fresh_registry):
    fresh_registry.register("fake")(_FakeApplier)

    cls = fresh_registry.get("fake")
    # Caller is responsible for invocation
    assert cls is _FakeApplier
    assert callable(cls)


def test_caller_invokes_with_payload(fresh_registry):
    fresh_registry.register("fake")(_FakeApplier)

    cls = fresh_registry.get("fake")
    instance = cls(**{"k": 1})
    assert instance.payload == {"k": 1}


def test_get_unknown_name_raises_with_available(fresh_registry):
    fresh_registry.register("fake")(_FakeApplier)

    with pytest.raises(ValueError, match="Unknown precondition: 'missing'"):
        fresh_registry.get("missing")


def test_register_rejects_class_missing_protocol_method(fresh_registry):
    class Bad:
        # Missing apply() method
        pass

    with pytest.raises(TypeError, match="missing method.*apply"):
        fresh_registry.register("bad")(Bad)


def test_register_rejects_duplicate_with_different_class(fresh_registry):
    fresh_registry.register("fake")(_FakeApplier)

    class OtherApplier:
        def apply(self, state):
            return state

    with pytest.raises(ValueError, match="already registered"):
        fresh_registry.register("fake")(OtherApplier)


def test_register_idempotent_with_same_class(fresh_registry):
    fresh_registry.register("fake")(_FakeApplier)
    # Re-registering the same class is a no-op (decorator-on-import safety)
    fresh_registry.register("fake")(_FakeApplier)
    assert fresh_registry.get("fake") is _FakeApplier


def test_list_returns_sorted_names(fresh_registry):
    fresh_registry.register("zeta")(_FakeApplier)
    fresh_registry.register("alpha")(_FakeApplier)
    fresh_registry.register("mu")(_FakeApplier)

    assert fresh_registry.list() == ["alpha", "mu", "zeta"]


def test_contains_returns_bool(fresh_registry):
    fresh_registry.register("fake")(_FakeApplier)
    assert "fake" in fresh_registry
    assert "not_registered" not in fresh_registry


def test_fresh_registry_is_independent_from_default():
    fresh = _PreconditionRegistry()
    assert "state" not in fresh  # default has it; fresh doesn't
    assert "state" in precondition_registry
