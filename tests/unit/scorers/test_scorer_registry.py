# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the scorer registry class."""

import pytest

from agent_evals.core._registries import (
    _ScorerRegistry,
    scorer_registry,
)
from agent_evals.core.ports import AdapterRegistry


@pytest.fixture
def fresh_registry():
    """Return a fresh, empty registry for the test."""
    return _ScorerRegistry()


def test_default_registry_implements_protocol():
    assert isinstance(scorer_registry, AdapterRegistry)


def test_default_registry_has_known_scorers():
    # Spot-check: a few well-known names from autoevals/agentevals/state
    assert "StateMatch" in scorer_registry
    assert "ExactMatch" in scorer_registry
    assert "TrajectoryStrictMatch" in scorer_registry


def test_register_decorator_adds_factory(fresh_registry):
    @fresh_registry.register("dummy")
    def dummy_factory(value: float = 0.5):
        def scorer(result, expected=None, **context):
            return value

        return scorer

    factory = fresh_registry.get("dummy")
    assert factory is dummy_factory


def test_caller_invokes_factory_with_kwargs(fresh_registry):
    @fresh_registry.register("dummy")
    def dummy_factory(value: float = 0.5):
        def scorer(result, expected=None, **context):
            return value

        return scorer

    factory = fresh_registry.get("dummy")
    scorer = factory(value=0.9)
    assert scorer(None, None) == 0.9


def test_get_unknown_raises(fresh_registry):
    with pytest.raises(ValueError, match="Unknown scorer: 'missing'"):
        fresh_registry.get("missing")


def test_register_no_protocol_check(fresh_registry):
    # Scorers are duck-typed callables, not Protocol-checked classes.
    # Register accepts any callable; the burden is on the factory to
    # return a valid scorer at call time.
    @fresh_registry.register("any_callable")
    def factory():
        return lambda _o, _e: 1.0

    assert fresh_registry.get("any_callable") is factory


def test_register_rejects_duplicate_with_different_callable(fresh_registry):
    @fresh_registry.register("dummy")
    def factory_one():
        return lambda _o, _e: 1.0

    def factory_two():
        return lambda _o, _e: 2.0

    with pytest.raises(ValueError, match="already registered"):
        fresh_registry.register("dummy")(factory_two)


def test_register_idempotent_with_same_callable(fresh_registry):
    @fresh_registry.register("dummy")
    def factory_one():
        return lambda _o, _e: 1.0

    # Re-registering same callable is a no-op
    fresh_registry.register("dummy")(factory_one)
    assert fresh_registry.get("dummy") is factory_one


def test_list_returns_sorted(fresh_registry):
    @fresh_registry.register("zeta")
    def f1():
        pass

    @fresh_registry.register("alpha")
    def f2():
        pass

    assert fresh_registry.list() == ["alpha", "zeta"]


def test_contains_returns_bool(fresh_registry):
    @fresh_registry.register("dummy")
    def factory():
        pass

    assert "dummy" in fresh_registry
    assert "not_registered" not in fresh_registry


def test_fresh_registry_is_independent_from_default():
    fresh = _ScorerRegistry()
    assert "StateMatch" not in fresh
    assert "StateMatch" in scorer_registry
