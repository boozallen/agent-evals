# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the platform-adapter registry class."""

import pytest

from agent_evals.core._registries import (
    _PlatformRegistry,
    platform_registry,
)
from agent_evals.core.ports import AdapterRegistry


class _FakeAdapter:
    """Minimal duck-typed Platform for testing.

    Note: Platform has many methods (sync/async eval, log, finalize,
    pull_traces). Real platform adapters implement them all;
    this fake is permissive because the registry doesn't enforce them.
    """

    def initialize_experiment(self, config):
        pass

    def log_result(self, example):
        pass

    def finalize(self):
        pass

    def evaluate(self, runner):
        pass

    async def aevaluate(self, runner):
        pass

    def pull_traces(self, **kwargs):
        return []


@pytest.fixture
def fresh_registry():
    return _PlatformRegistry()


def test_default_registry_implements_protocol():
    assert isinstance(platform_registry, AdapterRegistry)


def test_default_registry_has_local_registered():
    assert "local" in platform_registry


def test_register_decorator_adds_class(fresh_registry):
    fresh_registry.register("fake")(_FakeAdapter)
    assert fresh_registry.get("fake") is _FakeAdapter


def test_get_returns_class(fresh_registry):
    fresh_registry.register("fake")(_FakeAdapter)
    cls = fresh_registry.get("fake")
    assert cls is _FakeAdapter


def test_get_unknown_raises_with_available(fresh_registry):
    fresh_registry.register("fake")(_FakeAdapter)
    with pytest.raises(ValueError, match="Unknown platform adapter: 'missing'"):
        fresh_registry.get("missing")


def test_register_rejects_duplicate(fresh_registry):
    fresh_registry.register("fake")(_FakeAdapter)

    class Other(_FakeAdapter):
        pass

    with pytest.raises(ValueError, match="already registered"):
        fresh_registry.register("fake")(Other)


def test_register_idempotent_with_same_class(fresh_registry):
    fresh_registry.register("fake")(_FakeAdapter)
    fresh_registry.register("fake")(_FakeAdapter)
    assert fresh_registry.get("fake") is _FakeAdapter


def test_list_sorted(fresh_registry):
    fresh_registry.register("zeta")(_FakeAdapter)
    fresh_registry.register("alpha")(_FakeAdapter)
    assert fresh_registry.list() == ["alpha", "zeta"]


def test_contains_returns_bool(fresh_registry):
    fresh_registry.register("fake")(_FakeAdapter)
    assert "fake" in fresh_registry
    assert "not_registered" not in fresh_registry


def test_fresh_registry_is_independent_from_default():
    fresh = _PlatformRegistry()
    assert "local" not in fresh
    assert "local" in platform_registry


def test_get_triggers_lazy_load_for_optional_adapters():
    """Verify the get() method triggers PEP 562 lazy-load on miss.

    The optional-adapter pattern: importing 'braintrust' (or 'mlflow',
    'langfuse') as an attribute on adapters.platforms triggers its
    module to import, which fires the @register decorator and populates
    the registry. get() must trigger this on miss before raising.

    This test is best-effort: if the optional package is installed, we
    expect get('braintrust') to succeed; if not, we expect ImportError
    bubbled up with the install hint.
    """
    # Don't actually require braintrust to be installed; just verify
    # that the lazy-import-on-miss machinery is exercised. We use a
    # try/except because the test environment may or may not have it.
    try:
        cls = platform_registry.get("braintrust")
        # If we got here, braintrust was lazy-loaded successfully
        assert cls is not None
    except ImportError:
        # Package not installed — ImportError must surface (with the
        # "uv add <pkg>" hint). The fix in registry.py:get() deliberately
        # propagates ImportError; only ValueError on genuinely-unknown
        # names is acceptable as a fall-through.
        pass


def test_get_propagates_import_hint_for_missing_optional_dep(monkeypatch):
    """Missing optional adapter packages must surface the 'uv add <pkg>'
    install hint from platforms/__init__.py:__getattr__, not be swallowed
    into a generic 'Unknown adapter' ValueError.
    """
    import agent_evals.adapters.platforms as platforms_pkg

    # Simulate a missing optional package by monkeypatching __getattr__
    # to raise the same hint-bearing ImportError shape produced by the
    # real platforms/__init__.py loader for an uninstalled package.
    original_getattr = platforms_pkg.__getattr__

    def fake_getattr(name):
        if name == "fake_optional":
            raise ImportError(
                "The 'fake_optional' adapter requires the 'fake_pkg' package.\n"
                "Install it with: uv add fake_pkg"
            )
        return original_getattr(name)

    monkeypatch.setattr(platforms_pkg, "__getattr__", fake_getattr)

    with pytest.raises(ImportError, match="uv add fake_pkg"):
        platform_registry.get("fake_optional")


def test_platform_registry_lazy_package_set_by_family_init():
    """adapters/platforms/__init__.py must set _lazy_package on import.

    Pins the data-driven hex-layers fix (spec 025): the registry's
    optional-extras lazy trigger uses self._lazy_package, which the
    family __init__.py writes at family-init time. Without this write,
    optional adapters (mlflow, braintrust, langfuse) silently lose
    their PEP 562 trigger.
    """
    import agent_evals.adapters.platforms  # noqa: F401 — triggers __init__.py
    from agent_evals.core._registries import platform_registry

    assert platform_registry._lazy_package == "agent_evals.adapters.platforms"
