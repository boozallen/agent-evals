# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the BenchmarkConfigReader registry class."""

import pytest

from agent_evals.core._registries import (
    _BenchmarkConfigReaderRegistry,
    config_reader_registry,
)
from agent_evals.core.ports import AdapterRegistry


class _FakeReader:
    """Conforms to BenchmarkConfigReader Protocol."""

    def read(self, uri: str) -> str:
        return "x"


@pytest.fixture
def fresh_registry():
    """Return a fresh, empty registry for the test."""
    return _BenchmarkConfigReaderRegistry()


def test_default_registry_implements_adapter_registry_protocol():
    assert isinstance(config_reader_registry, AdapterRegistry)


def test_default_registry_has_file_scheme_registered():
    assert "file" in config_reader_registry
    assert "file" in config_reader_registry.list()


def test_register_decorator_adds_class(fresh_registry):
    @fresh_registry.register("fake")
    class FakeReader:
        def read(self, uri: str) -> str:
            return "x"

    assert "fake" in fresh_registry
    cls = fresh_registry.get("fake")
    assert cls is FakeReader


def test_get_returns_class_not_instance(fresh_registry):
    fresh_registry.register("fake")(_FakeReader)

    cls = fresh_registry.get("fake")
    # Caller is responsible for invocation
    assert cls is _FakeReader
    assert callable(cls)


def test_get_unknown_scheme_raises_with_available_schemes(fresh_registry):
    fresh_registry.register("fake")(_FakeReader)

    with pytest.raises(ValueError, match="Unknown URI scheme: 's3'"):
        fresh_registry.get("s3")


def test_register_rejects_class_missing_protocol_method(fresh_registry):
    class Bad:
        # Missing read() method
        pass

    with pytest.raises(TypeError, match="missing method.*read"):
        fresh_registry.register("bad")(Bad)


def test_register_rejects_duplicate_scheme_with_different_class(fresh_registry):
    fresh_registry.register("fake")(_FakeReader)

    class OtherReader:
        def read(self, uri: str) -> str:
            return "y"

    with pytest.raises(ValueError, match="already registered"):
        fresh_registry.register("fake")(OtherReader)


def test_register_idempotent_with_same_class(fresh_registry):
    fresh_registry.register("fake")(_FakeReader)
    # Re-registering the same class is a no-op (decorator-on-import safety)
    fresh_registry.register("fake")(_FakeReader)
    assert fresh_registry.get("fake") is _FakeReader


def test_list_returns_sorted_schemes(fresh_registry):
    fresh_registry.register("zeta")(_FakeReader)
    fresh_registry.register("alpha")(_FakeReader)
    fresh_registry.register("mu")(_FakeReader)

    assert fresh_registry.list() == ["alpha", "mu", "zeta"]


def test_contains_returns_bool(fresh_registry):
    fresh_registry.register("fake")(_FakeReader)
    assert "fake" in fresh_registry
    assert "not_registered" not in fresh_registry


def test_fresh_registry_is_independent_from_default():
    fresh = _BenchmarkConfigReaderRegistry()
    assert "file" not in fresh  # default has it; fresh doesn't
    assert "file" in config_reader_registry
