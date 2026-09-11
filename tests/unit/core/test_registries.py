# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for core/_registries.py — generic base + 5 subclasses + 5 instances."""

import sys
import types

import pytest

from agent_evals.core._registries import (
    _BenchmarkConfigReaderRegistry,
    _PlatformRegistry,
    _PreconditionRegistry,
    _ScorerRegistry,
    _SuccessCheckerRegistry,
    config_reader_registry,
    platform_registry,
    precondition_registry,
    scorer_registry,
    success_checker_registry,
)
from agent_evals.core.ports import AdapterRegistry

# Module-level instances exist and are the right type ----------------------


def test_module_level_instances_exist():
    assert isinstance(scorer_registry, _ScorerRegistry)
    assert isinstance(platform_registry, _PlatformRegistry)
    assert isinstance(success_checker_registry, _SuccessCheckerRegistry)
    assert isinstance(precondition_registry, _PreconditionRegistry)
    assert isinstance(config_reader_registry, _BenchmarkConfigReaderRegistry)


def test_all_subclasses_satisfy_adapter_registry_protocol():
    for instance in (
        scorer_registry,
        platform_registry,
        success_checker_registry,
        precondition_registry,
        config_reader_registry,
    ):
        assert isinstance(instance, AdapterRegistry)


# Generic base class basics -------------------------------------------------


def test_base_class_register_get_list_contains():
    reg = _ScorerRegistry()

    @reg.register("fake")
    def fake_factory(**_kwargs):
        return lambda: None

    assert "fake" in reg
    assert reg.list() == ["fake"]
    assert reg.get("fake") is fake_factory


def test_get_unknown_includes_family_label_in_error():
    reg = _ScorerRegistry()
    with pytest.raises(ValueError, match="Unknown scorer: 'missing'"):
        reg.get("missing")


def test_duplicate_registration_with_same_object_is_noop():
    reg = _ScorerRegistry()

    def f(**_kwargs):
        return lambda: None

    reg.register("x")(f)
    reg.register("x")(f)
    assert reg.list() == ["x"]


def test_duplicate_registration_with_different_object_raises():
    reg = _ScorerRegistry()

    def f1(**_kwargs):
        return lambda: None

    def f2(**_kwargs):
        return lambda: None

    reg.register("x")(f1)
    with pytest.raises(ValueError, match="already registered"):
        reg.register("x")(f2)


# Per-family error-message contracts ---------------------------------------


@pytest.mark.parametrize(
    "subclass,family_noun",
    [
        (_ScorerRegistry, "scorer"),
        (_PlatformRegistry, "platform adapter"),
        (_SuccessCheckerRegistry, "success checker"),
        (_PreconditionRegistry, "precondition"),
        (_BenchmarkConfigReaderRegistry, "URI scheme"),
    ],
)
def test_per_family_error_message_noun(subclass, family_noun):
    reg = subclass()
    with pytest.raises(ValueError, match=f"Unknown {family_noun}: 'nope'"):
        reg.get("nope")


@pytest.mark.parametrize(
    "subclass,available_noun",
    [
        # Pinned to match the legacy registries' diagnostics byte-for-byte
        # (see src/agent_evals/adapters/<family>/registry.py). Several
        # families have asymmetric labels — _PlatformRegistry's family
        # noun is "platform adapter" but its plural is "adapters";
        # _BenchmarkConfigReaderRegistry's family noun is "URI scheme"
        # but its plural is "schemes". Production callers and a public
        # docstring (benchmark/config.py) reference these exact strings.
        (_ScorerRegistry, "scorers"),
        (_PlatformRegistry, "adapters"),
        (_SuccessCheckerRegistry, "success checkers"),
        (_PreconditionRegistry, "preconditions"),
        (_BenchmarkConfigReaderRegistry, "schemes"),
    ],
)
def test_per_family_available_noun_in_error_message(subclass, available_noun):
    reg = subclass()
    with pytest.raises(ValueError, match=f"Available {available_noun}: "):
        reg.get("nope")


# Protocol-method check on relevant subclasses -----------------------------


def test_success_checker_registry_rejects_class_missing_required_methods():
    reg = _SuccessCheckerRegistry()

    class Incomplete:
        # Missing to_spec().
        pass

    with pytest.raises(TypeError, match="missing method"):
        reg.register("bad")(Incomplete)


def test_precondition_registry_rejects_class_missing_apply():
    reg = _PreconditionRegistry()

    class Incomplete:
        pass

    with pytest.raises(TypeError, match="missing method.*apply"):
        reg.register("bad")(Incomplete)


def test_config_reader_registry_rejects_class_missing_read():
    reg = _BenchmarkConfigReaderRegistry()

    class Incomplete:
        pass

    with pytest.raises(TypeError, match="missing method.*read"):
        reg.register("bad")(Incomplete)


def test_scorer_registry_does_not_protocol_check():
    reg = _ScorerRegistry()

    def plain_callable(**_kwargs):
        return lambda: None

    reg.register("x")(plain_callable)
    assert reg.get("x") is plain_callable


def test_platform_registry_does_not_protocol_check():
    reg = _PlatformRegistry()

    class MinimalPlatform:
        pass

    reg.register("x")(MinimalPlatform)
    assert reg.get("x") is MinimalPlatform


# _lazy_package data-driven trigger (platforms) ----------------------------


def test_platform_registry_lazy_package_default_is_none():
    reg = _PlatformRegistry()
    assert reg._lazy_package is None


def test_platform_registry_get_unknown_raises_value_error_when_lazy_package_unset():
    reg = _PlatformRegistry()
    with pytest.raises(ValueError, match="Unknown platform adapter"):
        reg.get("nonexistent")


def test_platform_registry_lazy_trigger_imports_lazy_package(monkeypatch):
    reg = _PlatformRegistry()
    fake_module = types.ModuleType("fake_lazy_module_for_test")

    def fake_getattr(name):
        if name == "demo":

            class DemoPlatform: ...

            reg.register("demo")(DemoPlatform)
            return DemoPlatform
        raise AttributeError(name)

    fake_module.__getattr__ = fake_getattr  # ty: ignore[invalid-assignment]
    sys.modules["fake_lazy_module_for_test"] = fake_module
    monkeypatch.setattr(reg, "_lazy_package", "fake_lazy_module_for_test")

    cls = reg.get("demo")
    assert cls.__name__ == "DemoPlatform"


def test_platform_registry_lazy_trigger_propagates_import_error(monkeypatch):
    reg = _PlatformRegistry()
    monkeypatch.setattr(reg, "_lazy_package", "definitely_not_a_real_module_xyz")

    with pytest.raises(ImportError):
        reg.get("anything")


def test_platform_registry_lazy_trigger_swallows_attribute_error(monkeypatch):
    """An unknown adapter name surfaces as ``ValueError``, not ``AttributeError``.

    ``_PlatformRegistry._trigger_lazy`` deliberately catches
    ``AttributeError`` so a genuine miss falls through to the registry's
    standard ``Unknown platform adapter`` ``ValueError`` with the
    ``Available adapters: ...`` list. A regression that re-raises
    ``AttributeError`` (e.g., a future blanket ``except`` change) would
    surface a confusing diagnostic; pin the swallow contract here.
    """
    reg = _PlatformRegistry()
    fake_module = types.ModuleType("fake_lazy_module_for_attr_swallow_test")

    def fake_getattr(name):
        # Mirrors the real adapters/platforms/__init__.py shape: every
        # access raises AttributeError until a known adapter is queried.
        raise AttributeError(name)

    fake_module.__getattr__ = fake_getattr  # ty: ignore[invalid-assignment]
    sys.modules["fake_lazy_module_for_attr_swallow_test"] = fake_module
    monkeypatch.setattr(reg, "_lazy_package", "fake_lazy_module_for_attr_swallow_test")

    with pytest.raises(ValueError, match="Unknown platform adapter"):
        reg.get("nonexistent")


# _format_obj handles classes and Callables --------------------------------


def test_format_obj_handles_class():
    reg = _ScorerRegistry()

    class SomeClass: ...

    label = reg._format_obj(SomeClass)
    assert "SomeClass" in label


def test_format_obj_handles_function():
    reg = _ScorerRegistry()

    def some_function(**_kwargs):
        return lambda: None

    label = reg._format_obj(some_function)
    assert "some_function" in label
