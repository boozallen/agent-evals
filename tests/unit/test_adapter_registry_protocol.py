# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Protocol-conformance tests for AdapterRegistry[T]."""


def test_adapter_registry_protocol_importable_from_core_ports():
    from agent_evals.core.ports import AdapterRegistry

    assert AdapterRegistry is not None


def test_adapter_registry_protocol_has_required_methods():
    from agent_evals.core.ports.registry import AdapterRegistry

    method_names = {"get", "list", "__contains__"}
    for name in method_names:
        assert hasattr(AdapterRegistry, name), f"AdapterRegistry missing {name!r}"


def test_adapter_registry_protocol_is_runtime_checkable():
    from agent_evals.core.ports.registry import AdapterRegistry

    class _Conforming:
        def get(self, key: str):
            return None

        def list(self) -> list[str]:
            return []

        def __contains__(self, key: str) -> bool:
            return False

    instance = _Conforming()
    assert isinstance(instance, AdapterRegistry)


def test_adapter_registry_rejects_non_conforming_class():
    from agent_evals.core.ports.registry import AdapterRegistry

    class _Missing:
        def get(self, key: str):
            return None

        # Missing list() and __contains__

    instance = _Missing()
    assert not isinstance(instance, AdapterRegistry)


def test_adapter_registry_is_generic():
    from agent_evals.core.ports.registry import AdapterRegistry

    # AdapterRegistry[int] should be subscriptable without error
    parameterized = AdapterRegistry[int]
    assert parameterized is not None


def test_same_registered_identity_fast_path():
    from agent_evals.core.ports.registry import _same_registered

    def f():
        pass

    assert _same_registered(f, f)


def test_same_registered_module_reload_safety():
    """Two function objects with the same __module__ and __qualname__
    are treated as the same registered callable.

    This is the module-reload safety case: when sys.modules is cleared
    and a module re-imported, the @register decorator runs again with a
    new function object (different identity) for the same logical
    callable. Without this safety, the decorator would raise
    ValueError("already registered") on re-import.
    """
    from agent_evals.core.ports.registry import _same_registered

    def make_factory():
        def Factory():  # noqa: N802 - intentional cls-like name for the test
            pass

        return Factory

    f1 = make_factory()
    f2 = make_factory()
    # Different identities
    assert f1 is not f2
    # But same __module__ and __qualname__ (the inner Factory definition)
    assert f1.__module__ == f2.__module__
    assert f1.__qualname__ == f2.__qualname__
    # _same_registered treats them as equivalent
    assert _same_registered(f1, f2)


def test_same_registered_genuine_collision():
    """Two callables with different __module__ or __qualname__ are NOT
    the same — protects against accidental name collisions across
    different modules.
    """
    from agent_evals.core.ports.registry import _same_registered

    def Factory():  # noqa: N802 - intentional cls-like name for the test
        pass

    class Factory_OtherModule:  # noqa: N801 - intentional cls-like name
        pass

    # __qualname__ differs ("Factory" vs "Factory_OtherModule")
    assert not _same_registered(Factory, Factory_OtherModule)
