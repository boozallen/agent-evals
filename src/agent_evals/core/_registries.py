# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Generic adapter-family registry dispatch infrastructure.

Provides ``_AdapterRegistryBase[T]`` plus per-family subclasses, all
implementing the ``AdapterRegistry[T]`` Protocol (see
``core.ports.registry``). The base class encodes the shared
register / get / list / contains shape; subclasses absorb family-
specific asymmetries via class-attribute hooks (``_family_label``,
``_available_label``, ``_required_methods``) and an optional
``_trigger_lazy`` override.

Each subclass exposes a module-level singleton instance that serves
as the canonical default registry for its family. Concrete adapter
modules register against these instances at import time via
``@<instance>.register("name")`` decorators.

Hex-layers compliance: this module imports only from ``core.ports``.
The one subclass with a data-driven lazy trigger (``_PlatformRegistry``)
carries a ``_lazy_package`` string that the platform family's
``__init__.py`` writes at family-init time. The base class accesses it
only through ``_trigger_lazy``, never by importing
``agent_evals.adapters.*``. The higher layer pushes its identity
*down* as data; the lower layer never imports upward.
"""

import builtins
from collections.abc import Callable
from typing import Any, Generic, TypeVar

from agent_evals.core.ports import (
    BenchmarkConfigReader,
    Platform,
    PreconditionApplier,
)
from agent_evals.core.ports.registry import _same_registered

T = TypeVar("T")


class _AdapterRegistryBase(Generic[T]):  # noqa: UP046  # Generic[T] kept deliberately; adopting PEP-695 type parameters is its own change
    """Generic dispatch infrastructure — register, get, list, contains.

    Subclasses set ``_family_label`` (error-message noun for the
    "Unknown ..." half), ``_available_label`` (plural-noun for the
    "Available ..." half — defaults to ``_family_label`` when unset),
    and ``_required_methods`` (Protocol method-check at registration
    time; ``()`` skips the check, leaving registration duck-typed).

    Subclasses override ``_trigger_lazy(key)`` if they need to import
    additional modules on cache miss (platforms: optional-extras
    PEP 562 lookup).
    """

    _family_label: str = "adapter"
    # Default sentinel: subclasses that don't override _available_label
    # fall back to _family_label in the "Available ..." half. Each
    # subclass below sets this explicitly to byte-match the legacy
    # registries' diagnostics (e.g. "Available adapters" vs.
    # "platform adapter") which are user-facing contracts.
    _available_label: str = ""
    _required_methods: tuple[str, ...] = ()

    def __init__(self) -> None:
        self._registry: dict[str, T] = {}

    def _format_obj(self, obj: T) -> str:
        """Return a 'module.qualname' label for ``obj``.

        Handles both classes (which expose ``__qualname__``) and plain
        Callables (which carry ``__qualname__`` at runtime). Same fallback
        chain as the legacy ``_describe_callable`` helper this method
        replaces. ty sees only the wrapped getattr calls, not direct
        attribute access, so the merge is safe.
        """
        module = getattr(obj, "__module__", "?")
        qualname = getattr(
            obj,
            "__qualname__",
            getattr(obj, "__name__", repr(obj)),
        )
        return f"{module}.{qualname}"

    def _trigger_lazy(self, key: str) -> None:
        """Hook for subclasses that need to import modules on cache miss.

        Default is a no-op. Platforms override to fire PEP 562 lazy-load
        of optional-extras adapters.
        """
        return

    def register(self, name: str) -> Callable[[T], T]:
        """Decorator: register ``obj`` under ``name``.

        If ``_required_methods`` is non-empty, ``obj`` must expose each
        method as a callable attribute. Re-registration with the same
        object is a no-op (safe for decorator-on-import re-runs);
        re-registration with a different object raises ``ValueError``.
        """

        def decorator(obj: T) -> T:
            if self._required_methods:
                missing = [
                    method
                    for method in self._required_methods
                    if not callable(getattr(obj, method, None))
                ]
                if missing:
                    cls_name = getattr(obj, "__name__", repr(obj))
                    raise TypeError(
                        f"@register({name!r}): {cls_name} does not satisfy "
                        f"the required Protocol — missing method(s): "
                        f"{', '.join(missing)}."
                    )
            existing = self._registry.get(name)
            if existing is not None and not _same_registered(existing, obj):
                raise ValueError(
                    f"@register({name!r}): name already registered to "
                    f"{self._format_obj(existing)}; cannot reuse it for "
                    f"{self._format_obj(obj)}."
                )
            self._registry[name] = obj
            return obj

        return decorator

    def get(self, key: str) -> T:
        """Return the registered callable for ``key``.

        Triggers ``_trigger_lazy(key)`` on cache miss; subsequent
        registries' import-time decorators may populate the registry
        before the second membership check.
        """
        if key not in self._registry:
            self._trigger_lazy(key)
        if key not in self._registry:
            available = ", ".join(sorted(self._registry.keys()))
            available_label = self._available_label or self._family_label
            raise ValueError(
                f"Unknown {self._family_label}: {key!r}. "
                f"Available {available_label}: {available or 'none registered'}"
            )
        return self._registry[key]

    def list(self) -> builtins.list[str]:
        """Return registered keys, sorted."""
        return sorted(self._registry.keys())

    def __contains__(self, key: object) -> bool:
        return isinstance(key, str) and key in self._registry


class _ScorerRegistry(_AdapterRegistryBase[Callable[..., Callable[..., Any]]]):
    _family_label = "scorer"
    _available_label = "scorers"


class _PlatformRegistry(_AdapterRegistryBase[type[Platform]]):
    """Platform-adapter registry with package-driven lazy-load support.

    Carries an optional ``_lazy_package`` attribute (a dotted module
    path string). When set, the lazy trigger imports that package and
    performs an attribute lookup for the missing key, allowing the
    package's own ``__getattr__`` to register the requested adapter
    on demand. When unset, the trigger is a no-op.
    """

    _family_label = "platform adapter"
    # Asymmetric vs. _family_label: the user-facing plural drops the
    # "platform" qualifier. Part of the public error-message contract;
    # pinned by tests/unit/core/test_registries.py.
    _available_label = "adapters"

    def __init__(self) -> None:
        super().__init__()
        self._lazy_package: str | None = None

    def _trigger_lazy(self, key: str) -> None:
        if self._lazy_package is None:
            return
        try:
            pkg = __import__(self._lazy_package, fromlist=[key])
            getattr(pkg, key)
        except AttributeError:
            pass
        # ImportError intentionally NOT caught: a package may raise
        # ImportError with an actionable install hint when an optional
        # dependency is missing. Blanket-catching would swallow that
        # hint and degrade the user experience to a generic
        # "Unknown ..." error.


class _SuccessCheckerRegistry(_AdapterRegistryBase[type[Any]]):
    _family_label = "success checker"
    _available_label = "success checkers"
    _required_methods = ("to_spec",)


class _PreconditionRegistry(_AdapterRegistryBase[type[PreconditionApplier]]):
    _family_label = "precondition"
    _available_label = "preconditions"
    _required_methods = ("apply",)


class _BenchmarkConfigReaderRegistry(_AdapterRegistryBase[type[BenchmarkConfigReader]]):
    _family_label = "URI scheme"
    # Asymmetric vs. _family_label: the user-facing plural drops the
    # "URI" qualifier. Part of the public error-message contract;
    # pinned by tests/unit/core/test_registries.py.
    _available_label = "schemes"
    _required_methods = ("read",)


# Module-level instances — canonical defaults referenced by the composition
# root, the benchmark layer, and the package-root re-export surface.
scorer_registry: _ScorerRegistry = _ScorerRegistry()
platform_registry: _PlatformRegistry = _PlatformRegistry()
success_checker_registry: _SuccessCheckerRegistry = _SuccessCheckerRegistry()
precondition_registry: _PreconditionRegistry = _PreconditionRegistry()
config_reader_registry: _BenchmarkConfigReaderRegistry = (
    _BenchmarkConfigReaderRegistry()
)
