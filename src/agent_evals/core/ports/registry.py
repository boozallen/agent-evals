# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""AdapterRegistry[T] Protocol — generic dispatch contract.

Each adapter family (scorers, platforms, success_checkers, preconditions,
benchmark_config_readers) owns a concrete registry class that implements
this Protocol. The benchmark layer and composition root receive a
registry by injection and resolve names through it; concrete adapter
modules register against their family's module-level instance via
``<family>_registry.register("name")``.

This Protocol captures only the dispatch shape: ``get(key) -> callable``,
``list() -> list[str]``, and membership testing. Registration is the
family's concern (each registry class also exposes a ``register(name)``
decorator method, but ``register`` is NOT part of this Protocol — it's
a write operation, and consumers of the Protocol only read).

Returning the registered callable (rather than instantiating it) keeps
all five families aligned to the same shape: scorers store factory
callables that the caller invokes with kwargs, while
success_checkers/preconditions/config_readers/platforms store classes
the caller invokes (with payload, with no args, etc.). The differing
construction shapes belong at the call site, not in the Protocol.
"""

import builtins
from typing import Protocol, TypeVar, runtime_checkable

T = TypeVar("T", covariant=True)


@runtime_checkable
class AdapterRegistry(Protocol[T]):
    """Read-only dispatch contract for an adapter-family registry.

    Implementations resolve a string key to the registered callable of
    type ``T``. The caller invokes the returned callable with the
    family's specific call shape (factory kwargs, payload dict, etc.).

    The ``key`` parameters on ``get`` and ``__contains__`` are
    positional-only (the trailing ``/``) so concrete registries can use
    family-idiomatic parameter names — e.g. ``name`` for scorers,
    ``scheme`` for benchmark_config_readers — without breaking
    structural conformance.
    """

    def get(self, key: str, /) -> T:
        """Return the registered callable for ``key``.

        Raises:
            ValueError: ``key`` is not registered. Implementations
                should include the available keys in the error
                message for diagnostic aid.
        """
        ...

    def list(self) -> builtins.list[str]:
        """Return all registered keys, sorted."""
        ...

    def __contains__(self, key: str, /) -> bool:
        """Return True if ``key`` is registered."""
        ...


def _same_registered(a: object, b: object) -> bool:
    """Return True if ``a`` and ``b`` are the same callable for registration purposes.

    Identity check (fast path) plus module+qualname check (safety for module
    reload, where re-importing produces a new function/class object with
    the same logical identity). This lets registry decorators run multiple
    times during the same process (e.g., when a test calls
    ``del sys.modules[...]`` and re-imports) without spuriously raising
    "already registered to a different class".
    """
    if a is b:
        return True
    a_mod = getattr(a, "__module__", None)
    b_mod = getattr(b, "__module__", None)
    a_qual = getattr(a, "__qualname__", None) or getattr(a, "__name__", None)
    b_qual = getattr(b, "__qualname__", None) or getattr(b, "__name__", None)
    return (
        a_mod is not None and a_mod == b_mod and a_qual is not None and a_qual == b_qual
    )
