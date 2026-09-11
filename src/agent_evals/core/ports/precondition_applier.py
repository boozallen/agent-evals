# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Precondition-applier port.

A PreconditionApplier binds a YAML precondition-block payload to the
side effect that seeds scenario state *before* the agent runs. The
Pydantic class IS the applier — Pydantic validates the payload at
construction; ``apply(target)`` performs the side effect.

Unlike the success side, preconditions use a single Protocol (no
spec/mechanism split). There is no composition behind the precondition
side: each precondition shape (``state``, future ``cookies``,
``db_seed``, ...) has a 1:1 relationship with its applier. See
``docs/architecture.md`` for the rationale.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class PreconditionApplier(Protocol):
    """Protocol every concrete precondition applier must satisfy.

    Concrete preconditions are typically Pydantic models so the YAML
    payload validates at config-load time. The Protocol is structural;
    the ``precondition_registry.register`` decorator
    additionally enforces the method at registration time so
    registry-path appliers fail fast on a missing method.
    """

    def apply(self, target: object) -> None:
        """Seed scenario state into ``target``.

        ``target`` is supplied by the agent at run time. Its concrete
        shape (a state dict, a cookies map, ...) is the agent's choice;
        each concrete applier narrows ``target`` to the type it expects
        and documents that contract. ``object`` here means "opaque to
        the framework" — implementers are expected to narrow.

        The method mutates ``target`` in place and returns ``None``.
        Errors raised by ``apply()`` surface as per-scenario evaluation
        errors via the existing ``EvalResult`` plumbing.
        """
        ...
