# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Typed contract for the per-case payload the runner shim hands to ``run_case``.

Single source of truth for what flows through ``ExampleData.input`` between
the loader (producer) and the ``BenchmarkRunner._agent_task`` shim
(consumer). Adding a new per-case kwarg means adding a field here;
loader and shim both reference the same type, so the contract change
shows up in code review at every site that participates.

``TypedDict`` (not Pydantic) is intentional:

- ``ExampleData.input`` is typed ``Any`` and forwarded by platform
  adapters as-is; no serialization round-trip happens, so Pydantic's
  validation buys nothing here.
- ``preconditions`` carries live ``PreconditionApplier`` instances
  whose ``apply`` method is load-bearing. ``model_dump()`` would
  collapse them to dicts; we want object identity preserved.
- TypedDict gives static type checking + zero runtime cost + matches
  the existing dict shape, so no migration ripple through the platform
  adapters that already accept ``input: Any``.
"""

from __future__ import annotations

from typing import TypedDict

from agent_evals.core.ports import PreconditionApplier


class AgentCaseInput(TypedDict):
    """One per-case payload the loader produces and the shim unpacks.

    Every field becomes a kwarg on ``BaseAgent.run_case`` (with ``prompt``
    as the lone positional). Add new per-case kwargs as fields here; the
    runner shim's unpacking enumerates them explicitly.
    """

    prompt: str
    preconditions: list[PreconditionApplier]
