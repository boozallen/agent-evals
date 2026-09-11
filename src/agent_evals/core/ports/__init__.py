# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Port definitions: Protocol interfaces owned by core, implemented by adapters.

Exports ``AdapterRegistry``, ``Parser``, ``Platform``,
``Scorer``, ``PreconditionApplier``, and ``BenchmarkConfigReader``.

Note: success-checks have no port — per-scenario success criteria are a
closed DTO union (``CheckSpec`` in ``core/checks.py``). Converters have
no port either — each is a single free function with no registry dispatch.
"""

from agent_evals.core.ports.benchmark_config_reader import BenchmarkConfigReader
from agent_evals.core.ports.parser import Parser
from agent_evals.core.ports.platform import Platform
from agent_evals.core.ports.precondition_applier import PreconditionApplier
from agent_evals.core.ports.registry import AdapterRegistry
from agent_evals.core.ports.scorer import Scorer

__all__ = [
    "AdapterRegistry",
    "BenchmarkConfigReader",
    "Parser",
    "Platform",
    "PreconditionApplier",
    "Scorer",
]
