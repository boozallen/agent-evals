# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-scenario precondition appliers.

Concrete applier classes (``StatePreconditionApplier``) register against
``precondition_registry`` at import time. The benchmark loader
receives the registry by injection (with the default supplied by the
composition root) and dispatches names through it.
"""

from agent_evals.adapters.preconditions.state import StatePreconditionApplier
from agent_evals.core._registries import precondition_registry

__all__ = [
    "StatePreconditionApplier",
    "precondition_registry",
]
