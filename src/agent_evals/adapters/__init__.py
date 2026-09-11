# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Adapters for external platforms and scorer libraries.

This module provides adapters that connect the core evaluation system
to external platforms (Braintrust, MLflow, Local) and scorer libraries
(autoevals, agentevals).

Subpackages:
- platforms/: Platform adapters for logging evaluation results
- scorers/: Scorer adapters wrapping external evaluation libraries

Public API:
- platform_registry: AdapterRegistry[type[Platform]]
  instance. Concrete adapters register against it; the composition
  root injects this default into the public API.
"""

# Re-export platform adapter public API
from agent_evals.core._registries import platform_registry

__all__ = [
    "platform_registry",
]
