# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-scenario success-check parsers.

Each parser registers against ``success_checker_registry`` at import
time and exposes ``to_spec()`` returning a ``core.checks.CheckSpec``.
"""

from agent_evals.adapters.success_checkers.end_state import StateCheckParser
from agent_evals.adapters.success_checkers.tool_use import ToolUseCheckParser
from agent_evals.core._registries import success_checker_registry

__all__ = [
    "StateCheckParser",
    "ToolUseCheckParser",
    "success_checker_registry",
]
