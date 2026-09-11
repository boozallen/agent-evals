# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ensure the success-checker registry is populated before any test in this directory runs.

Importing the agent_evals.adapters.success_checkers package eagerly
imports end_state and tool_use, whose @register decorators populate
the registry. Without this fixture, focused runs
(`uv run pytest tests/unit/adapters/success_checkers/`) would see an
empty registry only if the test module under load happens not to import
the parser submodules itself.
"""

import agent_evals.adapters.success_checkers  # noqa: F401
