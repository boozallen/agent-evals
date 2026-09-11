# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ensure the precondition registry is populated before any test in this directory runs.

Importing the agent_evals.adapters.preconditions package triggers eager
registration of the StatePreconditionApplier via its @register decorator.
Without this fixture, focused runs (`uv run pytest tests/unit/adapters/preconditions/`)
would see an empty registry.
"""

import agent_evals.adapters.preconditions  # noqa: F401
