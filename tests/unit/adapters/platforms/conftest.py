# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ensure the platform registry is populated before any test in this directory runs.

Importing the agent_evals.adapters.platforms package triggers eager
registration of the local adapter and writes the _lazy_package hint
used by optional-extras platforms (mlflow, braintrust, langfuse).
Without this fixture, focused runs (`uv run pytest tests/unit/adapters/platforms/`)
would see an empty registry.
"""

import agent_evals.adapters.platforms  # noqa: F401
