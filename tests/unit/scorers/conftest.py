# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ensure the scorer registry is populated before any test in this directory runs.

The registry singleton lives in agent_evals.core._registries and is empty
until the concrete scorer modules' @register decorators run. Importing
the agent_evals.adapters.scorers package triggers eager import of
agentevals, autoevals, state, and tool_calls — each runs its
registration decorator at module-import time.

Without this fixture, any focused subset that does not transitively
import the family package (e.g. ``uv run pytest tests/unit/scorers/``)
would see an empty registry. The full suite happens to pass without
this conftest because other test files transitively trigger the import
chain — but that's fragile and depends on test execution order.
"""

import agent_evals.adapters.scorers  # noqa: F401
