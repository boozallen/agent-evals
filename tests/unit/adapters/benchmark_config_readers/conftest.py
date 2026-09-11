# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Ensure the config-reader registry is populated before any test in this directory runs.

Importing the agent_evals.adapters.benchmark_config_readers package
triggers eager registration of LocalFileConfigReader via its @register
decorator. Without this fixture, focused runs
(`uv run pytest tests/unit/adapters/benchmark_config_readers/`)
would see an empty registry.
"""

import agent_evals.adapters.benchmark_config_readers  # noqa: F401
