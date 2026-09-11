# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration test for run_benchmark_async.

Exercises:
- YAML loading (no agent field in YAML)
- Agent subclass passed as a keyword argument from Python
- Per-scenario `success:` block schema
- Both `state` and `tool_use` success checkers flowing into scorers
- Multi-checker scenario (doors has both state and tool_use)
- BenchmarkResult aggregation
"""

from pathlib import Path

import pytest
from integration.test_benchmark_fixtures.mock_home_agent import MockHomeAgent

FIXTURES = Path(__file__).parent / "test_benchmark_fixtures"
BENCHMARK_YAML = FIXTURES / "home_lights.yaml"


@pytest.mark.asyncio
async def test_benchmark_runs_both_scorer_families():
    from agent_evals._composition import run_benchmark_async

    result = await run_benchmark_async(BENCHMARK_YAML, agent=MockHomeAgent)

    assert result.benchmark == "home-lights-v1"
    assert set(result.eval_results) == {"lights", "doors"}

    # `tool_use` success checker binds `ToolCallExactMatch`;
    # `state` binds `StateMatch`. Both scorers run on the doors
    # scenarios that declare both checkers.
    assert result.aggregate_pass_rates["StateMatch"] == 1.0
    assert result.aggregate_pass_rates["ToolCallExactMatch"] == 1.0


@pytest.mark.asyncio
async def test_benchmark_metadata_carries_dimensions():
    from agent_evals._composition import run_benchmark_async

    result = await run_benchmark_async(BENCHMARK_YAML, agent=MockHomeAgent)

    lights_examples = result.eval_results["lights"].examples
    assert lights_examples[0].metadata["capability"] == "lights"
    assert lights_examples[0].metadata["phrasing"] == "imperative"
    assert lights_examples[0].metadata["depth"] == "literal"
    assert lights_examples[0].metadata["scenario_id"] == 1
    assert lights_examples[0].metadata["run_index"] == 0
