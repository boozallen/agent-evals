# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for multi-file benchmark composition (#177).

Runs the same agent against (a) the legacy single-file home_lights.yaml
and (b) an equivalent parent + per-capability multi-file decomposition,
and asserts the resulting BenchmarkResult is observably identical on
every dimension that matters for hill-climbing. Catches accidental
side effects of the resolution pass (ordering changes, dropped
capabilities, missing scorers, ...) at the run-time level rather than
at the validate-time level the unit tests cover.
"""

from pathlib import Path

import pytest
from integration.test_benchmark_fixtures.mock_home_agent import MockHomeAgent

FIXTURES = Path(__file__).parent / "test_benchmark_fixtures"
SINGLE_FILE_YAML = FIXTURES / "home_lights.yaml"
MULTI_FILE_YAML = FIXTURES / "multi_file" / "home_lights.yaml"


@pytest.mark.asyncio
async def test_multi_file_benchmark_runs_end_to_end():
    """Multi-file YAML produces a working BenchmarkResult."""
    from agent_evals._composition import run_benchmark_async

    result = await run_benchmark_async(MULTI_FILE_YAML, agent=MockHomeAgent)

    assert result.benchmark == "home-lights-v1"
    assert set(result.eval_results) == {"lights", "doors"}
    assert result.aggregate_pass_rates["StateMatch"] == 1.0
    assert result.aggregate_pass_rates["ToolCallExactMatch"] == 1.0


@pytest.mark.asyncio
async def test_multi_file_matches_single_file():
    """Single-file and multi-file YAMLs produce equivalent results.

    The multi-file YAML (parent + capabilities/lights.yaml +
    capabilities/doors.yaml) should produce the same BenchmarkResult
    as the equivalent single-file YAML. Compares the dimensions that
    matter for hill-climbing: capability ordering, per-capability
    scorers, pass rates, and example metadata.
    """
    from agent_evals._composition import run_benchmark_async

    single = await run_benchmark_async(SINGLE_FILE_YAML, agent=MockHomeAgent)
    multi = await run_benchmark_async(MULTI_FILE_YAML, agent=MockHomeAgent)

    assert single.benchmark == multi.benchmark
    assert list(single.eval_results) == list(multi.eval_results), (
        "capability ordering must match"
    )
    assert single.aggregate_pass_rates == multi.aggregate_pass_rates
    assert single.aggregate_scores == multi.aggregate_scores

    for cap_name in single.eval_results:
        s_er = single.eval_results[cap_name]
        m_er = multi.eval_results[cap_name]
        assert set(s_er.scores) == set(m_er.scores), (
            f"capability {cap_name!r} scorer set mismatch"
        )
        assert s_er.pass_rates == m_er.pass_rates, (
            f"capability {cap_name!r} pass rates mismatch"
        )
        # Per-example: same scenario ids in the same order, same metadata
        # (apart from the experiment id which is run-specific).
        s_ids = [ex.metadata["scenario_id"] for ex in s_er.examples]
        m_ids = [ex.metadata["scenario_id"] for ex in m_er.examples]
        assert s_ids == m_ids, f"capability {cap_name!r} example ordering mismatch"


@pytest.mark.asyncio
async def test_single_file_benchmark_unchanged():
    """Regression: existing single-file fixtures run identically to before #177.

    The resolution pass added in commit 5 must not affect benchmarks
    that don't use URI references. Asserts the headline metrics from
    the existing test_benchmark_runs_both_scorer_families test still
    hold.
    """
    from agent_evals._composition import run_benchmark_async

    result = await run_benchmark_async(SINGLE_FILE_YAML, agent=MockHomeAgent)

    assert result.benchmark == "home-lights-v1"
    assert set(result.eval_results) == {"lights", "doors"}
    assert result.aggregate_pass_rates["StateMatch"] == 1.0
    assert result.aggregate_pass_rates["ToolCallExactMatch"] == 1.0
