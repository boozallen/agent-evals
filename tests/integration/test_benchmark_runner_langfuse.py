# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark + langfuse smoke test.

Pins ``run_benchmark_async(..., platform="langfuse")`` end-to-end:
the YAML loader, the ``BaseAgent`` shim, the langfuse adapter, and
the result-merging path all interoperate against a real langfuse
server with no per-platform branching in user code.

The benchmark runner adapts ``BaseAgent.run_case`` to the platform
task-call contract via a shim that reads ``prompt`` directly off the
per-case input dict. That access only resolves if the platform
adapter calls the task with the bare dataset input rather than
re-wrapping it — so this test is also the cross-cutting verification
that langfuse honors the platform task-call contract from the
benchmark layer's perspective, not just from a hand-rolled
``run_eval_async`` call.

Live-gated on ``LANGFUSE_HOST``. Skipped on CI without credentials;
runs against a sandbox/dev langfuse server in the credentialed
nightly job.
"""

import os
import uuid
from pathlib import Path

import pytest
from integration.test_benchmark_fixtures.mock_home_agent import MockHomeAgent

FIXTURES = Path(__file__).parent / "test_benchmark_fixtures"
BENCHMARK_YAML = FIXTURES / "home_lights_langfuse.yaml"

pytestmark = pytest.mark.skipif(
    not os.environ.get("LANGFUSE_HOST"),
    reason="LANGFUSE_HOST not set; live langfuse benchmark test skipped",
)


@pytest.mark.asyncio
async def test_benchmark_runs_on_langfuse_platform(
    default_benchmark_registries,
) -> None:
    """run_benchmark_async completes against platform=langfuse.

    The benchmark runner reads ``prompt`` directly from the per-case
    input dict. The langfuse adapter must therefore deliver that
    exact dict to the shim — not a wrapper enclosing it.
    """
    from agent_evals.benchmark.loader import load_benchmark
    from agent_evals.benchmark.runner import BenchmarkRunner

    # Per-run unique experiment name keeps repeated runs against a
    # shared dev langfuse server from colliding. The YAML carries a
    # static placeholder; we override it after loading so the fixture
    # stays declarative.
    config = load_benchmark(BENCHMARK_YAML, **default_benchmark_registries)
    # Per-run unique experiment name: override via model_copy so the
    # BenchmarkConfig holds the updated platform instance.
    config.platform = config.platform.model_copy(
        update={"experiment": f"home-lights-lf-{uuid.uuid4().hex[:6]}"}
    )

    result = await BenchmarkRunner(
        config=config, agent=MockHomeAgent, config_path=str(BENCHMARK_YAML)
    ).run_async()

    assert result.benchmark == "home-lights-langfuse-v1"
    assert set(result.eval_results) == {"lights", "doors"}

    # MockHomeAgent emits a deterministic state for kitchen + front-door
    # prompts; both StateMatch checks pass. Direct index — `.get` with a
    # default would mask a missing-scorer regression.
    assert result.aggregate_pass_rates["StateMatch"] == 1.0
