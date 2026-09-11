# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark + mlflow smoke test.

Pins the benchmark path (``BenchmarkRunner.run_async``, the engine behind
``run_benchmark_async``) with ``platform=MlflowConfig(...)`` and an async
``BaseAgent`` end-to-end: the YAML loader, the ``BaseAgent`` shim, the mlflow
adapter, and the result-merging path all interoperate against a real MLflow
server with no per-platform branching in user code.

This is the regression guard for issue #239: ``BenchmarkRunner`` hands the
adapter an ``async def`` task (wrapping the always-async ``BaseAgent.run_case``).
Before the ``asyncio.to_thread`` offload in ``MLflowPlatform.aevaluate``, that
async task raised ``MlflowException: Detected a running event loop``, so the
"subclass BaseAgent + benchmark on mlflow" path was broken while it worked on
local/braintrust/langfuse. This test pins that it now works.

Live-gated on ``MLFLOW_TRACKING_URI``. Skipped on CI without a server; runs
against a sandbox/dev MLflow server in the credentialed job.

Mirrors tests/integration/test_benchmark_runner_langfuse.py.
"""

import os
import uuid
from pathlib import Path

import pytest
from integration.test_benchmark_fixtures.mock_home_agent import MockHomeAgent

FIXTURES = Path(__file__).parent / "test_benchmark_fixtures"
BENCHMARK_YAML = FIXTURES / "home_lights_mlflow.yaml"

pytestmark = pytest.mark.skipif(
    not os.environ.get("MLFLOW_TRACKING_URI"),
    reason="MLFLOW_TRACKING_URI not set; live mlflow benchmark test skipped",
)


@pytest.mark.asyncio
async def test_benchmark_runs_on_mlflow_platform(
    default_benchmark_registries,
) -> None:
    """run_benchmark_async completes against platform=mlflow with an async agent.

    The benchmark runner wraps ``BaseAgent.run_case`` (async) in an async
    task. The mlflow adapter must offload its blocking ``genai.evaluate`` so
    that async task runs to completion instead of raising on the nested
    ``asyncio.run`` (issue #239).
    """
    from agent_evals.benchmark.loader import load_benchmark
    from agent_evals.benchmark.runner import BenchmarkRunner

    # Per-run unique experiment name keeps repeated runs against a shared dev
    # MLflow server from colliding. The YAML carries a static placeholder; we
    # override it after loading so the fixture stays declarative.
    config = load_benchmark(BENCHMARK_YAML, **default_benchmark_registries)
    config.platform = config.platform.model_copy(
        update={"experiment": f"home-lights-mlflow-{uuid.uuid4().hex[:6]}"}
    )

    result = await BenchmarkRunner(
        config=config, agent=MockHomeAgent, config_path=str(BENCHMARK_YAML)
    ).run_async()

    assert result.benchmark == "home-lights-mlflow-v1"
    assert set(result.eval_results) == {"lights", "doors"}

    # MockHomeAgent emits a deterministic state for kitchen + front-door
    # prompts; both StateMatch checks pass. Direct index — `.get` with a
    # default would mask a missing-scorer regression.
    assert result.aggregate_pass_rates["StateMatch"] == 1.0
