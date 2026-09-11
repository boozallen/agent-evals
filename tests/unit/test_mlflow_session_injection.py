# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Seam-contract tests for MLflowPlatform.

Pin that the new session= parameter actually flows the substituted
session through both adapter call sites (aevaluate + pull_traces),
and that the lazy default + init-failure wrapping behave as documented.
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

from helpers.fake_mlflow import FakeMlflowSession
from helpers.scorers import passing_scorer

from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform
from agent_evals.core.types import (
    ExampleData,
    ExpectedResult,
)


def test_constructor_does_not_call_real_sdk_when_session_injected():
    """Injecting session= must skip the lazy _RealMlflowSession construction."""
    fake = FakeMlflowSession()
    adapter = MLflowPlatform(session=fake)
    assert adapter._session is fake


def test_aevaluate_uses_injected_session_not_real_wrapper():
    """aevaluate must flow through the injected fake; no real mlflow imports needed."""
    fake = FakeMlflowSession()
    adapter = MLflowPlatform(session=fake)

    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://localhost:5000"}):
        asyncio.run(
            adapter.aevaluate(
                task=lambda x: f"echo: {x}",
                dataset=[
                    ExampleData(
                        input="hi", expected=ExpectedResult(expected="echo: hi")
                    )
                ],
                evaluators=[passing_scorer],
                platform=MlflowConfig(experiment="test"),
            )
        )

    assert len(fake.configurations) == 1
    assert fake.configurations[0]["experiment"] == "test"
    assert len(fake.evaluations) == 1
    assert any(s["run_id"] is not None for s in fake.searches)


def test_aevaluate_drives_async_task_to_completion():
    """An async user task is awaited to a real value, not left as a coroutine (issue #239).

    Server-less regression guard for the ``asyncio.to_thread`` offload: the
    benchmark layer always hands ``aevaluate`` an ``async def`` task. The fake
    session reproduces MLflow's real failure mode — its ``new_event_loop()``
    handling raises ``RuntimeError: Cannot run the event loop while another
    loop is running`` if the adapter were to call ``session.evaluate`` inline
    on the running ``aevaluate`` loop (the pre-#239 bug). Because ``aevaluate``
    now offloads via ``asyncio.to_thread``, ``session.evaluate`` runs on a
    loop-free worker thread and the async task's awaited output flows through
    to the result. This pins the fix without needing a live MLflow server.
    """
    fake = FakeMlflowSession()
    adapter = MLflowPlatform(session=fake)

    async def async_task(x: str) -> str:
        await asyncio.sleep(0)
        return f"async-echo: {x}"

    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://localhost:5000"}):
        result = asyncio.run(
            adapter.aevaluate(
                task=async_task,
                dataset=[
                    ExampleData(
                        input="alpha",
                        expected=ExpectedResult(expected="async-echo: alpha"),
                    )
                ],
                evaluators=[passing_scorer],
                platform=MlflowConfig(experiment="async-test"),
            )
        )

    # The coroutine was awaited end-to-end: its return value (not an unawaited
    # coroutine repr) round-tripped through the fake's trace store to output.
    assert len(result.examples) == 1
    assert "async-echo: alpha" in result.examples[0].output


def test_pull_traces_uses_injected_session_not_real_wrapper():
    """pull_traces must flow through the injected fake."""
    from agent_evals.adapters.platforms.mlflow_client import TraceRecord

    fake = FakeMlflowSession()
    fake.seed_traces(
        experiment_id="fake-exp-test",
        traces=[
            TraceRecord(
                request={"x": "hi"},
                response="hello",
                assessments=[{"expectation": {"value": "hello"}, "name": "expected"}],
            )
        ],
    )

    adapter = MLflowPlatform(session=fake)
    pulled = adapter.pull_traces(
        config={"experiment": "test", "tracking_uri": "https://localhost:5000"}
    )

    assert len(fake.configurations) == 1
    assert any(s["run_id"] is None for s in fake.searches)
    assert len(pulled) == 1
    assert pulled[0].expected is not None
    assert pulled[0].expected.expected == "hello"


def test_get_session_lazy_constructs_real_wrapper_when_no_session_injected():
    """Default no-arg constructor lazy-creates a _RealMlflowSession on first call."""
    adapter = MLflowPlatform()
    assert adapter._session is None

    with patch("agent_evals.adapters.platforms.mlflow._RealMlflowSession") as mock_real:
        mock_real.return_value = "REAL-SESSION-STAND-IN"
        session = adapter._get_session()
        assert session == "REAL-SESSION-STAND-IN"
        assert adapter._session == "REAL-SESSION-STAND-IN"

        session2 = adapter._get_session()
        assert session2 == "REAL-SESSION-STAND-IN"
        assert mock_real.call_count == 1


def test_no_arg_construction_byte_identical_to_today():
    """MLflowPlatform() with no kwargs must work as before."""
    adapter = MLflowPlatform()
    assert adapter._session is None
    assert adapter._session_info is None
    assert adapter._scorer_thresholds == {}
