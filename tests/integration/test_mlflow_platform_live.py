# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Live integration tests for MLflowPlatform against a real tracking server.

Env-gated on MLFLOW_TRACKING_URI; skipped without a server. Exercises
the public API end-to-end via the real mlflow SDK. This is the
SDK-contract verification that catches behavioral drift the
wrapper-port pattern (issue #204) can't paper over — per
docs/architecture.md "Adapter Testability Pattern", the wrapper
protects against SDK *naming* drift; the live test is what catches
*behavioral* drift.

Following Cosmic Python ch.5 ("test pyramid: happy-path e2e per
feature") plus one extra to cover MLflow's dual-shape
``search_traces``:

- ``test_aevaluate_against_real_mlflow`` — happy path: ``genai.evaluate``
  + ``search_traces(return_type="list")``.
- ``test_pull_traces_against_real_mlflow`` — happy path: dual-shape
  ``search_traces(return_type="pandas")`` + ``DataFrame.to_dict("records")``.
- ``test_async_predict_fn_succeeds_against_real_mlflow`` — happy path:
  an ``async def`` user task is offloaded by ``aevaluate`` via
  ``asyncio.to_thread`` so MLflow's own async-predict_fn handling runs
  on a loop-free worker thread (issue #239).
- ``test_aevaluate_then_pull_traces_round_trip`` — write+read round
  trip with a UUID marker proving content survives the cycle.

Score-value coercion, threshold flow, and ``structured_response``
branch are all exercised at the service layer (against
``FakeMlflowSession``); they don't need real-SDK verification.

Mirrors tests/integration/test_braintrust_adapter_live.py and
tests/integration/test_langfuse_adapter_live.py.
"""

from __future__ import annotations

import asyncio
import os
import uuid

import pytest

from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform
from agent_evals.core.types import (
    EvalResult,
    ExampleData,
    ExpectedResult,
    Score,
    TaskResult,
)

pytestmark = pytest.mark.skipif(
    not os.getenv("MLFLOW_TRACKING_URI"),
    reason="Requires MLFLOW_TRACKING_URI environment variable",
)


def _echo_task(x: str) -> str:
    return f"echo: {x}"


def _length_scorer(result, expected: ExpectedResult | None = None) -> Score:
    # MLflow's scorer wrapper passes the bare predict_fn output as `result`
    # (a str here), not a TaskResult — duck-type defensively, mirroring
    # the pattern in tests/integration/test_langfuse_adapter_live.py.
    output_str = result.output if isinstance(result, TaskResult) else str(result)
    expected_str = expected.expected if expected is not None else ""
    same_len = len(output_str) == len(expected_str)
    return Score(
        name="length_match",
        value=1.0 if same_len else 0.0,
        passed=same_len,
        metadata={"_threshold": 1.0},
    )


def _live_config() -> MlflowConfig:
    return MlflowConfig(
        experiment=f"agent-evals-issue-204-{uuid.uuid4().hex[:8]}",
    )


@pytest.mark.asyncio
async def test_aevaluate_against_real_mlflow():
    """End-to-end: real mlflow.genai.evaluate + search_traces(return_type='list').

    Pins that the adapter round-trips Score → Feedback → Score through
    MLflow's trace store, including the threshold-based pass/fail logic
    in _convert_from_platform_result.
    """
    adapter = MLflowPlatform()
    dataset = [
        ExampleData(input="alpha", expected=ExpectedResult(expected="echo: alpha")),
        ExampleData(input="bravo", expected=ExpectedResult(expected="echo: bravo")),
        ExampleData(input="charlie", expected=ExpectedResult(expected="echo: ch")),
    ]

    result = await adapter.aevaluate(
        task=_echo_task,
        dataset=dataset,
        evaluators=[_length_scorer],
        platform=_live_config(),
    )

    assert isinstance(result, EvalResult)
    assert result.platform == "mlflow"
    assert len(result.examples) == 3

    # MLflow keys scores by Feedback.name, set from the user Score.name in
    # _convert_to_platform_scorer.
    score_key = "length_match"
    for ex in result.examples:
        assert score_key in ex.scores, (
            f"Expected score key {score_key!r}, got {list(ex.scores.keys())!r}"
        )

    # Two pass length-match ("echo: alpha" / "echo: bravo" each equal in
    # length to their expected values); one fails ("echo: charlie" 13
    # chars vs "echo: ch" 8 chars).
    passed_count = sum(1 for ex in result.examples if ex.scores[score_key].passed)
    assert passed_count == 2, f"expected 2/3 to pass length match, got {passed_count}/3"


@pytest.mark.asyncio
async def test_pull_traces_against_real_mlflow():
    """End-to-end: aevaluate writes traces; pull_traces reads them back.

    Exercises the search_traces(return_type='pandas') + DataFrame
    .to_dict('records') path, distinct from aevaluate's
    return_type='list' + Trace-object path. The wrapper-port refactor
    has to preserve both shapes; this test pins that.
    """
    adapter = MLflowPlatform()
    platform_cfg = _live_config()  # one experiment shared across both calls
    pull_traces_cfg = {
        "experiment": platform_cfg.experiment,
        "tracking_uri": os.environ["MLFLOW_TRACKING_URI"],
    }
    dataset = [
        ExampleData(input="alpha", expected=ExpectedResult(expected="echo: alpha")),
        ExampleData(input="bravo", expected=ExpectedResult(expected="echo: bravo")),
    ]

    await adapter.aevaluate(
        task=_echo_task,
        dataset=dataset,
        evaluators=[_length_scorer],
        platform=platform_cfg,
    )

    pulled = adapter.pull_traces(config=pull_traces_cfg)

    # Fresh experiment with two rows → exactly two traces. Use >= for
    # tolerance against future mlflow versions that may record extra
    # internal traces alongside the eval rows.
    assert len(pulled) >= 2, f"expected at least 2 pulled traces, got {len(pulled)}"
    for ex in pulled:
        assert isinstance(ex, ExampleData)
        assert isinstance(ex.expected, ExpectedResult)


@pytest.mark.asyncio
async def test_async_predict_fn_succeeds_against_real_mlflow():
    """An async user task is driven to completion by mlflow via aevaluate.

    Issue #239: ``aevaluate`` offloads the blocking ``mlflow.genai.evaluate``
    call to a worker thread (``asyncio.to_thread``). On that thread there is
    no running event loop, so MLflow's own ``_wrap_async_predict_fn`` can call
    ``asyncio.run`` on the coroutine without the "Detected a running event
    loop" failure.

    This is the agent-evals-side fix the previous version of this test
    anticipated: it used to assert the failure as a "known SDK boundary." The
    boundary was our call site, not MLflow.
    """
    adapter = MLflowPlatform()

    async def async_task(x: str) -> str:
        await asyncio.sleep(0.01)
        return f"async-echo: {x}"

    dataset = [
        ExampleData(
            input="alpha",
            expected=ExpectedResult(expected="async-echo: alpha"),
        ),
    ]

    result = await adapter.aevaluate(
        task=async_task,
        dataset=dataset,
        evaluators=[_length_scorer],
        platform=_live_config(),
    )

    assert isinstance(result, EvalResult)
    assert result.platform == "mlflow"
    assert len(result.examples) == 1
    # The async task ran: its output flowed through MLflow's trace store and
    # back. _extract_request_response pulls the response content; the async
    # task returns "async-echo: alpha".
    assert "async-echo: alpha" in result.examples[0].output
    # The scorer pipeline also ran on the async task's output: _length_scorer
    # emits a Score named "length_match" (parity with the sync happy-path test).
    assert "length_match" in result.examples[0].scores


@pytest.mark.asyncio
async def test_aevaluate_then_pull_traces_round_trip():
    """Traces written by aevaluate must be readable via pull_traces with content intact.

    Today's ``test_pull_traces_against_real_mlflow`` reuses the same
    experiment but only asserts trace count + shape. This test pins
    that the actual input/output content survives the write-then-read
    cycle, using a UUID marker to avoid ordering ambiguity.
    """
    adapter = MLflowPlatform()
    platform_cfg = _live_config()
    pull_traces_cfg = {
        "experiment": platform_cfg.experiment,
        "tracking_uri": os.environ["MLFLOW_TRACKING_URI"],
    }
    marker = uuid.uuid4().hex[:8]

    dataset = [
        ExampleData(
            input=f"unique-{marker}",
            expected=ExpectedResult(expected=f"echo: unique-{marker}"),
        ),
    ]

    await adapter.aevaluate(
        task=_echo_task,
        dataset=dataset,
        evaluators=[_length_scorer],
        platform=platform_cfg,
    )

    pulled = adapter.pull_traces(config=pull_traces_cfg)

    matching = [ex for ex in pulled if marker in str(ex.input)]
    first_input = repr(pulled[0].input) if pulled else "None"
    assert len(matching) >= 1, (
        f"Round-trip failed: marker {marker!r} not in any pulled trace's input. "
        f"Pulled {len(pulled)} traces; first input={first_input}."
    )
    pulled_ex = matching[0]
    assert "unique-" in str(pulled_ex.input)
    assert isinstance(pulled_ex.output, TaskResult)
    assert "echo: unique-" in pulled_ex.output.output
