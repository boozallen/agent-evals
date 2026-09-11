# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cross-platform trajectory-persistence contract.

Every platform adapter MUST surface trajectory data carried in
``TaskResult.context['outputs']`` on the persisted ``EvalExample.trajectory``.
Adapters that read from platform traces (mlflow) reach the same
observable contract via internal trace walking — the contract is on
the result, not the implementation.

Live-server adapters skip when their gate env vars are missing:
``MLFLOW_TRACKING_URI`` for mlflow, ``BRAINTRUST_API_KEY`` for
braintrust, ``LANGFUSE_HOST`` for langfuse. The merge gate covers
``local`` unconditionally; a credentialed nightly job covers the
other three.

This file is the LSP guard for issue #212. If a future adapter author
forgets to populate ``trajectory`` on their EvalExample, the relevant
gated test fires before merge.
"""

import os
import uuid

import pytest

from agent_evals import (
    ExampleData,
    ExpectedResult,
    Score,
    TaskResult,
    run_eval_async,
)

_TRAJECTORY = [
    {"role": "user", "content": "alpha"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "noop", "arguments": "{}"},
            }
        ],
    },
    {"role": "assistant", "content": "answer: ALPHA"},
]


def _trajectory_task(query: str) -> TaskResult:  # noqa: ARG001
    return TaskResult.from_messages(_TRAJECTORY)


def _passing_scorer(result, expected: ExpectedResult | None = None, **_kwargs) -> Score:  # noqa: ARG001
    return Score(name="trajectory_pin", value=1.0, passed=True)


def _dataset() -> list[ExampleData]:
    return [ExampleData(input="alpha", expected=ExpectedResult(expected="alpha"))]


_MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
_MLFLOW_GATED = pytest.mark.skipif(
    not _MLFLOW_TRACKING_URI,
    reason="MLFLOW_TRACKING_URI not set; mlflow trajectory contract test skipped",
)
_BRAINTRUST_GATED = pytest.mark.skipif(
    not os.environ.get("BRAINTRUST_API_KEY"),
    reason="BRAINTRUST_API_KEY not set; braintrust trajectory contract test skipped",
)
_LANGFUSE_GATED = pytest.mark.skipif(
    not os.environ.get("LANGFUSE_HOST"),
    reason="LANGFUSE_HOST not set; langfuse trajectory contract test skipped",
)


def _assert_trajectory_persisted(result) -> None:
    assert result.summary["successful_examples"] == 1
    ex = result.examples[0]
    assert ex.error is None
    assert ex.trajectory == _TRAJECTORY, (
        f"adapter dropped trajectory: got {ex.trajectory}"
    )
    assert ex.tool_calls == [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "noop", "arguments": "{}"},
        }
    ]


@pytest.mark.asyncio
async def test_local_persists_trajectory() -> None:
    from agent_evals.adapters.platforms.local import LocalConfig

    result = await run_eval_async(
        task=_trajectory_task,
        dataset=_dataset(),
        scorers=[_passing_scorer],
        platform=LocalConfig(experiment="trajectory-contract-local"),
    )
    _assert_trajectory_persisted(result)


@_MLFLOW_GATED
@pytest.mark.asyncio
async def test_mlflow_persists_trajectory() -> None:
    from agent_evals.adapters.platforms.mlflow import MlflowConfig

    result = await run_eval_async(
        task=_trajectory_task,
        dataset=_dataset(),
        scorers=[_passing_scorer],
        platform=MlflowConfig(experiment="trajectory-contract"),
    )
    _assert_trajectory_persisted(result)


@pytest.mark.requires_braintrust
@_BRAINTRUST_GATED
@pytest.mark.asyncio
async def test_braintrust_persists_trajectory() -> None:
    from agent_evals.adapters.platforms.braintrust import BraintrustConfig

    result = await run_eval_async(
        task=_trajectory_task,
        dataset=_dataset(),
        scorers=[_passing_scorer],
        platform=BraintrustConfig(
            project="trajectory-contract", experiment="braintrust"
        ),
    )
    _assert_trajectory_persisted(result)


@_LANGFUSE_GATED
@pytest.mark.asyncio
async def test_langfuse_persists_trajectory() -> None:
    from agent_evals.adapters.platforms.langfuse import LangfuseConfig

    result = await run_eval_async(
        task=_trajectory_task,
        dataset=_dataset(),
        scorers=[_passing_scorer],
        platform=LangfuseConfig(
            experiment=f"trajectory-contract-{uuid.uuid4().hex[:6]}"
        ),
    )
    _assert_trajectory_persisted(result)
