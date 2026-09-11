# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Cross-platform task-call contract.

A single task body must work on every platform adapter. The user
task is invoked as ``task(input_value)`` regardless of which adapter
is selected; SDK-specific wrapper shapes (e.g. ``task(item=dict)``)
are translated inside the adapter and never reach the user's task.

This file is the contract test: it is the failure mode triggered if
a future adapter author chooses a different calling shape. The
adapter Protocol does not encode the call shape structurally — it
must be pinned behaviorally here.

Live-server adapters skip when their gate env vars are missing:
``mlflow`` requires ``MLFLOW_TRACKING_URI``, ``braintrust`` requires
``BRAINTRUST_API_KEY``, ``langfuse`` requires ``LANGFUSE_HOST``. The
merge gate covers ``local`` unconditionally; a credentialed nightly
job covers the other three. The contract is enforced for every
adapter that runs at all — skipping does not weaken the guard.
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
from agent_evals.adapters.platforms.local import LocalConfig

_MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")


def _portable_task(query: str) -> TaskResult:
    """The contract: a single task body, identical across all platforms.

    Receives the bare ``ExampleData.input`` value. If any adapter passes
    a wrapper dict instead, accessing ``.upper()`` on a dict raises and
    surfaces as a captured task error — which the test catches as a
    ``successful_examples`` regression.
    """
    return TaskResult(output=f"answer: {query.upper()}")


def _contains_scorer(result, expected: ExpectedResult | None = None) -> Score:
    out = result.output if isinstance(result, TaskResult) else str(result)
    exp = expected.expected if expected else ""
    passed = bool(exp) and exp.upper() in out
    return Score(name="contains", value=1.0 if passed else 0.0, passed=passed)


def _dataset() -> list[ExampleData]:
    return [
        ExampleData(input="alpha", expected=ExpectedResult(expected="alpha")),
    ]


_MLFLOW_GATED = pytest.mark.skipif(
    not _MLFLOW_TRACKING_URI,
    reason="MLFLOW_TRACKING_URI not set; mlflow contract test skipped",
)

_BRAINTRUST_GATED = pytest.mark.skipif(
    not os.environ.get("BRAINTRUST_API_KEY"),
    reason="BRAINTRUST_API_KEY not set; braintrust contract test skipped",
)

_LANGFUSE_GATED = pytest.mark.skipif(
    not os.environ.get("LANGFUSE_HOST"),
    reason="LANGFUSE_HOST not set; langfuse contract test skipped",
)


def _assert_task_received_bare_input(result) -> None:
    """The strong contract pin.

    ``_portable_task`` calls ``query.upper()`` to produce ``"ALPHA"``.
    On a wrapper dict that call raises ``AttributeError`` and the
    captured error replaces the output. A successful run with
    ``"ALPHA"`` in the recorded output is positive evidence that the
    user-task body executed against a bare string — not just that the
    adapter completed without raising.
    """
    assert result.summary["successful_examples"] == 1
    assert result.examples[0].error is None
    assert "ALPHA" in result.examples[0].output


@pytest.mark.asyncio
async def test_local_passes_bare_input_to_task() -> None:
    """The ``local`` adapter calls ``task(input_value)``.

    The local adapter stores the bare ``ExampleData.input`` directly
    on ``EvalExample.input``, so the round-trip equality check
    ``examples[0].input == "alpha"`` doubles as a structural pin
    on top of the cross-adapter output check.
    """
    result = await run_eval_async(
        task=_portable_task,
        dataset=_dataset(),
        scorers=[_contains_scorer],
        platform=LocalConfig(experiment="task-shape-local"),
    )
    _assert_task_received_bare_input(result)
    assert result.examples[0].input == "alpha"


@_MLFLOW_GATED
@pytest.mark.asyncio
async def test_mlflow_passes_bare_input_to_task() -> None:
    """The ``mlflow`` adapter calls ``task(input_value)``.

    MLflow's SDK requires the task to be invoked with a parameter-name
    keyed dict; ``_convert_to_platform_dataset`` builds that dict
    using the user task's parameter name. That dict shape is what
    mlflow stores in the trace's ``request`` field and surfaces back
    in ``EvalExample.input`` — a per-platform storage detail. So the
    contract proof comes from the output content, not the stored input.
    """
    from agent_evals.adapters.platforms.mlflow import MlflowConfig

    result = await run_eval_async(
        task=_portable_task,
        dataset=_dataset(),
        scorers=[_contains_scorer],
        platform=MlflowConfig(experiment="task-shape-mlflow"),
    )
    _assert_task_received_bare_input(result)


@pytest.mark.requires_braintrust
@_BRAINTRUST_GATED
@pytest.mark.asyncio
async def test_braintrust_passes_bare_input_to_task() -> None:
    """The ``braintrust`` adapter calls ``task(input_value)``.

    Braintrust internally splits each case into a task span and a
    scorer span; the task span's input is the bare value, the scorer
    span's input is a wrapper dict. ``EvalExample`` aggregates the
    task-side projection, so the cross-adapter output check is the
    proof here.
    """
    from agent_evals.adapters.platforms.braintrust import BraintrustConfig

    result = await run_eval_async(
        task=_portable_task,
        dataset=_dataset(),
        scorers=[_contains_scorer],
        platform=BraintrustConfig(
            project="contract", experiment="task-shape-braintrust"
        ),
    )
    _assert_task_received_bare_input(result)


@_LANGFUSE_GATED
@pytest.mark.asyncio
async def test_langfuse_passes_bare_input_to_task() -> None:
    """The ``langfuse`` adapter calls ``task(input_value)``.

    The langfuse SDK natively delivers a wrapper dict
    (``{"input": ..., "expected_output": ..., "metadata": ...}``);
    the adapter must unwrap before invoking the user task. If a
    wrapper dict reaches ``_portable_task``, ``query.upper()`` raises
    ``AttributeError`` (dicts have no ``.upper``), which the wrapper
    captures as a task error — the output assertion below catches it.
    """
    from agent_evals.adapters.platforms.langfuse import LangfuseConfig

    result = await run_eval_async(
        task=_portable_task,
        dataset=_dataset(),
        scorers=[_contains_scorer],
        platform=LangfuseConfig(experiment=f"contract-{uuid.uuid4().hex[:6]}"),
    )
    _assert_task_received_bare_input(result)
