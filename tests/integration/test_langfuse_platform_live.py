# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Live LangFuse adapter contracts.

Drives the adapter against a real langfuse server to prove three
properties no mocked test can verify:

- ``EvalExample.error`` is populated when the user task raises
- ``EvalExample.duration`` reflects actual wall time
- Items whose task raises remain in ``EvalResult.examples``

Gated on ``LANGFUSE_HOST``; runs are written to a sandbox/dev server and
not cleaned up.
"""

import asyncio
import os
import time
import uuid

import pytest

from agent_evals import (
    ExampleData,
    ExpectedResult,
    Score,
    TaskResult,
    run_eval_async,
)
from agent_evals.adapters.platforms.langfuse import LangfuseConfig

pytestmark = pytest.mark.skipif(
    not os.environ.get("LANGFUSE_HOST"),
    reason="LANGFUSE_HOST not set; live langfuse tests skipped",
)


def _contains_scorer(result, expected=None):
    out = result.output if isinstance(result, TaskResult) else str(result)
    exp = expected.expected if expected else ""
    passed = bool(exp) and exp in out
    return Score(name="contains", value=1.0 if passed else 0.0, passed=passed)


def _two_item_dataset() -> list[ExampleData]:
    return [
        ExampleData(input="alpha", expected=ExpectedResult(expected="alpha")),
        ExampleData(input="beta", expected=ExpectedResult(expected="beta")),
    ]


@pytest.mark.asyncio
async def test_happy_path_records_duration() -> None:
    """Successful tasks record their wall-time in EvalExample.duration."""

    def task(input_value):
        time.sleep(0.05)
        return f"answer: {input_value}"

    result = await run_eval_async(
        task=task,
        dataset=_two_item_dataset(),
        scorers=[_contains_scorer],
        platform=LangfuseConfig(experiment=f"live-A-{uuid.uuid4().hex[:6]}"),
    )

    assert result.summary == {
        "total_examples": 2,
        "successful_examples": 2,
        "failed_examples": 0,
    }
    assert len(result.examples) == 2
    for ex in result.examples:
        assert ex.error is None
        assert ex.duration > 0.04, (
            f"duration {ex.duration!r} should reflect ~50ms task sleep"
        )


@pytest.mark.asyncio
async def test_wrong_output_distinguishes_score_failure_from_task_failure() -> None:
    """Wrong output is a scorer failure, not a task failure.

    ``compute_summary_counts`` defines successful as ``error is None``;
    pass_rate carries the scorer signal. This is consistent across all
    platform adapters.
    """

    def task(input_value):
        return "totally wrong answer"

    result = await run_eval_async(
        task=task,
        dataset=_two_item_dataset(),
        scorers=[_contains_scorer],
        platform=LangfuseConfig(experiment=f"live-B-{uuid.uuid4().hex[:6]}"),
    )

    assert list(result.pass_rates.values()) == [0.0]
    assert result.summary == {
        "total_examples": 2,
        "successful_examples": 2,
        "failed_examples": 0,
    }
    for ex in result.examples:
        assert ex.error is None
        assert ex.output == "totally wrong answer"


@pytest.mark.asyncio
async def test_task_exception_preserved_in_examples() -> None:
    """Failed items remain in EvalResult.examples with their error captured."""

    def task(input_value):
        raise RuntimeError(f"boom on {input_value}")

    result = await run_eval_async(
        task=task,
        dataset=_two_item_dataset(),
        scorers=[_contains_scorer],
        platform=LangfuseConfig(experiment=f"live-C-{uuid.uuid4().hex[:6]}"),
    )

    assert result.summary["total_examples"] == 2, (
        f"failed items dropped from examples: summary={result.summary!r}"
    )
    assert len(result.examples) == 2

    captured_errors = sorted(ex.error or "" for ex in result.examples)
    assert captured_errors == ["boom on alpha", "boom on beta"], (
        f"expected per-item RuntimeError messages, got {captured_errors!r}"
    )


@pytest.mark.asyncio
async def test_concurrent_failures_index_keyed_correctly() -> None:
    """Each failure is attributed to its own item under concurrent execution."""
    n = 10
    dataset = [
        ExampleData(
            input=f"item-{i}",
            expected=ExpectedResult(expected=f"item-{i}"),
        )
        for i in range(n)
    ]

    def task(input_value):
        raise RuntimeError(f"error-for-{input_value}")

    result = await run_eval_async(
        task=task,
        dataset=dataset,
        scorers=[_contains_scorer],
        platform=LangfuseConfig(experiment=f"live-stress-{uuid.uuid4().hex[:6]}"),
    )

    assert result.summary["total_examples"] == n
    assert len(result.examples) == n
    for ex in result.examples:
        assert ex.error == f"error-for-{ex.input}"


@pytest.mark.asyncio
async def test_task_exception_recorded_server_side() -> None:
    """Failures reach the langfuse server, not just EvalResult.

    Guards against a regression where the adapter satisfies the in-process
    EvalResult shape without actually running the user task.
    """
    from langfuse import get_client

    marker = uuid.uuid4().hex[:8]

    def task(input_value):
        raise RuntimeError(f"server-check-{marker}-{input_value}")

    await run_eval_async(
        task=task,
        dataset=_two_item_dataset(),
        scorers=[_contains_scorer],
        platform=LangfuseConfig(experiment=f"live-server-{marker}"),
    )

    # Allow pending trace writes to flush before reading them back.
    await asyncio.sleep(1.0)

    lf = get_client()
    traces = lf.api.trace.list(limit=50).data

    found_inputs = set()
    for t in traces[:20]:
        full = lf.api.trace.get(t.id)
        output_str = str(full.output) if full.output is not None else ""
        for suffix in ("alpha", "beta"):
            if f"server-check-{marker}-{suffix}" in output_str:
                found_inputs.add(suffix)

    assert found_inputs == {"alpha", "beta"}
