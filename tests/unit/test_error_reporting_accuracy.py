# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end reporting accuracy for failed runs.

A run that failed must report itself as failed. These tests drive
``run_eval_async`` through three distinct failure modes — the task raises, the
task returns a non-``TaskResult``, and the scorers raise — and assert on what a
CI gate would actually read: the aggregates, and whether the summary counts
agree with them.

The gate under test is the idiomatic one:

    all(v >= 0.9 for v in result.pass_rates.values())

which is vacuously True over an empty dict — the reason case 2 was silent.
"""

import pytest

from agent_evals import ExampleData, ExpectedResult, Score, TaskResult, run_eval_async


def _gate_passes(result) -> bool:
    """Evaluate the CI gate shape this ticket is about."""
    return all(v >= 0.9 for v in result.pass_rates.values())


def _dataset(n: int = 3) -> list[ExampleData]:
    return [
        ExampleData(input=chr(ord("a") + i), expected=ExpectedResult(expected="ok"))
        for i in range(n)
    ]


def _always_passing_scorer(result, expected=None, **context):
    """Indifferent to the output — the shape that made a crash look green."""
    return Score(name="AlwaysPasses", value=1.0, passed=True)


@pytest.mark.asyncio
async def test_case_1_task_raises_does_not_report_passing():
    """Every task crashes: previously reported pass_rates {'S': 1.0}."""

    async def crashing_task(_):
        raise RuntimeError("simulated total agent outage")

    result = await run_eval_async(
        task=crashing_task,
        dataset=_dataset(),
        scorers=[_always_passing_scorer],
    )

    assert not _gate_passes(result)
    assert result.summary["successful_examples"] == 0
    assert result.summary["failed_examples"] == 3


@pytest.mark.asyncio
async def test_case_2_task_returns_non_task_result_does_not_report_passing():
    """Framework-level failure: previously returned empty aggregates."""

    async def bad_return_task(_):
        return {"not": "a TaskResult"}

    result = await run_eval_async(
        task=bad_return_task,
        dataset=_dataset(),
        scorers=[_always_passing_scorer],
    )

    # The severe case: an empty dict would pass the gate vacuously.
    assert result.pass_rates, "aggregates must not be empty on total failure"
    assert not _gate_passes(result)
    assert result.summary["successful_examples"] == 0
    assert result.summary["failed_examples"] == 3


@pytest.mark.asyncio
async def test_case_3_scorer_raises_is_not_counted_successful():
    """Task fine, all scorers crash: previously counted 3 successful."""

    async def ok_task(_):
        return TaskResult(output="ok")

    def crashing_scorer(result, expected=None, **context):
        raise RuntimeError("scorer exploded")

    result = await run_eval_async(
        task=ok_task,
        dataset=_dataset(),
        scorers=[crashing_scorer],
    )

    assert not _gate_passes(result)
    assert result.summary["successful_examples"] == 0
    assert result.summary["failed_examples"] == 3

    # The failure is recorded on the score itself, not just in logs.
    assert result.examples[0].scorer_errors
    assert "scorer exploded" in "".join(result.examples[0].scorer_errors.values())


@pytest.mark.asyncio
async def test_summary_and_aggregates_agree_on_partial_failure():
    """Counts and scores must tell the same story about which examples failed."""

    async def flaky_task(x):
        if x == "b":
            raise RuntimeError("this one fails")
        return TaskResult(output="ok")

    def exact(result, expected=None, **context):
        ok = result.output == expected.expected
        return Score(name="ExactMatch", value=1.0 if ok else 0.0, passed=ok)

    result = await run_eval_async(task=flaky_task, dataset=_dataset(), scorers=[exact])

    assert result.summary["failed_examples"] == 1
    assert result.summary["successful_examples"] == 2
    # 2 of 3 passed — the crashed example is a zero, not excluded.
    assert result.pass_rates["ExactMatch"] == pytest.approx(2 / 3)
    assert not _gate_passes(result)


@pytest.mark.asyncio
async def test_healthy_run_still_reports_passing():
    """The fix must not make legitimate green runs report as failed."""

    async def ok_task(_):
        return TaskResult(output="ok")

    def exact(result, expected=None, **context):
        ok = result.output == expected.expected
        return Score(name="ExactMatch", value=1.0 if ok else 0.0, passed=ok)

    result = await run_eval_async(task=ok_task, dataset=_dataset(), scorers=[exact])

    assert _gate_passes(result)
    assert result.pass_rates["ExactMatch"] == pytest.approx(1.0)
    assert result.summary["successful_examples"] == 3
    assert result.summary["failed_examples"] == 0


@pytest.mark.asyncio
async def test_legitimate_zero_score_is_not_an_error():
    """A real failing verdict still counts as a successful example.

    Distinguishing "scored 0.0" from "could not score" is the whole point;
    a low score is data, not an infrastructure failure.
    """

    async def ok_task(_):
        return TaskResult(output="wrong")

    def exact(result, expected=None, **context):
        return Score(name="ExactMatch", value=0.0, passed=False)

    result = await run_eval_async(task=ok_task, dataset=_dataset(), scorers=[exact])

    assert not _gate_passes(result)
    # The scorers all ran and returned verdicts, so nothing "failed".
    assert result.summary["successful_examples"] == 3
    assert result.summary["failed_examples"] == 0
    assert not result.examples[0].scorer_errors


@pytest.mark.asyncio
async def test_per_example_isolation_preserved():
    """One bad example must not abort the rest of the run."""

    async def one_bad_task(x):
        if x == "b":
            raise RuntimeError("only this one")
        return TaskResult(output="ok")

    result = await run_eval_async(
        task=one_bad_task,
        dataset=_dataset(5),
        scorers=[_always_passing_scorer],
    )

    assert len(result.examples) == 5
    assert result.summary["failed_examples"] == 1
    assert result.summary["successful_examples"] == 4
