# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for Braintrust adapter context preservation.

Verifies that TaskResult.context and ExpectedResult.context survive the
adapter pipeline (dataset conversion, task wrapping, historical replay,
scorer wrapping) so that trajectory scorers receive the full context.

Tests drive aevaluate end-to-end via FakeBraintrustClient. Assertions
land on the fake's recorded `data` (for dataset-conversion behavior),
on EvalResult.examples[*] (for end-to-end output behavior), or on a
recording scorer's captured arguments (for scorer-wrapper context).
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import patch

import pytest
from helpers.fake_braintrust import FakeBraintrustClient
from helpers.scorers import make_context_check_scorer, passing_scorer

from agent_evals.adapters.platforms.braintrust import (
    BraintrustConfig,
    BraintrustPlatform,
)
from agent_evals.core.types import (
    ExampleData,
    ExpectedResult,
    Score,
    TaskResult,
)

pytestmark = pytest.mark.requires_braintrust

TRAJECTORY_CONTEXT = {
    "outputs": [
        {"role": "user", "content": "Book flight"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "book", "arguments": '{"id": "123"}'}}
            ],
        },
    ]
}

EXPECTED_TRAJECTORY_CONTEXT = {
    "reference_outputs": [
        {"role": "user", "content": "Book flight"},
        {
            "role": "assistant",
            "tool_calls": [
                {"function": {"name": "book", "arguments": '{"id": "123"}'}}
            ],
        },
    ]
}


class TestBraintrustDatasetContextPreservation:
    """ExpectedResult.context survives dataset conversion into the seam."""

    @pytest.mark.asyncio
    async def test_preserves_expected_result_context(self):
        """ExpectedResult.context survives into client.run_eval(data=...)."""
        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            fake = FakeBraintrustClient()
            adapter = BraintrustPlatform(client=fake)

            await adapter.aevaluate(
                task=lambda x: "out",
                dataset=[
                    ExampleData(
                        input="Book flight",
                        output=TaskResult(output="ignored"),
                        expected=ExpectedResult(
                            expected="Booked!",
                            context=EXPECTED_TRAJECTORY_CONTEXT,
                        ),
                    )
                ],
                evaluators=[passing_scorer],
                platform=BraintrustConfig(project="p", experiment="e"),
            )

            recorded = fake.evals[0]["data"][0]
            assert isinstance(recorded["expected"], ExpectedResult)
            assert recorded["expected"].context == EXPECTED_TRAJECTORY_CONTEXT
            # Pin spec 029: _convert_to_platform_dataset emits only
            # input/expected/metadata. No `output` key, even when the
            # user pre-populates ExampleData.output (which they didn't
            # here, but the assertion holds either way).
            assert "output" not in recorded, (
                f"Expected dict without 'output' key, got keys {list(recorded.keys())!r}"
            )


class TestBraintrustTaskWrapperContextPreservation:
    """The adapter's task wrapper preserves TaskResult on the way to the seam."""

    @pytest.mark.asyncio
    async def test_sync_task_preserves_task_result(self):
        """A sync task returning TaskResult: the fake records the TaskResult unchanged."""
        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            fake = FakeBraintrustClient()
            adapter = BraintrustPlatform(client=fake)

            def my_task(_x):
                return TaskResult(output="Booked!", context=TRAJECTORY_CONTEXT)

            await adapter.aevaluate(
                task=my_task,
                dataset=[ExampleData(input="Book flight")],
                evaluators=[passing_scorer],
                platform=BraintrustConfig(project="p", experiment="e"),
            )

            # The fake invoked the (adapter-wrapped) task; calling it again
            # returns the same TaskResult shape end-to-end.
            wrapped = fake.evals[0]["task"]
            result = wrapped("Book flight")
            assert isinstance(result, TaskResult)
            assert result.context == TRAJECTORY_CONTEXT

    @pytest.mark.asyncio
    async def test_async_task_preserves_task_result(self):
        """Async task returning TaskResult: same — wrapper preserves the full type."""
        import asyncio as _asyncio

        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            fake = FakeBraintrustClient()
            adapter = BraintrustPlatform(client=fake)

            async def my_task(_x):
                return TaskResult(output="Booked!", context=TRAJECTORY_CONTEXT)

            await adapter.aevaluate(
                task=my_task,
                dataset=[ExampleData(input="Book flight")],
                evaluators=[passing_scorer],
                platform=BraintrustConfig(project="p", experiment="e"),
            )

            wrapped = fake.evals[0]["task"]
            result = wrapped("Book flight")
            if _asyncio.iscoroutine(result):
                result = await result
            assert isinstance(result, TaskResult)
            assert result.context == TRAJECTORY_CONTEXT


class TestBraintrustHistoricalReplayContextPreservation:
    """Pre-populated TaskResult outputs survive historical-replay through the seam."""

    @pytest.mark.asyncio
    async def test_replay_preserves_task_result_with_context(self):
        """ExampleData.output (pre-populated TaskResult) round-trips into client.run_eval.

        Uses a recording scorer to capture what the seam actually delivers as
        the task output. The replay task is single-shot (queue-drained on
        first call), so we observe the TaskResult during the live run rather
        than by re-invoking the wrapped task afterward.
        """
        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            scorer = make_context_check_scorer(
                lambda result, expected: (
                    isinstance(result, TaskResult)
                    and result.context == TRAJECTORY_CONTEXT
                    and result.output == "Booked!"
                ),
                name="ctx_check",
            )

            fake = FakeBraintrustClient()
            adapter = BraintrustPlatform(client=fake)

            preloaded = TaskResult(output="Booked!", context=TRAJECTORY_CONTEXT)
            result = await adapter.aevaluate(
                task=None,  # historical-replay mode
                dataset=[
                    ExampleData(
                        input="Book flight",
                        output=preloaded,
                        expected=ExpectedResult(
                            expected="Booked!",
                            context=EXPECTED_TRAJECTORY_CONTEXT,
                        ),
                    )
                ],
                evaluators=[scorer],
                platform=BraintrustConfig(project="p", experiment="e"),
            )

            assert result.examples[0].scores["ctx_check"].passed is True

    @pytest.mark.asyncio
    async def test_replay_preserves_fifo_order_for_repeated_input(self):
        """Two examples sharing the same input value: outputs replayed FIFO.

        Pins the per-input FIFO contract documented in
        _prepare_historical_replay's docstring. A regression that
        replaced setdefault(...).append(out) with last-write-wins
        (e.g., precomputed[example.input] = [out]) would fail this:
        the second example's pre-populated output would shadow the
        first.
        """
        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            # Outcome-level FIFO proof: each example's replayed task output —
            # the same value the scorer receives as ``result.output`` — lands
            # on ``EvalExample.output``. A last-write-wins regression would
            # surface as both examples showing "second".
            fifo_ok = make_context_check_scorer(
                lambda result, expected: (
                    isinstance(result, TaskResult)
                    and result.output in {"first", "second"}
                ),
                name="fifo_ok",
            )

            fake = FakeBraintrustClient()
            adapter = BraintrustPlatform(client=fake)

            # Two examples with the same input "Book flight" but different
            # pre-populated outputs. FIFO ordering means we should observe
            # "first" before "second" at the scorer.
            result = await adapter.aevaluate(
                task=None,
                dataset=[
                    ExampleData(
                        input="Book flight",
                        output=TaskResult(output="first"),
                        expected=ExpectedResult(expected="x"),
                    ),
                    ExampleData(
                        input="Book flight",
                        output=TaskResult(output="second"),
                        expected=ExpectedResult(expected="y"),
                    ),
                ],
                evaluators=[fifo_ok],
                platform=BraintrustConfig(project="p", experiment="e"),
            )

            # Every example received a valid replayed TaskResult at the scorer.
            assert all(ex.scores["fifo_ok"].passed is True for ex in result.examples)
            # FIFO ordering: example[0] got "first", example[1] got "second".
            observed = [ex.output for ex in result.examples]
            assert observed == ["first", "second"], (
                f"Expected FIFO replay ['first', 'second'], got {observed!r}"
            )

    @pytest.mark.asyncio
    async def test_replay_substitutes_empty_taskresult_for_missing_output(self):
        """example.output=None becomes TaskResult(output="") at replay time.

        Pins the missing-output stand-in. A regression that returned
        None instead would propagate through _wrap_task_for_braintrust
        as the string "" instead of a TaskResult, breaking the contract
        that scorers receive TaskResult instances.
        """
        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            scorer = make_context_check_scorer(
                lambda result, expected: (
                    isinstance(result, TaskResult) and result.output == ""
                ),
                name="ctx_check",
            )

            fake = FakeBraintrustClient()
            adapter = BraintrustPlatform(client=fake)

            result = await adapter.aevaluate(
                task=None,
                dataset=[
                    ExampleData(
                        input="Book flight",
                        # No output= — defaults to None → stand-in.
                        expected=ExpectedResult(expected="x"),
                    ),
                ],
                evaluators=[scorer],
                platform=BraintrustConfig(project="p", experiment="e"),
            )

            assert len(result.examples) == 1
            assert result.examples[0].scores["ctx_check"].passed is True

    @pytest.mark.asyncio
    async def test_replay_handles_unhashable_input_via_fifo_fallback(self):
        """Unhashable inputs (dict, list) trigger the FIFO-list fallback.

        Pins the ``except TypeError`` branches in ``_prepare_historical_replay``.
        The build-time branch routes the example into ``fallback`` (since
        the hashable-keyed dict can't accept the input). The replay-time
        branch handles the lookup-side ``TypeError`` so the fallback
        iterator gets consulted instead.
        """
        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            received: list[Any] = []

            def recording_scorer(
                result, expected: ExpectedResult | None = None, **_kwargs
            ):
                received.append(
                    result.output if isinstance(result, TaskResult) else result
                )
                return Score(name="rec", value=1.0, passed=True)

            recording_scorer.__name__ = "rec"

            fake = FakeBraintrustClient()
            adapter = BraintrustPlatform(client=fake)

            await adapter.aevaluate(
                task=None,
                dataset=[
                    ExampleData(
                        input={"action": "book"},
                        output=TaskResult(output="alpha"),
                        expected=ExpectedResult(expected="x"),
                    ),
                    ExampleData(
                        input={"action": "cancel"},
                        output=TaskResult(output="beta"),
                        expected=ExpectedResult(expected="y"),
                    ),
                ],
                evaluators=[recording_scorer],
                platform=BraintrustConfig(project="p", experiment="e"),
            )

            # FIFO fallback iterates in dataset order regardless of input value.
            assert received == ["alpha", "beta"], (
                f"Expected FIFO fallback ['alpha', 'beta'], got {received!r}"
            )


class TestBraintrustScorerContextPreservation:
    """Scorers receive TaskResult / ExpectedResult with context preserved."""

    @pytest.mark.asyncio
    async def test_sync_scorer_receives_full_context(self):
        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            scorer = make_context_check_scorer(
                lambda result, expected: (
                    isinstance(result, TaskResult)
                    and result.context == TRAJECTORY_CONTEXT
                    and isinstance(expected, ExpectedResult)
                    and expected.context == EXPECTED_TRAJECTORY_CONTEXT
                ),
                name="ctx_check",
            )

            def my_task(_x):
                return TaskResult(output="Booked!", context=TRAJECTORY_CONTEXT)

            fake = FakeBraintrustClient()
            adapter = BraintrustPlatform(client=fake)
            result = await adapter.aevaluate(
                task=my_task,
                dataset=[
                    ExampleData(
                        input="Book flight",
                        expected=ExpectedResult(
                            expected="Booked!", context=EXPECTED_TRAJECTORY_CONTEXT
                        ),
                    )
                ],
                evaluators=[scorer],
                platform=BraintrustConfig(project="p", experiment="e"),
            )

            assert result.examples[0].scores["ctx_check"].passed is True

    @pytest.mark.asyncio
    async def test_async_scorer_receives_full_context(self):
        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            scorer = make_context_check_scorer(
                lambda result, expected: (
                    isinstance(result, TaskResult)
                    and result.context == TRAJECTORY_CONTEXT
                    and isinstance(expected, ExpectedResult)
                    and expected.context == EXPECTED_TRAJECTORY_CONTEXT
                ),
                is_async=True,
                name="ctx_check",
            )

            def my_task(_x):
                return TaskResult(output="Booked!", context=TRAJECTORY_CONTEXT)

            fake = FakeBraintrustClient()
            adapter = BraintrustPlatform(client=fake)
            result = await adapter.aevaluate(
                task=my_task,
                dataset=[
                    ExampleData(
                        input="Book flight",
                        expected=ExpectedResult(
                            expected="Booked!", context=EXPECTED_TRAJECTORY_CONTEXT
                        ),
                    )
                ],
                evaluators=[scorer],
                platform=BraintrustConfig(project="p", experiment="e"),
            )

            assert result.examples[0].scores["ctx_check"].passed is True
