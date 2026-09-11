# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LangFuse adapter context preservation.

Verifies that ExpectedResult.context survives the adapter pipeline
(dataset conversion, historical dataset conversion, scorer wrapping)
so that trajectory scorers receive the full context.

Drives aevaluate end-to-end against a FakeLangfuseClient. The fake's
recorded ``experiments`` list is the inspection point for what the
adapter passed as ``data``; recording scorers verify what the adapter
passed to evaluators.
"""

import pytest
from helpers.fake_langfuse import FakeLangfuseClient
from helpers.scorers import make_context_check_scorer

from agent_evals.adapters.platforms.langfuse import LangfuseConfig, LangFusePlatform
from agent_evals.core.types import (
    ExampleData,
    ExpectedResult,
    TaskResult,
)

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


class TestLangFuseDatasetContextPreservation:
    """ExpectedResult must survive dataset conversion into the langfuse data shape."""

    @pytest.mark.asyncio
    async def test_preserves_expected_result_object(self):
        """Full ExpectedResult (with context) reaches the SDK's data argument."""
        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        dataset = [
            ExampleData(
                input="Book flight",
                expected=ExpectedResult(
                    expected="Booked!",
                    context=EXPECTED_TRAJECTORY_CONTEXT,
                ),
            )
        ]

        await adapter.aevaluate(
            task=lambda x: TaskResult(output="Booked!"),
            dataset=dataset,
            evaluators=[],
            platform=LangfuseConfig(experiment="test-exp"),
        )

        assert len(fake.experiments) == 1
        sent_data = fake.experiments[0]["data"]
        assert len(sent_data) == 1
        sent_expected = sent_data[0]["expected_output"]
        assert isinstance(sent_expected, ExpectedResult), (
            f"adapter must pass ExpectedResult to the SDK, got {type(sent_expected).__name__}"
        )
        assert sent_expected.context == EXPECTED_TRAJECTORY_CONTEXT
        assert sent_expected.expected == "Booked!"


class TestLangFuseHistoricalDatasetContextPreservation:
    """ExpectedResult must survive the historical-data path (task=None)."""

    @pytest.mark.asyncio
    async def test_preserves_expected_result_object(self):
        """Full ExpectedResult (with context) reaches the user scorer
        even when the dataset path is the historical-replay variant (task=None).
        """
        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        scorer = make_context_check_scorer(
            lambda result, expected: (
                isinstance(expected, ExpectedResult)
                and expected.context == EXPECTED_TRAJECTORY_CONTEXT
                and expected.expected == "Booked!"
            ),
            name="ctx_check",
        )

        dataset = [
            ExampleData(
                input="Book flight",
                output=TaskResult(output="Booked!"),
                expected=ExpectedResult(
                    expected="Booked!",
                    context=EXPECTED_TRAJECTORY_CONTEXT,
                ),
            )
        ]

        result = await adapter.aevaluate(
            task=None,  # triggers _convert_to_langfuse_historical_dataset
            dataset=dataset,
            evaluators=[scorer],
            platform=LangfuseConfig(experiment="test-historical"),
        )

        assert result.examples[0].scores["ctx_check"].passed is True


class TestLangFuseScorerContextPreservation:
    """The scorer wrapper must pass ExpectedResult (with context) to user scorers."""

    @pytest.mark.asyncio
    async def test_sync_scorer_receives_expected_with_context(self):
        scorer = make_context_check_scorer(
            lambda result, expected: (
                isinstance(expected, ExpectedResult)
                and expected.context == EXPECTED_TRAJECTORY_CONTEXT
                and expected.expected == "Booked!"
            ),
            name="ctx_check",
        )

        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        result = await adapter.aevaluate(
            task=lambda x: TaskResult(output="Booked!"),
            dataset=[
                ExampleData(
                    input="Book flight",
                    expected=ExpectedResult(
                        expected="Booked!",
                        context=EXPECTED_TRAJECTORY_CONTEXT,
                    ),
                )
            ],
            evaluators=[scorer],
            platform=LangfuseConfig(experiment="scorer-context-sync"),
        )

        assert result.examples[0].scores["ctx_check"].passed is True

    @pytest.mark.asyncio
    async def test_async_scorer_receives_expected_with_context(self):
        scorer = make_context_check_scorer(
            lambda result, expected: (
                isinstance(expected, ExpectedResult)
                and expected.context == EXPECTED_TRAJECTORY_CONTEXT
                and expected.expected == "Booked!"
            ),
            is_async=True,
            name="ctx_check",
        )

        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        result = await adapter.aevaluate(
            task=lambda x: TaskResult(output="Booked!"),
            dataset=[
                ExampleData(
                    input="Book flight",
                    expected=ExpectedResult(
                        expected="Booked!",
                        context=EXPECTED_TRAJECTORY_CONTEXT,
                    ),
                )
            ],
            evaluators=[scorer],
            platform=LangfuseConfig(experiment="scorer-context-async"),
        )

        assert result.examples[0].scores["ctx_check"].passed is True

    @pytest.mark.asyncio
    async def test_string_expected_output_still_works(self):
        """Plain-string expected values still wrap into ExpectedResult."""
        scorer = make_context_check_scorer(
            lambda result, expected: (
                isinstance(expected, ExpectedResult)
                and expected.expected == "Booked!"
                and expected.context is None
            ),
            name="ctx_check",
        )

        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        result = await adapter.aevaluate(
            task=lambda x: TaskResult(output="Booked!"),
            dataset=[
                ExampleData(
                    input="Book flight",
                    expected=ExpectedResult(expected="Booked!"),
                )
            ],
            evaluators=[scorer],
            platform=LangfuseConfig(experiment="scorer-string-expected"),
        )

        assert result.examples[0].scores["ctx_check"].passed is True
