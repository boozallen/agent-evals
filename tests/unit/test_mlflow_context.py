# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for MLflowPlatform context preservation.

Drives MLflowPlatform.aevaluate against FakeMlflowSession; assertions
land on the data the fake's evaluate recorded (i.e., what the adapter
passed through the port) and on EvalResult.examples[*].
"""

from __future__ import annotations

import asyncio
import os
from unittest.mock import patch

import pytest
from helpers.fake_mlflow import FakeMlflowSession
from helpers.scorers import make_context_check_scorer, passing_scorer

from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform
from agent_evals.core.types import ExampleData, ExpectedResult


@pytest.fixture(autouse=True)
def mlflow_tracking_uri_env():
    """Set MLFLOW_TRACKING_URI for all tests in this module."""
    with patch.dict(os.environ, {"MLFLOW_TRACKING_URI": "https://x"}):
        yield


class TestMLflowDatasetContextPreservation:
    """Adapter must preserve ExpectedResult.context through to MLflow's expectations dict."""

    def test_preserves_expected_context_in_expectations(self):
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)

        dataset = [
            ExampleData(
                input="q",
                expected=ExpectedResult(
                    expected="a",
                    context={"category": "math", "difficulty": "easy"},
                ),
            ),
        ]

        def task(_q: str) -> str:
            return "answer"

        asyncio.run(
            adapter.aevaluate(
                task=task,
                dataset=dataset,
                evaluators=[passing_scorer],
                platform=MlflowConfig(experiment="ctx-1"),
            )
        )

        assert len(fake.evaluations) == 1
        data = fake.evaluations[0]["data"]
        assert len(data) == 1
        expectations = data[0]["expectations"]
        assert expectations["expected"] == "a"
        assert expectations["context"] == {"category": "math", "difficulty": "easy"}

    def test_no_context_when_expected_has_none(self):
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)

        dataset = [ExampleData(input="q", expected=ExpectedResult(expected="a"))]

        def task(_q: str) -> str:
            return "answer"

        asyncio.run(
            adapter.aevaluate(
                task=task,
                dataset=dataset,
                evaluators=[passing_scorer],
                platform=MlflowConfig(experiment="ctx-2"),
            )
        )

        expectations = fake.evaluations[0]["data"][0]["expectations"]
        assert expectations["expected"] == "a"
        assert "context" not in expectations


class TestMLflowScorerContextPreservation:
    """Adapter's scorer wrapper must propagate context through to user scorers."""

    def test_sync_scorer_receives_expected_with_context(self):
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)

        scorer = make_context_check_scorer(
            lambda result, expected: (
                expected is not None
                and expected.expected == "a"
                and expected.context == {"src": "test"}
            ),
            name="ctx_check",
        )

        dataset = [
            ExampleData(
                input="q",
                expected=ExpectedResult(expected="a", context={"src": "test"}),
            ),
        ]

        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _q: "answer",
                dataset=dataset,
                evaluators=[scorer],
                platform=MlflowConfig(experiment="ctx-3"),
            )
        )

        assert result.examples[0].scores["ctx_check"].passed is True

    def test_async_scorer_receives_expected_with_context(self):
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)

        scorer = make_context_check_scorer(
            lambda result, expected: (
                expected is not None
                and expected.expected == "a"
                and expected.context == {"src": "async-test"}
            ),
            is_async=True,
            name="ctx_check",
        )

        dataset = [
            ExampleData(
                input="q",
                expected=ExpectedResult(expected="a", context={"src": "async-test"}),
            ),
        ]

        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _q: "answer",
                dataset=dataset,
                evaluators=[scorer],
                platform=MlflowConfig(experiment="ctx-4"),
            )
        )

        assert result.examples[0].scores["ctx_check"].passed is True
