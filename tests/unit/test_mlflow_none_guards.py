# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Contract test: MLflowPlatform tolerates None in SDK-typed Optional fields.

Several MLflow SDK fields are Optional and may be None at runtime.
The wrapper (_RealMlflowSession) handles SDK-side Trace.data.request /
.response Nones by defaulting to empty strings. The adapter handles
adapter-side Feedback.feedback and Expectation.expectation Nones by
skipping them with a warning.

This test pins the adapter's consumer-side path: assessments containing
Feedback/Expectation with None payloads must not crash _convert_from_platform_result.
"""

from __future__ import annotations

from types import SimpleNamespace

from helpers.fake_mlflow import FakeMlflowSession

from agent_evals.adapters.platforms.mlflow import MLflowPlatform
from agent_evals.core.types import (
    EvalResult,
)


def test_adapter_tolerates_none_in_optional_fields():
    """Feedback.feedback None and Expectation.expectation None must not crash."""
    from mlflow.genai.evaluation.entities import Expectation, Feedback

    from agent_evals.adapters.platforms.mlflow_client import TraceRecord

    fake = FakeMlflowSession()
    adapter = MLflowPlatform(session=fake)

    # Build a Feedback with .feedback=None and an Expectation with .expectation=None.
    # These are the SDK-typed Optionals that historically crashed _convert_from_platform_result
    # before the adapter added defensive guards (issue #153).
    feedback_none = Feedback(name="scorer-1", value=0.5, rationale=None)
    feedback_none.feedback = None
    expectation_none = Expectation(name="exp-1", value=0.5)
    expectation_none.expectation = None

    # Pre-seed an empty run synthesis path: bypass evaluate's scorer
    # invocation by seeding a run-scoped trace store directly. The
    # fake's search_traces(run_id=non-None) consults _run_traces.
    fake._run_traces["fake-run-0"] = [
        TraceRecord(
            request=None,
            response=None,
            assessments=[feedback_none, expectation_none],
            execution_duration_ms=None,
            trace_metadata={},
        )
    ]
    fake._next_run_id = 1  # so subsequent evaluate calls don't collide

    # Call _convert_from_platform_result directly with a stub
    # SimpleNamespace(run_id='fake-run-0'). That's the unit under test.
    adapter._session_info = adapter._get_session().configure(
        tracking_uri="https://x", experiment="none-guard-test"
    )

    platform_result = SimpleNamespace(run_id="fake-run-0")
    # _convert_from_platform_result expects a real EvaluationResult;
    # SimpleNamespace duck-types the .run_id access we need.
    result = adapter._convert_from_platform_result(platform_result)

    assert isinstance(result, EvalResult)
    assert len(result.examples) == 1
    ex = result.examples[0]

    # The Feedback with .feedback=None must be skipped (not crash, not
    # appear in scores).
    assert ex.scores == {}

    # The Expectation with .expectation=None must be skipped (not crash,
    # not appear in expected).
    assert ex.expected == ""

    # request/response None must yield empty-string input/output.
    assert ex.input == ""
    assert ex.output == ""


def test_pull_traces_tolerates_none_request_response():
    """pull_traces must not crash when DataFrame rows have None request/response."""
    from agent_evals.adapters.platforms.mlflow_client import TraceRecord

    fake = FakeMlflowSession()
    fake.seed_traces(
        experiment_id="fake-exp-none-pull",
        traces=[
            TraceRecord(
                request=None,
                response=None,
                assessments=[],
            )
        ],
    )

    adapter = MLflowPlatform(session=fake)
    pulled = adapter.pull_traces(
        config={"experiment": "none-pull", "tracking_uri": "https://x"}
    )

    assert len(pulled) == 1
    ex = pulled[0]
    # request=None passes through as None to ExampleData.input (Pydantic-validated)
    # response=None becomes "" in TaskResult.output via the conversion
    assert ex.output is not None
    assert ex.output.output == ""
