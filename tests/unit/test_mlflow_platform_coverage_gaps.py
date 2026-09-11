# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Targeted unit tests for MLflowPlatform behaviors not covered by the
seam/wrapper/context/threshold tests.

These tests fill coverage gaps identified during PR #218's review:

- Score-value coercion in ``_convert_from_platform_result`` (bool / NaN /
  ``"yes"`` / ``"no"`` / invalid string / unsupported type) — was covered
  by the deleted ``tests/integration/test_mlflow_adapter.py`` block on
  ``TestMLflowPlatformResultConversion`` but not migrated.
- ``_extract_request_response`` ``structured_response`` branch + JSON
  fallback — the deleted file's ``test_result_conversion_structured_response``
  was the only direct exercise.
- ``_validate_filter_string`` — pure-function input/output checks for the
  empty / unmatched-quotes / no-prefix / no-operator / valid cases.
- ``_convert_to_platform_dataset`` task-signature validation — deleted
  ``TestMLflowPlatformTaskSignatureValidation`` had 5 tests; the new
  end-to-end style passes only single-arg lambdas, so the multi-arg /
  no-arg / self-stripping branches are unexercised.

These are pure-function tests — no SDK interaction required, no fake
session needed for filter / signature / extraction.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent_evals.adapters.platforms.mlflow import MLflowPlatform
from agent_evals.adapters.platforms.mlflow_client import SessionInfo, TraceRecord
from agent_evals.core.types import (
    EvalResult,
    ExampleData,
    ExpectedResult,
)

# ---------------------------------------------------------------------------
# Score-value coercion: drives _convert_from_platform_result via the fake's
# pre-seeded run-trace store.
# ---------------------------------------------------------------------------


def _make_adapter_with_seeded_assessment(assessment):
    """Build an adapter pre-populated with a session_info and one TraceRecord.

    Return type intentionally unannotated: ty cannot infer the exact type
    of a class decorated by the generic registry, and the inferred return
    type is sufficient for this internal test helper.
    """
    from helpers.fake_mlflow import FakeMlflowSession

    fake = FakeMlflowSession()
    adapter = MLflowPlatform(session=fake)
    adapter._session_info = SessionInfo(
        experiment_id="fake-exp-coverage",
        tracking_uri="https://localhost:5000",
    )
    fake._run_traces["fake-run-coverage"] = [
        TraceRecord(
            request='{"messages":[{"content":"hi"}]}',
            response='{"messages":[{"content":"hello"}]}',
            assessments=[assessment],
            execution_duration_ms=100,
            trace_metadata={},
        )
    ]
    return adapter


def _convert_with_assessment(assessment) -> EvalResult:
    from mlflow.genai.evaluation.entities import EvaluationResult  # noqa: F401

    adapter = _make_adapter_with_seeded_assessment(assessment)
    platform_result = SimpleNamespace(run_id="fake-run-coverage")
    return adapter._convert_from_platform_result(platform_result)


def test_score_coercion_bool_true_passes():
    """bool True → value 1.0, passed True."""
    from mlflow.genai.evaluation.entities import Feedback

    fb = Feedback(name="BoolTrue", value=True, rationale=None)
    result = _convert_with_assessment(fb)
    assert result.examples[0].scores["BoolTrue"].value == pytest.approx(1.0)
    assert result.examples[0].scores["BoolTrue"].passed is True


def test_score_coercion_bool_false_fails():
    """bool False → value 0.0, passed False."""
    from mlflow.genai.evaluation.entities import Feedback

    fb = Feedback(name="BoolFalse", value=False, rationale=None)
    result = _convert_with_assessment(fb)
    assert result.examples[0].scores["BoolFalse"].value == pytest.approx(0.0)
    assert result.examples[0].scores["BoolFalse"].passed is False


def test_score_coercion_string_yes_passes():
    """str 'yes' → value 1.0, passed True (case-insensitive)."""
    from mlflow.genai.evaluation.entities import Feedback

    fb = Feedback(name="YesScore", value="YES", rationale=None)
    result = _convert_with_assessment(fb)
    assert result.examples[0].scores["YesScore"].value == pytest.approx(1.0)
    assert result.examples[0].scores["YesScore"].passed is True


def test_score_coercion_string_no_fails():
    """str 'no' → value 0.0, passed False (case-insensitive)."""
    from mlflow.genai.evaluation.entities import Feedback

    fb = Feedback(name="NoScore", value="No", rationale=None)
    result = _convert_with_assessment(fb)
    assert result.examples[0].scores["NoScore"].value == pytest.approx(0.0)
    assert result.examples[0].scores["NoScore"].passed is False


def test_score_coercion_string_numeric_uses_threshold():
    """str '0.7' → parsed as 0.7, compared to default 0.5 threshold → passes."""
    from mlflow.genai.evaluation.entities import Feedback

    fb = Feedback(name="NumStr", value="0.7", rationale=None)
    result = _convert_with_assessment(fb)
    assert result.examples[0].scores["NumStr"].value == pytest.approx(0.7)
    assert result.examples[0].scores["NumStr"].passed is True


def test_score_coercion_invalid_string_skipped():
    """Non-numeric, non-yes/no string is silently dropped — score absent."""
    from mlflow.genai.evaluation.entities import Feedback

    fb = Feedback(name="Bogus", value="banana", rationale=None)
    result = _convert_with_assessment(fb)
    assert "Bogus" not in result.examples[0].scores


def test_score_coercion_nan_skipped():
    """NaN string parses to nan, must be dropped (not pass with passed=False)."""
    from mlflow.genai.evaluation.entities import Feedback

    fb = Feedback(name="NaNScore", value="nan", rationale=None)
    result = _convert_with_assessment(fb)
    assert "NaNScore" not in result.examples[0].scores


def test_score_coercion_none_value_skipped():
    """Feedback.feedback.value=None must be dropped, not crash."""
    from mlflow.genai.evaluation.entities import Feedback

    # Construct Feedback with None value (rare but typed Optional in SDK)
    fb = Feedback(name="NoneVal", value=None, rationale=None)
    result = _convert_with_assessment(fb)
    assert "NoneVal" not in result.examples[0].scores


# ---------------------------------------------------------------------------
# _extract_request_response branches
# ---------------------------------------------------------------------------


def test_extract_request_response_structured_response_branch():
    """response_obj['structured_response'] takes precedence over messages[-1]."""
    adapter = MLflowPlatform()

    request = '{"messages":[{"content":"what is 2+2?"}]}'
    response = '{"structured_response":{"answer":4,"explanation":"basic math"}}'

    inp, out = adapter._extract_request_response(request, response)
    assert inp == "what is 2+2?"
    assert "answer" in out  # str(dict) representation
    assert "4" in out


def test_extract_request_response_malformed_json_falls_back():
    """When JSON parsing fails, raw strings are returned verbatim."""
    adapter = MLflowPlatform()

    inp, out = adapter._extract_request_response("not json", "also not json")
    assert inp == "not json"
    assert out == "also not json"


def test_extract_request_response_missing_messages_key_falls_back():
    """When messages key is missing, raw strings are returned (KeyError caught)."""
    adapter = MLflowPlatform()

    request = '{"other_key": "value"}'
    response = '{"another_key": "value"}'
    inp, out = adapter._extract_request_response(request, response)
    assert inp == request
    assert out == response


def test_extract_request_response_empty_messages_falls_back():
    """When messages is an empty list, IndexError is caught and raw returned."""
    adapter = MLflowPlatform()

    request = '{"messages":[]}'
    response = '{"messages":[]}'
    inp, out = adapter._extract_request_response(request, response)
    assert inp == request
    assert out == response


def test_extract_request_response_both_empty_short_circuits():
    """Both empty inputs short-circuit to ('', '')."""
    adapter = MLflowPlatform()
    assert adapter._extract_request_response("", "") == ("", "")


# ---------------------------------------------------------------------------
# _validate_filter_string
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filter_str,expected_error",
    [
        ("", "cannot be empty"),
        ('trace.status = "OK', "unmatched quotes"),
        ("status = 'OK'", "valid field prefix"),
    ],
)
def test_validate_filter_rejects_invalid(filter_str, expected_error):
    """Filter validation rejects empty, unmatched-quote, and no-prefix inputs."""
    with pytest.raises(ValueError, match=expected_error):
        MLflowPlatform._validate_filter_string(filter_str)


def test_validate_filter_accepts_valid_filter():
    """A filter with a valid prefix and operator passes (returns None)."""
    MLflowPlatform._validate_filter_string('trace.status = "OK"')


# ---------------------------------------------------------------------------
# _convert_to_platform_dataset task-signature validation
# ---------------------------------------------------------------------------


def _no_args() -> str:
    return "x"


def _two_args(a: str, b: str) -> str:  # noqa: ARG001
    _ = a, b
    return "x"


@pytest.mark.parametrize(
    "bad_task", [_no_args, _two_args], ids=["no-param", "two-param"]
)
def test_dataset_conversion_rejects_invalid_task_signature(bad_task):
    """Tasks with 0 or 2+ parameters (excluding self) are rejected."""
    adapter = MLflowPlatform()
    dataset = [ExampleData(input="a", expected=ExpectedResult(expected="b"))]
    with pytest.raises(ValueError, match="exactly one parameter"):
        # Deliberate signature mismatch — exercises the validation path.
        adapter._convert_to_platform_dataset(bad_task, dataset)


def test_dataset_conversion_strips_self_for_methods():
    """A bound method has self + one arg — adapter must skip self.

    Also covers task=None inline (using example.input directly as the
    inputs payload), since the same dataset path branches on task.
    """
    adapter = MLflowPlatform()

    class _Caller:
        def task(self, x: str) -> str:
            _ = self, x
            return "answer"

    bound = _Caller().task
    dataset = [
        ExampleData(
            input={"prompt": "hi", "lang": "en"},
            expected=ExpectedResult(expected="hello"),
        ),
    ]

    # Bound method: x parameter wraps the dict input.
    converted = adapter._convert_to_platform_dataset(bound, dataset)
    assert converted[0]["inputs"] == {"x": {"prompt": "hi", "lang": "en"}}

    # task=None: input passed directly.
    converted = adapter._convert_to_platform_dataset(None, dataset)
    assert converted[0]["inputs"] == {"prompt": "hi", "lang": "en"}
