# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test the _safe_scorer wrapper that prevents scorer exceptions from crashing the eval.

The wrapper is private (underscore-prefixed in adapters/scorers/tool_calls.py).
Each tool-call scorer factory wraps its body via _safe_scorer; this test
exercises the wrapper by forcing the inner body to raise and verifying
the wrapper:
  - catches the exception
  - returns a failing Score with passed=False, value=0.0
  - puts the exception class in score.reasoning
  - puts the exception message in score.metadata["error"]
  - calls logger.exception (preserving the traceback)
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest

from agent_evals.adapters.scorers.tool_calls import ToolCallExactMatch
from agent_evals.core.types import ExpectedResult, Score, TaskResult


@pytest.fixture
def matching_inputs() -> tuple[TaskResult, ExpectedResult]:
    """Inputs that would normally produce a passing score."""
    actual = TaskResult(
        output="",
        context={
            "outputs": [
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "lock", "args": {"door": "front"}}],
                }
            ]
        },
    )
    expected = ExpectedResult(
        expected="",
        context={
            "reference_outputs": [
                {
                    "role": "assistant",
                    "tool_calls": [{"name": "lock", "args": {"door": "front"}}],
                }
            ]
        },
    )
    return actual, expected


def test_safe_scorer_catches_exception_and_returns_failing_score(
    matching_inputs: tuple[TaskResult, ExpectedResult],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the inner scorer body raises, _safe_scorer returns a failing Score."""
    actual, expected = matching_inputs
    scorer = ToolCallExactMatch()

    # Force the inner scorer to raise by patching one of its helpers.
    # `_extract_inputs` is called near the top of every scorer body and
    # is a clean injection point.
    boom = RuntimeError("synthetic scorer failure")

    with (
        caplog.at_level(
            logging.ERROR, logger="agent_evals.adapters.scorers.tool_calls"
        ),
        patch(
            "agent_evals.adapters.scorers.tool_calls._extract_inputs",
            side_effect=boom,
        ),
    ):
        score = scorer(actual, expected)

    # The wrapper returned a Score, not propagated the exception.
    assert isinstance(score, Score)
    assert score.passed is False
    assert score.value == 0.0
    assert score.name == "ToolCallExactMatch"
    assert "RuntimeError" in (score.reasoning or "")
    assert score.metadata.get("error") == "synthetic scorer failure"

    # logger.exception was called (verifies the traceback was preserved
    # in the log, not just a bare error message).
    assert any(
        "ToolCallExactMatch" in record.message and record.levelno >= logging.ERROR
        for record in caplog.records
    ), (
        f"Expected logger.exception call mentioning the scorer name; got {caplog.records!r}"
    )


def test_safe_scorer_does_not_swallow_passing_scores(
    matching_inputs: tuple[TaskResult, ExpectedResult],
) -> None:
    """The wrapper is transparent on the happy path: no exception → original Score."""
    actual, expected = matching_inputs
    scorer = ToolCallExactMatch()
    score = scorer(actual, expected)

    assert isinstance(score, Score)
    assert score.passed is True
    assert score.name == "ToolCallExactMatch"


def test_every_tool_call_factory_wraps_with_safe_scorer() -> None:
    """All 5 tool-call scorer factories return wrapped scorers — i.e., scorer
    exceptions never propagate.

    Repeats the same patch test against every factory, briefly. This
    catches the regression class where someone removes _safe_scorer from
    one factory's return path; that factory's exceptions would propagate
    while the others are still safe.
    """
    from agent_evals.adapters.scorers.tool_calls import (
        ToolCallAnyMatch,
        ToolCallExactMatch,
        ToolCallSubsetMatch,
        ToolCallSupersetMatch,
        ToolCallUnorderedMatch,
    )

    factories = [
        ToolCallExactMatch,
        ToolCallSubsetMatch,
        ToolCallSupersetMatch,
        ToolCallUnorderedMatch,
        ToolCallAnyMatch,
    ]

    actual = TaskResult(output="", context={"outputs": []})
    expected = ExpectedResult(expected="", context={"reference_outputs": []})
    boom = RuntimeError("synthetic")

    with patch(
        "agent_evals.adapters.scorers.tool_calls._extract_inputs",
        side_effect=boom,
    ):
        for factory in factories:
            scorer = factory()
            score = scorer(actual, expected)
            assert isinstance(score, Score), (
                f"{factory.__name__} returned non-Score on exception"
            )
            assert score.passed is False, (
                f"{factory.__name__} did not produce failing Score on exception"
            )
            assert "RuntimeError" in (score.reasoning or ""), (
                f"{factory.__name__} did not capture exception class in reasoning"
            )
