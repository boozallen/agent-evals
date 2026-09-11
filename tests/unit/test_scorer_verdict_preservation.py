# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""The scorer owns its pass/fail verdict; adapters must not recompute it.

Companion to ``test_threshold_preservation.py``, which pins the *threshold*
contract using spy scorers that always emit ``_threshold`` metadata. That
property is exactly what blinded the suite to this defect: only the autoevals
family writes ``_threshold``, so the fallback path was never exercised by a
scorer shape that reaches it in production.

These tests therefore build scorers from the **registered factories** —
``StateMatch``, ``ToolCallExactMatch`` — and from plain callables that emit no
``_threshold`` at all, mirroring what real users pass.

The defect: each adapter's scorer wrapper discards ``Score.passed`` and the
reconstruction path recomputes it as ``value >= threshold``, defaulting to
``PASS_THRESHOLD = 0.5``. ``StateMatch`` returns ``value = matched/total`` with
``passed = (no failures)``, so a 3-of-4 partial match round-trips as passing.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any
from unittest.mock import patch

import pytest

from agent_evals.core.types import ExampleData, ExpectedResult, Score, TaskResult

# ---------------------------------------------------------------------------
# Fixtures shared across the three adapters
# ---------------------------------------------------------------------------

# Four expected state paths; the task's final_state matches three of them.
# StateMatch → value == 0.75, passed is False.
_EXPECTED_STATE = {
    "lights.kitchen": "on",
    "lights.hallway": "on",
    "thermostat.target": 70,
    "lock.front": "locked",
}

_ACTUAL_STATE = {
    "lights": {"kitchen": "on", "hallway": "on"},
    "thermostat": {"target": 70},
    "lock": {"front": "unlocked"},  # the one mismatch
}


def _state_task(_input: Any) -> TaskResult:
    """Task returning a final_state matching 3 of the 4 expected paths."""
    return TaskResult(output="done", context={"final_state": _ACTUAL_STATE})


def _state_dataset() -> list[ExampleData]:
    return [ExampleData(input="run the house", expected=ExpectedResult(expected="ok"))]


def _make_verdict_scorer(name: str, value: float, passed: bool):
    """A scorer emitting a fixed (value, passed) pair and NO _threshold.

    Deliberately not a spy over a threshold: the point is a verdict that is
    not a function of the value, which no threshold can reproduce.
    """

    def scorer(result: TaskResult, expected: ExpectedResult | None = None, **kwargs):
        _ = result, expected, kwargs  # vulture: intentionally unused
        return Score(name=name, value=value, passed=passed)

    scorer.__name__ = name
    scorer.__qualname__ = name
    return scorer


def _make_per_input_scorer(name: str, verdicts: dict[str, tuple[float, bool]]):
    """A scorer returning a different (value, passed) per task output.

    Keyed on the task's output string so each example in one run gets its own
    verdict — the fixture for per-example distinctness.
    """

    def scorer(result: TaskResult, expected: ExpectedResult | None = None, **kwargs):
        _ = expected, kwargs  # vulture: intentionally unused
        value, passed = verdicts[str(result.output)]
        return Score(name=name, value=value, passed=passed)

    scorer.__name__ = name
    scorer.__qualname__ = name
    return scorer


# ---------------------------------------------------------------------------
# Per-adapter drivers — each returns EvalResult.examples
# ---------------------------------------------------------------------------


async def _run_braintrust(task, dataset, evaluators, experiment: str):
    from helpers.fake_braintrust import FakeBraintrustClient

    from agent_evals.adapters.platforms.braintrust import (
        BraintrustConfig,
        BraintrustPlatform,
    )

    adapter = BraintrustPlatform(client=FakeBraintrustClient())
    with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
        result = await adapter.aevaluate(
            task=task,
            dataset=dataset,
            evaluators=evaluators,
            platform=BraintrustConfig(project="p", experiment=experiment),
        )
    return result.examples


async def _run_mlflow(task, dataset, evaluators, experiment: str):
    from helpers.fake_mlflow import FakeMlflowSession

    from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform

    adapter = MLflowPlatform(session=FakeMlflowSession())
    result = await adapter.aevaluate(
        task=task,
        dataset=dataset,
        evaluators=evaluators,
        platform=MlflowConfig(experiment=experiment),
    )
    return result.examples


async def _run_langfuse(task, dataset, evaluators, experiment: str):
    from helpers.fake_langfuse import FakeLangfuseClient

    from agent_evals.adapters.platforms.langfuse import (
        LangfuseConfig,
        LangFusePlatform,
    )

    adapter = LangFusePlatform(client=FakeLangfuseClient())
    result = await adapter.aevaluate(
        task=task,
        dataset=dataset,
        evaluators=evaluators,
        platform=LangfuseConfig(experiment=experiment),
    )
    return result.examples


_ADAPTERS = [
    pytest.param(
        _run_braintrust, id="braintrust", marks=pytest.mark.requires_braintrust
    ),
    pytest.param(_run_mlflow, id="mlflow"),
    pytest.param(_run_langfuse, id="langfuse"),
]


@pytest.fixture(autouse=True)
def _mlflow_tracking_uri(monkeypatch):
    """MLflowPlatform._resolve_tracking_uri raises without this.

    Spelled ``https``: the session is faked, so nothing connects, but the value
    still passes through transport validation.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://x")


# ===================================================================
# Task 2.1 — direct-call reference (passes today)
# ===================================================================


class TestStateMatchDirectCall:
    """The baseline every adapter assertion below is compared against."""

    def test_three_of_four_paths_is_fractional_and_failing(self):
        from agent_evals.adapters.scorers.state import StateMatch

        scorer = StateMatch(expected_state=_EXPECTED_STATE)
        score = scorer(_state_task(None), ExpectedResult(expected="ok"))

        assert score.value == pytest.approx(0.75)
        assert score.passed is False
        # No _threshold: nothing an adapter could use to rebuild this verdict.
        assert "_threshold" not in score.metadata


# ===================================================================
# Tasks 2.2 / 2.3 / 2.4 — AC 1, one test per adapter
# ===================================================================


class TestStateMatchVerdictSurvivesRoundTrip:
    """AC 1: StateMatch 3-of-4 must report value=0.75, passed=False."""

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_partial_state_match_reports_failure(self, run_adapter):
        from agent_evals.adapters.scorers.state import StateMatch

        examples = asyncio.run(
            run_adapter(
                _state_task,
                _state_dataset(),
                [StateMatch(expected_state=_EXPECTED_STATE)],
                "verdict-statematch",
            )
        )

        score = examples[0].scores["StateMatch"]
        assert score.value == pytest.approx(0.75)
        assert score.passed is False, (
            "StateMatch matched 3 of 4 paths and returned passed=False; the "
            "adapter must report that verdict, not recompute 0.75 >= 0.5."
        )


# ===================================================================
# Task 2.5 — the real guard for AC 2's intent
# ===================================================================


class TestNonMonotoneVerdict:
    """A verdict is not required to be a function of the value.

    No threshold can reproduce both of these, so this is the test that
    actually distinguishes 'report the scorer's verdict' from 'derive a
    better threshold'. See proposal.md's scope note on AC 2.
    """

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_high_value_failing_verdict_is_preserved(self, run_adapter):
        scorer = _make_verdict_scorer("StrictRubric", value=0.9, passed=False)

        examples = asyncio.run(
            run_adapter(
                lambda _x: TaskResult(output="hello"),
                _state_dataset(),
                [scorer],
                "verdict-high-fail",
            )
        )

        score = examples[0].scores["StrictRubric"]
        assert score.value == pytest.approx(0.9)
        assert score.passed is False, (
            "0.9 with passed=False must stay failing; a disqualifying "
            "criterion can fail despite a high aggregate value."
        )

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_low_value_passing_verdict_is_preserved(self, run_adapter):
        scorer = _make_verdict_scorer("LenientRubric", value=0.1, passed=True)

        examples = asyncio.run(
            run_adapter(
                lambda _x: TaskResult(output="hello"),
                _state_dataset(),
                [scorer],
                "verdict-low-pass",
            )
        )

        score = examples[0].scores["LenientRubric"]
        assert score.value == pytest.approx(0.1)
        assert score.passed is True, (
            "0.1 with passed=True must stay passing; the scorer, not the "
            "adapter, decides what counts as a pass."
        )


# ===================================================================
# Task 2.6 — per-example distinctness
# ===================================================================


class TestPerExampleVerdicts:
    """One cache slot per scorer would report the last verdict for all."""

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_each_example_keeps_its_own_verdict(self, run_adapter):
        # Three examples, three different (value, passed) pairs. Values are
        # all on the same side of 0.5 as each other's verdicts are NOT, so a
        # threshold cannot reproduce the pattern.
        verdicts = {
            "out-a": (0.8, False),
            "out-b": (0.2, True),
            "out-c": (0.9, False),
        }
        scorer = _make_per_input_scorer("PerExample", verdicts)

        dataset = [
            ExampleData(input=key, expected=ExpectedResult(expected="ok"))
            for key in ("in-a", "in-b", "in-c")
        ]
        outputs = {"in-a": "out-a", "in-b": "out-b", "in-c": "out-c"}

        examples = asyncio.run(
            run_adapter(
                lambda x: TaskResult(output=outputs[str(x)]),
                dataset,
                [scorer],
                "verdict-per-example",
            )
        )

        assert len(examples) == 3
        observed = sorted(
            (ex.scores["PerExample"].value, ex.scores["PerExample"].passed)
            for ex in examples
        )
        assert observed == sorted(verdicts.values()), (
            "Each example must report its own verdict; a per-scorer cache "
            "slot would report one example's verdict for all three."
        )


# ===================================================================
# Task 2.7 — AC 2 satisfied literally
# ===================================================================


class TestBinaryScorerFamilyCoverage:
    """AC 2's literal requirement: a tool_calls-family scorer, no _threshold.

    NOTE: this test passes identically before and after the fix, so on its own
    it guards nothing. ``tool_calls`` only ever returns (1.0, True) or
    (0.0, False), and ``value >= 0.5`` reproduces both. It is here because AC 2
    asks for it; the real guard for AC 2's intent is
    ``TestNonMonotoneVerdict`` above. Do not read this test's green status as
    evidence that verdict preservation works.
    """

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_tool_call_exact_match_verdict_round_trips(self, run_adapter):
        from agent_evals.adapters.scorers.tool_calls import ToolCallExactMatch

        def _call(city: str) -> list[dict[str, Any]]:
            return [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "function": {
                                "name": "get_weather",
                                "arguments": f'{{"city": "{city}"}}',
                            }
                        }
                    ],
                }
            ]

        def task(_x):
            return TaskResult(output="answer", context={"outputs": _call("Berlin")})

        dataset = [
            ExampleData(
                input="weather?",
                expected=ExpectedResult(
                    expected="ok",
                    context={"reference_outputs": _call("Paris")},
                ),
            )
        ]

        examples = asyncio.run(
            run_adapter(task, dataset, [ToolCallExactMatch()], "verdict-toolcalls")
        )

        score = examples[0].scores["ToolCallExactMatch"]
        # Wrong city → args mismatch → (0.0, False). Binary either way.
        assert score.value == pytest.approx(0.0)
        assert score.passed is False
