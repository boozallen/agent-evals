# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Verdict preservation changed nothing that a platform receives.

The rejected alternative to the in-process cache was to smuggle the verdict
through a vendor SDK field — MLflow ``Feedback.metadata``, langfuse
``Evaluation.metadata``, or a dict-shaped braintrust return. That approach was
rejected because its load-bearing assumption cannot be tested without live
credentials for all three platforms. These tests pin the consequence: each
scorer wrapper's return value is byte-for-byte the shape it was before, so no
reviewer has to diff the adapters to confirm the ingest path is untouched.

Two things are asserted:

- **Shape** — braintrust returns a bare ``float``; mlflow returns a ``Feedback``
  carrying only name/value/rationale; langfuse returns an ``Evaluation``
  carrying only name/value/comment. Every optional metadata field is left
  ``None``, which is what makes "the verdict does not ride the payload"
  checkable rather than merely claimed.
- **Prose is not a wire format** — langfuse's ``comment=f"Passed: {...}"``
  predates this change and stays human-readable. No reconstruction path parses
  it, proven by feeding the reconstruction a comment that contradicts the
  value and watching it be ignored.

Companion to ``test_scorer_verdict_preservation.py`` (the verdict survives),
``test_verdict_fallback.py`` (what happens when it does not) and
``test_verdict_cache.py`` (the cache alone).
"""

from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest

import agent_evals.adapters.platforms.braintrust as braintrust_module
import agent_evals.adapters.platforms.langfuse as langfuse_module
import agent_evals.adapters.platforms.mlflow as mlflow_module
from agent_evals.core.types import ExampleData, ExpectedResult, Score, TaskResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _scorer(name: str, value: float, passed: bool):
    """A scorer whose verdict a threshold cannot reproduce.

    ``value=0.9, passed=False`` is the interesting case: if an adapter were
    smuggling the verdict into a payload field, that is the payload that would
    have to carry it.
    """

    def scorer(result: TaskResult, expected: ExpectedResult | None = None, **kwargs):
        _ = result, expected, kwargs  # vulture: intentionally unused
        return Score(name=name, value=value, passed=passed)

    scorer.__name__ = name
    scorer.__qualname__ = name
    return scorer


def _make_async_scorer(name: str, value: float, passed: bool):
    """The async counterpart of :func:`_scorer` — a separate wrapper branch."""

    async def scorer(
        result: TaskResult, expected: ExpectedResult | None = None, **kwargs
    ):
        _ = result, expected, kwargs  # vulture: intentionally unused
        return Score(name=name, value=value, passed=passed)

    scorer.__name__ = name
    scorer.__qualname__ = name
    return scorer


# ===================================================================
# Braintrust — the wrapper still returns a bare float
# ===================================================================


class TestBraintrustReturnShape:
    """``float(score.value)`` and nothing else.

    ``tests/helpers/fake_braintrust.py`` calls ``float(ev)`` on whatever a
    wrapper returns, and the real SDK is no more forgiving. A dict-shaped
    return — the SDK-metadata approach — would break both.
    """

    def test_sync_wrapper_returns_a_bare_float(self):
        adapter = braintrust_module.BraintrustPlatform()
        (wrapper,) = adapter._convert_to_platform_scorer(
            [_scorer("Rubric", 0.9, False)]
        )

        returned = wrapper(TaskResult(output="a"), ExpectedResult(expected="gold"))

        assert type(returned) is float, (
            "Braintrust ingests the scorer's return directly; anything but a "
            "float changes what the platform receives."
        )
        assert returned == pytest.approx(0.9)

    def test_async_wrapper_returns_a_bare_float(self):
        adapter = braintrust_module.BraintrustPlatform()
        (wrapper,) = adapter._convert_to_platform_scorer(
            [_make_async_scorer("Rubric", 0.9, False)]
        )

        returned = asyncio.run(
            wrapper(TaskResult(output="a"), ExpectedResult(expected="gold"))
        )

        assert type(returned) is float
        assert returned == pytest.approx(0.9)

    def test_the_failing_verdict_was_recorded_without_riding_the_return(self):
        """The verdict is recoverable, and the float carries no trace of it."""
        from agent_evals.adapters.platforms.utils import verdict_key

        adapter = braintrust_module.BraintrustPlatform()
        (wrapper,) = adapter._convert_to_platform_scorer(
            [_scorer("Rubric", 0.9, False)]
        )

        task_result = TaskResult(output="a")
        expected = ExpectedResult(expected="gold")
        returned = wrapper(task_result, expected)

        key = verdict_key(task_result, expected)
        assert adapter._verdicts.lookup("Rubric", key, 0.9) is False
        assert returned == pytest.approx(0.9)


# ===================================================================
# MLflow — the wrapper still returns Feedback(name, value, rationale)
# ===================================================================


class TestMLflowReturnShape:
    """``Feedback`` carries the value and the rationale; metadata stays None."""

    def _feedback(self, value: float, passed: bool, reasoning: str | None = None):
        adapter = mlflow_module.MLflowPlatform()

        def scorer(result, expected=None, **kwargs):
            _ = result, expected, kwargs  # vulture: intentionally unused
            return Score(name="Rubric", value=value, passed=passed, reasoning=reasoning)

        scorer.__name__ = "Rubric"
        (spec,) = adapter._convert_to_platform_scorer([scorer])
        return spec.fn(outputs=TaskResult(output="a"), inputs={"question": "q"})

    def test_returns_a_feedback_with_only_name_value_rationale(self):
        from mlflow.genai.evaluation.entities import Feedback

        feedback = self._feedback(0.9, False, reasoning="three of four paths")

        assert isinstance(feedback, Feedback)
        assert feedback.name == "Rubric"
        assert feedback.value == pytest.approx(0.9)
        assert feedback.rationale == "three of four paths"

    def test_feedback_metadata_is_untouched(self):
        feedback = self._feedback(0.9, False)

        assert feedback.metadata is None, (
            "Feedback.metadata is the field the rejected SDK approach would "
            "have used. It must stay empty: the verdict is kept in-process."
        )

    #: ``Feedback`` stamps these itself at construction, so two payloads built
    #: microseconds apart differ on them whenever the clock ticks in between.
    #: They say nothing about the verdict, so they are excluded rather than
    #: allowed to make this a time-dependent test.
    _SDK_STAMPED_FIELDS = ("create_time", "last_update_time")

    def test_the_failing_verdict_is_absent_from_the_feedback(self):
        """No field on the payload distinguishes passed=False from passed=True."""
        failing = self._feedback(0.9, False).to_dictionary()
        passing = self._feedback(0.9, True).to_dictionary()

        for field in self._SDK_STAMPED_FIELDS:
            assert field in failing, (
                f"{field} is excluded below because MLflow stamps it. If the "
                "SDK stopped emitting it, drop it from _SDK_STAMPED_FIELDS "
                "rather than leaving a comparison narrower than it needs to be."
            )
            failing.pop(field)
            passing.pop(field)

        assert failing == passing, (
            "Two scores differing only in `passed` must produce identical "
            "MLflow payloads — that is what 'the platform receives the same "
            "thing as before' means."
        )


# ===================================================================
# Langfuse — the wrapper still returns Evaluation(name, value, comment)
# ===================================================================


class TestLangfuseReturnShape:
    """``Evaluation`` carries the value plus the pre-existing prose comment."""

    def _evaluation(self, value: float, passed: bool):
        adapter = langfuse_module.LangFusePlatform()
        (wrapper,) = adapter._convert_to_platform_scorer(
            [_scorer("Rubric", value, passed)]
        )
        return wrapper(input="q", output=TaskResult(output="a"), expected_output="gold")

    def test_returns_an_evaluation_with_the_value_and_the_prose_comment(self):
        from langfuse import Evaluation

        evaluation = self._evaluation(0.9, False)

        assert isinstance(evaluation, Evaluation)
        assert evaluation.name == "Rubric"
        assert evaluation.value == pytest.approx(0.9)
        # Pre-existing prose, unchanged by this change.
        assert evaluation.comment == "Passed: False"

    def test_evaluation_metadata_is_untouched(self):
        evaluation = self._evaluation(0.9, False)

        assert evaluation.metadata is None, (
            "Evaluation.metadata is the field the rejected SDK approach would "
            "have used. It must stay empty."
        )
        assert evaluation.data_type is None
        assert evaluation.config_id is None


# ===================================================================
# The langfuse comment is prose, not a wire format
# ===================================================================


class TestLangfuseCommentIsNotParsed:
    """``comment=f"Passed: {score.passed}"`` is for human readers only.

    It is the one place a verdict is already visible in a platform payload, so
    it is the obvious shortcut — and the wrong one: a comment is free text a
    user or the SDK may rewrite. These tests prove the reconstruction path
    ignores it by handing it a comment that contradicts the value.
    """

    def _examples(self, value: float, comment: str):
        adapter = langfuse_module.LangFusePlatform()
        platform_result = SimpleNamespace(
            item_results=[
                SimpleNamespace(
                    item={"input": "q", "expected_output": "gold"},
                    output="a",
                    evaluations=[
                        SimpleNamespace(name="Rubric", value=value, comment=comment)
                    ],
                    trace_id=None,
                    dataset_run_id=None,
                )
            ],
        )
        dataset = [ExampleData(input="q", expected=ExpectedResult(expected="gold"))]
        return adapter._build_examples_from_platform_result(
            platform_result,  # type: ignore[arg-type]
            dataset,
        )

    def test_a_passing_comment_does_not_override_a_failing_value(self):
        # Cache is empty (nothing went through a wrapper), so the threshold
        # comparison owns the verdict: 0.2 < 0.5 → False. If the comment were
        # parsed, this would report True.
        examples = self._examples(0.2, comment="Passed: True")

        assert examples[0].scores["Rubric"].passed is False

    def test_a_failing_comment_does_not_override_a_passing_value(self):
        examples = self._examples(0.8, comment="Passed: False")

        assert examples[0].scores["Rubric"].passed is True

    def test_garbled_prose_is_harmless(self):
        examples = self._examples(0.8, comment="passed?? nope -- Passed: False!!")

        assert examples[0].scores["Rubric"].passed is True

    def test_a_missing_comment_is_harmless(self):
        adapter = langfuse_module.LangFusePlatform()
        platform_result = SimpleNamespace(
            item_results=[
                SimpleNamespace(
                    item={"input": "q", "expected_output": "gold"},
                    output="a",
                    # No comment attribute at all.
                    evaluations=[SimpleNamespace(name="Rubric", value=0.8)],
                    trace_id=None,
                    dataset_run_id=None,
                )
            ],
        )
        dataset = [ExampleData(input="q", expected=ExpectedResult(expected="gold"))]

        examples = adapter._build_examples_from_platform_result(
            platform_result,  # type: ignore[arg-type]
            dataset,
        )

        assert examples[0].scores["Rubric"].passed is True


# ===================================================================
# Structural backstop: no adapter reads a verdict off a payload field
# ===================================================================


class TestNoAdapterParsesAPayloadForTheVerdict:
    """A grep-style guard, so a future edit cannot quietly add the shortcut.

    The behavioural tests above cover the paths that exist today. This one
    covers the paths someone might add tomorrow: reading ``.comment`` back, or
    reaching into ``Feedback.metadata`` / ``Evaluation.metadata``.
    """

    @pytest.mark.parametrize(
        "module",
        [
            pytest.param(braintrust_module, id="braintrust"),
            pytest.param(mlflow_module, id="mlflow"),
            pytest.param(langfuse_module, id="langfuse"),
        ],
    )
    def test_no_reconstruction_path_reads_a_comment(self, module):
        source = Path(inspect.getfile(module)).read_text(encoding="utf-8")

        # The only permitted mentions of `comment` are the two Evaluation
        # kwargs on the write side and the comments explaining them.
        reads = [
            line
            for line in source.splitlines()
            if "comment" in line
            and "comment=" not in line
            and not line.lstrip().startswith("#")
        ]
        assert reads == [], (
            "A reconstruction path appears to read a comment field. The "
            f"langfuse comment is human-readable prose: {reads}"
        )

    @pytest.mark.parametrize(
        ("module", "sdk_type", "allowed_kwargs"),
        [
            pytest.param(
                mlflow_module,
                "Feedback",
                {"name", "value", "rationale"},
                id="mlflow-Feedback",
            ),
            pytest.param(
                langfuse_module,
                "Evaluation",
                {"name", "value", "comment"},
                id="langfuse-Evaluation",
            ),
        ],
    )
    def test_sdk_payloads_are_built_from_the_same_kwargs_as_before(
        self, module, sdk_type, allowed_kwargs
    ):
        """Parse the adapter and check every SDK payload construction.

        A string search cannot do this — the calls span several lines — so walk
        the AST instead. Adding ``metadata=`` to either call is exactly the
        rejected SDK-metadata approach, and it would fail here.
        """
        tree = ast.parse(Path(inspect.getfile(module)).read_text(encoding="utf-8"))

        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == sdk_type
        ]
        assert calls, f"No {sdk_type}(...) construction found in {module.__name__}"

        for call in calls:
            assert not call.args, (
                f"{sdk_type} is constructed positionally at line {call.lineno}; "
                "keyword-only construction is what makes this check meaningful."
            )
            kwargs = {kw.arg for kw in call.keywords}
            assert kwargs == allowed_kwargs, (
                f"{sdk_type}(...) at line {call.lineno} is built from {kwargs}, "
                f"expected exactly {allowed_kwargs}. A new field here changes "
                "what the platform receives; the verdict is kept in-process "
                "instead (design.md Decision 1)."
            )
