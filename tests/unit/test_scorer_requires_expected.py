# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Every comparison scorer reports a missing reference instead of scoring one.

A scorer that compares an output against a reference has nothing to do
when the reference is absent. The shared autoevals wrappers used to
substitute an empty string — ``expected.expected if expected else ""`` —
so the comparison still ran and still returned a number. The caller read
that number as a verdict about their agent.

Three failure shapes came out of that substitution, and they are ordered
here worst-first:

1. **A false pass.** ``Battle`` asks an LLM "is the first response better
   than the second?" with the reference as the second. Against an empty
   second response the answer is yes, and the scorer reports **1.0**.
   ``Summary`` behaves the same way: asked which of an expert summary and
   a candidate better describes a text, with the expert summary blank,
   the judge picks the candidate. A scorer reporting success because no
   reference existed is worse than one reporting nothing.
2. **A false failure indistinguishable from a real one.** ``Levenshtein``
   against ``""`` yields a real similarity number, and ``0.0`` from a
   missing reference looked exactly like ``0.0`` from a wrong answer.
3. **A false pass by coincidence.** ``ExactMatch`` on an empty output
   *matched* the absent reference and reported 1.0.

The guarantee these tests pin: the wrapper reports the absence, the
underlying scorer is never called, and the result is distinguishable in
``metadata["error"]`` from a genuine low score. Scorers that genuinely do
not read a reference are unaffected, and that too is asserted — a blanket
guard would break the reference-free judges.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import pytest

from agent_evals.adapters.scorers import autoevals as autoevals_adapters
from agent_evals.adapters.scorers.autoevals import (
    _create_autoevals_async_scorer,
    _create_autoevals_scorer,
)
from agent_evals.core.types import ExpectedResult, TaskResult

_SHARED_WRAPPERS = {"_create_autoevals_scorer", "_create_autoevals_async_scorer"}

# Derived from what each wrapped autoevals class does with `expected`, NOT
# from the scorer's name. For the LLM judges the deciding evidence is
# whether the class's prompt template interpolates `{{expected}}`:
#
#   requires it  — factuality, sql, summary, translation, battle
#   ignores it   — closed_q_a, humor, security, possible
#
# `Battle` and `Summary` are the two that read reference-free from their
# names and are not; both are listed as requiring one because their
# templates place the reference in the comparison itself.
_EXPECTED_TRUTH_TABLE = {
    "Levenshtein": True,
    "ExactMatch": True,
    "EmbeddingSimilarity": True,
    "Factuality": True,
    "Sql": True,
    "Summary": True,
    "Translation": True,
    "Battle": True,
    "ClosedQA": False,
    "Humor": False,
    "Security": False,
    "Possible": False,
}


class _SpyScorer:
    """Stands in for an autoevals class, recording whether it was called."""

    calls: list[dict[str, Any]] = []

    def __init__(self, **_: Any) -> None:
        pass

    def __call__(self, **kwargs: Any):
        type(self).calls.append(kwargs)
        return _SpyResult()

    async def eval_async(self, **kwargs: Any):
        type(self).calls.append(kwargs)
        return _SpyResult()


class _SpyResult:
    score = 1.0
    metadata: dict[str, Any] = {}
    rationale = "spy"


@pytest.fixture(autouse=True)
def _reset_spy():
    _SpyScorer.calls = []
    yield
    _SpyScorer.calls = []


def _sync(requires_expected: bool):
    return _create_autoevals_scorer(
        _SpyScorer,
        scorer_name="Spy",
        threshold=0.5,
        requires_expected=requires_expected,
    )


def _async(requires_expected: bool):
    return _create_autoevals_async_scorer(
        _SpyScorer,
        scorer_name="SpyAsync",
        threshold=0.5,
        requires_expected=requires_expected,
    )


class TestDeclarationIsExplicit:
    """No factory may silently inherit the lenient default."""

    def test_every_shared_wrapper_call_site_declares_the_flag(self):
        """Parsed from the source, so a new factory cannot omit it quietly.

        The parameter defaults to ``False``, which is the right default for
        an internal helper but the wrong thing to inherit by accident: a
        new comparison scorer added without the keyword would score
        against an empty reference and nobody would be told. Reading the
        call sites out of the AST makes the omission a test failure at the
        moment it is written.
        """
        source = pathlib.Path(inspect.getfile(autoevals_adapters)).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        undeclared: list[str] = []
        call_count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = func.id if isinstance(func, ast.Name) else None
            if name not in _SHARED_WRAPPERS:
                continue
            call_count += 1
            keywords = {kw.arg for kw in node.keywords}
            if "requires_expected" not in keywords:
                scorer_name = next(
                    (
                        kw.value.value
                        for kw in node.keywords
                        if kw.arg == "scorer_name"
                        and isinstance(kw.value, ast.Constant)
                    ),
                    f"<line {node.lineno}>",
                )
                undeclared.append(str(scorer_name))

        assert call_count == len(_EXPECTED_TRUTH_TABLE), (
            f"expected {len(_EXPECTED_TRUTH_TABLE)} shared-wrapper call sites, "
            f"found {call_count} — update the truth table alongside the code"
        )
        assert undeclared == [], (
            "these factories route through a shared wrapper without declaring "
            f"requires_expected: {undeclared}"
        )

    def test_declared_values_match_the_derived_truth_table(self):
        """Pins each verdict so a flip has to be argued for, not typo'd in."""
        source = pathlib.Path(inspect.getfile(autoevals_adapters)).read_text(
            encoding="utf-8"
        )
        tree = ast.parse(source)

        declared: dict[str, bool] = {}
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not (
                isinstance(node.func, ast.Name) and node.func.id in _SHARED_WRAPPERS
            ):
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            name_node = kwargs.get("scorer_name")
            flag_node = kwargs.get("requires_expected")
            assert isinstance(name_node, ast.Constant)
            assert isinstance(flag_node, ast.Constant)
            declared[name_node.value] = flag_node.value

        assert declared == _EXPECTED_TRUTH_TABLE


class TestMissingReferenceIsReported:
    """The absence is named, and no comparison is attempted."""

    @pytest.mark.parametrize(
        "expected",
        [None, ExpectedResult(expected=""), ExpectedResult(expected="   ")],
        ids=["absent", "empty", "whitespace"],
    )
    def test_no_usable_reference_reports_the_cause(self, expected):
        score = _sync(requires_expected=True)(TaskResult(output="an answer"), expected)

        assert score.passed is False
        assert score.value == 0.0
        assert score.metadata["error"] == "Expected value is required for Spy"
        assert score.metadata["expected"] is None
        assert "requires an expected value" in (score.reasoning or "")

    def test_the_underlying_scorer_is_never_called(self):
        """The substituted comparison must not happen at all.

        Guarding *after* the call would still produce a correct-looking
        Score while having asked an LLM to judge against nothing — a real
        billed request whose answer is discarded.
        """
        _sync(requires_expected=True)(TaskResult(output="an answer"), None)

        assert _SpyScorer.calls == []

    async def test_async_guard_precedes_the_model_call(self):
        score = await _async(requires_expected=True)(
            TaskResult(output="an answer"), None
        )

        assert _SpyScorer.calls == [], "no LLM request may be issued"
        assert score.metadata["error"] == "Expected value is required for SpyAsync"

    def test_result_is_distinguishable_from_a_genuine_low_score(self):
        """Both are 0.0; only one carries a cause.

        Before this change the two were byte-identical, so a caller could
        not tell "your agent answered wrongly" from "you gave me nothing
        to compare against" — and the second is a harness bug, not a
        finding about the agent.
        """
        missing = _sync(requires_expected=True)(TaskResult(output="wrong"), None)
        genuine = _sync(requires_expected=True)(
            TaskResult(output="wrong"), ExpectedResult(expected="right")
        )

        assert missing.value == 0.0 and missing.passed is False
        assert "error" in missing.metadata
        assert "error" not in genuine.metadata
        assert len(_SpyScorer.calls) == 1, "only the genuine comparison ran"

    def test_two_comparison_scorers_behave_consistently(self):
        """One convention, not one per factory."""
        first = _create_autoevals_scorer(
            _SpyScorer, scorer_name="First", threshold=0.5, requires_expected=True
        )
        second = _create_autoevals_scorer(
            _SpyScorer, scorer_name="Second", threshold=0.5, requires_expected=True
        )

        a = first(TaskResult(output="o"), None)
        b = second(TaskResult(output="o"), None)

        assert a.metadata.keys() == b.metadata.keys()
        assert (a.value, a.passed) == (b.value, b.passed)
        assert a.metadata["error"] != b.metadata["error"], "each names itself"

    def test_report_matches_the_convention_the_explicit_guards_already_use(self):
        """Same shape as NumericDiff / ListContains / JSONDiff.

        Those three already guarded by hand. A second convention for the
        same condition would mean a caller aggregating on
        ``metadata["error"]`` catches some scorers and not others.
        """
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        existing = NumericDiff()(TaskResult(output="5"), None)
        new = _sync(requires_expected=True)(TaskResult(output="5"), None)

        assert existing.value == new.value
        assert existing.passed == new.passed
        assert existing.metadata["error"].startswith("Expected value is required for")
        assert new.metadata["error"].startswith("Expected value is required for")
        assert "requires an expected value for comparison" in (existing.reasoning or "")
        assert "requires an expected value for comparison" in (new.reasoning or "")


class TestReferenceFreeScorersAreUnaffected:
    """A blanket guard would break the judges that read output alone."""

    def test_reference_free_scorer_scores_normally_without_a_reference(self):
        score = _sync(requires_expected=False)(TaskResult(output="funny"), None)

        assert score.value == 1.0
        assert score.passed is True
        assert "error" not in score.metadata
        assert len(_SpyScorer.calls) == 1

    def test_reference_free_scorer_reports_no_missing_expected_condition(self):
        score = _sync(requires_expected=False)(TaskResult(output="funny"), None)

        assert "Expected value is required" not in str(score.metadata)
        assert "requires an expected value" not in (score.reasoning or "")

    async def test_reference_free_async_scorer_still_receives_an_empty_reference(self):
        """The call shape passes ``expected=`` unconditionally.

        These scorers ignore it — their templates never interpolate it —
        so ``""`` is the honest value to send rather than a reason to
        change the call.
        """
        await _async(requires_expected=False)(TaskResult(output="funny"), None)

        assert _SpyScorer.calls[0]["expected"] == ""

    def test_mixed_scorer_set_yields_one_score_and_one_named_failure(self):
        """The same example, scored by both kinds.

        This is the trace-sourced case end to end: an export with no answer
        key is still fully useful to the reference-free scorers, and the
        comparison scorers say why they abstained rather than dragging the
        aggregate down with fabricated zeros.
        """
        comparison = _create_autoevals_scorer(
            _SpyScorer, scorer_name="Comparison", threshold=0.5, requires_expected=True
        )
        reference_free = _create_autoevals_scorer(
            _SpyScorer,
            scorer_name="ReferenceFree",
            threshold=0.5,
            requires_expected=False,
        )
        example = TaskResult(output="an answer")

        scores = [comparison(example, None), reference_free(example, None)]

        by_name = {s.name: s for s in scores}
        assert by_name["Comparison"].metadata["error"] == (
            "Expected value is required for Comparison"
        )
        assert by_name["ReferenceFree"].passed is True
        assert "error" not in by_name["ReferenceFree"].metadata
        assert len(_SpyScorer.calls) == 1, "only the reference-free scorer ran"


class TestPublicFactoriesHonourTheirDeclaration:
    """The wiring reaches the public factories, not just the internal helper."""

    def test_exact_match_no_longer_passes_on_an_absent_reference(self):
        """The coincidental false pass, pinned.

        ``ExactMatch`` on an empty output against a substituted empty
        reference compared equal and reported 1.0 — a pass earned by there
        being nothing on either side.
        """
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        score = ExactMatch()(TaskResult(output=""), None)

        assert score.passed is False
        assert score.value == 0.0
        assert score.metadata["error"] == "Expected value is required for ExactMatch"

    def test_levenshtein_reports_absence_rather_than_a_similarity(self):
        from agent_evals.adapters.scorers.autoevals import Levenshtein

        score = Levenshtein()(TaskResult(output="an answer"), None)

        assert score.metadata["error"] == "Expected value is required for Levenshtein"

    async def test_battle_does_not_report_a_pass_without_a_reference(self):
        """The worst case: a 1.0 from an empty second response.

        Reached without a network call because the guard runs before the
        scorer is instantiated, which is itself part of the contract.
        """
        from agent_evals.adapters.scorers.autoevals import Battle

        score = await Battle(model="gpt-4o-mini")(TaskResult(output="a solution"), None)

        assert score.passed is False
        assert score.value == 0.0
        assert score.metadata["error"] == "Expected value is required for Battle"

    async def test_summary_does_not_report_a_pass_without_a_reference(self):
        from agent_evals.adapters.scorers.autoevals import Summary

        score = await Summary(model="gpt-4o-mini")(TaskResult(output="a summary"), None)

        assert score.passed is False
        assert score.metadata["error"] == "Expected value is required for Summary"
