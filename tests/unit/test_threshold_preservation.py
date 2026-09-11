# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for scorer threshold preservation across platform adapters.

Regression tests for the bug where platform adapters (Braintrust, MLflow,
Langfuse) rebuilt Score objects using a hardcoded 0.5 threshold instead of
the scorer-configured threshold, causing scores like 0.996 to incorrectly
pass when the scorer required threshold=1.0.

The fix stores `_threshold` in Score.metadata at the scorer level, captures
it via `self._scorer_thresholds` in each adapter's wrapper, and uses it when
rebuilding Score objects from platform results.
"""

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_evals.core.types import ExpectedResult, Score, TaskResult

# ---------------------------------------------------------------------------
# Helpers: spy scorers that return configurable Score with _threshold metadata
# ---------------------------------------------------------------------------


def _make_sync_scorer(name: str, value: float, threshold: float):
    """Build a sync scorer returning a Score with the given value and _threshold."""

    def scorer(result: TaskResult, expected: ExpectedResult | None = None, **kwargs):
        return Score(
            name=name,
            value=value,
            passed=value >= threshold,
            metadata={"_threshold": threshold},
        )

    scorer.__name__ = name
    scorer.__qualname__ = name
    return scorer


def _make_async_scorer(name: str, value: float, threshold: float):
    """Build an async scorer returning a Score with the given value and _threshold."""

    async def scorer(
        result: TaskResult, expected: ExpectedResult | None = None, **kwargs
    ):
        return Score(
            name=name,
            value=value,
            passed=value >= threshold,
            metadata={"_threshold": threshold},
        )

    scorer.__name__ = name
    scorer.__qualname__ = name
    return scorer


# ===================================================================
# Braintrust adapter
# ===================================================================


@pytest.mark.requires_braintrust
class TestBraintrustThresholdPreservation:
    """Braintrust adapter must honour scorer thresholds, not hardcoded 0.5."""

    @pytest.mark.asyncio
    async def test_score_below_custom_threshold_fails(self):
        """Score of 0.996 must FAIL when the scorer requires threshold=1.0."""
        from helpers.fake_braintrust import FakeBraintrustClient

        from agent_evals.adapters.platforms.braintrust import (
            BraintrustConfig,
            BraintrustPlatform,
        )
        from agent_evals.core.types import ExampleData, ExpectedResult, TaskResult

        scorer = _make_sync_scorer("JSONDiff", value=0.996, threshold=1.0)

        fake = FakeBraintrustClient()
        adapter = BraintrustPlatform(client=fake)

        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            result = await adapter.aevaluate(
                task=lambda _x: TaskResult(output="hello"),
                dataset=[
                    ExampleData(input="hi", expected=ExpectedResult(expected="x"))
                ],
                evaluators=[scorer],
                platform=BraintrustConfig(project="p", experiment="threshold-below"),
            )

        score = result.examples[0].scores["JSONDiff"]
        assert score.value == pytest.approx(0.996)
        assert score.passed is False, (
            "0.996 must fail against a threshold of 1.0 — see issue #205's "
            "Braintrust threshold-preservation contract."
        )

    @pytest.mark.asyncio
    async def test_score_at_threshold_passes(self):
        """Score exactly equal to the threshold must pass (>= not >)."""
        from helpers.fake_braintrust import FakeBraintrustClient

        from agent_evals.adapters.platforms.braintrust import (
            BraintrustConfig,
            BraintrustPlatform,
        )
        from agent_evals.core.types import ExampleData, ExpectedResult, TaskResult

        scorer = _make_sync_scorer("Levenshtein", value=0.95, threshold=0.95)

        fake = FakeBraintrustClient()
        adapter = BraintrustPlatform(client=fake)

        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            result = await adapter.aevaluate(
                task=lambda _x: TaskResult(output="hello"),
                dataset=[
                    ExampleData(input="hi", expected=ExpectedResult(expected="x"))
                ],
                evaluators=[scorer],
                platform=BraintrustConfig(project="p", experiment="threshold-equal"),
            )

        assert result.examples[0].scores["Levenshtein"].passed is True

    @pytest.mark.asyncio
    async def test_unknown_scorer_falls_back_to_default_threshold(self):
        """A scorer that emits no _threshold uses PASS_THRESHOLD (0.5)."""
        from helpers.fake_braintrust import FakeBraintrustClient

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

        def scorer_no_threshold(result, expected=None, **_kwargs):
            return Score(name="CustomScorer", value=0.6, passed=True)

        scorer_no_threshold.__name__ = "CustomScorer"
        scorer_no_threshold.__qualname__ = "CustomScorer"

        fake = FakeBraintrustClient()
        adapter = BraintrustPlatform(client=fake)

        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            result = await adapter.aevaluate(
                task=lambda _x: TaskResult(output="hello"),
                dataset=[
                    ExampleData(input="hi", expected=ExpectedResult(expected="x"))
                ],
                evaluators=[scorer_no_threshold],
                platform=BraintrustConfig(project="p", experiment="threshold-default"),
            )

        # 0.6 >= 0.5 (default) → passes
        assert result.examples[0].scores["CustomScorer"].passed is True

    @pytest.mark.asyncio
    async def test_async_scorer_threshold_metadata_flows_to_passed(self):
        """Async scorer wrapper path must also honour _threshold metadata."""
        from helpers.fake_braintrust import FakeBraintrustClient

        from agent_evals.adapters.platforms.braintrust import (
            BraintrustConfig,
            BraintrustPlatform,
        )
        from agent_evals.core.types import ExampleData, ExpectedResult, TaskResult

        scorer = _make_async_scorer("Factuality", value=0.8, threshold=0.9)

        fake = FakeBraintrustClient()
        adapter = BraintrustPlatform(client=fake)

        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            result = await adapter.aevaluate(
                task=lambda _x: TaskResult(output="hello"),
                dataset=[
                    ExampleData(input="hi", expected=ExpectedResult(expected="x"))
                ],
                evaluators=[scorer],
                platform=BraintrustConfig(project="p", experiment="threshold-async"),
            )

        # 0.8 < 0.9 → fails
        assert result.examples[0].scores["Factuality"].passed is False

    @pytest.mark.asyncio
    async def test_threshold_capture_persists_across_items(self):
        """A threshold learned from one item's scorer-metadata applies to subsequent items."""
        from helpers.fake_braintrust import FakeBraintrustClient

        from agent_evals.adapters.platforms.braintrust import (
            BraintrustConfig,
            BraintrustPlatform,
        )
        from agent_evals.core.types import ExampleData, ExpectedResult, TaskResult

        scorer = _make_sync_scorer("JSONDiff", value=0.996, threshold=1.0)

        fake = FakeBraintrustClient()
        adapter = BraintrustPlatform(client=fake)

        with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
            result = await adapter.aevaluate(
                task=lambda _x: TaskResult(output="hello"),
                dataset=[
                    ExampleData(input="a", expected=ExpectedResult(expected="x")),
                    ExampleData(input="b", expected=ExpectedResult(expected="y")),
                ],
                evaluators=[scorer],
                platform=BraintrustConfig(project="p", experiment="threshold-persist"),
            )

        # Both items reflect the captured 1.0 threshold (0.996 < 1.0 → fail).
        assert result.examples[0].scores["JSONDiff"].passed is False
        assert result.examples[1].scores["JSONDiff"].passed is False


@pytest.mark.requires_braintrust
class TestBraintrustScoreNormalization:
    """Direct unit tests for BraintrustPlatform._convert_case_scores raw-score shapes.

    These cover the string/dict/below-default-float branches that the
    end-to-end FakeBraintrustClient-driven tests cannot reach. The fake
    float-coerces every scorer return before the value enters
    _convert_case_scores, so the wrapper-driven test path only exercises
    the int/float branch. These three tests reach the production
    branches directly, since _convert_case_scores normalizes raw scores
    that the real Braintrust SDK can emit in any of these shapes.
    """

    def _get_adapter(self):
        # _convert_case_scores does not touch self._client, so the default
        # (no-arg) constructor is sufficient and keeps ty's typed-port
        # contract (`BraintrustClientPort | None`) clean.
        from agent_evals.adapters.platforms.braintrust import BraintrustPlatform

        return BraintrustPlatform()

    def test_string_score_uses_custom_threshold(self):
        """Numeric string scores must respect stored threshold (0.996 < 1.0 → fail)."""
        adapter = self._get_adapter()
        adapter._scorer_thresholds["JSONDiff"] = 1.0
        case = SimpleNamespace(scores={"JSONDiff": "0.996"})
        scores = adapter._convert_case_scores(case)
        assert scores["JSONDiff"].passed is False

    def test_dict_score_uses_custom_threshold(self):
        """Dict scores ({'score': value}) must respect stored threshold."""
        adapter = self._get_adapter()
        adapter._scorer_thresholds["JSONDiff"] = 1.0
        case = SimpleNamespace(scores={"JSONDiff": {"score": 0.996}})
        scores = adapter._convert_case_scores(case)
        assert scores["JSONDiff"].passed is False

    def test_float_below_default_threshold_fails(self):
        """Unknown scorer below default 0.5 threshold must fail."""
        adapter = self._get_adapter()
        case = SimpleNamespace(scores={"CustomScorer": 0.4})
        scores = adapter._convert_case_scores(case)
        assert scores["CustomScorer"].passed is False

    def test_none_value_is_silently_skipped(self):
        """A score with value=None is dropped from the result.

        Pins the silent-skip contract at braintrust.py:_convert_case_scores
        — if a future change starts emitting `Score(value=0.0)` instead,
        this test fails loudly.
        """
        adapter = self._get_adapter()
        case = SimpleNamespace(scores={"S": None})
        scores = adapter._convert_case_scores(case)
        assert "S" not in scores

    def test_non_numeric_string_is_silently_skipped(self):
        """Non-numeric strings outside {yes/no/true/false/pass/fail} are dropped."""
        adapter = self._get_adapter()
        case = SimpleNamespace(scores={"S": "maybe"})
        scores = adapter._convert_case_scores(case)
        assert "S" not in scores

    def test_unsupported_value_type_is_silently_skipped(self):
        """Unsupported value types (lists, dicts without a 'score' key, etc.) are dropped."""
        adapter = self._get_adapter()
        case = SimpleNamespace(scores={"S": [1, 2, 3]})
        scores = adapter._convert_case_scores(case)
        assert "S" not in scores


# ===================================================================
# MLflow adapter
# ===================================================================


@pytest.fixture(autouse=False)
def _mlflow_tracking_uri(monkeypatch):
    """Set MLFLOW_TRACKING_URI so MLflowPlatform._resolve_tracking_uri doesn't raise."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://x")


class TestMLflowThresholdPreservation:
    """MLflow adapter must honour scorer thresholds, not hardcoded 0.5.

    End-to-end tests driving aevaluate against FakeMlflowSession;
    assertions land on EvalResult.examples[*].scores[name].passed.
    """

    @pytest.fixture(autouse=True)
    def _set_mlflow_tracking_uri(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://x")

    def _build_dataset(self):
        from agent_evals.core.types import ExampleData, ExpectedResult

        return [ExampleData(input="x", expected=ExpectedResult(expected="y"))]

    def test_sync_scorer_below_custom_threshold_fails(self):
        """Sync scorer with _threshold=1.0 returning 0.996 yields passed=False."""
        from helpers.fake_mlflow import FakeMlflowSession

        from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform

        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)

        scorer = _make_sync_scorer("JSONDiff", value=0.996, threshold=1.0)

        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _x: "answer",
                dataset=self._build_dataset(),
                evaluators=[scorer],
                platform=MlflowConfig(experiment="th-1"),
            )
        )

        score = result.examples[0].scores["JSONDiff"]
        assert score.value == pytest.approx(0.996)
        assert score.passed is False

    def test_async_scorer_below_custom_threshold_fails(self):
        """Async scorer with _threshold=0.9 returning 0.8 yields passed=False."""
        from helpers.fake_mlflow import FakeMlflowSession

        from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform

        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)

        scorer = _make_async_scorer("Factuality", value=0.8, threshold=0.9)

        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _x: "answer",
                dataset=self._build_dataset(),
                evaluators=[scorer],
                platform=MlflowConfig(experiment="th-2"),
            )
        )

        score = result.examples[0].scores["Factuality"]
        assert score.value == pytest.approx(0.8)
        assert score.passed is False

    def test_score_exactly_at_threshold_passes(self):
        """Score equal to threshold must pass (>= not >)."""
        from helpers.fake_mlflow import FakeMlflowSession

        from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform

        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)

        scorer = _make_sync_scorer("Levenshtein", value=0.95, threshold=0.95)

        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _x: "answer",
                dataset=self._build_dataset(),
                evaluators=[scorer],
                platform=MlflowConfig(experiment="th-3"),
            )
        )

        score = result.examples[0].scores["Levenshtein"]
        assert score.passed is True

    def test_default_threshold_used_when_metadata_absent(self):
        """A scorer that omits _threshold metadata uses PASS_THRESHOLD (0.5)."""
        from helpers.fake_mlflow import FakeMlflowSession

        from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform
        from agent_evals.core.types import Score

        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)

        def NoMeta(  # noqa: N802
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            _ = result, expected  # vulture: intentionally unused
            return Score(name="NoMeta", value=0.6, passed=True)

        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _x: "answer",
                dataset=self._build_dataset(),
                evaluators=[NoMeta],
                platform=MlflowConfig(experiment="th-4"),
            )
        )

        score = result.examples[0].scores["NoMeta"]
        assert score.value == pytest.approx(0.6)
        # 0.6 >= 0.5 default threshold → passes
        assert score.passed is True


# ===================================================================
# Factory-built scorers
# ===================================================================
#
# The spy helpers above hand-set __name__/__qualname__, so an adapter that
# derives its threshold-cache key from __name__ accidentally agrees with one
# that looks the key up by declared name. Real factories don't: every scorer
# returned by an autoevals factory or by StateMatch is a nested closure
# literally named "scorer". These tests build their scorers the way callers
# do, which is the only shape that reaches the mismatch.


def _jsondiff_near_miss_dataset():
    """A JSONDiff(threshold=1.0) case scoring ~0.9166 — a near miss, not a match."""
    from agent_evals.core.types import ExampleData

    output = json.dumps({"name": "abcdef", "status": "ok"})
    expected = json.dumps({"name": "abcdeX", "status": "ok"})
    return TaskResult(output=output), [
        ExampleData(input="q", expected=ExpectedResult(expected=expected)),
    ]


class TestFactoryBuiltScorerThresholds:
    """A threshold declared by a registered factory must survive the round-trip.

    Regression tests: the MLflow adapter captured
    ``_scorer_thresholds`` keyed by ``evaluator.__name__`` (always the literal
    ``"scorer"`` for factory-built scorers) while looking it up by
    ``assessment.name`` (``"JSONDiff"``), so every declared threshold silently
    fell back to ``PASS_THRESHOLD``.
    """

    @pytest.fixture(autouse=True)
    def _set_mlflow_tracking_uri(self, monkeypatch):
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://x")

    def test_mlflow_honours_factory_declared_threshold(self):
        """JSONDiff(threshold=1.0) scoring ~0.9166 must fail through MLflow."""
        from helpers.fake_mlflow import FakeMlflowSession

        from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform
        from agent_evals.adapters.scorers.autoevals import JSONDiff

        output, dataset = _jsondiff_near_miss_dataset()

        adapter = MLflowPlatform(session=FakeMlflowSession())
        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _x: output,
                dataset=dataset,
                evaluators=[JSONDiff(threshold=1.0)],
                platform=MlflowConfig(experiment="factory-threshold"),
            )
        )

        score = result.examples[0].scores["JSONDiff"]
        assert score.value == pytest.approx(0.9166666666666667)
        assert score.passed is False, (
            "0.9166 must fail against the declared threshold of 1.0 — a pass "
            "means the threshold lookup missed and fell back to PASS_THRESHOLD."
        )

    def test_mlflow_threshold_cache_key_matches_reported_score_name(self):
        """The captured cache key must be the name the reconstruction path looks up.

        Pins the mismatch structurally rather than only through its symptom: a
        key of ``"scorer"`` alongside a reported score named ``"JSONDiff"``
        makes the lookup unreachable no matter what the values are.
        """
        from helpers.fake_mlflow import FakeMlflowSession

        from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform
        from agent_evals.adapters.scorers.autoevals import JSONDiff

        output, dataset = _jsondiff_near_miss_dataset()

        adapter = MLflowPlatform(session=FakeMlflowSession())
        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _x: output,
                dataset=dataset,
                evaluators=[JSONDiff(threshold=1.0)],
                platform=MlflowConfig(experiment="factory-key"),
            )
        )

        reported_names = set(result.examples[0].scores)
        assert set(adapter._scorer_thresholds) == reported_names
        assert adapter._scorer_thresholds["JSONDiff"] == pytest.approx(1.0)

    def test_mlflow_keeps_two_factory_scorers_on_distinct_keys(self):
        """Two factory-built scorers must not share one threshold cache slot.

        Both are closures named ``scorer``, so a ``__name__``-derived key put
        them in the same slot and whichever ran first decided the threshold for
        both. Their declared thresholds differ (1.0 vs 0.5) and so do the
        verdicts those thresholds produce, so a collision is observable.
        """
        from helpers.fake_mlflow import FakeMlflowSession

        from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform
        from agent_evals.adapters.scorers.autoevals import JSONDiff, Levenshtein

        output, dataset = _jsondiff_near_miss_dataset()

        adapter = MLflowPlatform(session=FakeMlflowSession())
        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _x: output,
                dataset=dataset,
                evaluators=[JSONDiff(threshold=1.0), Levenshtein(threshold=0.5)],
                platform=MlflowConfig(experiment="two-factories"),
            )
        )

        assert adapter._scorer_thresholds == {
            "JSONDiff": pytest.approx(1.0),
            "Levenshtein": pytest.approx(0.5),
        }

        scores = result.examples[0].scores
        assert scores["JSONDiff"].value == pytest.approx(0.9166666666666667)
        assert scores["JSONDiff"].passed is False
        assert scores["Levenshtein"].value == pytest.approx(0.9705882352941176)
        assert scores["Levenshtein"].passed is True

    def test_mlflow_falls_back_to_pass_threshold_when_none_declared(self):
        """A scorer declaring no ``_threshold`` still gets the documented default.

        ``StateMatch`` writes no ``_threshold``, so the ``PASS_THRESHOLD``
        fallback is what the cache records. Pinned on the cached threshold
        rather than on the resulting verdict, because whether recomputing that
        verdict is correct at all is a separate open question — this test
        guards that the fallback stays reachable, not that its outcome is
        right.
        """
        from helpers.fake_mlflow import FakeMlflowSession

        from agent_evals.adapters.platforms.mlflow import (
            PASS_THRESHOLD,
            MlflowConfig,
            MLflowPlatform,
        )
        from agent_evals.adapters.scorers.state import StateMatch
        from agent_evals.core.types import ExampleData

        expected_state = {"a": 1, "b": 2, "c": 3, "d": 4}
        output = TaskResult(
            output="done",
            context={"final_state": {"a": 1, "b": 2, "c": 3, "d": 99}},
        )
        dataset = [ExampleData(input="q", expected=ExpectedResult(expected="done"))]

        adapter = MLflowPlatform(session=FakeMlflowSession())
        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _x: output,
                dataset=dataset,
                evaluators=[StateMatch(expected_state=expected_state)],
                platform=MlflowConfig(experiment="no-declared-threshold"),
            )
        )

        assert adapter._scorer_thresholds == {"StateMatch": PASS_THRESHOLD}
        assert result.examples[0].scores["StateMatch"].value == pytest.approx(0.75)

    def test_mlflow_honours_threshold_when_score_name_differs_from_callable(self):
        """The cache key must follow the name the score is *reported* under.

        MLflow reports ``Feedback(name=score.name)`` — the string the scorer put
        in its own ``Score``, which need not equal the name derived from the
        callable. Deriving the cache key only from the callable leaves the
        ``assessment.name`` lookup missing whenever the two differ, which is the
        same silent fallback to ``PASS_THRESHOLD`` by another route.
        """
        from helpers.fake_mlflow import FakeMlflowSession

        from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform
        from agent_evals.core.types import ExampleData

        def StrictFactory(threshold: float = 1.0):  # noqa: N802
            """Derived name is ``StrictFactory``; reported name is ``strict_check``."""

            def scorer(
                result: TaskResult, expected: ExpectedResult | None = None
            ) -> Score:
                return Score(
                    name="strict_check",
                    value=0.9,
                    passed=False,
                    metadata={"_threshold": threshold},
                )

            return scorer

        dataset = [ExampleData(input="q", expected=ExpectedResult(expected="x"))]
        adapter = MLflowPlatform(session=FakeMlflowSession())
        result = asyncio.run(
            adapter.aevaluate(
                task=lambda _x: TaskResult(output="x"),
                dataset=dataset,
                evaluators=[StrictFactory(threshold=1.0)],
                platform=MlflowConfig(experiment="divergent-name"),
            )
        )

        score = result.examples[0].scores["strict_check"]
        assert score.value == pytest.approx(0.9)
        assert score.passed is False, (
            "0.9 must fail against the declared threshold of 1.0 — a pass means "
            "the cache was keyed by the callable's name while the lookup asked "
            "for the reported name."
        )
        assert adapter._scorer_thresholds["strict_check"] == pytest.approx(1.0)

    def test_mlflow_warns_when_two_scorers_share_one_reported_name(self, caplog):
        """A shared ``Score.name`` is unresolvable, so it must be reported.

        Both scorers report ``Feedback(name="shared_name")``, so the
        reconstruction path has nothing to tell them apart — it only sees
        ``assessment.name``. The first threshold cached wins and the second is
        dropped, meaning one scorer is judged against a threshold it never
        declared. The adapter cannot fix that, but it must not hide it.
        """
        import logging

        from helpers.fake_mlflow import FakeMlflowSession

        from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform
        from agent_evals.core.types import ExampleData

        dataset = [ExampleData(input="q", expected=ExpectedResult(expected="x"))]
        adapter = MLflowPlatform(session=FakeMlflowSession())

        with caplog.at_level(
            logging.WARNING, logger="agent_evals.adapters.platforms.mlflow"
        ):
            asyncio.run(
                adapter.aevaluate(
                    task=lambda _x: TaskResult(output="x"),
                    dataset=dataset,
                    evaluators=[
                        LenientShared(threshold=0.1),
                        StrictShared(threshold=1.0),
                    ],
                    platform=MlflowConfig(experiment="colliding-names"),
                )
            )

        assert any(
            "already has threshold" in record.message for record in caplog.records
        ), (
            "a second scorer reporting under an already-cached name must warn, "
            f"got {[r.message for r in caplog.records]}"
        )
        # First writer wins, and the distinct per-callable keys still hold each
        # scorer's own declared threshold.
        assert adapter._scorer_thresholds["shared_name"] == pytest.approx(0.1)
        assert adapter._scorer_thresholds["LenientShared"] == pytest.approx(0.1)
        assert adapter._scorer_thresholds["StrictShared"] == pytest.approx(1.0)


def LenientShared(threshold: float = 0.1):  # noqa: N802
    """Derived name ``LenientShared``; reports under ``shared_name``.

    Module-level, matching the registered factories: a factory nested inside a
    test method carries that method in its ``__qualname__``, which derives a
    different name than production ever would.
    """

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        return Score(
            name="shared_name",
            value=0.5,
            passed=True,
            metadata={"_threshold": threshold},
        )

    return scorer


def StrictShared(threshold: float = 1.0):  # noqa: N802
    """Derived name ``StrictShared``; reports under the same ``shared_name``."""

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        return Score(
            name="shared_name",
            value=0.5,
            passed=False,
            metadata={"_threshold": threshold},
        )

    return scorer


def scorer(result: TaskResult, expected: ExpectedResult | None = None):
    """A scorer whose name derivation is genuinely unresolvable.

    Module-level and literally named ``scorer``, so neither the closure
    freevar nor the ``__qualname__`` heuristic can recover anything better.
    Defined here rather than nested so its ``__qualname__`` carries no
    ``.<locals>.`` prefix to fall back on.
    """
    return Score(name="anonymous", value=1.0, passed=True, metadata={})


class TestIndeterminateScorerName:
    """Name derivation that cannot resolve must say so rather than pass silently."""

    def test_warns_when_derivation_falls_through(self, caplog):
        """Two such scorers collide on one cache key, so the fallthrough is logged."""
        import logging

        from agent_evals.adapters.platforms.utils import infer_scorer_name

        with caplog.at_level(
            logging.WARNING, logger="agent_evals.adapters.platforms.utils"
        ):
            derived = infer_scorer_name(scorer)

        assert derived == "scorer"
        assert any(
            "Could not determine a declared name" in record.message
            for record in caplog.records
        ), f"expected a fallthrough warning, got {[r.message for r in caplog.records]}"

    def test_no_warning_when_a_name_is_recoverable(self, caplog):
        """A factory-built scorer resolves, so it must not trip the warning."""
        import logging

        from agent_evals.adapters.platforms.utils import infer_scorer_name
        from agent_evals.adapters.scorers.autoevals import JSONDiff

        with caplog.at_level(
            logging.WARNING, logger="agent_evals.adapters.platforms.utils"
        ):
            derived = infer_scorer_name(JSONDiff(threshold=1.0))

        assert derived == "JSONDiff"
        assert caplog.records == []


# ===================================================================
# Langfuse adapter
# ===================================================================


class TestLangfuseThresholdPreservation:
    """Langfuse adapter must honour scorer thresholds, not hardcoded 0.5."""

    @pytest.mark.asyncio
    async def test_score_below_custom_threshold_fails(self):
        """Score of 0.996 must FAIL when the scorer requires threshold=1.0."""
        from helpers.fake_langfuse import FakeLangfuseClient

        from agent_evals.adapters.platforms.langfuse import (
            LangfuseConfig,
            LangFusePlatform,
        )
        from agent_evals.core.types import ExampleData, ExpectedResult, TaskResult

        scorer = _make_sync_scorer("JSONDiff", value=0.996, threshold=1.0)

        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        result = await adapter.aevaluate(
            task=lambda x: TaskResult(output="hello"),
            dataset=[ExampleData(input="hi", expected=ExpectedResult(expected="x"))],
            evaluators=[scorer],
            platform=LangfuseConfig(experiment="threshold-below"),
        )

        score = result.examples[0].scores["JSONDiff"]
        assert score.value == pytest.approx(0.996)
        assert score.passed is False, (
            "0.996 must fail against a threshold of 1.0 — see issue #201's "
            "Langfuse threshold-preservation contract."
        )

    @pytest.mark.asyncio
    async def test_score_at_threshold_passes(self):
        """Score exactly equal to the threshold must pass (>= not >)."""
        from helpers.fake_langfuse import FakeLangfuseClient

        from agent_evals.adapters.platforms.langfuse import (
            LangfuseConfig,
            LangFusePlatform,
        )
        from agent_evals.core.types import ExampleData, ExpectedResult, TaskResult

        scorer = _make_sync_scorer("Levenshtein", value=0.95, threshold=0.95)

        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        result = await adapter.aevaluate(
            task=lambda x: TaskResult(output="hello"),
            dataset=[ExampleData(input="hi", expected=ExpectedResult(expected="x"))],
            evaluators=[scorer],
            platform=LangfuseConfig(experiment="threshold-equal"),
        )

        assert result.examples[0].scores["Levenshtein"].passed is True

    @pytest.mark.asyncio
    async def test_unknown_scorer_falls_back_to_default_threshold(self):
        """A scorer that emits no _threshold uses PASS_THRESHOLD (0.5)."""
        from helpers.fake_langfuse import FakeLangfuseClient

        from agent_evals.adapters.platforms.langfuse import (
            LangfuseConfig,
            LangFusePlatform,
        )
        from agent_evals.core.types import (
            ExampleData,
            ExpectedResult,
            Score,
            TaskResult,
        )

        def scorer_no_threshold(
            result: TaskResult, expected: ExpectedResult | None = None, **kwargs
        ):
            # No _threshold in metadata — adapter must fall back to 0.5.
            return Score(name="CustomScorer", value=0.6, passed=True)

        scorer_no_threshold.__name__ = "CustomScorer"
        scorer_no_threshold.__qualname__ = "CustomScorer"

        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        result = await adapter.aevaluate(
            task=lambda x: TaskResult(output="hello"),
            dataset=[ExampleData(input="hi", expected=ExpectedResult(expected="x"))],
            evaluators=[scorer_no_threshold],
            platform=LangfuseConfig(experiment="threshold-default"),
        )

        # 0.6 >= 0.5 (default) → passes
        assert result.examples[0].scores["CustomScorer"].passed is True

    @pytest.mark.asyncio
    async def test_async_scorer_threshold_metadata_flows_to_passed(self):
        """The async scorer wrapper path must also honour _threshold metadata."""
        from helpers.fake_langfuse import FakeLangfuseClient

        from agent_evals.adapters.platforms.langfuse import (
            LangfuseConfig,
            LangFusePlatform,
        )
        from agent_evals.core.types import ExampleData, ExpectedResult, TaskResult

        scorer = _make_async_scorer("Factuality", value=0.8, threshold=0.9)

        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        result = await adapter.aevaluate(
            task=lambda x: TaskResult(output="hello"),
            dataset=[ExampleData(input="hi", expected=ExpectedResult(expected="x"))],
            evaluators=[scorer],
            platform=LangfuseConfig(experiment="threshold-async"),
        )

        # 0.8 < 0.9 → fails despite the scorer reporting passed=False itself
        assert result.examples[0].scores["Factuality"].passed is False

    @pytest.mark.asyncio
    async def test_threshold_capture_persists_across_items(self):
        """A threshold learned from one item's scorer-metadata applies to subsequent items."""
        from helpers.fake_langfuse import FakeLangfuseClient

        from agent_evals.adapters.platforms.langfuse import (
            LangfuseConfig,
            LangFusePlatform,
        )
        from agent_evals.core.types import ExampleData, ExpectedResult, TaskResult

        scorer = _make_sync_scorer("JSONDiff", value=0.996, threshold=1.0)

        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        result = await adapter.aevaluate(
            task=lambda x: TaskResult(output="hello"),
            dataset=[
                ExampleData(input="a", expected=ExpectedResult(expected="x")),
                ExampleData(input="b", expected=ExpectedResult(expected="y")),
            ],
            evaluators=[scorer],
            platform=LangfuseConfig(experiment="threshold-persist"),
        )

        # Both items must reflect the captured 1.0 threshold (0.996 < 1.0 → fail).
        assert result.examples[0].scores["JSONDiff"].passed is False
        assert result.examples[1].scores["JSONDiff"].passed is False
