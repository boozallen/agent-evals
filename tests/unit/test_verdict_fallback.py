# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""The threshold comparison stays reachable, and every cache miss reaches it.

Verdict preservation is opportunistic: the cache key is derived from content,
not from a real example id, so it can miss. AC 4 requires the pre-existing
``value >= threshold`` path to remain live for the cases that legitimately have
no originating agent-evals ``Score`` — platform-native scores and historical
traces — and the design requires every failure mode to degrade to that path
rather than to a confident verdict attributed to the wrong example.

Companion to ``test_scorer_verdict_preservation.py`` (the happy path) and
``test_verdict_cache.py`` (the cache in isolation). This file pins the seam
between them: what each adapter does when the lookup does not answer.
"""

from __future__ import annotations

import asyncio
import logging
import os
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent_evals.core.types import ExampleData, ExpectedResult, Score, TaskResult

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _mlflow_tracking_uri(monkeypatch):
    """MLflowPlatform._resolve_tracking_uri raises without this.

    Spelled ``https``: the session is faked, so nothing connects, but the value
    still passes through transport validation.
    """
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://x")


def _dataset(*inputs: str) -> list[ExampleData]:
    return [
        ExampleData(input=value, expected=ExpectedResult(expected="ok"))
        for value in inputs
    ]


def _make_scorer(name: str, verdicts: list[tuple[float, bool]]):
    """A scorer returning the given (value, passed) pairs in call order.

    Call-ordered rather than input-keyed on purpose: these tests need two
    *indistinguishable* examples to receive different verdicts, which an
    input-keyed scorer cannot express.
    """
    remaining = list(verdicts)

    def scorer(result: TaskResult, expected: ExpectedResult | None = None, **kwargs):
        _ = result, expected, kwargs  # vulture: intentionally unused
        value, passed = remaining.pop(0) if remaining else verdicts[-1]
        return Score(name=name, value=value, passed=passed)

    scorer.__name__ = name
    scorer.__qualname__ = name
    return scorer


async def _run_braintrust(task, dataset, evaluators, experiment: str, adapter=None):
    from helpers.fake_braintrust import FakeBraintrustClient

    from agent_evals.adapters.platforms.braintrust import (
        BraintrustConfig,
        BraintrustPlatform,
    )

    adapter = adapter or BraintrustPlatform(client=FakeBraintrustClient())
    with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
        result = await adapter.aevaluate(
            task=task,
            dataset=dataset,
            evaluators=evaluators,
            platform=BraintrustConfig(project="p", experiment=experiment),
        )
    return result.examples


async def _run_mlflow(task, dataset, evaluators, experiment: str, adapter=None):
    from helpers.fake_mlflow import FakeMlflowSession

    from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform

    adapter = adapter or MLflowPlatform(session=FakeMlflowSession())
    result = await adapter.aevaluate(
        task=task,
        dataset=dataset,
        evaluators=evaluators,
        platform=MlflowConfig(experiment=experiment),
    )
    return result.examples


async def _run_langfuse(task, dataset, evaluators, experiment: str, adapter=None):
    from helpers.fake_langfuse import FakeLangfuseClient

    from agent_evals.adapters.platforms.langfuse import (
        LangfuseConfig,
        LangFusePlatform,
    )

    adapter = adapter or LangFusePlatform(client=FakeLangfuseClient())
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


def _fresh_adapter(runner):
    """Build a reusable adapter instance matching the given runner."""
    from helpers.fake_braintrust import FakeBraintrustClient
    from helpers.fake_langfuse import FakeLangfuseClient
    from helpers.fake_mlflow import FakeMlflowSession

    from agent_evals.adapters.platforms.braintrust import BraintrustPlatform
    from agent_evals.adapters.platforms.langfuse import LangFusePlatform
    from agent_evals.adapters.platforms.mlflow import MLflowPlatform

    if runner is _run_braintrust:
        return BraintrustPlatform(client=FakeBraintrustClient())
    if runner is _run_mlflow:
        return MLflowPlatform(session=FakeMlflowSession())
    return LangFusePlatform(client=FakeLangfuseClient())


# ===================================================================
# Task 5.1 — platform-native scores with no originating agent-evals Score
# ===================================================================


class TestPlatformNativeScoresUseTheThreshold:
    """A score with no cache entry must still get a verdict, from the threshold.

    These are the real AC 4 cases: a score computed by the platform itself, or
    read back off a historical trace, has no agent-evals ``Score`` behind it, so
    there is nothing to preserve. Each adapter's reconstruction path is driven
    directly, which is the only way to inject a score the wrappers never saw.
    """

    @pytest.mark.requires_braintrust
    def test_braintrust_native_score_uses_the_default_threshold(self):
        from agent_evals.adapters.platforms.braintrust import (
            PASS_THRESHOLD,
            BraintrustPlatform,
        )

        adapter = BraintrustPlatform()
        assert PASS_THRESHOLD == 0.5

        case = SimpleNamespace(
            output="native", expected="native", scores={"NativeScorer": 0.6}
        )
        scores = adapter._convert_case_scores(case)

        assert scores["NativeScorer"].value == pytest.approx(0.6)
        assert scores["NativeScorer"].passed is True, (
            "No cache entry exists for a platform-native score, so the "
            "threshold comparison must still produce the verdict."
        )

    @pytest.mark.requires_braintrust
    def test_braintrust_native_score_uses_a_declared_threshold(self):
        from agent_evals.adapters.platforms.braintrust import BraintrustPlatform

        adapter = BraintrustPlatform()
        adapter._scorer_thresholds["NativeScorer"] = 0.9

        case = SimpleNamespace(
            output="native", expected="native", scores={"NativeScorer": 0.6}
        )
        scores = adapter._convert_case_scores(case)

        assert scores["NativeScorer"].passed is False

    def test_mlflow_native_score_uses_the_default_threshold(self):
        from helpers.fake_mlflow import FakeMlflowSession
        from mlflow.genai.evaluation.entities import Feedback

        from agent_evals.adapters.platforms.mlflow import (
            PASS_THRESHOLD,
            MLflowPlatform,
        )
        from agent_evals.adapters.platforms.mlflow_client import TraceRecord

        assert PASS_THRESHOLD == 0.5

        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)
        adapter._session_info = fake.configure(
            tracking_uri="https://x", experiment="native"
        )
        # A trace whose Feedback never passed through a scorer wrapper.
        fake._run_traces["native-run"] = [
            TraceRecord(
                request='{"messages": [{"content": "q"}]}',
                response='{"messages": [{"content": "a"}]}',
                assessments=[Feedback(name="NativeScorer", value=0.6)],
                execution_duration_ms=1,
                trace_metadata={},
            )
        ]

        result = adapter._convert_from_platform_result(
            SimpleNamespace(run_id="native-run")
        )
        score = result.examples[0].scores["NativeScorer"]

        assert score.value == pytest.approx(0.6)
        assert score.passed is True, (
            "A Feedback read off a trace has no originating agent-evals "
            "Score, so the threshold comparison must produce the verdict."
        )

    def test_mlflow_native_score_uses_a_declared_threshold(self):
        from helpers.fake_mlflow import FakeMlflowSession
        from mlflow.genai.evaluation.entities import Feedback

        from agent_evals.adapters.platforms.mlflow import MLflowPlatform
        from agent_evals.adapters.platforms.mlflow_client import TraceRecord

        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)
        adapter._session_info = fake.configure(
            tracking_uri="https://x", experiment="native"
        )
        adapter._scorer_thresholds["NativeScorer"] = 0.9
        fake._run_traces["native-run"] = [
            TraceRecord(
                request='{"messages": [{"content": "q"}]}',
                response='{"messages": [{"content": "a"}]}',
                assessments=[Feedback(name="NativeScorer", value=0.6)],
                execution_duration_ms=1,
                trace_metadata={},
            )
        ]

        result = adapter._convert_from_platform_result(
            SimpleNamespace(run_id="native-run")
        )
        assert result.examples[0].scores["NativeScorer"].passed is False

    def test_langfuse_native_score_uses_the_default_threshold(self):
        from agent_evals.adapters.platforms.langfuse import (
            PASS_THRESHOLD,
            LangFusePlatform,
        )

        assert PASS_THRESHOLD == 0.5

        adapter = LangFusePlatform()
        # An Evaluation that never passed through a scorer wrapper.
        platform_result = SimpleNamespace(
            item_results=[
                SimpleNamespace(
                    item={"input": "q", "expected_output": "gold"},
                    output="a",
                    evaluations=[SimpleNamespace(name="NativeScorer", value=0.6)],
                    trace_id=None,
                    dataset_run_id=None,
                )
            ],
        )

        examples = adapter._build_examples_from_platform_result(
            platform_result,  # type: ignore[arg-type]
            _dataset("q"),
        )
        score = examples[0].scores["NativeScorer"]

        assert score.value == pytest.approx(0.6)
        assert score.passed is True

    def test_langfuse_native_score_uses_a_declared_threshold(self):
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        adapter = LangFusePlatform()
        adapter._scorer_thresholds["NativeScorer"] = 0.9
        platform_result = SimpleNamespace(
            item_results=[
                SimpleNamespace(
                    item={"input": "q", "expected_output": "gold"},
                    output="a",
                    evaluations=[SimpleNamespace(name="NativeScorer", value=0.6)],
                    trace_id=None,
                    dataset_run_id=None,
                )
            ],
        )

        examples = adapter._build_examples_from_platform_result(
            platform_result,  # type: ignore[arg-type]
            _dataset("q"),
        )
        assert examples[0].scores["NativeScorer"].passed is False


# ===================================================================
# Task 5.2 — a value disagreement falls back and says so
# ===================================================================


class TestValueDisagreementFallsBack:
    """A recorded verdict whose value does not match must not be reported.

    A value mismatch means the key was derived from something other than the
    example the value came from. Reporting the verdict anyway would attribute
    one example's verdict to another; falling back is the safe answer, and the
    warning is how an operator finds out preservation silently stopped working.
    """

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_mismatched_value_uses_the_threshold_and_warns(self, run_adapter, caplog):
        # Record each verdict against a value the platform will never report
        # back. That is precisely what a mis-derived key looks like from the
        # lookup's side: the slot is found, but its value belongs to a
        # different example.
        adapter = _fresh_adapter(run_adapter)
        scorer = _make_scorer("Drifter", [(0.8, False)])

        real_record = adapter._verdicts.record

        def record_a_drifted_value(scorer_name, example_key, score):
            # Stay inside Score's 0.0-1.0 range; only the disagreement matters.
            real_record(
                scorer_name,
                example_key,
                Score(name=score.name, value=score.value - 0.5, passed=score.passed),
            )

        adapter._verdicts.record = record_a_drifted_value  # type: ignore[method-assign]

        with caplog.at_level(logging.WARNING):
            examples = asyncio.run(
                run_adapter(
                    lambda _x: TaskResult(output="hello"),
                    _dataset("in-a"),
                    [scorer],
                    "verdict-drift",
                    adapter,
                )
            )

        score = examples[0].scores["Drifter"]
        assert score.value == pytest.approx(0.8)
        assert score.passed is True, (
            "The recorded value no longer matches, so the verdict cannot be "
            "trusted; the threshold comparison (0.8 >= 0.5) must decide."
        )
        assert "Drifter" in caplog.text, (
            "A silent fallback would hide that preservation stopped working."
        )

    def test_the_guard_is_the_cache_not_the_adapter(self):
        """Pin the mechanism directly, independent of any adapter's plumbing."""
        from agent_evals.adapters.platforms.utils import VerdictCache

        cache = VerdictCache()
        cache.record("Drifter", "k", Score(name="Drifter", value=0.3, passed=False))
        assert cache.lookup("Drifter", "k", 0.8) is None


# ===================================================================
# Task 5.3 — duplicate inputs within one dataset
# ===================================================================


class TestDuplicateInputs:
    """Two examples can be indistinguishable; the key cannot separate them.

    ``ExampleData`` has no id field, so two examples with the same input and
    expected value derive the same key. A deterministic scorer returns the same
    verdict for both and there is nothing to disambiguate. A non-deterministic
    one may not, and then neither example may borrow the other's verdict.
    """

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_identical_examples_with_the_same_verdict_both_keep_it(self, run_adapter):
        scorer = _make_scorer("Deterministic", [(0.75, False), (0.75, False)])

        examples = asyncio.run(
            run_adapter(
                lambda _x: TaskResult(output="same"),
                _dataset("dup", "dup"),
                [scorer],
                "verdict-dup-agree",
                None,
            )
        )

        assert len(examples) == 2
        for example in examples:
            score = example.scores["Deterministic"]
            assert score.value == pytest.approx(0.75)
            assert score.passed is False, (
                "Both examples got the same verdict, so a shared key is not "
                "a conflict and the verdict must survive for both."
            )

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_identical_examples_with_conflicting_verdicts_both_fall_back(
        self, run_adapter, caplog
    ):
        # A non-deterministic scorer: same inputs, different verdicts. Values
        # are equal so the value-agreement guard cannot catch this — only the
        # ambiguity rule can.
        scorer = _make_scorer("Flaky", [(0.75, False), (0.75, True)])

        with caplog.at_level(logging.WARNING):
            examples = asyncio.run(
                run_adapter(
                    lambda _x: TaskResult(output="same"),
                    _dataset("dup", "dup"),
                    [scorer],
                    "verdict-dup-conflict",
                    None,
                )
            )

        assert len(examples) == 2
        for example in examples:
            score = example.scores["Flaky"]
            # 0.75 >= 0.5 → the threshold fallback says True for both. The
            # point is that neither example reports the *other's* verdict as if
            # it had been preserved.
            assert score.passed is True, (
                "Indistinguishable examples with conflicting verdicts must "
                "both fall back, not borrow each other's verdict."
            )
        assert "Flaky" in caplog.text


# ===================================================================
# Task 5.4 — a reused adapter instance across two runs
# ===================================================================


class TestReusedAdapterInstance:
    """Run two's examples must not inherit run one's verdicts."""

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_second_run_does_not_inherit_the_first_runs_verdict(self, run_adapter):
        adapter = _fresh_adapter(run_adapter)

        # Run one: a failing verdict at a value above the threshold.
        first = asyncio.run(
            run_adapter(
                lambda _x: TaskResult(output="shared"),
                _dataset("same-input"),
                [_make_scorer("Reused", [(0.8, False)])],
                "verdict-run-one",
                adapter,
            )
        )
        assert first[0].scores["Reused"].passed is False

        # Run two: identical input and output, so the key is identical, but the
        # scorer now returns a passing verdict. A stale entry would win.
        second = asyncio.run(
            run_adapter(
                lambda _x: TaskResult(output="shared"),
                _dataset("same-input"),
                [_make_scorer("Reused", [(0.8, True)])],
                "verdict-run-two",
                adapter,
            )
        )

        assert second[0].scores["Reused"].passed is True, (
            "The second run recorded its own verdict; a cache not reset per "
            "run would report the first run's verdict instead."
        )

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_a_second_run_with_no_scorers_reports_nothing_stale(self, run_adapter):
        """Without a reset, run one's entry would still be sitting in the cache."""
        adapter = _fresh_adapter(run_adapter)

        asyncio.run(
            run_adapter(
                lambda _x: TaskResult(output="shared"),
                _dataset("same-input"),
                [_make_scorer("Reused", [(0.8, False)])],
                "verdict-stale-one",
                adapter,
            )
        )
        assert adapter._verdicts.lookup("Reused", "any-key", 0.8) is None

        asyncio.run(
            run_adapter(
                lambda _x: TaskResult(output="shared"),
                _dataset("same-input"),
                [],
                "verdict-stale-two",
                adapter,
            )
        )
        assert adapter._verdicts._entries == {}, (
            "aevaluate must clear the cache at the start of every run."
        )


# ===================================================================
# Task 5.5 — PASS_THRESHOLD keeps a live consumer in all three adapters
# ===================================================================


class TestPassThresholdRemainsLive:
    """AC 4: the constant remains defined and tested, not merely defined.

    The tests above exercise it behaviourally. This one pins that it is still
    the default each adapter reaches for, so a future change that deletes the
    fallback cannot leave the constant sitting unused.
    """

    @pytest.mark.parametrize(
        "module_path",
        [
            pytest.param(
                "agent_evals.adapters.platforms.braintrust",
                marks=pytest.mark.requires_braintrust,
            ),
            "agent_evals.adapters.platforms.mlflow",
            "agent_evals.adapters.platforms.langfuse",
        ],
    )
    def test_each_adapter_defines_the_default_threshold(self, module_path):
        import importlib

        module = importlib.import_module(module_path)
        assert module.PASS_THRESHOLD == 0.5

    @pytest.mark.parametrize(
        "module_path",
        [
            pytest.param(
                "agent_evals.adapters.platforms.braintrust",
                marks=pytest.mark.requires_braintrust,
            ),
            "agent_evals.adapters.platforms.mlflow",
            "agent_evals.adapters.platforms.langfuse",
        ],
    )
    def test_each_adapter_still_reads_the_default_threshold(self, module_path):
        """The constant must be referenced by the reconstruction path itself."""
        import importlib
        import inspect

        module = importlib.import_module(module_path)
        source = inspect.getsource(module)
        # Two references minimum: the definition and at least one consumer.
        assert source.count("PASS_THRESHOLD") >= 2, (
            f"{module_path} defines PASS_THRESHOLD but no longer consumes it; "
            "AC 4 requires the fallback to stay reachable."
        )


# ===================================================================
# A hostile input degrades the key, never the run
# ===================================================================


class _BadStr:
    """An object whose ``str()`` raises but whose ``repr()`` works."""

    def __str__(self) -> str:
        raise RuntimeError("str is unavailable")

    def __repr__(self) -> str:
        return "<BadStr>"


class _BadRepr:
    """An object neither ``str()`` nor ``repr()`` can render."""

    def __str__(self) -> str:
        raise RuntimeError("str is unavailable")

    def __repr__(self) -> str:
        raise RuntimeError("repr is unavailable")


class TestHostileInputsDegradeRatherThanCrash:
    """Key derivation calls user code, so it must never be the thing that fails.

    Deriving a key means calling ``str`` and ``repr`` on the user's own input
    objects. A lazy proxy, a partially-initialized ORM row, or a strict mock can
    raise from either, and deeply nested or self-referential structures can
    exhaust the stack. Before verdict preservation existed these inputs scored
    fine; losing a verdict is an acceptable cost, failing the run is not.
    """

    def test_a_raising_str_still_yields_usable_renderings(self):
        from agent_evals.adapters.platforms.mlflow import _input_renderings

        renderings = _input_renderings({"q": _BadStr()})

        assert renderings, "the dict itself is always a candidate"
        assert any(isinstance(r, dict) for r in renderings)

    def test_a_raising_repr_yields_no_renderings_instead_of_raising(self):
        """Every candidate is deduplicated by repr, so none survive — that is fine."""
        from agent_evals.adapters.platforms.mlflow import _input_renderings

        assert _input_renderings({"q": _BadRepr()}) == []

    def test_a_self_referential_input_does_not_raise(self):
        from agent_evals.adapters.platforms.mlflow import _input_renderings

        loop: dict[str, object] = {"q": "x"}
        loop["self"] = loop

        # json.dumps raises ValueError on a circular reference; str/repr both
        # handle it via the recursion marker.
        assert _input_renderings(loop)

    def test_a_deeply_nested_input_does_not_raise(self):
        """repr on deep nesting raises RecursionError, which is not a ValueError."""
        from agent_evals.adapters.platforms.mlflow import _input_renderings

        deep: object = "leaf"
        for _ in range(5000):
            deep = [deep]

        # No assertion on the contents: whether any rendering survives depends
        # on the interpreter's remaining stack. Not raising is the requirement.
        _input_renderings({"q": deep})

    def test_an_oversized_rendering_is_refused_rather_than_retained(self):
        """The mlflow renderings are bounded like the shared key parts are.

        ``str`` / ``repr`` / ``json.dumps`` on platform-supplied data have no
        inherent size limit, and every rendering they produce is hashed into a
        key and held for the length of a run. An input too large to render
        within the bound yields no candidates at all, which costs the verdict
        for that example and nothing else.
        """
        from agent_evals.adapters.platforms.mlflow import (
            MAX_KEY_PART_LEN,
            _input_renderings,
            _safe_repr,
            _safe_str,
        )

        oversized = "x" * (MAX_KEY_PART_LEN + 1)

        assert _safe_str(oversized) is None
        assert _safe_repr(oversized) is None
        assert _input_renderings({"q": oversized}) == []

        # An ordinary input is unaffected: the bound is generous enough that
        # preservation is not lost on real data.
        assert _input_renderings({"q": "ordinary"})

    @pytest.mark.parametrize("run_adapter", _ADAPTERS)
    def test_a_hostile_input_still_produces_a_verdict(self, run_adapter):
        """The verdict may be lost; the run is not.

        ``_BadStr`` rather than ``_BadRepr``: MLflow — real and faked alike —
        serializes the inputs itself before this adapter ever sees them back, so
        an object with no usable ``repr`` at all fails outside the code under
        test. An object whose ``str`` raises is the hostile case that actually
        reaches key derivation, and it raised there before the guard existed.
        """
        scorer = _make_scorer("Hostile", [(0.8, False)])

        examples = asyncio.run(
            run_adapter(
                lambda _x: TaskResult(output="out"),
                [ExampleData(input=_BadStr(), expected=ExpectedResult(expected="ok"))],
                [scorer],
                "verdict-hostile-input",
                None,
            )
        )

        assert len(examples) == 1
        assert examples[0].scores["Hostile"].value == pytest.approx(0.8)
        assert isinstance(examples[0].scores["Hostile"].passed, bool), (
            "An input whose str raises must still produce a verdict rather "
            "than failing the run."
        )
