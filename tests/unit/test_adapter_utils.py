# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for shared adapter utilities."""

import pytest

from agent_evals.adapters.platforms.utils import (
    NO_SCORES_KEY,
    compute_aggregate_scores,
    compute_summary_counts,
)
from agent_evals.core.types import EvalExample, Score


def test_compute_aggregate_scores_empty_examples():
    """No examples should yield empty aggregates and pass rates."""
    aggregate, pass_rates = compute_aggregate_scores([])

    assert aggregate == {}
    assert pass_rates == {}


def test_compute_aggregate_scores_single_scorer_multiple_examples():
    """Average and pass rate are computed correctly for a single scorer."""
    examples = [
        EvalExample(
            input="a",
            output="out-a",
            scores={"scorer": Score(name="scorer", value=0.5, passed=True)},
        ),
        EvalExample(
            input="b",
            output="out-b",
            scores={"scorer": Score(name="scorer", value=1.0, passed=False)},
        ),
    ]

    aggregate, pass_rates = compute_aggregate_scores(examples)

    # Mean of [0.5, 1.0]
    assert aggregate == {"scorer": pytest.approx(0.75)}
    # 1 of 2 passed
    assert pass_rates == {"scorer": pytest.approx(0.5)}


def test_compute_aggregate_scores_multiple_scorers_and_missing_values():
    """Scorers present on only some examples are aggregated over their own counts."""
    examples = [
        EvalExample(
            input="a",
            output="out-a",
            scores={
                "s1": Score(name="s1", value=0.2, passed=False),
                "s2": Score(name="s2", value=0.8, passed=True),
            },
        ),
        EvalExample(
            input="b",
            output="out-b",
            scores={
                "s1": Score(name="s1", value=0.4, passed=True),
                # s2 missing on this example
            },
        ),
        EvalExample(
            input="c",
            output="out-c",
            scores={
                "s2": Score(name="s2", value=1.0, passed=True),
            },
        ),
    ]

    aggregate, pass_rates = compute_aggregate_scores(examples)

    # s1: values [0.2, 0.4] → mean 0.3, passes [False, True] → 1/2
    assert aggregate["s1"] == pytest.approx(0.3)
    assert pass_rates["s1"] == pytest.approx(0.5)

    # s2: values [0.8, 1.0] (on examples a and c) → mean 0.9, passes [True, True] → 1.0
    assert aggregate["s2"] == pytest.approx(0.9)
    assert pass_rates["s2"] == pytest.approx(1.0)


def test_compute_summary_counts_empty_examples():
    """Empty example list should yield all-zero counts."""
    total, success, failed = compute_summary_counts([])

    assert total == 0
    assert success == 0
    assert failed == 0


def test_compute_summary_counts_mixed_success_and_failure():
    """Examples with error=None are counted as successful; others as failed."""
    examples = [
        EvalExample(
            input="a",
            output="out-a",
            error=None,
            scores={},
        ),
        EvalExample(
            input="b",
            output="out-b",
            error="task failed",
            scores={},
        ),
        EvalExample(
            input="c",
            output="out-c",
            error=None,
            scores={},
        ),
    ]

    total, success, failed = compute_summary_counts(examples)

    assert total == 3
    assert success == 2
    assert failed == 1


# --- Errored runs must not aggregate as passing ------------------------------


def test_failed_examples_count_as_zero_not_absent():
    """A failed example contributes 0.0 rather than shrinking the denominator.

    Skipping it would average only the examples that worked, so one healthy
    example among failures would report a perfect score.
    """
    examples = [
        EvalExample(
            input="a",
            output="out-a",
            scores={"s": Score(name="s", value=1.0, passed=True)},
        ),
        EvalExample(input="b", output="", error="task exploded", scores={}),
        EvalExample(input="c", output="", error="task exploded", scores={}),
    ]

    aggregate, pass_rates = compute_aggregate_scores(examples)

    # 1 pass out of 3 examples, not 1 out of 1.
    assert aggregate["s"] == pytest.approx(1 / 3)
    assert pass_rates["s"] == pytest.approx(1 / 3)


def test_absent_scorer_on_healthy_example_is_not_penalised():
    """Scorer sets are heterogeneous by design; absence is not failure.

    A benchmark binds scorers per capability, so a healthy example that simply
    does not use a given scorer must stay out of that scorer's denominator.
    """
    examples = [
        EvalExample(
            input="a",
            output="out-a",
            scores={"only_here": Score(name="only_here", value=1.0, passed=True)},
        ),
        EvalExample(
            input="b",
            output="out-b",
            scores={"other": Score(name="other", value=1.0, passed=True)},
        ),
    ]

    aggregate, pass_rates = compute_aggregate_scores(examples)

    assert pass_rates["only_here"] == pytest.approx(1.0)
    assert aggregate["only_here"] == pytest.approx(1.0)


def test_errored_score_is_not_averaged_as_genuine_performance():
    """A scorer that crashed produced no verdict, so it cannot count as one."""
    examples = [
        EvalExample(
            input="a",
            output="out-a",
            scores={"s": Score(name="s", value=1.0, passed=True)},
        ),
        EvalExample(
            input="b",
            output="out-b",
            scores={
                "s": Score(name="s", value=1.0, passed=True, error="scorer exploded")
            },
        ),
    ]

    aggregate, pass_rates = compute_aggregate_scores(examples)

    # The errored score is a zero despite carrying value=1.0/passed=True.
    assert aggregate["s"] == pytest.approx(0.5)
    assert pass_rates["s"] == pytest.approx(0.5)


def test_run_with_no_scores_at_all_is_not_vacuously_passing():
    """Examples that all failed before scoring must not yield empty aggregates.

    ``all(v >= threshold for v in {}.values())`` is True, so an empty dict is
    read as a pass by the common CI gate. An explicit failing key is emitted
    instead.
    """
    examples = [
        EvalExample(input=x, output="", error="framework failure", scores={})
        for x in "abc"
    ]

    aggregate, pass_rates = compute_aggregate_scores(examples)

    assert pass_rates == {NO_SCORES_KEY: 0.0}
    assert aggregate == {NO_SCORES_KEY: 0.0}
    assert not all(v >= 0.9 for v in pass_rates.values())


def test_summary_counts_see_scorer_failures():
    """An example whose task ran but whose scorers crashed is not successful."""
    examples = [
        EvalExample(
            input="a",
            output="out-a",
            error=None,
            scores={
                "s": Score(name="s", value=0.0, passed=False, error="scorer exploded")
            },
        ),
        EvalExample(
            input="b",
            output="out-b",
            error=None,
            scores={"s": Score(name="s", value=0.0, passed=False)},
        ),
    ]

    total, success, failed = compute_summary_counts(examples)

    # Example a failed (scorer error); example b legitimately scored 0.0.
    assert (total, success, failed) == (2, 1, 1)
