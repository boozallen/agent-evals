# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the shared verdict cache in adapters/platforms/utils.py.

The cache exists because a scorer owns its ``Score.passed`` but platforms
accept only a numeric value. Its key is derived from content rather than from a
real example id, so the guards matter as much as the happy path: every failure
mode must degrade to the caller's threshold comparison, never to a confident
wrong verdict attributed to the wrong example.
"""

from __future__ import annotations

import logging

import pytest

from agent_evals.adapters.platforms.utils import (
    MAX_KEY_PART_LEN,
    UNRENDERABLE_KEY,
    VerdictCache,
    verdict_key,
)
from agent_evals.core.types import ExpectedResult, Score, TaskResult


def _score(value: float, passed: bool, name: str = "S") -> Score:
    return Score(name=name, value=value, passed=passed)


class TestVerdictKey:
    """Both adapter sides must derive identical keys from equivalent inputs."""

    def test_same_parts_produce_same_key(self):
        assert verdict_key("out", "exp") == verdict_key("out", "exp")

    def test_different_parts_produce_different_keys(self):
        assert verdict_key("out", "exp") != verdict_key("out", "other")

    def test_task_result_unwraps_to_its_output(self):
        """A wrapper holds a TaskResult; the reconstruction path holds the string."""
        assert verdict_key(TaskResult(output="hello")) == verdict_key("hello")

    def test_expected_result_unwraps_to_its_expected(self):
        assert verdict_key(ExpectedResult(expected="gold")) == verdict_key("gold")

    def test_context_does_not_affect_the_key(self):
        """Only the output identifies the example; context may not round-trip."""
        with_ctx = TaskResult(output="hello", context={"final_state": {"a": 1}})
        assert verdict_key(with_ctx) == verdict_key(TaskResult(output="hello"))

    def test_part_order_matters(self):
        assert verdict_key("a", "b") != verdict_key("b", "a")

    def test_distinct_parts_do_not_collide_by_concatenation(self):
        """('ab', 'c') and ('a', 'bc') must not collapse to one key."""
        assert verdict_key("ab", "c") != verdict_key("a", "bc")

    def test_non_string_parts_are_accepted(self):
        """mlflow passes a dict of inputs; None appears when expected is absent."""
        assert verdict_key({"q": "x"}, None) == verdict_key({"q": "x"}, None)
        assert verdict_key({"q": "x"}, None) != verdict_key({"q": "y"}, None)


class TestVerdictCacheHit:
    """The verdict a scorer returned is the verdict the caller reads back."""

    def test_records_and_returns_a_failing_verdict_at_a_high_value(self):
        cache = VerdictCache()
        cache.record("StateMatch", "k1", _score(0.75, False))
        assert cache.lookup("StateMatch", "k1", 0.75) is False

    def test_records_and_returns_a_passing_verdict_at_a_low_value(self):
        cache = VerdictCache()
        cache.record("Lenient", "k1", _score(0.1, True))
        assert cache.lookup("Lenient", "k1", 0.1) is True

    def test_distinct_keys_keep_distinct_verdicts(self):
        cache = VerdictCache()
        cache.record("S", "k1", _score(0.8, False))
        cache.record("S", "k2", _score(0.2, True))
        assert cache.lookup("S", "k1", 0.8) is False
        assert cache.lookup("S", "k2", 0.2) is True

    def test_same_key_under_different_scorers_is_independent(self):
        cache = VerdictCache()
        cache.record("A", "k1", _score(0.6, True))
        cache.record("B", "k1", _score(0.6, False))
        assert cache.lookup("A", "k1", 0.6) is True
        assert cache.lookup("B", "k1", 0.6) is False


class TestVerdictCacheMiss:
    """Every miss returns None so the caller falls back to its own comparison."""

    def test_unknown_key_misses(self):
        cache = VerdictCache()
        assert cache.lookup("S", "never-recorded", 0.5) is None

    def test_unknown_scorer_misses(self):
        cache = VerdictCache()
        cache.record("S", "k1", _score(0.9, False))
        assert cache.lookup("Other", "k1", 0.9) is None

    def test_value_disagreement_outside_tolerance_misses(self):
        """A mis-derived key or rewritten value must not yield a verdict."""
        cache = VerdictCache()
        cache.record("S", "k1", _score(0.75, False))
        assert cache.lookup("S", "k1", 0.42) is None

    def test_value_disagreement_within_tolerance_still_hits(self):
        """Values round-trip through JSON; exact comparison would misfire."""
        cache = VerdictCache()
        cache.record("S", "k1", _score(0.1 + 0.2, False))
        assert cache.lookup("S", "k1", 0.30000000000000004) is False

    def test_value_disagreement_logs_a_warning_naming_the_scorer(self, caplog):
        cache = VerdictCache()
        cache.record("JSONDiff", "k1", _score(0.75, False))
        with caplog.at_level(logging.WARNING):
            assert cache.lookup("JSONDiff", "k1", 0.2) is None
        assert "JSONDiff" in caplog.text


class TestVerdictCacheAmbiguity:
    """Indistinguishable examples with conflicting verdicts must not guess."""

    def test_same_verdict_written_twice_stays_usable(self):
        """A deterministic scorer over duplicate examples is not a conflict."""
        cache = VerdictCache()
        cache.record("S", "dup", _score(0.75, False))
        cache.record("S", "dup", _score(0.75, False))
        assert cache.lookup("S", "dup", 0.75) is False

    def test_conflicting_verdicts_mark_the_slot_ambiguous(self):
        cache = VerdictCache()
        cache.record("S", "dup", _score(0.6, True))
        cache.record("S", "dup", _score(0.6, False))
        assert cache.lookup("S", "dup", 0.6) is None

    def test_conflict_logs_a_warning_naming_the_scorer(self, caplog):
        cache = VerdictCache()
        cache.record("LLMJudge", "dup", _score(0.6, True))
        with caplog.at_level(logging.WARNING):
            cache.record("LLMJudge", "dup", _score(0.6, False))
        assert "LLMJudge" in caplog.text

    def test_ambiguity_is_sticky(self):
        """A third write agreeing with the first must not un-ambiguate the slot."""
        cache = VerdictCache()
        cache.record("S", "dup", _score(0.6, True))
        cache.record("S", "dup", _score(0.6, False))
        cache.record("S", "dup", _score(0.6, True))
        assert cache.lookup("S", "dup", 0.6) is None

    def test_ambiguity_does_not_leak_to_other_keys(self):
        cache = VerdictCache()
        cache.record("S", "dup", _score(0.6, True))
        cache.record("S", "dup", _score(0.6, False))
        cache.record("S", "clean", _score(0.9, False))
        assert cache.lookup("S", "dup", 0.6) is None
        assert cache.lookup("S", "clean", 0.9) is False


class TestVerdictCacheClear:
    """An adapter instance may be reused across runs."""

    def test_clear_drops_every_entry(self):
        cache = VerdictCache()
        cache.record("S", "k1", _score(0.9, False))
        cache.record("T", "k2", _score(0.1, True))
        cache.clear()
        assert cache.lookup("S", "k1", 0.9) is None
        assert cache.lookup("T", "k2", 0.1) is None

    def test_clear_resets_ambiguity(self):
        cache = VerdictCache()
        cache.record("S", "dup", _score(0.6, True))
        cache.record("S", "dup", _score(0.6, False))
        cache.clear()
        cache.record("S", "dup", _score(0.6, True))
        assert cache.lookup("S", "dup", 0.6) is True


class TestVerdictCacheTypes:
    """Values arrive as ints and bools from some platforms."""

    @pytest.mark.parametrize("value", [1, 0])
    def test_integer_values_compare_against_recorded_floats(self, value):
        cache = VerdictCache()
        cache.record("S", "k1", _score(float(value), True))
        assert cache.lookup("S", "k1", value) is True


class TestVerdictKeyPartIsolation:
    """A part may not smuggle the separator and shift a key boundary.

    Joining raw ``repr`` output on a delimiter is not injective: an object whose
    ``__repr__`` emits that delimiter moves the boundary between two parts, so
    two genuinely different invocations collapse onto one slot. Each part is
    hashed separately to make the boundary unforgeable.
    """

    def test_a_part_cannot_forge_the_separator(self):
        """The regression: these two collided when parts were joined on \\x1f."""

        class Raw:
            def __init__(self, text: str) -> None:
                self.text = text

            def __repr__(self) -> str:
                return self.text

        assert verdict_key(Raw("X\x1fY"), Raw("Z")) != verdict_key(
            Raw("X"), Raw("Y\x1fZ")
        )

    def test_an_empty_part_is_still_a_part(self):
        assert verdict_key("", "a") != verdict_key("a", "")
        assert verdict_key("a") != verdict_key("a", "")

    def test_a_single_part_key_stays_stable(self):
        assert verdict_key("only") == verdict_key("only")


class TestVerdictKeyUnrenderableParts:
    """A user object that cannot be rendered must cost a miss, not a run.

    ``repr`` invokes user code. A lazy proxy, a partially-initialized ORM row,
    or a strict mock can raise, and preserving a verdict is never worth failing
    an eval that scored fine before this feature existed.
    """

    def test_a_raising_repr_yields_the_sentinel_instead_of_propagating(self):
        class BadRepr:
            def __repr__(self):
                raise RuntimeError("repr is unavailable")

        assert verdict_key(BadRepr()) == UNRENDERABLE_KEY

    def test_one_bad_part_poisons_the_whole_key(self):
        """A key built from a part that could not be read is not trustworthy."""

        class BadRepr:
            def __repr__(self):
                raise RuntimeError("repr is unavailable")

        assert verdict_key("fine", BadRepr()) == UNRENDERABLE_KEY

    def test_a_recursion_error_is_caught_too(self):
        """RecursionError is neither TypeError nor ValueError."""

        class Deep:
            def __repr__(self):
                raise RecursionError("too deep")

        assert verdict_key(Deep()) == UNRENDERABLE_KEY

    def test_the_sentinel_is_never_stored(self):
        """Every unrenderable example would otherwise share one slot."""
        cache = VerdictCache()
        cache.record("S", UNRENDERABLE_KEY, _score(0.9, False))
        assert cache.lookup("S", UNRENDERABLE_KEY, 0.9) is None

    def test_the_sentinel_cannot_borrow_another_examples_verdict(self):
        """Two different unrenderable examples must not resolve to each other."""
        cache = VerdictCache()
        cache.record("S", UNRENDERABLE_KEY, _score(0.9, True))
        cache.record("S", UNRENDERABLE_KEY, _score(0.9, False))
        assert cache.lookup("S", UNRENDERABLE_KEY, 0.9) is None


class TestVerdictKeyLengthBound:
    """A rendering is bounded before it is hashed or retained.

    ``repr`` runs user code on values that arrive from a platform, so nothing in
    this module controls how long its output is. An oversized rendering is
    refused for the same reason a raising one is: the verdict for that example
    is worth less than an unbounded string held for the length of a run.
    """

    def test_a_rendering_at_the_bound_still_keys(self):
        """The bound must not cost preservation on ordinary data."""
        at_limit = "x" * (MAX_KEY_PART_LEN - 2)  # -2 for repr's quote marks
        assert verdict_key(at_limit) != UNRENDERABLE_KEY

    def test_a_rendering_over_the_bound_yields_the_sentinel(self):
        assert verdict_key("x" * (MAX_KEY_PART_LEN + 1)) == UNRENDERABLE_KEY

    def test_an_oversized_repr_is_refused_even_when_the_object_is_small(self):
        """The object's own size is irrelevant; the rendering's is what is kept."""

        class Verbose:
            def __repr__(self):
                return "v" * (MAX_KEY_PART_LEN + 1)

        assert verdict_key(Verbose()) == UNRENDERABLE_KEY

    def test_one_oversized_part_poisons_the_whole_key(self):
        assert verdict_key("fine", "x" * (MAX_KEY_PART_LEN + 1)) == UNRENDERABLE_KEY

    def test_an_oversized_part_is_logged_at_debug(self, caplog):
        """Losing preservation is worth a trace, not a warning: it is not an error."""
        with caplog.at_level(logging.DEBUG):
            verdict_key("x" * (MAX_KEY_PART_LEN + 1))
        assert "bound" in caplog.text

    def test_an_oversized_example_degrades_rather_than_borrowing(self):
        """Two oversized examples must not collapse into one cache slot."""
        cache = VerdictCache()
        big_a = verdict_key("a" * (MAX_KEY_PART_LEN + 1))
        big_b = verdict_key("b" * (MAX_KEY_PART_LEN + 1))
        assert big_a == big_b == UNRENDERABLE_KEY
        cache.record("S", big_a, _score(0.9, True))
        assert cache.lookup("S", big_b, 0.9) is None


class TestVerdictCacheNaN:
    """NaN fails every comparison, so it must not satisfy the value guard.

    ``abs(recorded - nan) > TOLERANCE`` is False, which would read as "the
    values agree" and return a confident verdict at exactly the moment the
    value read back off the platform is not a number.
    """

    def test_a_nan_value_misses_rather_than_hitting(self):
        cache = VerdictCache()
        cache.record("S", "k1", _score(0.9, False))
        assert cache.lookup("S", "k1", float("nan")) is None

    def test_a_nan_value_logs_a_warning_naming_the_scorer(self, caplog):
        cache = VerdictCache()
        cache.record("StateMatch", "k1", _score(0.9, False))
        with caplog.at_level(logging.WARNING):
            assert cache.lookup("StateMatch", "k1", float("nan")) is None
        assert "StateMatch" in caplog.text

    @pytest.mark.parametrize("value", [float("inf"), float("-inf")])
    def test_infinities_miss_on_the_ordinary_value_guard(self, value):
        cache = VerdictCache()
        cache.record("S", "k1", _score(0.9, False))
        assert cache.lookup("S", "k1", value) is None


class TestVerdictCacheRunLifetime:
    """One adapter instance may serve runs in sequence, never concurrently.

    The cache is instance state. Two overlapping runs would share it, and the
    second's reset would discard verdicts the first had already recorded,
    silently reverting that run to the threshold-derived pass/fail this class
    exists to replace. Failing loudly is the only safe option.
    """

    def test_sequential_runs_are_allowed(self):
        cache = VerdictCache()
        cache.begin_run("Adapter")
        cache.record("S", "k1", _score(0.9, False))
        cache.end_run()

        cache.begin_run("Adapter")
        assert cache.lookup("S", "k1", 0.9) is None, "a run must start empty"
        cache.end_run()

    def test_an_overlapping_run_on_one_instance_is_refused(self):
        cache = VerdictCache()
        cache.begin_run("MLflowPlatform")
        with pytest.raises(RuntimeError, match="concurrent runs"):
            cache.begin_run("MLflowPlatform")

    def test_the_refusal_names_the_adapter(self):
        cache = VerdictCache()
        cache.begin_run("BraintrustPlatform")
        with pytest.raises(RuntimeError, match="BraintrustPlatform"):
            cache.begin_run("BraintrustPlatform")

    def test_a_failed_run_does_not_block_the_next_one(self):
        """end_run belongs in a finally; a raising run must still release."""
        cache = VerdictCache()
        cache.begin_run("Adapter")
        try:
            raise ValueError("the eval blew up")
        except ValueError:
            pass
        finally:
            cache.end_run()

        cache.begin_run("Adapter")  # must not raise
        cache.end_run()

    def test_end_run_releases_the_entries(self):
        """Entries must not stay reachable from a held traceback."""
        cache = VerdictCache()
        cache.begin_run("Adapter")
        cache.record("S", "k1", _score(0.9, False))
        cache.end_run()
        assert cache.lookup("S", "k1", 0.9) is None
