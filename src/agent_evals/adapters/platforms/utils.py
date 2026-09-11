# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Shared utility functions for adapter implementations.

These helpers operate purely on core Agent Evals types and are safe to use
across multiple platform adapters when they want common aggregation or
summary behavior.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
import math
from collections.abc import Callable, Iterable
from typing import Any

from agent_evals.core.types import EvalExample

logger = logging.getLogger(__name__)

#: Aggregate key emitted when examples ran but *nothing* scored. An empty
#: aggregate dict is indistinguishable from a perfect run to the common
#: ``all(v >= threshold for v in pass_rates.values())`` gate, which is
#: vacuously True over no values. Reporting an explicit 0.0 under this key
#: makes total failure fail that gate instead of passing it silently.
NO_SCORES_KEY = "__no_scores__"

# Name that factory-produced scorer closures carry in ``__name__``. Reaching
# this as a derived name means no declared name was recoverable.
_INDETERMINATE_NAME = "scorer"


def infer_scorer_name(evaluator: Callable[..., Any]) -> str:
    """Infer a scorer's declared name from the callable.

    Adapters that key per-scorer state by name must derive that key the same
    way wherever they touch it. ``__name__`` alone will not do: a scorer built
    by a factory is a nested closure literally named ``scorer``, so every such
    scorer collapses onto one key.

    Handles:
    - Closures that capture ``scorer_name`` in their closure
    - Functions returned by factories like ``NumericDiff()`` whose
      ``__qualname__`` looks like ``NumericDiff.<locals>.scorer``
    - Plain callables or lambdas using their ``__name__`` / ``__qualname__``

    Derivation is best-effort. A hand-rolled closure that captures no
    ``scorer_name`` and carries an uninformative ``__qualname__`` still yields
    ``scorer``; two such scorers in one evaluation share a key and the second
    overwrites the first. That case logs a warning rather than passing
    silently, since the callable alone does not carry enough to resolve it.

    Args:
        evaluator: The scorer callable to name.

    Returns:
        The scorer's declared name, or ``scorer`` when none is recoverable.
    """
    # 1) Try to pull ``scorer_name`` out of the closure (used by autoevals helpers)
    code = getattr(evaluator, "__code__", None)
    closure = getattr(evaluator, "__closure__", None)
    if code is not None and closure is not None and code.co_freevars:
        try:
            cells = dict(zip(code.co_freevars, closure, strict=True))
            cell = cells.get("scorer_name")
            if cell is not None:
                value = cell.cell_contents
                if isinstance(value, str) and value:
                    return value
        except ValueError, TypeError:
            # Introspecting a closure is best-effort: ``zip(strict=True)`` raises
            # ValueError on a freevar/cell length mismatch, ``cell_contents``
            # raises ValueError for an empty cell, and a ``__closure__`` that is
            # not iterable raises TypeError. None of those are recoverable here,
            # and none should abort scoring — fall back to the name heuristics
            # below. Logged at debug so the fallback is traceable.
            logger.debug(
                "Closure introspection failed for scorer %r; "
                "falling back to name heuristics.",
                evaluator,
                exc_info=True,
            )

    # 2) Fallback to qualname/name heuristics
    raw_name = (
        getattr(evaluator, "__qualname__", None)
        or getattr(evaluator, "__name__", None)
        or evaluator.__class__.__name__
    )

    # Normalize closure names like "NumericDiff.<locals>.scorer" → "NumericDiff"
    if isinstance(raw_name, str) and ".<locals>." in raw_name:
        raw_name = raw_name.split(".<locals>.")[0]

    name = str(raw_name)
    if name == _INDETERMINATE_NAME:
        logger.warning(
            "Could not determine a declared name for scorer %r; using %r. "
            "Per-scorer state keyed by this name will collide with any other "
            "scorer that also resolves to it.",
            evaluator,
            name,
        )
    return name


def is_async_callable(func: Callable[..., Any]) -> bool:
    """Report whether calling ``func`` produces an awaitable.

    ``inspect.iscoroutinefunction`` inspects the object it is handed, so it
    returns False for an instance whose ``__call__`` is ``async def`` — a
    supported callable shape that would otherwise be misread as sync. This
    falls back to inspecting ``__call__`` for non-function objects.

    Detection is best-effort: a callable that hides its asyncness from
    introspection entirely (e.g. ``functools.partial`` wrapping an async
    callable class) still reports False, so callers that must not leak a
    coroutine should also handle an awaitable return value.

    Args:
        func: The callable to classify.

    Returns:
        True if calling ``func`` is expected to return an awaitable.
    """
    if inspect.iscoroutinefunction(func):
        return True
    # Plain functions and methods are fully described by the check above; only
    # other callables can carry an async __call__ that it misses. Look it up on
    # the type, mirroring how Python resolves the call itself.
    if not inspect.isfunction(func) and not inspect.ismethod(func):
        return inspect.iscoroutinefunction(type(func).__call__)
    return False


def compute_aggregate_scores(
    examples: Iterable[EvalExample],
    expected_scorer_names: Iterable[str] | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute per-scorer average score and pass rate across examples.

    Two different reasons a score can be missing, treated differently:

    - **The example failed** (task crashed, or it blew up before scoring).
      Counts as a **zero**. Skipping it would shrink the denominator to only
      the examples that worked, so a run where everything failed would report
      an empty aggregate that an ``all(v >= 0.9 for v in pass_rates.values())``
      gate reads as passing.
    - **The example succeeded and simply has no score for this scorer.**
      Skipped, as before. Scorer sets are legitimately heterogeneous — a
      benchmark binds scorers per capability from its ``success:`` block — so
      absence on a healthy example means "not applicable here", and zeroing it
      would penalise every example that correctly did not run that scorer.

    A score that recorded an ``error`` instead of a verdict is always a zero:
    the scorer ran but produced no opinion.

    ``expected_scorer_names`` is how a caller states which scorers *should*
    have run. Without it, names can only be recovered from the scores that
    exist, so a run where nothing scored yields no names at all. Callers that
    know their scorer list should pass it.

    Args:
        examples: The examples to aggregate over.
        expected_scorer_names: Scorer names that should be present on every
            example. Names absent from every example still appear in the
            output, scored 0.0.

    Returns:
        A tuple of two dicts:
        - aggregate_scores: mean numeric score per scorer name
        - pass_rates: fraction of examples that passed per scorer name

    Both dicts are empty only when there are genuinely no examples and no
    expected scorer names.
    """
    examples_list = list(examples)
    aggregate_scores: dict[str, float] = {}
    pass_rates: dict[str, float] = {}

    all_names: set[str] = set(expected_scorer_names or ())
    for ex in examples_list:
        all_names.update(ex.scores.keys())

    if not examples_list:
        return aggregate_scores, pass_rates

    if not all_names:
        # Examples ran but produced no scores at all — every one of them
        # failed before scoring. Returning {} here would read as "nothing to
        # check, therefore fine"; emit an explicit failing entry instead.
        return {NO_SCORES_KEY: 0.0}, {NO_SCORES_KEY: 0.0}

    for name in sorted(all_names):
        values: list[float] = []
        flags: list[bool] = []
        for ex in examples_list:
            score = ex.scores.get(name)
            if score is None:
                # Absent because the example failed → a zero. Absent on a
                # healthy example → this scorer does not apply to it, so it
                # does not belong in this scorer's denominator at all.
                if not ex.failed:
                    continue
                values.append(0.0)
                flags.append(False)
            elif score.error is not None:
                # The scorer ran and produced no verdict. Not performance.
                values.append(0.0)
                flags.append(False)
            else:
                values.append(score.value)
                flags.append(score.passed)

        if values:
            aggregate_scores[name] = sum(values) / len(values)
            pass_rates[name] = sum(1 for p in flags if p) / len(flags)

    return aggregate_scores, pass_rates


def compute_summary_counts(
    examples: Iterable[EvalExample],
) -> tuple[int, int, int]:
    """Compute simple summary counts for the evaluation.

    A successful example is one where the task ran *and* every scorer returned
    a real verdict — see ``EvalExample.failed``. Counting only task errors
    would report an example as successful when the task ran fine but all of
    its scorers crashed.

    Returns:
        A tuple of:
        (total_examples, successful_examples, failed_examples).
    """
    examples_list = list(examples)
    total_examples = len(examples_list)
    failed_examples = sum(1 for ex in examples_list if ex.failed)
    successful_examples = total_examples - failed_examples
    return total_examples, successful_examples, failed_examples


# ---------------------------------------------------------------------------
# Verdict preservation
#
# A scorer owns its ``Score.passed``. Platforms accept only a numeric value, so
# the verdict would be lost on the round-trip and each adapter used to rebuild
# it as ``value >= threshold``. That is only correct while ``passed`` is a
# monotone function of ``value``, which no scorer promises: ``StateMatch``
# returns ``matched/total`` with ``passed = (no failures)``.
#
# These two helpers keep the verdict in-process instead. The recording side
# (each adapter's scorer wrapper, which holds the real ``Score``) stores it; the
# reconstruction side reads it back. Nothing the platform receives changes.
# ---------------------------------------------------------------------------


#: Returned by :func:`verdict_key` when a part cannot be rendered at all. The
#: cache refuses to store or match it, so callers fall back to their threshold
#: comparison. A user object whose ``__repr__``/``__str__`` raises must not
#: crash an eval run that scored fine before verdict preservation existed.
UNRENDERABLE_KEY = "<unrenderable>"

#: Upper bound on the rendered form of a single verdict-key part, in characters.
#: ``repr`` runs user code on values that arrive from a platform, so its output
#: length is not something this module controls; an oversized rendering is
#: refused rather than hashed and retained. The bound is deliberately generous —
#: a long agent trajectory renders well inside it — because exceeding it costs
#: the verdict for that example (:data:`UNRENDERABLE_KEY`, hence a cache miss
#: and the threshold comparison), and preservation should not be lost on
#: ordinary data. Mirrors ``_MAX_LITERAL_EVAL_LEN`` in
#: ``adapters/scorers/autoevals.py``, which bounds untrusted input to the same
#: end.
MAX_KEY_PART_LEN = 1 << 20


def _render_part(part: Any) -> bytes | None:
    """Render one key part to bytes, or ``None`` if it cannot be rendered.

    ``repr`` invokes user code and may raise anything at all; a lazy proxy, a
    partially-initialized ORM row, or a strict mock all do. ``repr`` on a
    deeply nested structure can also exhaust the stack, and ``RecursionError``
    is neither ``TypeError`` nor ``ValueError``. Catching broadly here is
    deliberate: the only consequence of a failure is a cache miss.

    A rendering longer than :data:`MAX_KEY_PART_LEN` is refused for the same
    reason, so that no unbounded string derived from platform-supplied data is
    hashed or held in the cache. The bound cannot prevent a hostile ``__repr__``
    from allocating before it returns — nothing in-process can — but it does
    stop that allocation from being retained for the length of a run.
    """
    try:
        rendered = repr(part)
    except Exception:  # noqa: BLE001 - any user __repr__ failure means "no key"
        logger.debug(
            "Could not render a verdict-key part of type %s; the scorer's own "
            "verdict cannot be preserved for this example and the threshold "
            "comparison will be used instead.",
            type(part).__name__,
        )
        return None

    if len(rendered) > MAX_KEY_PART_LEN:
        logger.debug(
            "A verdict-key part of type %s rendered to %d characters, over the "
            "%d-character bound; the scorer's own verdict cannot be preserved "
            "for this example and the threshold comparison will be used "
            "instead.",
            type(part).__name__,
            len(rendered),
            MAX_KEY_PART_LEN,
        )
        return None

    return rendered.encode("utf-8", errors="replace")


def verdict_key(*parts: Any) -> str:
    """Build a stable cache key from the parts a scorer invocation saw.

    Unwraps the two agent-evals envelope types so the recording and
    reconstruction sides agree even when only one of them holds the envelope:
    a wrapper is handed a ``TaskResult`` while the reconstruction path may hold
    the bare output string it unwrapped to.

    Both sides MUST call this same function; a divergent derivation yields a
    miss, not a wrong verdict, but it silently disables preservation.

    Each part is hashed separately and the digests are then hashed together, so
    a part cannot smuggle the separator: joining raw ``repr`` output on a
    delimiter is not injective, because an object's ``__repr__`` may emit that
    delimiter itself and shift the boundary between two parts.

    Args:
        *parts: The values identifying one scorer invocation. Which parts an
            adapter passes is its own choice — see each adapter's call site for
            why those parts round-trip on that platform.

    Returns:
        A digest string usable as a dict key, or ``UNRENDERABLE_KEY`` when a
        part could not be rendered at all.
    """
    # Imported here rather than at module scope: core.types is already a
    # dependency of this module, but keeping the unwrap local documents that
    # these are the only two envelope types this helper knows about.
    from agent_evals.core.types import ExpectedResult, TaskResult

    digests: list[str] = []
    for part in parts:
        if isinstance(part, TaskResult):
            part = part.output
        elif isinstance(part, ExpectedResult):
            part = part.expected

        rendered = _render_part(part)
        if rendered is None:
            # One unrenderable part makes the whole key untrustworthy. Return a
            # sentinel the cache refuses to store or match, so the caller
            # degrades to its threshold comparison instead of raising into the
            # user's eval run.
            return UNRENDERABLE_KEY
        digests.append(hashlib.sha256(rendered).hexdigest())

    return hashlib.sha256("".join(digests).encode("ascii")).hexdigest()


class VerdictCache:
    """Per-run store of the ``(value, passed)`` pairs scorers actually returned.

    Keyed by ``(scorer_name, example_key)``. A verdict is per-example, so a
    single slot per scorer would report one example's verdict for every
    example.

    Because the key is derived from content rather than from a real example id
    (``ExampleData`` has no id field), lookups are guarded so that every failure
    mode degrades to the caller's pre-existing threshold comparison rather than
    to a confident wrong verdict:

    - **Unknown key** → miss.
    - **Conflicting verdicts for one key** → the slot is marked ambiguous and
      every later read misses. Two examples can share a key only when the
      scorer saw identical inputs; a deterministic scorer then returns the same
      verdict and there is no conflict, but a non-deterministic one (an LLM
      judge) may not. Picking one of two verdicts and attributing it to both
      examples would be worse than falling back.
    - **Value disagreement** → miss. The value read back off the platform must
      match the value recorded alongside the verdict, within a float tolerance
      since values round-trip through JSON. This catches a mis-derived key, a
      reshaped payload, or a platform that rewrites values.
    """

    #: Values round-trip through JSON on two of the three platforms, so an
    #: exact comparison would produce spurious misses.
    TOLERANCE = 1e-9

    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[float, bool] | None] = {}
        self._run_active = False

    def record(self, scorer_name: str, example_key: str, score: Any) -> None:
        """Store the verdict a scorer returned for one example.

        A second write carrying a different ``passed`` for the same key marks
        the slot ambiguous (stored as ``None``); it stays ambiguous thereafter.

        Args:
            scorer_name: The name the reconstruction side will look up.
            example_key: A key from :func:`verdict_key`.
            score: The ``Score`` the scorer returned.
        """
        if example_key == UNRENDERABLE_KEY:
            # Every unrenderable example would share this one slot, so storing
            # it would let one example's verdict be read for another's.
            return

        slot = (scorer_name, example_key)
        incoming = (float(score.value), bool(score.passed))

        if slot not in self._entries:
            self._entries[slot] = incoming
            return

        existing = self._entries[slot]
        if existing is None:
            return  # already ambiguous
        if existing[1] != incoming[1]:
            logger.warning(
                "Scorer %s returned conflicting verdicts (%s and %s) for two "
                "indistinguishable examples; falling back to the threshold "
                "comparison for both rather than guessing which is which. "
                "Either the dataset contains duplicate examples, or the scorer "
                "is non-deterministic (an LLM judge scoring the same example "
                "twice can disagree with itself).",
                scorer_name,
                existing[1],
                incoming[1],
            )
            self._entries[slot] = None

    def lookup(self, scorer_name: str, example_key: str, value: float) -> bool | None:
        """Return the recorded verdict, or ``None`` when the caller must fall back.

        Args:
            scorer_name: The scorer whose verdict is wanted.
            example_key: A key from :func:`verdict_key`, derived the same way
                the recording side derived it.
            value: The numeric value read back off the platform, used as a
                consistency check against the recorded one.

        Returns:
            The preserved ``passed``, or ``None`` on an unknown key, an
            ambiguous slot, or a value disagreement.
        """
        if example_key == UNRENDERABLE_KEY:
            return None

        slot = (scorer_name, example_key)
        if slot not in self._entries:
            return None

        entry = self._entries[slot]
        if entry is None:
            return None

        recorded_value, passed = entry
        if math.isnan(value):
            # NaN fails every comparison, so the tolerance check below would
            # accept it as agreeing with anything. A value that is not a number
            # is precisely when the recorded pair should not be trusted.
            logger.warning(
                "Scorer %s reported a NaN value where %r was recorded with its "
                "verdict; falling back to the threshold comparison.",
                scorer_name,
                recorded_value,
            )
            return None

        if abs(recorded_value - value) > self.TOLERANCE:
            logger.warning(
                "Scorer %s reported value %r but %r was recorded with its "
                "verdict; the two do not correspond, so falling back to the "
                "threshold comparison.",
                scorer_name,
                value,
                recorded_value,
            )
            return None

        return passed

    def clear(self) -> None:
        """Drop every entry. Called when an adapter starts a run."""
        self._entries.clear()

    def begin_run(self, adapter_name: str) -> None:
        """Start a run: drop the previous run's entries, refusing to overlap.

        The cache is instance state, so two concurrent runs on one adapter
        instance would share it — and the second run's :meth:`clear` would
        discard verdicts the first had already recorded, silently reverting it
        to the threshold comparison this class exists to replace. That is the
        original defect, so it fails loudly instead.

        Args:
            adapter_name: Named in the error, so the caller knows which adapter
                instance was reused.

        Raises:
            RuntimeError: If a run is already in progress on this instance.
        """
        if self._run_active:
            raise RuntimeError(
                f"{adapter_name} is already running an evaluation. One adapter "
                "instance cannot serve two concurrent runs: they share the "
                "verdict cache, so one run would discard the other's verdicts "
                "and silently fall back to threshold-derived pass/fail. "
                "Construct a separate adapter instance per concurrent run."
            )
        self._run_active = True
        self._entries.clear()

    def end_run(self) -> None:
        """Finish a run and release its entries.

        Callers invoke this in a ``finally`` block: an eval that raises must
        not leave a run marked active, nor leave its entries reachable from the
        traceback for as long as the exception is held.
        """
        self._run_active = False
        self._entries.clear()
