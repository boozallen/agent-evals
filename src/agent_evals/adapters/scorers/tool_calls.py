# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scorers that compare tool-call sequences.

Five list-level relations:

- ``ToolCallExactMatch`` — same calls, same order, same count.
- ``ToolCallSupersetMatch`` — every reference call appears in actual,
  in order; extras allowed.
- ``ToolCallSubsetMatch`` — every actual call appears in reference,
  in order; agent may skip; extras forbidden.
- ``ToolCallUnorderedMatch`` — same multiset of calls, any order.
- ``ToolCallAnyMatch`` — at least one match against any reference
  entry; extras allowed; silence fails.

Frame filtering and per-call args comparison are shared concerns —
see ``_extract_tool_calls`` and ``_args_match`` below.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Callable
from typing import Any, Literal

from agent_evals.core._registries import scorer_registry
from agent_evals.core.types import ExpectedResult, Score, TaskResult

logger = logging.getLogger(__name__)


def _normalize_call(call: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """Return ``(name, args)`` from either call shape.

    Unknown shapes degrade to ``("", {})``; callers see a downstream
    mismatch rather than a crash.
    """
    if not isinstance(call, dict):
        return "", {}
    if "function" in call and isinstance(call["function"], dict):
        name = call["function"].get("name", "") or ""
        raw_args = call["function"].get("arguments", "")
        if isinstance(raw_args, dict):
            args: dict[str, Any] = raw_args
        else:
            try:
                parsed = json.loads(raw_args) if raw_args else {}
                args = parsed if isinstance(parsed, dict) else {}
            except ValueError, TypeError:
                args = {}
        return name, args
    if "name" in call:
        name = call.get("name", "") or ""
        raw = call.get("args", {})
        args = raw if isinstance(raw, dict) else {}
        return name, args
    return "", {}


def _extract_tool_calls(
    messages: list[dict[str, Any]] | None,
) -> list[tuple[str, dict[str, Any]]]:
    """Pull tool calls from a chat-message list, dropping every other frame."""
    out: list[tuple[str, dict[str, Any]]] = []
    for msg in messages or []:
        if not isinstance(msg, dict):
            continue
        tool_calls = msg.get("tool_calls") or []
        for tc in tool_calls:
            out.append(_normalize_call(tc))
    return out


def _safe_scorer(
    name: str,
    body: Callable[[TaskResult, ExpectedResult | None], Score],
) -> Callable[[TaskResult, ExpectedResult | None], Score]:
    """Wrap a scorer so uncaught exceptions log + return a failing Score.

    Without the log, a scorer bug would aggregate as silent ``passed=False``
    rows in the benchmark report — indistinguishable from real model failures.
    """

    @functools.wraps(body)
    def wrapped(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        try:
            return body(result, expected)
        except Exception as e:  # noqa: BLE001 -- intentional catch-all at the scorer boundary
            logger.exception("%s scorer failed", name)
            return _fail(
                name,
                f"Scorer failed with exception: {type(e).__name__}",
                error=str(e),
            )

    return wrapped


def _extract_inputs(
    name: str,
    result: TaskResult,
    expected: ExpectedResult | None,
) -> tuple[list[tuple[str, dict[str, Any]]], list[tuple[str, dict[str, Any]]]] | Score:
    """Return ``(actual, reference)`` calls, or an error Score on malformed context."""
    outputs = (result.context or {}).get("outputs")
    reference_outputs: Any = None
    if expected and expected.context:
        reference_outputs = expected.context.get("reference_outputs")

    if outputs is None:
        return _fail(
            name,
            "Actual trajectory not provided at context['outputs']",
            error="'outputs' missing from context",
        )
    if reference_outputs is None:
        return _fail(
            name,
            "Reference trajectory not provided at "
            "expected.context['reference_outputs']",
            error="'reference_outputs' missing from ExpectedResult.context",
        )

    return _extract_tool_calls(outputs), _extract_tool_calls(reference_outputs)


def _pass(name: str, reasoning: str, **metadata: Any) -> Score:
    return Score(
        name=name,
        value=1.0,
        passed=True,
        metadata=dict(metadata),
        reasoning=reasoning,
    )


def _fail(name: str, reasoning: str, **metadata: Any) -> Score:
    return Score(
        name=name,
        value=0.0,
        passed=False,
        metadata=dict(metadata),
        reasoning=reasoning,
    )


def _args_match(actual: dict[str, Any], reference: dict[str, Any], mode: str) -> bool:
    """Compare tool-call args under the selected mode.

    Direction matches the list-level ``match_calls`` vocabulary so the
    same word means the same containment regardless of axis.

    - ``"ignore"``: always true (names still compared elsewhere).
    - ``"exact"``: actual == reference (same keys, same values).
    - ``"subset"``: actual ⊆ reference — every kv the agent passed is
      allowed by the reference; **no extras** beyond what reference lists.
    - ``"superset"``: reference ⊆ actual — every kv the reference lists
      is present and equal in actual; **extras OK**.
    """
    if mode == "ignore":
        return True
    if mode == "exact":
        return actual == reference
    if mode == "subset":
        return all(k in reference and reference[k] == v for k, v in actual.items())
    if mode == "superset":
        return all(k in actual and actual[k] == v for k, v in reference.items())
    return False


@scorer_registry.register("ToolCallExactMatch")
def ToolCallExactMatch(  # noqa: N802
    tool_args_match_mode: Literal["exact", "subset", "superset", "ignore"] = "exact",
) -> Callable:
    """Pass when actual and reference tool-call sequences match pairwise, in order.

    Same count, same order, same names; per-call args compared per
    ``tool_args_match_mode``. Non-tool-call frames are filtered out
    before comparison, so YAML references list only the expected calls
    without tool-response or prose padding.

    Args:
        tool_args_match_mode: ``exact`` (default), ``subset`` (actual
            kvs ⊆ reference; no extras), ``superset`` (reference kvs ⊆
            actual; extras OK), or ``ignore`` (names only).

    Returns:
        A scorer callable conforming to the Scorer Protocol.
    """

    name = "ToolCallExactMatch"

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        inputs = _extract_inputs(name, result, expected)
        if isinstance(inputs, Score):
            return inputs
        actual_calls, reference_calls = inputs

        if len(actual_calls) != len(reference_calls):
            return _fail(
                name,
                f"Expected {len(reference_calls)} tool call(s), "
                f"got {len(actual_calls)}",
                actual_count=len(actual_calls),
                reference_count=len(reference_calls),
                tool_args_match_mode=tool_args_match_mode,
            )

        for i, ((a_name, a_args), (r_name, r_args)) in enumerate(
            zip(actual_calls, reference_calls, strict=True)
        ):
            if a_name != r_name:
                return _fail(
                    name,
                    f"Tool #{i}: expected {r_name!r}, got {a_name!r}",
                    mismatch_index=i,
                    expected_name=r_name,
                    actual_name=a_name,
                    tool_args_match_mode=tool_args_match_mode,
                )
            if not _args_match(a_args, r_args, tool_args_match_mode):
                return _fail(
                    name,
                    f"Tool #{i} ({a_name!r}) args did not match "
                    f"under mode={tool_args_match_mode!r}",
                    mismatch_index=i,
                    tool_name=a_name,
                    expected_args=r_args,
                    actual_args=a_args,
                    tool_args_match_mode=tool_args_match_mode,
                )

        return _pass(
            name,
            f"All {len(actual_calls)} tool call(s) matched",
            call_count=len(actual_calls),
            tool_args_match_mode=tool_args_match_mode,
        )

    return _safe_scorer(name, scorer)


@scorer_registry.register("ToolCallSupersetMatch")
def ToolCallSupersetMatch(  # noqa: N802
    tool_args_match_mode: Literal["exact", "subset", "superset", "ignore"] = "exact",
) -> Callable:
    """Pass when reference is an in-order subsequence of actual.

    Each reference call must match some later actual call by name and
    args (under ``tool_args_match_mode``); intervening and trailing
    actual calls are tolerated.
    """

    name = "ToolCallSupersetMatch"

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        inputs = _extract_inputs(name, result, expected)
        if isinstance(inputs, Score):
            return inputs
        actual_calls, reference_calls = inputs

        # Subsequence search: each reference call must appear later in
        # actual than the previous match. Advancing a single actual cursor
        # preserves order without backtracking.
        a_idx = 0
        for r_idx, (r_name, r_args) in enumerate(reference_calls):
            while a_idx < len(actual_calls):
                a_name, a_args = actual_calls[a_idx]
                a_idx += 1
                if a_name == r_name and _args_match(
                    a_args, r_args, tool_args_match_mode
                ):
                    break
            else:
                return _fail(
                    name,
                    f"Reference call #{r_idx} ({r_name!r}) not found in actual "
                    f"under mode={tool_args_match_mode!r}",
                    unmatched_reference_index=r_idx,
                    unmatched_reference_name=r_name,
                    actual_count=len(actual_calls),
                    reference_count=len(reference_calls),
                    tool_args_match_mode=tool_args_match_mode,
                )

        return _pass(
            name,
            f"All {len(reference_calls)} reference call(s) matched",
            actual_count=len(actual_calls),
            reference_count=len(reference_calls),
            tool_args_match_mode=tool_args_match_mode,
        )

    return _safe_scorer(name, scorer)


@scorer_registry.register("ToolCallSubsetMatch")
def ToolCallSubsetMatch(  # noqa: N802
    tool_args_match_mode: Literal["exact", "subset", "superset", "ignore"] = "exact",
) -> Callable:
    """Pass when actual is an in-order subsequence of reference.

    The agent may skip reference calls, but each actual call must match
    some later reference call by name and args (under
    ``tool_args_match_mode``). An actual call with no remaining
    reference match fails.
    """

    name = "ToolCallSubsetMatch"

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        inputs = _extract_inputs(name, result, expected)
        if isinstance(inputs, Score):
            return inputs
        actual_calls, reference_calls = inputs

        # Subsequence search, mirror of ToolCallSupersetMatch: actual must
        # be a subsequence of reference. If the reference cursor exhausts
        # while actual calls remain, the agent introduced an extra.
        r_idx = 0
        for a_idx, (a_name, a_args) in enumerate(actual_calls):
            while r_idx < len(reference_calls):
                r_name, r_args = reference_calls[r_idx]
                r_idx += 1
                if a_name == r_name and _args_match(
                    a_args, r_args, tool_args_match_mode
                ):
                    break
            else:
                return _fail(
                    name,
                    f"Actual call #{a_idx} ({a_name!r}) not found in reference "
                    f"under mode={tool_args_match_mode!r}",
                    unmatched_actual_index=a_idx,
                    unmatched_actual_name=a_name,
                    actual_count=len(actual_calls),
                    reference_count=len(reference_calls),
                    tool_args_match_mode=tool_args_match_mode,
                )

        return _pass(
            name,
            f"All {len(actual_calls)} actual call(s) matched against reference",
            actual_count=len(actual_calls),
            reference_count=len(reference_calls),
            tool_args_match_mode=tool_args_match_mode,
        )

    return _safe_scorer(name, scorer)


@scorer_registry.register("ToolCallUnorderedMatch")
def ToolCallUnorderedMatch(  # noqa: N802
    tool_args_match_mode: Literal["exact", "subset", "superset", "ignore"] = "exact",
) -> Callable:
    """Pass when actual and reference share the same multiset of calls.

    Counts must agree; each reference call is paired with a distinct
    actual call by name and args (under ``tool_args_match_mode``).
    Order is not constrained.
    """

    name = "ToolCallUnorderedMatch"

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        inputs = _extract_inputs(name, result, expected)
        if isinstance(inputs, Score):
            return inputs
        actual_calls, reference_calls = inputs

        if len(actual_calls) != len(reference_calls):
            return _fail(
                name,
                f"Expected {len(reference_calls)} tool call(s), "
                f"got {len(actual_calls)}",
                actual_count=len(actual_calls),
                reference_count=len(reference_calls),
                tool_args_match_mode=tool_args_match_mode,
            )

        # Greedy pairing. With ``tool_args_match_mode`` in {ignore, subset},
        # multiple reference calls can match the same actual call, so a
        # greedy first-fit can fail on inputs that a backtracking matcher
        # would accept. The simple form is a deliberate choice — pin a
        # failing case in tests/unit/scorers/ if a benchmark surfaces one.
        consumed = [False] * len(actual_calls)
        for r_idx, (r_name, r_args) in enumerate(reference_calls):
            found = False
            for a_idx, (a_name, a_args) in enumerate(actual_calls):
                if consumed[a_idx]:
                    continue
                if a_name == r_name and _args_match(
                    a_args, r_args, tool_args_match_mode
                ):
                    consumed[a_idx] = True
                    found = True
                    break
            if not found:
                return _fail(
                    name,
                    f"Reference call #{r_idx} ({r_name!r}) has no matching "
                    f"actual call under mode={tool_args_match_mode!r}",
                    unmatched_reference_index=r_idx,
                    unmatched_reference_name=r_name,
                    actual_count=len(actual_calls),
                    reference_count=len(reference_calls),
                    tool_args_match_mode=tool_args_match_mode,
                )

        return _pass(
            name,
            f"All {len(reference_calls)} call(s) matched (any order)",
            actual_count=len(actual_calls),
            reference_count=len(reference_calls),
            tool_args_match_mode=tool_args_match_mode,
        )

    return _safe_scorer(name, scorer)


@scorer_registry.register("ToolCallAnyMatch")
def ToolCallAnyMatch(  # noqa: N802
    tool_args_match_mode: Literal["exact", "subset", "superset", "ignore"] = "exact",
) -> Callable:
    """Pass when any actual call matches any reference entry; silence fails.

    Disjunctive existence check — the YAML lists *acceptable* shapes
    (e.g., a status query that accepts ``device_type=lights`` or
    ``device_type=all``) and the agent passes by hitting any one. Extras
    are allowed; an empty actual trajectory fails.

    Args:
        tool_args_match_mode: ``exact`` (default), ``subset`` (actual
            kvs ⊆ reference; no extras), ``superset`` (reference kvs ⊆
            actual; extras OK), or ``ignore`` (names only).

    Returns:
        A scorer callable conforming to the Scorer Protocol.
    """

    name = "ToolCallAnyMatch"

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        inputs = _extract_inputs(name, result, expected)
        if isinstance(inputs, Score):
            return inputs
        actual_calls, reference_calls = inputs

        # Silence fails — distinct failure mode from "had calls but
        # none matched", surfaced separately so debug logs are clear.
        if not actual_calls:
            return _fail(
                name,
                "Agent made no tool calls; any_of requires at least one match.",
                actual_count=0,
                reference_count=len(reference_calls),
                tool_args_match_mode=tool_args_match_mode,
                failure_mode="silence",
            )

        for a_name, a_args in actual_calls:
            for r_name, r_args in reference_calls:
                if a_name == r_name and _args_match(
                    a_args, r_args, tool_args_match_mode
                ):
                    return _pass(
                        name,
                        f"Matched call {a_name!r} against reference "
                        f"under mode={tool_args_match_mode!r}",
                        actual_count=len(actual_calls),
                        reference_count=len(reference_calls),
                        tool_args_match_mode=tool_args_match_mode,
                        matched_name=a_name,
                    )

        return _fail(
            name,
            f"None of {len(actual_calls)} actual call(s) matched any of "
            f"{len(reference_calls)} reference entr"
            f"{'y' if len(reference_calls) == 1 else 'ies'} "
            f"under mode={tool_args_match_mode!r}",
            actual_count=len(actual_calls),
            reference_count=len(reference_calls),
            tool_args_match_mode=tool_args_match_mode,
            failure_mode="no_match",
        )

    return _safe_scorer(name, scorer)


__all__ = [
    "ToolCallExactMatch",
    "ToolCallSupersetMatch",
    "ToolCallSubsetMatch",
    "ToolCallUnorderedMatch",
    "ToolCallAnyMatch",
]
