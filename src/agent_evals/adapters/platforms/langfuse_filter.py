# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Clause grammar for the Langfuse adapter's caller-supplied ``filter``.

The library's ``filter`` is the *library's* own form, shared in shape
across platform adapters (``"trace.status = 'OK'"``,
``"scores.Factuality > 0.8"``). It is not Langfuse's form, and it is
applied here rather than forwarded.

**Why the clause is not handed to the SDK.** ``api.trace.list`` accepts
its own ``filter`` — a JSON array of typed conditions — and its docstring
states that when supplied it *takes precedence over* the ``userId``,
``sessionId``, ``name``, ``tags``, ``version``, ``release``,
``environment``, and timestamp query parameters. Routing a caller's
clause through it would not merely fail to parse: it would silently
discard the session and user selection the adapter had just expressed,
returning a plausible result set answering a question nobody asked.
Client-side application is the correct seam here, not a shortcut.

**Why the grammar is small.** ``AND``-only, six comparison operators, a
closed field allowlist. A caller who wants ``OR``, ``LIKE``, or
parentheses gets a ``ValueError`` naming the token rather than a wrong
result set — the limitation surfaces at the call, not in the numbers.

This module is deliberately Langfuse-local. MLflow validates its own
clause form in ``mlflow.py``, and platform adapters must not import one
another (``import-linter`` Contract 2 forbids the edge in both
directions). Extracting a shared grammar is a plausible follow-up;
reaching across the family here would trade an architecture boundary for
a few saved lines.
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from agent_evals.adapters.platforms.langfuse_client import TraceRecord

# Fields a clause may name, mapped onto TraceRecord attributes.
#
# Closed by design: an unknown field is a parse error, not an
# always-false clause. A silently-false clause returns an empty set that
# reads exactly like "no trace matched", so a typo would look like a
# finding about the data.
_RECORD_FIELDS: dict[str, str] = {
    "id": "id",
    "session_id": "session_id",
    "user_id": "user_id",
    "input": "input",
    "output": "output",
}

# Prefix for reading an arbitrary key out of a trace's metadata, e.g.
# `metadata.environment = 'prod'`.
#
# This is what makes the clause worth having. `id`, `session_id`, and
# `user_id` are already expressible as server-side selection parameters,
# so a grammar limited to them would duplicate the query rather than
# extend it. A trace's distinguishing attributes live in its metadata.
_METADATA_PREFIX = "metadata."

# Score *values* are deliberately absent from the allowlist. The trace
# listing returns `scores` as a list of score IDs (`Optional[List[str]]`),
# not values, so `scores.Factuality > 0.8` cannot be evaluated from a
# listing at all — resolving it would mean an extra API call per trace.
# Naming such a field raises, with a message that says so, rather than
# comparing against an ID string and reporting the result as a score
# threshold.
_SCORES_HINT = (
    "score values are not available from a trace listing (the API returns "
    "score IDs, not values), so they cannot be filtered on here"
)

_OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
    # Two-character operators are matched first; see _CLAUSE_RE.
    "!=": lambda a, b: a != b,
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    "=": lambda a, b: a == b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
}

# `!=`, `>=`, `<=` precede `=`, `>`, `<` so the two-character forms win.
_CLAUSE_RE = re.compile(
    r"^\s*(?P<field>[A-Za-z_][\w.]*)\s*(?P<op>!=|>=|<=|=|>|<)\s*(?P<literal>.+?)\s*$"
)

# Splits on `AND` as a standalone word, case-insensitively. Quoted
# literals containing the word (`metadata.note = 'A and B'`) are not
# split, because the delimiter must be surrounded by whitespace and
# unquoted — a bare `and` inside quotes has no whitespace-delimited
# match outside them only when the quotes are balanced, which the
# literal parser then verifies.
_AND_RE = re.compile(r"\s+AND\s+", re.IGNORECASE)

# The literal parser runs on caller-supplied text. Bounding the input
# keeps pathological or oversized strings out of CPython's parser
# entirely rather than depending on its recursion guards holding across
# versions. Mirrors the bound the scorer family applies to untrusted LLM
# output; kept local because adapter families must not import each other.
_MAX_LITERAL_LEN = 1 << 10


@dataclass(frozen=True)
class _Clause:
    """One parsed ``<field> <op> <literal>`` comparison."""

    field: str
    operator: str
    literal: Any
    source: str


def _parse_literal(raw: str, clause: str) -> Any:
    """Parse a clause's right-hand side into a Python value.

    Uses ``ast.literal_eval`` (never ``eval`` / ``exec``), length-bounded
    first. A bare word that is not a Python literal — ``prod`` rather
    than ``'prod'`` — is accepted as a string, because requiring quotes
    everywhere is a papercut that produces no safety: the value never
    reaches an evaluator, only a comparison.
    """
    if len(raw) > _MAX_LITERAL_LEN:
        raise ValueError(
            f"filter clause literal is too long "
            f"({len(raw)} > {_MAX_LITERAL_LEN} chars): {clause!r}"
        )
    try:
        return ast.literal_eval(raw)
    except ValueError, SyntaxError, MemoryError, RecursionError:
        # Not a Python literal. A bare unquoted word is a common and
        # unambiguous intent; anything containing a quote character was
        # trying to be a quoted string and got it wrong, so that is an
        # error rather than a bare word.
        if '"' in raw or "'" in raw:
            raise ValueError(
                f"filter clause has an unparseable literal: {clause!r} "
                f"(check for unbalanced quotes)"
            ) from None
        return raw


def _parse_clause(clause: str) -> _Clause:
    """Parse one clause, raising ``ValueError`` naming the offending token."""
    match = _CLAUSE_RE.match(clause)
    if match is None:
        raise ValueError(
            f"filter clause is not understood: {clause!r}. Expected "
            f"'<field> <operator> <value>' with operator one of "
            f"{', '.join(sorted(_OPERATORS))}, clauses joined by AND."
        )

    field = match.group("field")
    if field.startswith("scores.") or field == "scores":
        raise ValueError(f"filter field {field!r} is not supported: {_SCORES_HINT}")
    if not field.startswith(_METADATA_PREFIX) and field not in _RECORD_FIELDS:
        supported = ", ".join([*sorted(_RECORD_FIELDS), f"{_METADATA_PREFIX}<key>"])
        raise ValueError(
            f"filter field {field!r} is not supported. Supported fields: {supported}."
        )
    if field == _METADATA_PREFIX.rstrip("."):  # bare `metadata`
        raise ValueError(
            f"filter field 'metadata' needs a key, e.g. "
            f"'{_METADATA_PREFIX}environment = prod'"
        )
    if field.startswith(_METADATA_PREFIX) and not field[len(_METADATA_PREFIX) :]:
        raise ValueError(f"filter clause names an empty metadata key: {clause!r}")

    return _Clause(
        field=field,
        operator=match.group("op"),
        literal=_parse_literal(match.group("literal"), clause),
        source=clause,
    )


def parse_filter(filter_clause: str) -> list[_Clause]:
    """Parse a filter string into its clauses.

    Args:
        filter_clause: One or more ``<field> <op> <literal>`` clauses
            joined by ``AND``.

    Returns:
        The parsed clauses, all of which must hold for a record to match.

    Raises:
        ValueError: If any clause is unparseable, names an unsupported
            field, or uses an unsupported operator. Naming the offending
            token matters: returning the unfiltered set instead would be
            the inert-parameter defect relocated — a caller receives a
            plausible result set that does not answer their question,
            with nothing to indicate the difference.
    """
    if not filter_clause or not filter_clause.strip():
        raise ValueError("filter clause is empty")

    clauses = [part for part in _AND_RE.split(filter_clause.strip()) if part.strip()]
    if not clauses:
        raise ValueError(f"filter clause is empty after parsing: {filter_clause!r}")
    return [_parse_clause(clause) for clause in clauses]


def _read_field(record: TraceRecord, field: str) -> Any:
    if field.startswith(_METADATA_PREFIX):
        key = field[len(_METADATA_PREFIX) :]
        return record.metadata.get(key)
    return getattr(record, _RECORD_FIELDS[field])


def _compare(clause: _Clause, actual: Any) -> bool:
    """Evaluate one clause against a record's value.

    Equality is compared as-is; ordering falls back to a string
    comparison when the operands are not mutually comparable, so
    ``metadata.version >= '2'`` behaves rather than raising. A comparison
    that is meaningless even as strings (``None`` against a number) is
    False, not an error: the record simply does not match.
    """
    operator = _OPERATORS[clause.operator]
    expected = clause.literal

    if clause.operator in ("=", "!="):
        if actual == expected:
            return clause.operator == "="
        # A JSON payload read back as a dict/list will not equal a string
        # literal even when the caller meant the rendered form, so retry
        # equality on the string rendering before concluding no-match.
        if isinstance(expected, str) and str(actual) == expected:
            return clause.operator == "="
        return clause.operator == "!="

    if actual is None:
        return False
    try:
        return operator(actual, expected)
    except TypeError:
        try:
            return operator(str(actual), str(expected))
        except TypeError:
            return False


def matches_filter(record: TraceRecord, clauses: list[_Clause]) -> bool:
    """Return True when every clause holds for ``record``.

    Clauses are conjunctive: the grammar has no ``OR``, so a record must
    satisfy all of them. This is applied *after* the platform-side
    session / user / count selection and must never widen it.
    """
    return all(
        _compare(clause, _read_field(record, clause.field)) for clause in clauses
    )
