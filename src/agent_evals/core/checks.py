# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Domain DTOs for success criteria.

Frozen dataclass union (`CheckSpec`) carrying YAML-parsed criteria
between the parser layer and the dispatch layer. This module imports
only from the standard library; the DTO carries data, not behavior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One expected tool call: a name plus an args mapping.

    `args` is `dict[str, Any]` because YAML payloads are arbitrary
    nested user data. Treat the dict as immutable by convention — the
    dataclass freeze is shallow.
    """

    name: str
    args: dict[str, Any]


@dataclass(frozen=True, slots=True)
class StateCheck:
    """Pass when the agent's final state matches `expected_state`."""

    expected_state: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolUseCheck:
    """Pass when the agent's tool-call sequence matches `calls` under the chosen modes.

    `match_calls` selects the list-level relation between actual and
    reference call sequences; `match_args` selects the per-call args
    comparison. The two axes are orthogonal. Both `Literal` types are
    part of the DTO's schema — every consumer must know the same
    vocabulary to dispatch on the value.
    """

    calls: tuple[ToolCall, ...]
    match_calls: Literal["exact", "subset", "superset", "unordered", "any_of"] = "exact"
    match_args: Literal["exact", "subset", "superset", "ignore"] = "exact"


CheckSpec = StateCheck | ToolUseCheck
"""Closed union over every supported check type.

Adding a new check type extends this union; the dispatch site is
exhaustively type-checked, so omissions fail at static-analysis time
rather than runtime.
"""
