# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tool-use success-check parser: YAML key ``tool_use``.

The Literal types for ``match_calls`` and ``match_args`` must stay in
sync with the corresponding fields on ``core.checks.ToolUseCheck`` —
``to_spec()`` constructs the DTO with these exact values.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent_evals.core._registries import success_checker_registry
from agent_evals.core.checks import ToolCall, ToolUseCheck


class _ToolCallPayload(BaseModel):
    """Private boundary type for one ``tool_use.calls`` entry.

    ``extra="forbid"`` rejects unrecognised YAML keys at parse time,
    surfacing a structured ``ValidationError`` rather than a silent
    ignore or a bare ``KeyError`` inside ``to_spec()``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    args: dict[str, Any] = Field(default_factory=dict)


@success_checker_registry.register("tool_use")
class ToolUseCheckParser(BaseModel):
    """Parses ``success: { tool_use: {...} }`` into a ``ToolUseCheck``.

    Both ``match_calls`` and ``match_args`` default to ``"exact"`` when
    omitted from the YAML payload.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    calls: list[_ToolCallPayload]
    match_calls: Literal["exact", "subset", "superset", "unordered", "any_of"] = "exact"
    match_args: Literal["exact", "subset", "superset", "ignore"] = "exact"

    def to_spec(self) -> ToolUseCheck:
        return ToolUseCheck(
            calls=tuple(ToolCall(name=c.name, args=c.args) for c in self.calls),
            match_calls=self.match_calls,
            match_args=self.match_args,
        )


__all__ = ["ToolUseCheckParser"]
