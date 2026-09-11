# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Native state-assertion scorer.

Validates that an agent's final world state matches expected dotted-path
values. Unlike autoevals/agentevals adapters, this scorer has no external
dependencies.

Agent convention:
    Task must return TaskResult with context={"final_state": <state_dict>}.

Providing expected state:
    - Factory-level (shared): pass expected_state= when creating the scorer.
    - Per-example: attach ExpectedResult.context["expected_state"] to each
      ExampleData. Per-example value overrides factory default when present.
"""

import logging
from collections.abc import Callable
from typing import Any

from agent_evals.core._registries import scorer_registry
from agent_evals.core.types import ExpectedResult, Score, TaskResult

logger = logging.getLogger(__name__)


_MISSING = object()


def _dotted_get(d: Any, path: str) -> Any:
    """Traverse a nested dict via dotted path. Returns _MISSING if absent."""
    current = d
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return _MISSING
        current = current[part]
    return current


@scorer_registry.register("StateMatch")
def StateMatch(  # noqa: N802
    expected_state: dict[str, Any] | None = None,
) -> Callable:
    """Factory for a dotted-path state assertion scorer.

    Checks that every path in expected_state exists in the agent's final
    state and equals the declared value.

    Args:
        expected_state: Default {dotted_path: value} map applied to every
            example. Overridden per-example when ExpectedResult.context
            contains an "expected_state" key.

    Returns:
        Callable scorer conforming to Scorer Protocol.
    """

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        try:
            per_example = None
            if expected and expected.context:
                per_example = expected.context.get("expected_state")
            paths = per_example if per_example is not None else expected_state

            if paths is None:
                return Score(
                    name="StateMatch",
                    value=0.0,
                    passed=False,
                    metadata={"error": "no expected_state provided"},
                    reasoning=(
                        "StateMatch requires expected_state at factory level "
                        "or per-example via ExpectedResult.context['expected_state']"
                    ),
                )

            state = (result.context or {}).get("final_state")
            if not isinstance(state, dict):
                return Score(
                    name="StateMatch",
                    value=0.0,
                    passed=False,
                    metadata={
                        "error": "'final_state' missing or not a dict",
                        "got_type": type(state).__name__,
                    },
                    reasoning=(
                        "Expected dict at TaskResult.context['final_state']; "
                        f"got {type(state).__name__}"
                    ),
                )

            failures: dict[str, dict[str, Any]] = {}
            for path, want in paths.items():
                got = _dotted_get(state, path)
                if got is _MISSING:
                    failures[path] = {
                        "expected": want,
                        "actual": None,
                        "reason": "path missing",
                    }
                elif got != want:
                    failures[path] = {"expected": want, "actual": got}

            total = len(paths)
            value = (total - len(failures)) / total if total else 1.0
            return Score(
                name="StateMatch",
                value=value,
                passed=not failures,
                metadata={
                    "failures": failures,
                    "paths_checked": total,
                },
                reasoning=(
                    f"All {total} paths matched"
                    if not failures
                    else f"{len(failures)}/{total} path(s) failed: {sorted(failures)}"
                ),
            )

        except Exception as e:  # noqa: BLE001 -- matches existing scorer pattern
            return Score(
                name="StateMatch",
                value=0.0,
                passed=False,
                metadata={"error": str(e)},
                reasoning=f"Scorer failed with exception: {type(e).__name__}",
            )

    return scorer


__all__ = ["StateMatch"]
