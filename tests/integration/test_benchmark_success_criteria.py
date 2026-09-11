# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for per-scenario success checkers.

Locks the user-visible invariants:

1. A capability mixing heterogeneous ``success:`` blocks runs end-to-end.
2. Per-scorer denominators in the merged ``EvalResult`` reflect only the
   scenarios that declared the matching success checker (a scorer that
   ran on 2 of 3 scenarios shows in ``pass_rates`` with denominator 2,
   not 3).
3. The markdown report renders truthfully — no false-fail rows for
   scenarios that didn't declare a given scorer's success checker.

The fake agent is constructed so we get a *fractional* pass rate
(``StateMatch = 2/2``, ``ToolCallExactMatch = 1/2``) — that's the
distinction between "ran on N" and "passed on N" we want to assert.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_evals import BaseAgent
from agent_evals.core.types import TaskResult

# ---------------------------------------------------------------------------
# Fixture YAML — the canonical "mixed-shape lights capability" scenario set.
# ---------------------------------------------------------------------------
#
# Three scenarios:
#   1. state-only       — kitchen on
#   2. tool_use-only    — turn on living room (we make the agent NOT match,
#                          so ToolCallExactMatch FAILS for this scenario)
#   3. both             — turn on hallway, agent matches BOTH state and
#                          trajectory, so both checkers PASS
#
# With the agent below this gives:
#   StateMatch:           2/2 = 1.0   (scenarios 1 + 3, both pass)
#   ToolCallExactMatch: 1/2 = 0.5  (scenario 3 passes, scenario 2 fails)
MIXED_LIGHTS_YAML = textwrap.dedent(
    """
    benchmark: mixed-lights-v1
    description: Heterogeneous-success integration test
    platform:
      name: local
      experiment: mixed-lights
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: kitchen on (state only)
            prompt: Turn on the kitchen light
            dimensions: { phrasing: imperative }
            success:
              state:
                lights.kitchen.state: "on"

          - id: 2
            name: living room on (tool_use only)
            prompt: Turn on the living room light
            dimensions: { phrasing: imperative }
            success:
              tool_use:
                calls:
                  - name: set_light_state
                    args: { room: living_room, state: "on" }

          - id: 3
            name: hallway on (both)
            prompt: Turn on the hallway light
            dimensions: { phrasing: imperative }
            success:
              state:
                lights.hallway.state: "on"
              tool_use:
                calls:
                  - name: set_light_state
                    args: { room: hallway, state: "on" }
    """
).strip()


class MixedLightsAgent(BaseAgent):
    """Stub agent that makes one scenario per scorer fail.

    Behavior:
      - Scenario 1 (kitchen):  matches state.   StateMatch passes.
      - Scenario 2 (living):   emits a *wrong* tool call.
                                ToolCallExactMatch fails.
      - Scenario 3 (hallway):  emits the right tool call AND matches state.
                                Both checkers pass.

    This produces:
        StateMatch:            2 of 2 declared scenarios pass = 1.0
        ToolCallExactMatch: 1 of 2 declared scenarios pass = 0.5
    """

    name = "mixed-lights-agent"
    version = "test"

    async def run_case(self, prompt: str, **kwargs) -> TaskResult:
        prompt_lower = prompt.lower()

        tool_calls: list[dict] = []
        final_state: dict = {"lights": {}}

        if "kitchen" in prompt_lower:
            # Scenario 1: state-only. Match state; tool calls don't matter.
            final_state["lights"]["kitchen"] = {"state": "on"}
            tool_calls.append(
                {"name": "set_light_state", "args": {"room": "kitchen", "state": "on"}}
            )
        elif "living" in prompt_lower:
            # Scenario 2: tool_use-only. Emit a WRONG trajectory so
            # ToolCallExactMatch fails. (No state assertion to satisfy.)
            tool_calls.append({"name": "noop", "args": {}})
        elif "hallway" in prompt_lower:
            # Scenario 3: both. Match BOTH state and trajectory.
            final_state["lights"]["hallway"] = {"state": "on"}
            tool_calls.append(
                {"name": "set_light_state", "args": {"room": "hallway", "state": "on"}}
            )

        trajectory = [{"role": "assistant", "tool_calls": tool_calls}]
        return TaskResult(
            output="",
            context={"outputs": trajectory, "final_state": final_state},
        )


@pytest.fixture
def mixed_lights_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "mixed_lights.yaml"
    p.write_text(MIXED_LIGHTS_YAML)
    return p


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mixed_lights_capability_runs_to_completion(mixed_lights_yaml: Path):
    """Heterogeneous success-checker scenarios run end-to-end and merge correctly.

    Locks invariants:
      - Capability is present in result.eval_results.
      - Examples list is the union of all three scenarios.
      - Per-scorer denominators count only the scenarios that declared
        the matching success checker (NOT the total scenario count).
      - Numerators reflect actual pass/fail across declared scenarios.
    """
    from agent_evals._composition import run_benchmark_async

    result = await run_benchmark_async(mixed_lights_yaml, agent=MixedLightsAgent)

    assert result.benchmark == "mixed-lights-v1"
    assert "lights" in result.eval_results

    er = result.eval_results["lights"]

    # Union of all three scenarios.
    assert len(er.examples) == 3, (
        f"expected 3 merged examples (one per scenario), got {len(er.examples)}"
    )

    # Per-scorer denominators: StateMatch ran on 2 scenarios (1 + 3),
    # ToolCallExactMatch ran on 2 scenarios (2 + 3).
    sm_examples = [ex for ex in er.examples if "StateMatch" in ex.scores]
    tcs_examples = [ex for ex in er.examples if "ToolCallExactMatch" in ex.scores]

    assert len(sm_examples) == 2, (
        "StateMatch denominator should be 2 (only scenarios 1 and 3 declared "
        f"state); got {len(sm_examples)}"
    )
    assert len(tcs_examples) == 2, (
        "ToolCallExactMatch denominator should be 2 (only scenarios 2 and 3 "
        f"declared tool_use); got {len(tcs_examples)}"
    )

    # Numerators: StateMatch passes on both declared, TCS passes on 1 of 2.
    assert er.pass_rates["StateMatch"] == pytest.approx(1.0), (
        "StateMatch should pass on both scenarios that declared state; "
        f"got pass_rate={er.pass_rates['StateMatch']}"
    )
    assert er.pass_rates["ToolCallExactMatch"] == pytest.approx(0.5), (
        "ToolCallExactMatch should pass on 1 of 2 scenarios that declared "
        f"tool_use; got pass_rate={er.pass_rates['ToolCallExactMatch']}"
    )

    # No scenario carries a score for a scorer it didn't declare —
    # cross-scenario leakage is the very thing per-scenario success blocks
    # exist to prevent.
    for ex in er.examples:
        scenario_id = ex.metadata["scenario_id"]
        if scenario_id == 1:
            assert set(ex.scores) == {"StateMatch"}
        elif scenario_id == 2:
            assert set(ex.scores) == {"ToolCallExactMatch"}
        elif scenario_id == 3:
            assert set(ex.scores) == {"StateMatch", "ToolCallExactMatch"}
        else:
            pytest.fail(f"unexpected scenario_id {scenario_id}")


@pytest.mark.asyncio
async def test_markdown_report_truthful_cells(mixed_lights_yaml: Path):
    """Markdown report renders without false-fail cells for non-applicable scorers.

    Asserts:
      - The "lights" capability section appears in the report.
      - Each scenario's PASS/FAIL row reflects only the scorers it declared
        — scenario 1 (state-only) is PASS, scenario 2 (tool_use-only,
        agent-mismatched) is FAIL, scenario 3 (both, agent-matched) is PASS.
      - The aggregate score line in the report counts 2/3 scenarios passing
        (scenarios 1 and 3 — every-scorer-passed; scenario 2 fails).
    """
    from agent_evals._composition import run_benchmark_async

    result = await run_benchmark_async(mixed_lights_yaml, agent=MixedLightsAgent)
    md = result.to_markdown()

    # Capability heading is present.
    assert "### lights" in md, "expected per-capability heading in tasks section"

    # Aggregate score: 2/3 passed (scenarios 1 and 3 had every declared
    # scorer pass; scenario 2's only declared scorer failed).
    assert "2/3" in md, (
        f"expected aggregate '2/3' (passed/total) in report; report was:\n{md}"
    )

    # Per-scenario rows. Scenario 1 PASS, 2 FAIL, 3 PASS.
    # Use the human-readable "name" column to anchor each row.
    lines = md.splitlines()

    def _row_for(name_substring: str) -> str:
        for line in lines:
            if name_substring in line:
                return line
        raise AssertionError(
            f"no report row contains {name_substring!r}; report was:\n{md}"
        )

    kitchen_row = _row_for("kitchen on")
    living_row = _row_for("living room on")
    hallway_row = _row_for("hallway on")

    assert "PASS" in kitchen_row and "FAIL" not in kitchen_row, (
        f"kitchen scenario should be PASS (state-only, agent matched); "
        f"row was: {kitchen_row!r}"
    )
    assert "FAIL" in living_row, (
        f"living-room scenario should be FAIL (agent emitted wrong tool call); "
        f"row was: {living_row!r}"
    )
    assert "PASS" in hallway_row and "FAIL" not in hallway_row, (
        f"hallway scenario should be PASS (both checkers matched); "
        f"row was: {hallway_row!r}"
    )

    # Aggregate pass-rate accessors (used by user-written CI gates) reflect
    # the merged per-scorer denominators, not capability-wide counts.
    assert result.aggregate_pass_rates["StateMatch"] == pytest.approx(1.0)
    assert result.aggregate_pass_rates["ToolCallExactMatch"] == pytest.approx(0.5)
