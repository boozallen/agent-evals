# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests that ``precondition:`` blocks reach the agent intact.

Exercises YAML load -> ``compile_capability`` -> runner shim ->
``BaseAgent.run_case`` against a real stateful agent.
"""

from __future__ import annotations

import copy
import textwrap
from pathlib import Path
from typing import Any

import pytest

from agent_evals import BaseAgent, TaskResult

# --- Fake stateful agents -------------------------------------------------


def _default_world() -> dict[str, Any]:
    return {"lights": {"kitchen": {"state": "off"}}}


class StatefulHomeAgent(BaseAgent):
    """A fake agent that consumes preconditions and mutates world state.

    Behavior:
        - Accepts ``preconditions=()`` and applies each via
          ``applier.apply(self._world)``.
        - Hand-coded prompt-to-mutation: ``"kitchen ... off"`` sets
          ``lights.kitchen.state = "off"``; ``"kitchen ... on"`` sets it
          to ``"on"``.
        - Returns ``TaskResult`` whose ``context["final_state"]`` is a
          deep copy of the post-run world (so the success-side state
          checker can read it).
    """

    name = "stateful-home"
    version = "test"

    async def setup(self) -> None:
        self._world: dict[str, Any] = _default_world()

    async def run_case(self, prompt: str, *, preconditions=(), **_) -> TaskResult:
        # Reset per-case so successive cases don't bleed state.
        self._world = _default_world()
        for applier in preconditions:
            applier.apply(self._world)

        prompt_lower = prompt.lower()
        if "kitchen" in prompt_lower:
            if "off" in prompt_lower:
                self._world["lights"]["kitchen"]["state"] = "off"
            elif "on" in prompt_lower:
                self._world["lights"]["kitchen"]["state"] = "on"

        return TaskResult(
            output="",
            context={"final_state": copy.deepcopy(self._world)},
        )


class DoNothingHomeAgent(BaseAgent):
    """Negative-control agent: applies preconditions, then ignores the prompt.

    Use to demonstrate that *without* a precondition a "kitchen off"
    scenario tautologically passes (default already matches), but *with*
    the precondition (kitchen pre-set to ``on``) the same scenario now
    fails — proving the precondition makes the assertion meaningful.
    """

    name = "do-nothing"
    version = "test"

    async def setup(self) -> None:
        self._world: dict[str, Any] = _default_world()

    async def run_case(self, prompt: str, *, preconditions=(), **_) -> TaskResult:
        self._world = _default_world()
        for applier in preconditions:
            applier.apply(self._world)
        # Intentionally do nothing in response to the prompt.
        return TaskResult(
            output="",
            context={"final_state": copy.deepcopy(self._world)},
        )


# --- YAML fixtures ---------------------------------------------------------


WITH_PRECONDITION_YAML = textwrap.dedent("""\
    benchmark: precondition-on-then-off
    description: Seed kitchen on, then assert state off.
    platform:
      name: local
      experiment: with-precond
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: kitchen off when on
            prompt: Turn the kitchen light off
            dimensions: { phrasing: imperative }
            precondition:
              state:
                lights.kitchen.state: 'on'
            success:
              state:
                lights.kitchen.state: 'off'
    """)


WITHOUT_PRECONDITION_YAML = textwrap.dedent("""\
    benchmark: no-precondition-tautology
    description: No precondition; default already matches the goal.
    platform:
      name: local
      experiment: without-precond
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: kitchen off (tautology)
            prompt: Turn the kitchen light off
            dimensions: { phrasing: imperative }
            success:
              state:
                lights.kitchen.state: 'off'
    """)


PLAIN_LEGACY_YAML = textwrap.dedent("""\
    benchmark: legacy-no-precondition
    description: Pre-spec scenario shape, no precondition block.
    platform:
      name: local
      experiment: legacy
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: kitchen on
            prompt: Turn the kitchen light on
            dimensions: { phrasing: imperative }
            success:
              state:
                lights.kitchen.state: 'on'
    """)


def _write_yaml(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content)
    return path


# --- Tests -----------------------------------------------------------------


@pytest.mark.asyncio
async def test_state_precondition_seeds_before_run(tmp_path: Path) -> None:
    """Precondition flips kitchen=on; agent turns it off; success passes.

    Includes a negative-control run with a do-nothing agent (same YAML
    fails) and a tautology run with no precondition (do-nothing agent
    passes), confirming the precondition is what makes the assertion
    informative.
    """
    from agent_evals._composition import run_benchmark_async

    yaml_path = _write_yaml(tmp_path, "with_precond.yaml", WITH_PRECONDITION_YAML)

    # Working agent: seeded kitchen=on, prompt asks for off, agent flips it off.
    result = await run_benchmark_async(yaml_path, agent=StatefulHomeAgent)
    assert result.aggregate_pass_rates["StateMatch"] == 1.0
    examples = result.eval_results["lights"].examples
    assert len(examples) == 1
    assert examples[0].scores["StateMatch"].passed is True

    # Negative control: same YAML, but the agent does nothing in response
    # to the prompt. With the precondition seeding kitchen=on, the
    # post-run state stays ``on`` and the success check correctly fails —
    # demonstrating the precondition makes the assertion informative.
    nc_result = await run_benchmark_async(yaml_path, agent=DoNothingHomeAgent)
    assert nc_result.aggregate_pass_rates["StateMatch"] == 0.0
    nc_examples = nc_result.eval_results["lights"].examples
    assert nc_examples[0].scores["StateMatch"].passed is False

    # And the tautology: same scenario but *without* the precondition,
    # the do-nothing agent passes anyway (default already matches). This
    # is the problem preconditions are designed to solve.
    no_precond_path = _write_yaml(
        tmp_path, "without_precond.yaml", WITHOUT_PRECONDITION_YAML
    )
    tauto_result = await run_benchmark_async(no_precond_path, agent=DoNothingHomeAgent)
    assert tauto_result.aggregate_pass_rates["StateMatch"] == 1.0


@pytest.mark.asyncio
async def test_scenario_without_precondition_unaffected(tmp_path: Path) -> None:
    """Pre-spec YAML (no ``precondition:``) runs unchanged with a legacy agent."""
    from agent_evals._composition import run_benchmark_async

    class LegacyAgent(BaseAgent):
        name = "legacy"
        version = "test"

        async def run_case(self, prompt: str, **_) -> TaskResult:
            # No `preconditions` parameter on purpose. The shim should
            # forward `preconditions=[]` and the **_ slot should swallow
            # it silently.
            world = _default_world()
            if "kitchen" in prompt.lower() and "on" in prompt.lower():
                world["lights"]["kitchen"]["state"] = "on"
            return TaskResult(
                output="",
                context={"final_state": world},
            )

    yaml_path = _write_yaml(tmp_path, "legacy.yaml", PLAIN_LEGACY_YAML)
    result = await run_benchmark_async(yaml_path, agent=LegacyAgent)
    assert result.aggregate_pass_rates["StateMatch"] == 1.0
