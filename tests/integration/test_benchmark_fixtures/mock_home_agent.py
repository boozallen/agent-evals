# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Mock home-automation agent for benchmark integration tests.

Deterministic: parses keywords from the prompt, emits a trajectory and a
resulting final_state. Not a real LLM.
"""

from agent_evals import BaseAgent
from agent_evals.core.types import TaskResult


class MockHomeAgent(BaseAgent):
    """Deterministic stand-in for a real home-automation agent."""

    name = "mock-home"
    version = "test"

    async def run_case(self, prompt: str, **kwargs) -> TaskResult:
        prompt_lower = prompt.lower()

        tool_calls = []
        final_state: dict = {"lights": {}, "doors": {}}

        if "kitchen" in prompt_lower and "on" in prompt_lower:
            tool_calls.append(
                {"name": "set_light_state", "args": {"room": "kitchen", "state": "on"}}
            )
            final_state["lights"]["kitchen"] = {"state": "on"}

        if "lock" in prompt_lower and "front" in prompt_lower:
            tool_calls.append({"name": "lock_door", "args": {"door": "front"}})
            final_state["doors"]["front"] = "locked"

        trajectory = [{"role": "assistant", "tool_calls": tool_calls}]

        return TaskResult(
            output="",
            context={
                "outputs": trajectory,
                "final_state": final_state,
            },
        )
