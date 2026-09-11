# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the user-facing `BaseAgent` base class.

Covers only the class contract -- lifecycle ordering tests that involve the
benchmark runner live in tests/unit/benchmark/test_runner_lifecycle.py.
"""

import inspect

import pytest

from agent_evals import BaseAgent, TaskResult


class _MinimalAgent(BaseAgent):
    """Subclass that implements only the required `run_case` method."""

    async def run_case(self, prompt: str, **kwargs) -> TaskResult:
        return TaskResult(output=f"echo:{prompt}")


class TestBaseAgentDefaults:
    """Behaviors of the class itself, independent of the runner."""

    def test_class_attr_defaults(self) -> None:
        assert BaseAgent.name == ""
        assert BaseAgent.version is None

    def test_subclass_overrides_identity(self) -> None:
        class NamedAgent(_MinimalAgent):
            name = "named"
            version = "1.2.3"

        assert NamedAgent.name == "named"
        assert NamedAgent.version == "1.2.3"

    def test_setup_and_teardown_are_no_op_coroutines(self) -> None:
        agent = _MinimalAgent()
        setup_result = agent.setup()
        teardown_result = agent.teardown()
        try:
            assert inspect.iscoroutine(setup_result)
            assert inspect.iscoroutine(teardown_result)
        finally:
            setup_result.close()
            teardown_result.close()

    @pytest.mark.asyncio
    async def test_default_setup_and_teardown_return_none(self) -> None:
        agent = _MinimalAgent()
        assert await agent.setup() is None
        assert await agent.teardown() is None

    @pytest.mark.asyncio
    async def test_minimal_subclass_run_case(self) -> None:
        agent = _MinimalAgent()
        result = await agent.run_case("hello")
        assert isinstance(result, TaskResult)
        assert result.output == "echo:hello"

    @pytest.mark.asyncio
    async def test_run_case_ignores_unknown_kwargs(self) -> None:
        """**kwargs slot accepts any kwargs without error."""
        agent = _MinimalAgent()
        result = await agent.run_case(
            "hi", capability="x", scenario_id=42, unknown=True
        )
        assert result.output == "echo:hi"

    def test_instantiating_without_run_case_raises_typeerror(self) -> None:
        """ABC + @abstractmethod: missing `run_case` fails at instantiation."""

        class BrokenAgent(BaseAgent):
            pass

        with pytest.raises(TypeError) as exc_info:
            BrokenAgent()  # type: ignore[abstract]

        assert "run_case" in str(exc_info.value)

    def test_cannot_instantiate_bare_base_agent(self) -> None:
        with pytest.raises(TypeError):
            BaseAgent()  # type: ignore[abstract]
