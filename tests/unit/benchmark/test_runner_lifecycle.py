# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lifecycle guarantees for BenchmarkRunner + BaseAgent.

Exercises the runner's `setup` -> case-loop -> `teardown` ordering and the
`finally` contract that teardown runs even when something raises. Uses a
FakeAgent with call-counters; no external services.
"""

import textwrap
from pathlib import Path

import pytest

from agent_evals import BaseAgent
from agent_evals.core.types import TaskResult

BENCHMARK_YAML = textwrap.dedent(
    """
    benchmark: lifecycle-v1
    platform:
      name: local
      experiment: lifecycle
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: t1
            prompt: Turn on the kitchen light
            success:
              state:
                lights.kitchen.state: "on"
          - id: 2
            name: t2
            prompt: Turn on the kitchen light
            success:
              state:
                lights.kitchen.state: "on"
    """
).strip()


@pytest.fixture
def bench_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "bench.yaml"
    p.write_text(BENCHMARK_YAML)
    return p


def _state_from_prompt(prompt: str) -> dict:
    state = {"lights": {}}
    if "kitchen" in prompt.lower() and "on" in prompt.lower():
        state["lights"]["kitchen"] = {"state": "on"}
    return state


class _FakeAgent(BaseAgent):
    """Counting agent. Records call order into a shared list on the instance."""

    name = "fake"
    version = "0"

    def __init__(self):
        self.calls: list[str] = []
        self.setup_count = 0
        self.teardown_count = 0
        self.run_count = 0

    async def setup(self) -> None:
        self.calls.append("setup")
        self.setup_count += 1

    async def run_case(self, prompt: str, **kwargs) -> TaskResult:
        self.calls.append(f"run:{prompt[:20]}")
        self.run_count += 1
        return TaskResult(
            output="", context={"final_state": _state_from_prompt(prompt)}
        )

    async def teardown(self) -> None:
        self.calls.append("teardown")
        self.teardown_count += 1


class _FailingRunAgent(_FakeAgent):
    """Raises in run_case so we can verify teardown still fires."""

    async def run_case(self, prompt: str, **kwargs) -> TaskResult:
        self.calls.append(f"run:{prompt[:20]}")
        self.run_count += 1
        raise RuntimeError("boom in run")


class _FailingSetupAgent(_FakeAgent):
    """Raises in setup so we can verify teardown still fires."""

    async def setup(self) -> None:
        self.calls.append("setup")
        self.setup_count += 1
        raise RuntimeError("boom in setup")


class TestLifecycleOrdering:
    @pytest.mark.asyncio
    async def test_setup_precedes_run_which_precedes_teardown(self, bench_yaml: Path):
        from agent_evals._composition import run_benchmark_async

        agent = _FakeAgent()
        await run_benchmark_async(bench_yaml, agent=agent)

        # Exactly one setup, one teardown, setup first, teardown last.
        assert agent.setup_count == 1
        assert agent.teardown_count == 1
        assert agent.calls[0] == "setup"
        assert agent.calls[-1] == "teardown"
        # At least one run between them.
        assert any(call.startswith("run:") for call in agent.calls[1:-1])

    @pytest.mark.asyncio
    async def test_setup_runs_exactly_once_with_multiple_tests(self, bench_yaml: Path):
        from agent_evals._composition import run_benchmark_async

        agent = _FakeAgent()
        await run_benchmark_async(bench_yaml, agent=agent)

        # Two tests in the YAML; setup/teardown still once each.
        assert agent.run_count == 2
        assert agent.setup_count == 1
        assert agent.teardown_count == 1


class TestTeardownOnError:
    @pytest.mark.asyncio
    async def test_teardown_runs_when_run_case_raises(self, bench_yaml: Path):
        """Exceptions raised inside run_case are captured as task_error by the
        platform adapter (existing framework behavior). Teardown still fires
        because the capability loop completes; the guarantee we care about is
        that teardown always runs after setup."""
        from agent_evals._composition import run_benchmark_async

        agent = _FailingRunAgent()
        # Does not raise out; platform records task_error on each example.
        await run_benchmark_async(bench_yaml, agent=agent)

        assert agent.setup_count == 1
        assert agent.teardown_count == 1
        assert agent.calls[-1] == "teardown"
        assert agent.run_count == 2  # both tests attempted

    @pytest.mark.asyncio
    async def test_teardown_runs_when_capability_loop_raises(
        self, bench_yaml: Path, monkeypatch
    ):
        """Simulate an exception escaping the test loop (e.g., platform adapter
        failure). Teardown must still fire because the runner wraps the loop
        in try/finally."""
        from agent_evals._composition import run_benchmark_async
        from agent_evals.benchmark import runner as runner_module

        async def fake_run_eval_async(**kwargs):
            raise RuntimeError("platform explosion")

        monkeypatch.setattr(runner_module, "run_eval_async", fake_run_eval_async)

        agent = _FakeAgent()
        with pytest.raises(RuntimeError, match="platform explosion"):
            await run_benchmark_async(bench_yaml, agent=agent)

        assert agent.setup_count == 1
        assert agent.teardown_count == 1
        assert agent.calls[-1] == "teardown"

    @pytest.mark.asyncio
    async def test_teardown_runs_when_setup_raises(self, bench_yaml: Path):
        """Subclass must tolerate partial initialization; we only assert
        that teardown is attempted."""
        from agent_evals._composition import run_benchmark_async

        agent = _FailingSetupAgent()
        with pytest.raises(RuntimeError, match="boom in setup"):
            await run_benchmark_async(bench_yaml, agent=agent)

        assert agent.setup_count == 1
        assert agent.teardown_count == 1
        assert agent.run_count == 0


class TestInstanceVsClassDispatch:
    @pytest.mark.asyncio
    async def test_class_instantiated_once(self, bench_yaml: Path):
        from agent_evals._composition import run_benchmark_async

        instances: list[_FakeAgent] = []

        class TrackingAgent(_FakeAgent):
            def __init__(self):
                super().__init__()
                instances.append(self)

        await run_benchmark_async(bench_yaml, agent=TrackingAgent)

        assert len(instances) == 1
        assert instances[0].setup_count == 1
        assert instances[0].teardown_count == 1

    @pytest.mark.asyncio
    async def test_instance_is_used_directly(self, bench_yaml: Path):
        from agent_evals._composition import run_benchmark_async

        agent = _FakeAgent()
        await run_benchmark_async(bench_yaml, agent=agent)
        assert agent.setup_count == 1
        assert agent.teardown_count == 1
