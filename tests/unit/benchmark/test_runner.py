# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BenchmarkRunner and run_benchmark_async.

The benchmark runner now requires a `BaseAgent` subclass (or instance).
These tests exercise the class + instance paths with minimal `BaseAgent`
subclasses; lifecycle ordering (setup-before-run_case, teardown-in-finally)
lives in test_runner_lifecycle.py.
"""

import asyncio
import textwrap
from pathlib import Path

import pytest

from agent_evals import BaseAgent
from agent_evals.core.types import TaskResult


def _state_from_prompt(prompt: str) -> dict:
    """Return a final_state dict derived from a prompt -- mirrors the
    behavior the scorers expect in these tests.
    """
    state = {"lights": {}}
    if "kitchen" in prompt.lower() and "on" in prompt.lower():
        state["lights"]["kitchen"] = {"state": "on"}
    return state


class StateMatchAgent(BaseAgent):
    """Minimal agent that populates final_state from the prompt."""

    name = "state-match-agent"
    version = "test"

    async def run_case(self, prompt: str, **kwargs) -> TaskResult:
        return TaskResult(
            output="", context={"final_state": _state_from_prompt(prompt)}
        )


BENCHMARK_YAML = textwrap.dedent(
    """
    benchmark: test-v1
    platform:
      name: local
      experiment: t1
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: kitchen on
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


class TestRunBenchmarkAsync:
    @pytest.mark.asyncio
    async def test_accepts_agent_class(self, bench_yaml: Path):
        from agent_evals._composition import run_benchmark_async

        result = await run_benchmark_async(bench_yaml, agent=StateMatchAgent)
        assert result.benchmark == "test-v1"
        assert "lights" in result.eval_results
        assert result.aggregate_pass_rates["StateMatch"] == 1.0

    @pytest.mark.asyncio
    async def test_accepts_agent_instance(self, bench_yaml: Path):
        """Passing an instance (escape hatch for constructor-arg subclasses)."""
        from agent_evals._composition import run_benchmark_async

        result = await run_benchmark_async(bench_yaml, agent=StateMatchAgent())
        assert result.aggregate_pass_rates["StateMatch"] == 1.0

    @pytest.mark.asyncio
    async def test_populates_agent_identity_on_result(self, bench_yaml: Path):
        from agent_evals._composition import run_benchmark_async

        result = await run_benchmark_async(bench_yaml, agent=StateMatchAgent)
        assert result.agent_name == "state-match-agent"
        assert result.agent_version == "test"

    @pytest.mark.asyncio
    async def test_rejects_non_agent_argument(self, bench_yaml: Path):
        """A plain callable is no longer valid -- the rewrite is intentional."""
        from agent_evals._composition import run_benchmark_async

        def not_an_agent(input_):
            return TaskResult(output="")

        with pytest.raises(TypeError):
            # Passing a callable instead of a BaseAgent subclass is exactly what
            # we expect to raise; silence the static-type warning that would
            # otherwise flag this intentional mismatch.
            await run_benchmark_async(bench_yaml, agent=not_an_agent)


class TestBenchmarkRunnerClass:
    @pytest.mark.asyncio
    async def test_runner_accepts_config_and_agent_directly(
        self, bench_yaml: Path, default_benchmark_registries
    ):
        from agent_evals.benchmark.loader import load_benchmark
        from agent_evals.benchmark.runner import BenchmarkRunner

        config = load_benchmark(bench_yaml, **default_benchmark_registries)
        runner = BenchmarkRunner(config, agent=StateMatchAgent)
        result = await runner.run_async()
        assert result.benchmark == "test-v1"


EXECUTION_YAML = textwrap.dedent(
    """
    benchmark: exec-test-v1
    platform:
      name: local
      experiment: exec
    execution:
      n_parallel_runs: 5
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: kitchen on
            prompt: Turn on the kitchen light
            success:
              state:
                lights.kitchen.state: "on"
    """
).strip()


class TestEvalConfigPropagation:
    """YAML execution block -> EvalConfig reaching run_eval_async.

    Monkeypatch note: `runner.py` imports `run_eval_async` via
    `from agent_evals import run_eval_async`, which binds the symbol onto
    the `agent_evals.benchmark.runner` module's namespace. Patching the
    attribute on that module (as done below) replaces the reference the
    runner actually uses at call time.
    """

    @pytest.mark.asyncio
    async def test_yaml_execution_block_propagates_to_run_eval_async(
        self, tmp_path: Path, monkeypatch
    ):
        from agent_evals._composition import run_benchmark_async
        from agent_evals.benchmark import runner as runner_module

        p = tmp_path / "exec.yaml"
        p.write_text(EXECUTION_YAML)

        captured: dict = {}

        async def fake_run_eval_async(**kwargs):
            captured.update(kwargs)
            from agent_evals import EvalResult

            return EvalResult(
                experiment_id="x",
                experiment_url="file:///tmp/x",
                platform="local",
                scores={"StateMatch": 1.0},
                pass_rates={"StateMatch": 1.0},
                examples=[],
                summary={"total_examples": 1, "successful_examples": 1},
            )

        monkeypatch.setattr(runner_module, "run_eval_async", fake_run_eval_async)

        await run_benchmark_async(p, agent=StateMatchAgent)

        exec_cfg = captured.get("config")
        assert exec_cfg is not None, "config (EvalConfig) not forwarded"
        assert exec_cfg.max_concurrent_tests == 5  # from YAML's n_parallel_runs: 5

        # Per-capability platform config must have the capability name suffixed onto
        # the base experiment name so each capability's traces are distinct.
        platform_cfg = captured.get("platform")
        assert platform_cfg is not None, "platform config not forwarded"
        assert platform_cfg.experiment is not None, (
            "experiment must be set on platform config"
        )
        assert platform_cfg.experiment.endswith("-lights"), (
            f"expected platform.experiment to end with '-lights', got {platform_cfg.experiment!r}"
        )

    @pytest.mark.asyncio
    async def test_yaml_without_execution_block_uses_defaults(
        self, bench_yaml: Path, monkeypatch
    ):
        from agent_evals._composition import run_benchmark_async
        from agent_evals.benchmark import runner as runner_module

        captured: dict = {}

        async def fake_run_eval_async(**kwargs):
            captured.update(kwargs)
            from agent_evals import EvalResult

            return EvalResult(
                experiment_id="x",
                experiment_url="file:///tmp/x",
                platform="local",
                scores={"StateMatch": 1.0},
                pass_rates={"StateMatch": 1.0},
                examples=[],
                summary={"total_examples": 1, "successful_examples": 1},
            )

        monkeypatch.setattr(runner_module, "run_eval_async", fake_run_eval_async)

        await run_benchmark_async(bench_yaml, agent=StateMatchAgent)

        exec_cfg = captured.get("config")
        assert exec_cfg is not None
        # YAML's n_parallel_runs defaults to 1 (safe for stateful agents).
        assert exec_cfg.max_concurrent_tests == 1


def _multi_test_yaml(n_parallel_runs: int, n_tests: int) -> str:
    """Build a benchmark YAML with N scenarios, all targeting the kitchen light."""
    scenario_entries = []
    for i in range(n_tests):
        scenario_entries.append(
            f"      - id: {i}\n"
            f"        name: kitchen on {i}\n"
            f"        prompt: Turn on the kitchen light\n"
            f"        success:\n"
            f"          state:\n"
            f'            lights.kitchen.state: "on"'
        )
    scenarios_block = "\n".join(scenario_entries)
    return (
        "benchmark: concurrency-v1\n"
        "platform:\n"
        "  name: local\n"
        "  experiment: conc\n"
        "execution:\n"
        f"  n_parallel_runs: {n_parallel_runs}\n"
        "capabilities:\n"
        "  - name: lights\n"
        "    scenarios:\n"
        f"{scenarios_block}\n"
    )


class _Probe:
    """Tracks peak concurrent run_case invocations."""

    def __init__(self) -> None:
        self.active = 0
        self.peak = 0
        self._lock = asyncio.Lock()

    async def enter(self) -> None:
        async with self._lock:
            self.active += 1
            self.peak = max(self.peak, self.active)

    async def exit(self) -> None:
        async with self._lock:
            self.active -= 1


class TestBenchmarkConcurrencyEnforcement:
    """End-to-end: YAML's n_parallel_runs actually caps agent invocations.

    Translation tests (above) prove the value reaches EvalConfig.
    These prove the value enforces concurrency during a real benchmark run.
    """

    @pytest.mark.asyncio
    async def test_n_parallel_runs_one_serializes_agent_calls(self, tmp_path: Path):
        from agent_evals._composition import run_benchmark_async

        probe = _Probe()

        class ProbedAgent(BaseAgent):
            name = "probed"

            async def run_case(self, prompt: str, **kwargs) -> TaskResult:
                await probe.enter()
                try:
                    await asyncio.sleep(0.01)
                    return TaskResult(
                        output="", context={"final_state": _state_from_prompt(prompt)}
                    )
                finally:
                    await probe.exit()

        yaml_path = tmp_path / "serial.yaml"
        yaml_path.write_text(_multi_test_yaml(n_parallel_runs=1, n_tests=6))

        await run_benchmark_async(yaml_path, agent=ProbedAgent)

        assert probe.peak == 1, f"expected serial execution, saw peak={probe.peak}"

    @pytest.mark.asyncio
    async def test_n_parallel_runs_caps_agent_concurrency(self, tmp_path: Path):
        from agent_evals._composition import run_benchmark_async

        probe = _Probe()
        cap = 3

        class ProbedAgent(BaseAgent):
            name = "probed"

            async def run_case(self, prompt: str, **kwargs) -> TaskResult:
                await probe.enter()
                try:
                    await asyncio.sleep(0.02)
                    return TaskResult(
                        output="", context={"final_state": _state_from_prompt(prompt)}
                    )
                finally:
                    await probe.exit()

        yaml_path = tmp_path / "parallel.yaml"
        yaml_path.write_text(_multi_test_yaml(n_parallel_runs=cap, n_tests=16))

        await run_benchmark_async(yaml_path, agent=ProbedAgent)

        assert probe.peak <= cap, f"peak={probe.peak} exceeded cap={cap}"
        assert probe.peak >= 2, (
            f"test is not exercising parallelism; peak={probe.peak} "
            "(expected at least 2 concurrent before the cap kicked in)"
        )


# ---------------------------------------------------------------------------
# Per-scenario grouping + sequential execution + merge.
# ---------------------------------------------------------------------------

MIXED_YAML = textwrap.dedent(
    """
    benchmark: mixed-runner-v1
    platform:
      name: local
      experiment: mixed
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: state only
            prompt: Turn on the kitchen light
            success:
              state: { lights.kitchen.state: "on" }
          - id: 2
            name: trajectory only
            prompt: Check status
            success:
              tool_use:
                calls:
                  - name: get_status
                    args: {}
          - id: 3
            name: both
            prompt: Turn on and verify
            success:
              state: { lights.kitchen.state: "on" }
              tool_use:
                calls:
                  - name: turn_on
                    args: { device: kitchen }
    """
).strip()


class TestPerGroupRunEvalCalls:
    @pytest.mark.asyncio
    async def test_run_async_calls_run_eval_per_group(
        self, tmp_path: Path, monkeypatch
    ):
        """`run_async` must invoke `run_eval_async` once per compiled group."""
        from agent_evals._composition import run_benchmark_async
        from agent_evals.benchmark import runner as runner_module

        p = tmp_path / "mixed.yaml"
        p.write_text(MIXED_YAML)

        call_count = {"n": 0}

        async def fake_run_eval_async(**kwargs):
            call_count["n"] += 1
            from agent_evals import EvalResult

            return EvalResult(
                experiment_id=f"x{call_count['n']}",
                experiment_url=f"file:///tmp/x{call_count['n']}",
                platform="local",
                scores={},
                pass_rates={},
                examples=[],
                summary={},
            )

        monkeypatch.setattr(runner_module, "run_eval_async", fake_run_eval_async)

        await run_benchmark_async(p, agent=StateMatchAgent)

        # Three distinct success-block shapes -> three groups -> three calls.
        assert call_count["n"] == 3, (
            f"expected 3 run_eval_async calls (one per group); got {call_count['n']}"
        )

    @pytest.mark.asyncio
    async def test_groups_run_sequentially(self, tmp_path: Path, monkeypatch):
        """Within a capability, groups run sequentially -- never overlap.

        Spy on `run_eval_async` and assert that no second call begins before
        the prior call returns. Sequential group execution preserves the
        user-visible "≤n_parallel_runs LLM calls in flight" contract.
        """
        from agent_evals._composition import run_benchmark_async
        from agent_evals.benchmark import runner as runner_module

        p = tmp_path / "mixed.yaml"
        p.write_text(MIXED_YAML)

        in_flight = {"n": 0, "max": 0}

        async def fake_run_eval_async(**kwargs):
            in_flight["n"] += 1
            in_flight["max"] = max(in_flight["max"], in_flight["n"])
            # Yield to the event loop so any concurrent caller would interleave.
            await asyncio.sleep(0.01)
            assert in_flight["n"] == 1, (
                "groups within a capability must run sequentially; "
                f"saw {in_flight['n']} run_eval_async calls in flight."
            )
            in_flight["n"] -= 1
            from agent_evals import EvalResult

            return EvalResult(
                experiment_id="x",
                experiment_url="file:///tmp/x",
                platform="local",
                scores={},
                pass_rates={},
                examples=[],
                summary={},
            )

        monkeypatch.setattr(runner_module, "run_eval_async", fake_run_eval_async)

        await run_benchmark_async(p, agent=StateMatchAgent)

        # Defensive: the in-call assertion already failed loudly if peak >1,
        # but record the peak for diagnosability when the suite goes red.
        assert in_flight["max"] == 1, (
            f"expected peak in-flight=1, saw {in_flight['max']}"
        )


class TestMergeEvalResults:
    """Direct tests for the private `_merge_eval_results` helper."""

    def _make_example(self, scenario_id: int, scores: dict[str, float]):
        from agent_evals.core.types import EvalExample, Score

        score_objs = {
            name: Score(name=name, value=val, passed=val >= 0.5)
            for name, val in scores.items()
        }
        return EvalExample(
            input={"prompt": f"p{scenario_id}"},
            output="",
            expected="",
            scores=score_objs,
            metadata={"scenario_id": scenario_id},
            duration=0.0,
            error=None,
        )

    def _make_result(
        self, exp_id: str, examples: list, scores: dict, pass_rates: dict, dur: float
    ):
        from agent_evals.core.types import EvalResult

        return EvalResult(
            experiment_id=exp_id,
            experiment_url=f"file:///tmp/{exp_id}",
            platform="local",
            scores=scores,
            pass_rates=pass_rates,
            examples=examples,
            summary={"total_examples": len(examples)},
            metadata={"src": exp_id},
            duration=dur,
        )

    def test_merge_eval_results_correct_denominators(self):
        """Scorer that ran in 3/5 examples shows pass_rates over denominator 3."""
        from agent_evals.benchmark.runner import _merge_eval_results

        # Group A: 3 examples, all scored by StateMatch (2 pass, 1 fail).
        ex_a = [
            self._make_example(1, {"StateMatch": 1.0}),
            self._make_example(2, {"StateMatch": 1.0}),
            self._make_example(3, {"StateMatch": 0.0}),
        ]
        result_a = self._make_result(
            "a",
            ex_a,
            scores={"StateMatch": 2 / 3},
            pass_rates={"StateMatch": 2 / 3},
            dur=1.0,
        )

        # Group B: 2 examples, all scored by ToolCallExactMatch (1 pass, 1 fail).
        ex_b = [
            self._make_example(4, {"ToolCallExactMatch": 1.0}),
            self._make_example(5, {"ToolCallExactMatch": 0.0}),
        ]
        result_b = self._make_result(
            "b",
            ex_b,
            scores={"ToolCallExactMatch": 0.5},
            pass_rates={"ToolCallExactMatch": 0.5},
            dur=2.0,
        )

        merged = _merge_eval_results([result_a, result_b])

        # Examples concatenated.
        assert len(merged.examples) == 5

        # StateMatch ran on 3 of 5 -> denominator 3, value 2/3.
        assert merged.scores["StateMatch"] == pytest.approx(2 / 3)
        assert merged.pass_rates["StateMatch"] == pytest.approx(2 / 3)

        # ToolCallExactMatch ran on 2 of 5 -> denominator 2, value 0.5.
        assert merged.scores["ToolCallExactMatch"] == pytest.approx(0.5)
        assert merged.pass_rates["ToolCallExactMatch"] == pytest.approx(0.5)

        # Duration sums.
        assert merged.duration == pytest.approx(3.0)

        # Experiment id from first group.
        assert merged.experiment_id == "a"
        assert merged.experiment_url == "file:///tmp/a"
        assert merged.platform == "local"

    def test_merge_preserves_example_metadata(self):
        from agent_evals.benchmark.runner import _merge_eval_results

        ex_a = [
            self._make_example(1, {"StateMatch": 1.0}),
            self._make_example(2, {"StateMatch": 0.0}),
        ]
        # Tag one example's metadata with extra keys to confirm preservation.
        ex_a[0].metadata["phrasing"] = "imperative"
        ex_a[1].metadata["depth"] = "literal"

        result_a = self._make_result(
            "a",
            ex_a,
            scores={"StateMatch": 0.5},
            pass_rates={"StateMatch": 0.5},
            dur=1.0,
        )

        ex_b = [self._make_example(3, {"ToolCallExactMatch": 1.0})]
        ex_b[0].metadata["phrasing"] = "polite"

        result_b = self._make_result(
            "b",
            ex_b,
            scores={"ToolCallExactMatch": 1.0},
            pass_rates={"ToolCallExactMatch": 1.0},
            dur=0.5,
        )

        merged = _merge_eval_results([result_a, result_b])

        assert len(merged.examples) == 3
        # Re-key by scenario_id for clarity.
        by_id = {ex.metadata["scenario_id"]: ex for ex in merged.examples}
        assert by_id[1].metadata["phrasing"] == "imperative"
        assert by_id[2].metadata["depth"] == "literal"
        assert by_id[3].metadata["phrasing"] == "polite"

    def test_merge_single_group_short_circuit(self):
        """Single-group input is returned unchanged (homogeneous-case fast path)."""
        from agent_evals.benchmark.runner import _merge_eval_results

        ex = [self._make_example(1, {"StateMatch": 1.0})]
        result = self._make_result(
            "only",
            ex,
            scores={"StateMatch": 1.0},
            pass_rates={"StateMatch": 1.0},
            dur=0.5,
        )

        merged = _merge_eval_results([result])
        # Same object reference: short-circuit returns the input unchanged.
        assert merged is result


class TestAgentTaskShimPreconditions:
    """The ``_agent_task`` shim forwards ``preconditions`` to ``run_case`` via kwargs.

    Built on top of ``BenchmarkRunner.run_async`` rather than reaching into the
    private shim closure: we drive a real benchmark run, capture the kwargs the
    runner forwarded into ``run_case``, and assert on the captured value.
    """

    @pytest.mark.asyncio
    async def test_shim_forwards_preconditions_to_run_case(self, tmp_path: Path):
        """Given an ``input_`` containing ``preconditions``, the shim forwards
        them to ``agent.run_case`` as ``preconditions=[...]``."""
        from agent_evals._composition import run_benchmark_async
        from agent_evals.adapters.preconditions.state import StatePreconditionApplier

        captured: dict = {}

        class RecordingAgent(BaseAgent):
            name = "recording"

            async def run_case(self, prompt: str, **kwargs) -> TaskResult:
                # Record the kwargs the shim forwarded for the first call only.
                if "preconditions" not in captured:
                    captured["prompt"] = prompt
                    captured["preconditions"] = kwargs.get("preconditions")
                return TaskResult(
                    output="",
                    context={"final_state": _state_from_prompt(prompt)},
                )

        precond_yaml = textwrap.dedent(
            """
            benchmark: precond-shim-v1
            platform:
              name: local
              experiment: t
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: kitchen on (preseeded off)
                    prompt: Turn on the kitchen light
                    precondition:
                      state:
                        lights.kitchen.state: "off"
                    success:
                      state:
                        lights.kitchen.state: "on"
            """
        ).strip()
        p = tmp_path / "precond_shim.yaml"
        p.write_text(precond_yaml)

        await run_benchmark_async(p, agent=RecordingAgent)

        assert "preconditions" in captured
        preconds = captured["preconditions"]
        assert isinstance(preconds, list)
        assert len(preconds) == 1
        assert isinstance(preconds[0], StatePreconditionApplier)

    @pytest.mark.asyncio
    async def test_legacy_agent_without_preconditions_kwarg_still_works(
        self, tmp_path: Path
    ):
        """Agents whose ``run_case`` swallows extra kwargs via ``**_`` keep
        working when the shim forwards ``preconditions=`` -- backward compat."""
        from agent_evals._composition import run_benchmark_async

        class LegacyAgent(BaseAgent):
            name = "legacy"

            async def run_case(self, prompt: str, **_) -> TaskResult:
                # Does not consume preconditions explicitly. Must not raise.
                return TaskResult(
                    output="",
                    context={"final_state": _state_from_prompt(prompt)},
                )

        precond_yaml = textwrap.dedent(
            """
            benchmark: precond-legacy-v1
            platform:
              name: local
              experiment: t
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: kitchen on (with precondition)
                    prompt: Turn on the kitchen light
                    precondition:
                      state:
                        lights.kitchen.state: "off"
                    success:
                      state:
                        lights.kitchen.state: "on"
            """
        ).strip()
        p = tmp_path / "precond_legacy.yaml"
        p.write_text(precond_yaml)

        # The bug-form would be a TypeError on the forwarded kwarg.
        result = await run_benchmark_async(p, agent=LegacyAgent)
        assert "lights" in result.eval_results
