# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark orchestrator -- async-first, mirrors run_eval_async.

BenchmarkRunner owns the per-capability loop.
``_run_benchmark_async_internal`` is the engine entry point that wraps
``BenchmarkRunner.run_async``; the public ``run_benchmark_async``
delegates here after loading the YAML and wiring registries.

The benchmark-layer agent contract is `BaseAgent` (see `benchmark/agent.py`):
users subclass it, implement `run_case`, and optionally override `setup`
/ `teardown`. The runner:
  - Instantiates the class if a `type[BaseAgent]` is passed (otherwise
    uses the provided instance directly).
  - Awaits `setup` once before the case loop.
  - Wraps `agent.run_case(prompt)` in a per-case shim that matches the
    `task(input_: dict)` signature `run_eval_async` expects -- this keeps
    `run_eval_async` unchanged while letting `BaseAgent.run_case` take a
    plain `prompt: str`.
  - Awaits `teardown` in a `finally` block so it runs even if `setup`
    raised partway or `run_case` raised mid-loop. Subclasses are
    responsible for tolerating partial initialization.
  - Populates `BenchmarkResult.agent_name` / `agent_version` from the
    instance's class attributes.

Each capability compiles into a *list* of ``(dataset, scorers)``
groups (see ``benchmark/loader.py``). Within a capability, groups run
**sequentially** so the user's ``n_parallel_runs`` knob
stays an honest "≤N LLM calls in flight" cap. After all groups for a
capability finish, we merge their ``EvalResult`` objects into a single
``EvalResult`` whose per-scorer denominators reflect only the examples
that actually scored under each scorer.
"""

from statistics import fmean

from pydantic import TypeAdapter, ValidationError

from agent_evals import run_eval_async
from agent_evals.benchmark.agent import BaseAgent
from agent_evals.benchmark.agent_input import AgentCaseInput
from agent_evals.benchmark.config import BenchmarkConfig, ExecutionBlock
from agent_evals.benchmark.loader import compile_capability
from agent_evals.benchmark.types import BenchmarkResult
from agent_evals.core.types import (
    EvalConfig,
    EvalResult,
    PathSafeIdentifier,
    TaskResult,
)

# Applies the field-level identifier rule to a value that never passes through
# a model field. Built once at import rather than per capability.
_EXPERIMENT_NAME_ADAPTER: TypeAdapter[str] = TypeAdapter(PathSafeIdentifier)


def _validated_experiment_name(composed: str) -> str:
    """Return ``composed`` if it is a safe path segment, else raise.

    The composed per-capability experiment name is assembled from two
    already-constrained values, but the result is installed with
    ``model_copy(update=...)``, which does not re-validate — so nothing has
    checked the composition itself. For the local platform adapter that
    string becomes a directory name, which makes this the point where the
    value reaches a filesystem sink.

    Raises:
        ValueError: ``composed`` is not a safe path segment. The message
            names the offending value so the capability that produced it is
            identifiable from the config.
    """
    try:
        return _EXPERIMENT_NAME_ADAPTER.validate_python(composed)
    except ValidationError as exc:
        raise ValueError(
            f"composed experiment name {composed!r} is not a safe path "
            f"segment; it is built from the benchmark name and a capability "
            f"name, and becomes a directory name for local runs"
        ) from exc


def _to_execution_config(block: ExecutionBlock) -> EvalConfig:
    """Translate the YAML-facing ExecutionBlock into the core EvalConfig.

    YAML's ``n_parallel_runs`` is the same axis as
    ``EvalConfig.max_concurrent_tests`` — both cap how many runs
    (dataset examples) execute concurrently. The benchmark layer speaks
    "run"; core speaks "test" (one ``task(input)`` call).
    """
    return EvalConfig(max_concurrent_tests=block.n_parallel_runs)


def _resolve_agent(agent: type[BaseAgent] | BaseAgent) -> BaseAgent:
    """Accept a class or an instance; return an instance ready to `setup`.

    Passing the class is the common case. Passing an instance is the escape
    hatch for subclasses that take constructor arguments.
    """
    if isinstance(agent, BaseAgent):
        return agent
    if isinstance(agent, type) and issubclass(agent, BaseAgent):
        return agent()
    raise TypeError(
        f"agent must be a BaseAgent subclass or instance; got {type(agent).__name__}"
    )


def _merge_eval_results(results: list[EvalResult]) -> EvalResult:
    """Merge the per-group EvalResults of one capability into a single result.

    Behavior:
      - **Single-group fast path:** if ``len(results) == 1`` return the
        input unchanged. Homogeneous capabilities (the common case)
        therefore produce byte-identical output to pre-016 behavior;
        no merge machinery runs.
      - **examples:** concatenated across groups in the input order.
      - **scores / pass_rates:** union of scorer names across groups;
        for each scorer, mean over only the examples whose ``scores``
        dict contains that scorer (denominator = count of examples with
        the scorer, *not* total examples). Mirrors
        ``BenchmarkResult._macro_average`` so the macro-vs-micro story
        stays consistent across the codebase.
      - **duration:** sum across groups (each group is a serial
        ``run_eval_async`` call, so durations stack).
      - **experiment_id / experiment_url / platform:** taken from the
        first group's result. All groups within a capability share the
        same per-capability platform config (see ``run_async`` below),
        so platforms that allocate distinct experiment ids per call
        only surface the first one. That's a known limitation of
        per-capability merging and acceptable for v1; if it bites in
        practice, the platform-side ``experiment`` config can be made
        per-group.
      - **summary / metadata:** dict union; on key collision the first
        group's value wins. Heterogeneous-capability integration tests
        guard against subtle drift here.
    """
    if len(results) == 1:
        return results[0]
    if not results:
        raise ValueError("_merge_eval_results requires at least one EvalResult")

    # Concatenate examples in group order.
    merged_examples = [ex for er in results for ex in er.examples]

    # Collect every scorer name that appears in any example.
    all_scorer_names: set[str] = set()
    for ex in merged_examples:
        all_scorer_names.update(ex.scores.keys())

    merged_scores: dict[str, float] = {}
    merged_pass_rates: dict[str, float] = {}
    for name in all_scorer_names:
        # Denominator counts only examples where this scorer ran.
        applicable = [ex for ex in merged_examples if name in ex.scores]
        if not applicable:
            continue
        merged_scores[name] = fmean(ex.scores[name].value for ex in applicable)
        merged_pass_rates[name] = fmean(
            1.0 if ex.scores[name].passed else 0.0 for ex in applicable
        )

    # First-wins merge for summary and metadata.
    merged_summary: dict = {}
    for er in results:
        for k, v in er.summary.items():
            if k not in merged_summary:
                merged_summary[k] = v
    merged_metadata: dict = {}
    for er in results:
        for k, v in er.metadata.items():
            if k not in merged_metadata:
                merged_metadata[k] = v

    first = results[0]
    return EvalResult(
        experiment_id=first.experiment_id,
        experiment_url=first.experiment_url,
        platform=first.platform,
        scores=merged_scores,
        pass_rates=merged_pass_rates,
        examples=merged_examples,
        summary=merged_summary,
        metadata=merged_metadata,
        duration=sum(er.duration for er in results),
    )


class BenchmarkRunner:
    """Orchestrates one run_eval_async per group per capability.

    Capabilities run sequentially. Within a capability, the loader
    compiles a list of ``(dataset, scorers)`` groups (one per distinct
    scorer-set across the capability's scenarios); the runner awaits
    each group's ``run_eval_async`` call sequentially and merges the
    per-group ``EvalResult`` objects into a single capability-level
    ``EvalResult``.

    Within a single ``run_eval_async`` call, ``EvalConfig`` controls
    test-level parallelism. Sequential group execution preserves the
    user-visible "≤ ``n_parallel_runs`` LLM calls in flight" contract.
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        agent: type[BaseAgent] | BaseAgent,
        config_path: str | None = None,
    ):
        self.config = config
        self.agent = _resolve_agent(agent)
        self.config_path = config_path

    async def run_async(self) -> BenchmarkResult:
        execution_config = _to_execution_config(self.config.execution)

        # Adapts BaseAgent.run_case(prompt, **kwargs) to the task(input_: dict)
        # signature run_eval_async expects. Unpacks every AgentCaseInput
        # field explicitly: prompt is positional, the rest become kwargs on
        # run_case. Adding a new per-case kwarg = add a field to
        # AgentCaseInput and add a new line here; the diff makes the
        # contract change visible at every site.
        agent = self.agent

        async def _agent_task(input_: AgentCaseInput) -> TaskResult:
            return await agent.run_case(
                input_["prompt"],
                preconditions=input_["preconditions"],
            )

        eval_results: dict = {}
        # setup is inside the try so that teardown runs even if setup fails
        # partway (subclass is responsible for tolerating partial state --
        # e.g., checking `hasattr(self, "_client")` before closing it).
        try:
            await agent.setup()
            for capability in self.config.capabilities:
                groups = compile_capability(capability, self.config.execution)

                # Per-capability platform config. All groups within a
                # capability share the same experiment id; merge logic
                # in `_merge_eval_results` reflects this.
                # `experiment` is `str | None = None`, so `or` falls through
                # to the benchmark name when unset.
                base_exp = self.config.platform.experiment or self.config.benchmark
                # Per-capability copy so each capability's experiment suffix
                # doesn't leak into the next.
                #
                # ``model_copy(update=...)`` installs the value without
                # re-validating, so a field constraint on ``experiment`` does
                # not cover this path. Both operands are individually
                # constrained, but the composition is a new string that no
                # field validator has seen — and for the local adapter it
                # becomes a directory name. Validate the composed value here,
                # at the point of use, before it is installed.
                composed_experiment = _validated_experiment_name(
                    f"{base_exp}-{capability.name}"
                )
                platform_cfg = self.config.platform.model_copy(
                    update={"experiment": composed_experiment}
                )

                per_group_results: list[EvalResult] = []
                # Sequential group execution. Do not switch to
                # `asyncio.gather` here: it would multiply
                # max-tests-in-flight by num_groups and break the
                # user-visible ``n_parallel_runs`` cap.
                for dataset, scorers in groups:
                    per_group_results.append(
                        await run_eval_async(
                            task=_agent_task,
                            dataset=dataset,
                            scorers=scorers,
                            platform=platform_cfg,
                            config=execution_config,
                        )
                    )

                eval_results[capability.name] = _merge_eval_results(per_group_results)
        finally:
            await agent.teardown()

        return BenchmarkResult(
            benchmark=self.config.benchmark,
            agent_name=agent.name or None,
            agent_version=agent.version,
            config=self.config,
            config_path=self.config_path,
            eval_results=eval_results,
        )


async def _run_benchmark_async_internal(
    config: BenchmarkConfig,
    *,
    agent: type[BaseAgent] | BaseAgent,
    config_path: str | None,
) -> BenchmarkResult:
    """Run a pre-built ``BenchmarkConfig``. Parallels ``_run_eval_async_internal``."""
    return await BenchmarkRunner(
        config, agent=agent, config_path=config_path
    ).run_async()
