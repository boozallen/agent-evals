# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""YAML benchmark loader and capability-to-ExampleData compiler.

Parses a benchmark YAML file into ``BenchmarkConfig``, and compiles each
capability into a list of ``(dataset, scorers)`` groups ready for
``run_eval_async``.
"""

from pathlib import Path
from typing import Any

import yaml

from agent_evals import ExampleData, ExpectedResult, _compile_check
from agent_evals.benchmark.agent_input import AgentCaseInput
from agent_evals.benchmark.config import (
    BenchmarkConfig,
    CapabilitySpec,
    ExecutionBlock,
    ScenarioSpec,
)
from agent_evals.benchmark.includes import _resolve_list_entries
from agent_evals.core.checks import CheckSpec
from agent_evals.core.ports import (
    AdapterRegistry,
    BenchmarkConfigReader,
    Parser,
    Platform,
    PreconditionApplier,
    Scorer,
)


def load_benchmark(
    path: str | Path,
    *,
    platform_registry: AdapterRegistry[type[Platform]],
    precondition_registry: AdapterRegistry[type[PreconditionApplier]],
    success_checker_registry: AdapterRegistry[type[Parser[CheckSpec]]],
    config_reader_registry: AdapterRegistry[type[BenchmarkConfigReader]],
) -> BenchmarkConfig:
    """Parse a benchmark YAML file into a validated BenchmarkConfig.

    Multi-file composition: list entries under ``capabilities:`` may
    use a registered URI scheme (``file://...``) to reference content
    in another file. Resolution happens *before* Pydantic validation
    so the model never sees URI strings; ``extra="forbid"`` stays
    honest and validation errors point at structural mistakes, not URI
    ambiguity. See ``benchmark/includes.py`` for the resolver.

    Platform resolution happens inside the ``BenchmarkConfig``
    before-validator ``_resolve_platform``, which reads
    ``platform_registry`` from the Pydantic validation context. This
    mirrors how ``precondition_registry`` and
    ``success_checker_registry`` are resolved in ``ScenarioSpec``
    validators — one uniform "registry per family, resolve in a
    validator" pattern across all four adapter families.

    Args:
        path: Path to the benchmark YAML file.
        platform_registry: Resolves platform adapter names to their
            adapter class (and ``config_class``). Supplied by the
            composition root.
        precondition_registry: Resolves precondition keys.
        success_checker_registry: Resolves success-checker keys.
        config_reader_registry: Resolves URI schemes for multi-file
            composition.
    """
    path = Path(path)
    parent_uri = path.absolute().as_uri()
    # encoding="utf-8" matches LocalFileConfigReader so root and included
    # files have identical encoding behavior across platforms (Windows
    # platform-default would be cp1252).
    with path.open(encoding="utf-8") as fh:
        try:
            raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"failed to parse YAML from {parent_uri!r}: {exc}"
            ) from exc
    if isinstance(raw, dict) and isinstance(raw.get("capabilities"), list):
        raw["capabilities"] = _resolve_list_entries(
            raw["capabilities"],
            base_uri=parent_uri,
            registry=config_reader_registry,
        )
    context: dict[str, Any] = {
        "platform_registry": platform_registry,
        "precondition_registry": precondition_registry,
        "success_checker_registry": success_checker_registry,
    }
    return BenchmarkConfig.model_validate(raw, context=context)


def _freeze(obj: Any) -> Any:
    """Recursively convert ``obj`` into a hashable form.

    Dicts become frozensets of (key, frozen-value) pairs; lists/tuples
    become tuples of frozen elements; other values pass through. Leaves
    must already be hashable.
    """
    if isinstance(obj, dict):
        return frozenset((k, _freeze(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        return tuple(_freeze(x) for x in obj)
    return obj


def _scenario_group_key(scenario: ScenarioSpec) -> tuple[tuple[str, Any], ...]:
    """Return a stable hashable key identifying a scenario's scorer-set.

    Two scenarios share a key when their CheckSpecs produce equivalent
    scorers. The key combines each spec's class name with its
    `compile_check(spec).scorer_config`, so specs of the same type with
    different scorer-affecting configuration produce distinct keys.

    Entries sort by ``(class_name, repr(frozen_config))`` rather than
    ``(class_name, frozen_config)`` because frozensets do not support
    ``<``. ``repr`` is sufficient — its only role is to give ``sorted``
    a deterministic total order, not to round-trip the value.
    """

    def _entry(spec: Any) -> tuple[str, Any]:
        compiled = _compile_check(spec)
        cfg = compiled.scorer_config
        if not isinstance(cfg, dict):
            raise TypeError(
                f"compile_check({type(spec).__name__}).scorer_config must "
                f"return dict, got {type(cfg).__name__}"
            )
        return (type(spec).__name__, _freeze(cfg))

    entries = (_entry(s) for s in scenario.parsed_check_specs)
    return tuple(sorted(entries, key=lambda x: (x[0], repr(x[1]))))


def _expected_context_for(scenario: ScenarioSpec) -> dict:
    """Merge each spec's compile_check().expected_context for one scenario."""
    ctx: dict = {}
    for spec in scenario.parsed_check_specs:
        ctx.update(_compile_check(spec).expected_context)
    return ctx


def compile_capability(
    capability: CapabilitySpec, execution: ExecutionBlock
) -> list[tuple[list[ExampleData], list[Scorer]]]:
    """Compile one capability into a list of ``(dataset, scorers)`` groups.

    Walks the capability's scenarios, parses each scenario's success
    checkers (already validated at config-load time), and groups
    scenarios whose checkers produce equivalent scorers. Returns one
    ``(dataset, scorers)`` pair per group; the runner invokes
    ``run_eval_async`` once per group and merges the results.

    Each scenario is replicated by ``execution.runs_per_scenario`` inside
    its group. Each replicated example carries the per-scenario merged
    ``ExpectedResult.context`` produced by its checkers.

    Agent input contract:
        Each ExampleData.input is an ``AgentCaseInput`` TypedDict (see
        ``benchmark/agent_input.py``). Every field on it becomes a kwarg
        on ``BaseAgent.run_case`` (with ``prompt`` as the lone
        positional); the runner shim unpacks fields explicitly so adding
        a new per-case kwarg requires updating ``AgentCaseInput`` and
        the shim together — the contract is visible in code review at
        every participating site.

    Metadata contract:
        Each ExampleData.metadata contains:
            {"scenario_id": int, "scenario_name": str, "capability": str,
             "run_index": int (0..runs_per_scenario-1),
             ...each key from scenario.dimensions}

    Args:
        capability: The capability spec from the benchmark YAML.
        execution: Execution block (used for runs_per_scenario replication).

    Returns:
        A list of ``(dataset, scorers)`` pairs, one per distinct
        scorer-set across the capability's scenarios. The list is
        ordered by first-occurrence of each scorer-set, which keeps
        the runner's experiment-id sequence stable across runs.
    """
    # Insertion-ordered dict so groups iterate in first-occurrence order.
    groups: dict[
        tuple[tuple[str, Any], ...], tuple[list[ExampleData], list[Scorer]]
    ] = {}

    for scenario in capability.scenarios:
        try:
            key = _scenario_group_key(scenario)
            if key not in groups:
                scorers = [
                    _compile_check(s).scorer for s in scenario.parsed_check_specs
                ]
                groups[key] = ([], scorers)
            dataset, _scorers = groups[key]

            ctx = _expected_context_for(scenario)
            # Per-replication examples share a single applier-list reference.
            # Safe because PreconditionApplier instances are stateless: apply()
            # mutates its `target` argument, never `self`. If a future applier
            # ever carries per-call state, copy each instance per-replication
            # (a `list(...)` shallow copy is not enough — the appliers inside
            # would still be shared).
            preconditions = list(scenario.parsed_preconditions)
            case_input: AgentCaseInput = {
                "prompt": scenario.prompt,
                "preconditions": preconditions,
            }
            for run_idx in range(execution.runs_per_scenario):
                dataset.append(
                    ExampleData(
                        input=case_input,
                        expected=ExpectedResult(expected="", context=ctx),
                        metadata={
                            "scenario_id": scenario.id,
                            "scenario_name": scenario.name,
                            "capability": capability.name,
                            "run_index": run_idx,
                            **scenario.dimensions,
                        },
                    )
                )
        except Exception as exc:
            raise type(exc)(
                f"Capability {capability.name!r}, scenario {scenario.id!r} "
                f"({scenario.name!r}): {exc}"
            ) from exc

    return list(groups.values())
