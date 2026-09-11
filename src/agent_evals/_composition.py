# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Composition root: wires registries into the public API.

Top-level imports stay minimal so ``import agent_evals`` doesn't pull in
adapter packages or registries. Each ``_get_<family>_registry()`` helper
imports its family function-locally — accidentally hoisting one of those
imports to module scope breaks the lazy-import contract (pinned by
``tests/validation/test_composition_static_imports.py``).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, assert_never

from agent_evals.core.checks import CheckSpec, StateCheck, ToolUseCheck
from agent_evals.core.ports import Scorer
from agent_evals.core.runner import (
    _run_eval_async_internal,
    _run_eval_internal,
)
from agent_evals.core.types import (
    Dataset,
    EvalConfig,
    EvalResult,
    PlatformConfig,
)

if TYPE_CHECKING:
    from agent_evals.benchmark.agent import BaseAgent
    from agent_evals.benchmark.types import BenchmarkResult


def _get_platform_registry() -> Any:
    """Return ``platform_registry`` after firing the family's ``__init__`` side effects.

    Importing the family package registers the ``local`` adapter and writes
    ``_lazy_package`` on the registry (the hook ``_PlatformRegistry`` uses
    to PEP-562-load mlflow / braintrust / langfuse on demand).
    """
    import agent_evals.adapters.platforms  # noqa: F401
    from agent_evals.core._registries import platform_registry

    return platform_registry


def _get_precondition_registry() -> Any:
    """Return ``precondition_registry``.

    Importing the family package fires its ``@register`` decorators.
    """
    import agent_evals.adapters.preconditions  # noqa: F401
    from agent_evals.core._registries import precondition_registry

    return precondition_registry


def _get_success_checker_registry() -> Any:
    """Return ``success_checker_registry``.

    Importing the family package fires its ``@register`` decorators.
    """
    import agent_evals.adapters.success_checkers  # noqa: F401
    from agent_evals.core._registries import success_checker_registry

    return success_checker_registry


def _get_config_reader_registry() -> Any:
    """Return ``config_reader_registry``.

    Importing the family package fires its ``@register`` decorators.
    """
    import agent_evals.adapters.benchmark_config_readers  # noqa: F401
    from agent_evals.core._registries import config_reader_registry

    return config_reader_registry


@dataclass(frozen=True, slots=True)
class CompiledCheck:
    """The triple ``compile_check`` returns for each ``CheckSpec``.

    ``scorer`` is the configured scorer instance; ``expected_context``
    is the dict of keys the scorer reads from ``ExpectedResult.context``;
    ``scorer_config`` distinguishes scorer-affecting variants for
    grouping.
    """

    scorer: Scorer
    expected_context: dict[str, Any]
    scorer_config: dict[str, Any]


def compile_check(spec: CheckSpec) -> CompiledCheck:
    """Map a ``CheckSpec`` to its scorer + context + config triple.

    Adding a new ``CheckSpec`` member without a matching ``case`` arm
    fails type-checking via ``assert_never``. Scorer imports are
    function-local to keep ``import agent_evals`` from eagerly loading
    adapter scorer modules.
    """
    from agent_evals.adapters.scorers.state import StateMatch
    from agent_evals.adapters.scorers.tool_calls import (
        ToolCallAnyMatch,
        ToolCallExactMatch,
        ToolCallSubsetMatch,
        ToolCallSupersetMatch,
        ToolCallUnorderedMatch,
    )

    match spec:
        case StateCheck(expected_state=s):
            return CompiledCheck(
                scorer=StateMatch(),
                expected_context={"expected_state": s},
                scorer_config={},
            )
        case ToolUseCheck(calls=cs, match_calls=mc, match_args=ma):
            scorer_cls = {
                "exact": ToolCallExactMatch,
                "subset": ToolCallSubsetMatch,
                "superset": ToolCallSupersetMatch,
                "unordered": ToolCallUnorderedMatch,
                "any_of": ToolCallAnyMatch,
            }[mc]
            return CompiledCheck(
                scorer=scorer_cls(tool_args_match_mode=ma),
                expected_context={
                    "reference_outputs": [
                        {
                            "role": "assistant",
                            "tool_calls": [
                                {"name": c.name, "args": c.args} for c in cs
                            ],
                        }
                    ]
                },
                scorer_config={"match_calls": mc, "match_args": ma},
            )
        case _:
            assert_never(spec)


def run_eval(
    task: Callable[[Any], Any] | None = None,
    dataset: Dataset | None = None,
    scorers: Sequence[Scorer] | None = None,
    platform: PlatformConfig | None = None,
    config: EvalConfig | None = None,
) -> EvalResult:
    """Run an evaluation synchronously.

    Synchronous wrapper around ``run_eval_async`` (calls ``asyncio.run``).
    Inside an event loop, call ``run_eval_async`` directly.

    Args:
        task: Callable that processes an input and returns a ``TaskResult``
            (sync or async). If ``None``, every example must carry a
            pre-populated ``output`` (historical-data replay).
        dataset: List of ``ExampleData`` instances.
        scorers: Scorer callables (sync or async, may mix). Each receives
            ``(output: TaskResult, expected: ExpectedResult)``.
        platform: Typed ``PlatformConfig`` subclass selecting the adapter.
            Defaults to ``LocalConfig()`` when ``None``.
        config: Optional ``EvalConfig`` governing how the eval runs; see
            ``EvalConfig`` for the available settings.

    Returns:
        ``EvalResult`` with aggregate scores and per-example results.

    Raises:
        ValueError: ``dataset`` is empty, ``scorers`` is empty, or
            ``task`` is ``None`` and an example lacks a pre-populated
            output.
    """
    if platform is None:
        from agent_evals.adapters.platforms.local import LocalConfig

        platform = LocalConfig()
    registry = _get_platform_registry()
    return _run_eval_internal(
        task=task,
        dataset=dataset,
        scorers=scorers,
        platform=platform,
        config=config,
        platform_registry=registry,
    )


async def run_eval_async(
    task: Callable[[Any], Any] | None = None,
    dataset: Dataset | None = None,
    scorers: Sequence[Scorer] | None = None,
    platform: PlatformConfig | None = None,
    config: EvalConfig | None = None,
) -> EvalResult:
    """Run an evaluation asynchronously, fanning scorers out per example.

    Args:
        task: Callable that processes an input and returns a ``TaskResult``
            (sync or async). If ``None``, every example must carry a
            pre-populated ``output`` (historical-data replay).
        dataset: List of ``ExampleData`` instances.
        scorers: Scorer callables (sync or async, may mix). Each receives
            ``(output: TaskResult, expected: ExpectedResult)``.
        platform: Typed ``PlatformConfig`` subclass selecting the adapter.
            Defaults to ``LocalConfig()`` when ``None``.
        config: Optional ``EvalConfig`` governing how the eval runs; see
            ``EvalConfig`` for the available settings.

    Returns:
        ``EvalResult`` with aggregate scores and per-example results.

    Raises:
        ValueError: ``dataset`` is empty, ``scorers`` is empty, or
            ``task`` is ``None`` and an example lacks a pre-populated
            output.
    """
    if platform is None:
        from agent_evals.adapters.platforms.local import LocalConfig

        platform = LocalConfig()
    registry = _get_platform_registry()
    return await _run_eval_async_internal(
        task=task,
        dataset=dataset,
        scorers=scorers,
        platform=platform,
        config=config,
        platform_registry=registry,
    )


async def run_benchmark_async(
    yaml_path: str | Path,
    *,
    agent: type[BaseAgent] | BaseAgent,
) -> BenchmarkResult:
    """Run a benchmark described by a YAML file (async).

    Primary public API for benchmarks. Mirrors ``run_eval_async``.

    Args:
        yaml_path: Path to the benchmark YAML file.
        agent: Either a ``BaseAgent`` subclass (the runner instantiates
            it) or a ``BaseAgent`` instance (used directly). Subclass
            ``agent_evals.BaseAgent`` and implement ``run_case`` — see
            ``docs/benchmarks.md`` for the pattern.
    """
    from agent_evals.benchmark.loader import load_benchmark
    from agent_evals.benchmark.runner import _run_benchmark_async_internal

    config = load_benchmark(
        yaml_path,
        platform_registry=_get_platform_registry(),
        precondition_registry=_get_precondition_registry(),
        success_checker_registry=_get_success_checker_registry(),
        config_reader_registry=_get_config_reader_registry(),
    )
    return await _run_benchmark_async_internal(
        config, agent=agent, config_path=str(yaml_path)
    )
