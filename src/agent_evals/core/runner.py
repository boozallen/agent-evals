# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation orchestration and execution.

Defines:
- run_eval: Synchronous evaluation API
- run_eval_async: Asynchronous evaluation API
- EvalRunner: Synchronous orchestration logic
- AsyncEvalRunner: Asynchronous orchestration logic

Note:
    Core never imports from adapters (hexagonal architecture).
    The platform registry is injected via parameters.
"""

import asyncio
import logging
import time
from collections.abc import Callable, Sequence
from typing import Any

from agent_evals.core.ports import AdapterRegistry, Platform, Scorer
from agent_evals.core.types import (
    Dataset,
    EvalConfig,
    EvalResult,
    ExampleData,
    PlatformConfig,
)

logger = logging.getLogger(__name__)


def _validate_inputs(dataset: Dataset, scorers: Sequence[Scorer]) -> None:
    """Validate evaluation inputs.

    Args:
        dataset: List of ExampleData instances
        scorers: List of scorer callables

    Raises:
        ValueError: If dataset is empty, scorers is empty, or contains non-ExampleData
    """
    if not dataset:
        raise ValueError("Dataset cannot be empty")
    if not scorers:
        raise ValueError("At least one scorer is required")

    # Verify all items are ExampleData instances
    for i, example in enumerate(dataset):
        if not isinstance(example, ExampleData):
            raise ValueError(
                f"Dataset item {i} must be ExampleData instance, "
                f"got {type(example).__name__}. "
                f"Use ExampleData(input=..., expected=...) to create examples."
            )


class EvalRunner:
    """Core evaluation orchestration engine.

    EvalRunner controls the complete evaluation lifecycle:
    1. Validation: Ensure inputs are valid
    2. Initialization: Set up platform adapter
    3. Execution: Delegate to platform adapter
    4. Finalization: Return results from adapter

    The runner owns the control flow - adapters handle the evaluation logic.
    """

    def __init__(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        scorers: Sequence[Scorer],
        platform: PlatformConfig,
        config: EvalConfig | None = None,
        platform_registry: AdapterRegistry[type[Platform]] | None = None,
    ):
        """Initialize evaluation runner.

        Args:
            task: Optional callable that processes inputs and returns TaskResult.
                  If None, examples must have pre-populated outputs
                  (historical data pattern).
            dataset: List of ExampleData instances
            scorers: List of scorer callables
            platform: Typed platform configuration (PlatformConfig subclass).
            config: Optional ``EvalConfig`` governing how the eval runs;
                see ``EvalConfig`` for the available settings.
            platform_registry: Registry that resolves platform names to
                adapter classes. The runner instantiates the resolved
                class with no args.

        Raises:
            ValueError: If dataset is empty or scorers is empty
        """
        # Validate inputs
        _validate_inputs(dataset, scorers)

        self.task = task
        self.dataset = dataset
        self.scorers = scorers
        self.platform = platform
        self.config = config or EvalConfig()
        self.platform_registry = platform_registry

        # Initialize adapter (will be created in run())
        self.adapter = None

    def run(self) -> EvalResult:
        """Execute the evaluation.

        Returns:
            EvalResult with all examples and aggregate scores

        Raises:
            ValueError: If adapter initialization fails
        """
        start_time = time.perf_counter()

        logger.info(
            "Starting evaluation: %d examples, %d scorers",
            len(self.dataset),
            len(self.scorers),
        )

        if self.platform_registry is None:
            raise RuntimeError(
                "platform_registry must be provided; "
                "use run_eval from agent_evals package."
            )
        logger.info("Initializing %s adapter", self.platform.name)
        self.adapter = self.platform_registry.get(self.platform.name)()

        # Validation: task must be provided for live evaluation
        if self.task is None:
            raise ValueError(
                "Task callable is required for live evaluation. "
                "For historical data patterns, use async execution instead."
            )

        result = self.adapter.evaluate(
            task=self.task,
            dataset=self.dataset,
            evaluators=self.scorers,
            platform=self.platform,
            config=self.config,
        )

        duration = time.perf_counter() - start_time
        logger.info("Evaluation complete in %.2fs", duration)

        return result


def _run_eval_internal(
    task: Callable[[Any], Any] | None = None,
    dataset: Dataset | None = None,
    scorers: Sequence[Scorer] | None = None,
    platform: PlatformConfig | None = None,
    config: EvalConfig | None = None,
    platform_registry: AdapterRegistry[type[Platform]] | None = None,
) -> EvalResult:
    """Internal implementation of run_eval - requires platform_registry.

    This is the pure core implementation that has no adapter dependencies.
    The public run_eval function (in _composition.py) wires the default
    registry.
    """
    # Validate required parameters
    if dataset is None:
        raise ValueError("dataset parameter is required")
    if scorers is None:
        raise ValueError("scorers parameter is required")
    if platform_registry is None:
        raise ValueError(
            "platform_registry is required. "
            "Use run_eval / run_eval_async from agent_evals (not core.runner) "
            "for the public API which injects the default registry."
        )

    # Wrap async implementation with asyncio.run() for backward compatibility
    return asyncio.run(
        _run_eval_async_internal(
            task=task,
            dataset=dataset,
            scorers=scorers,
            platform=platform,
            config=config,
            platform_registry=platform_registry,
        )
    )


# Keep run_eval for backward compatibility - will be overwritten by __init__.py
def run_eval(
    task: Callable[[Any], Any] | None = None,
    dataset: Dataset | None = None,
    scorers: Sequence[Scorer] | None = None,
    platform: PlatformConfig | None = None,
    config: EvalConfig | None = None,
) -> EvalResult:
    """Run an evaluation synchronously.

    Main entry point for evaluating a task on a dataset with multiple scorers.
    This is a synchronous wrapper around the async implementation (run_eval_async).

    For new code, consider using `run_eval_async` directly for better performance
    with async scorers and parallel execution.

    Note:
        This function is typically imported from agent_evals package which
        provides the default platform registry. If importing directly from
        core.runner, you must use _run_eval_internal with a platform_registry.

    Args:
        task: Optional callable that processes inputs and returns TaskResult
              (sync or async). If None, examples must have pre-populated outputs
              (historical data pattern).
        dataset: List of ExampleData instances (with ExpectedResult for expected field)
        scorers: List of scorer callables tuples
                 (sync or async, can mix). Scorers receive TaskResult as output
                 parameter and ExpectedResult as expected parameter.
        platform: Typed platform configuration (PlatformConfig subclass) selecting
                  the adapter. Defaults to the local filesystem platform when None.
        config: Optional ``EvalConfig`` governing how the eval runs; see
                ``EvalConfig`` for the available settings.

    Returns:
        EvalResult with aggregate scores and all example results

    Raises:
        ValueError: If dataset is empty or scorers is empty or if task is None and
                    examples don't have pre-populated outputs

    Examples:
        >>> def my_task(input_val):
        ...     return TaskResult(output=f"Output: {input_val}")
        >>>
        >>> def exact_match(result: TaskResult, expected: ExpectedResult):
        ...     matches = result.output == expected.expected
        ...     value = 1.0 if matches else 0.0
        ...     return Score(name="ExactMatch", value=value, passed=matches)
        >>>
        >>> result = run_eval(
        ...     task=my_task,
        ...     dataset=[
        ...         ExampleData(
        ...             input="test",
        ...             expected=ExpectedResult(expected="Output: test")
        ...         )
        ...     ],
        ...     scorers=[exact_match],
        ... )
        >>> print(result.scores)
        {'ExactMatch': 1.0}

    Note:
        This function wraps the async implementation using asyncio.run().
        For async contexts, use run_eval_async directly to avoid
        "RuntimeError: asyncio.run() cannot be called from a running event loop".
    """
    # This stub exists for type checking and documentation.
    # The real implementation is wired in __init__.py with the default factory.
    raise NotImplementedError(
        "run_eval must be imported from agent_evals package, "
        "not agent_evals.core.runner. Use: from agent_evals import run_eval"
    )


class AsyncEvalRunner:
    """Async evaluation orchestration engine.

    AsyncEvalRunner provides async-first evaluation.
    It controls the complete evaluation lifecycle:
    1. Validation: Ensure inputs are valid
    2. Initialization: Set up platform adapter (async)
    3. Execution: Run task on dataset
    4. Finalization: Compute aggregates and return results (async)

    The runner owns the control flow - adapters are passive logging backends.
    """

    def __init__(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        scorers: Sequence[Scorer],
        platform: PlatformConfig,
        config: EvalConfig | None = None,
        platform_registry: AdapterRegistry[type[Platform]] | None = None,
    ):
        """Initialize async evaluation runner.

        Args:
            task: Optional callable that processes inputs and returns TaskResult
                  (sync or async). If None, examples must have pre-populated outputs
                  (historical data pattern).
            dataset: List of ExampleData instances
            scorers: List of scorer callables (sync or async)
            platform: Typed platform configuration (PlatformConfig subclass).
            config: Optional ``EvalConfig`` governing how the eval runs;
                see ``EvalConfig`` for the available settings.
            platform_registry: Registry that resolves platform names to
                adapter classes. The runner instantiates the resolved
                class with no args.

        Raises:
            ValueError: If dataset is empty or scorers is empty
        """
        # Validate inputs
        _validate_inputs(dataset, scorers)

        self.task = task
        self.dataset = dataset
        self.scorers = scorers
        self.platform = platform
        self.config = config or EvalConfig()
        self.platform_registry = platform_registry

        # Initialize adapter (will be created in run_async())
        self.adapter = None

    async def run_async(self) -> EvalResult:
        """Execute the evaluation asynchronously.

        Returns:
            EvalResult with all examples and aggregate scores

        Raises:
            ValueError: If adapter initialization fails
        """
        try:
            start_time = time.perf_counter()

            logger.info(
                "Starting async evaluation: %d examples, %d scorers",
                len(self.dataset),
                len(self.scorers),
            )

            if self.platform_registry is None:
                raise RuntimeError(
                    "platform_registry must be provided; "
                    "use run_eval_async from agent_evals package."
                )
            logger.info("Initializing %s adapter", self.platform.name)
            self.adapter = self.platform_registry.get(self.platform.name)()

            result = await self.adapter.aevaluate(
                task=self.task,
                dataset=self.dataset,
                evaluators=self.scorers,
                platform=self.platform,
                config=self.config,
            )

            duration = time.perf_counter() - start_time
            logger.info("Evaluation complete in %.2fs", duration)

            return result
        finally:
            pass  # No runner-held resources to clean up


async def _run_eval_async_internal(
    task: Callable[[Any], Any] | None = None,
    dataset: Dataset | None = None,
    scorers: Sequence[Scorer] | None = None,
    platform: PlatformConfig | None = None,
    config: EvalConfig | None = None,
    platform_registry: AdapterRegistry[type[Platform]] | None = None,
) -> EvalResult:
    """Internal implementation of run_eval_async - requires platform_registry.

    This is the pure core implementation that has no adapter dependencies.
    The public run_eval_async function (in _composition.py) wires the
    default registry.
    """
    # Validate required parameters
    if dataset is None:
        raise ValueError("dataset parameter is required")
    if scorers is None:
        raise ValueError("scorers parameter is required")
    if platform_registry is None:
        raise ValueError(
            "platform_registry is required. "
            "Use run_eval / run_eval_async from agent_evals (not core.runner) "
            "for the public API which injects the default registry."
        )

    # _composition.py always resolves platform before calling here.
    if platform is None:
        raise RuntimeError(
            "platform must be resolved before _run_eval_async_internal is called. "
            "Use run_eval_async from agent_evals (not core.runner) for the public API."
        )

    runner = AsyncEvalRunner(
        task=task,
        dataset=dataset,
        scorers=scorers,
        platform=platform,
        config=config,
        platform_registry=platform_registry,
    )

    return await runner.run_async()


# Keep run_eval_async for backward compatibility - will be overwritten by __init__.py
async def run_eval_async(
    task: Callable[[Any], Any] | None = None,
    dataset: Dataset | None = None,
    scorers: Sequence[Scorer] | None = None,
    platform: PlatformConfig | None = None,
    config: EvalConfig | None = None,
) -> EvalResult:
    """Run an evaluation asynchronously with parallel scorer execution.

    Primary async API for evaluating a task on a dataset with multiple scorers.
    Scorers execute in parallel per example for dramatic performance improvements
    with I/O-bound scorers (e.g., LLM-as-judge).

    Note:
        This function is typically imported from agent_evals package which
        provides the default platform registry. If importing directly from
        core.runner, you must use _run_eval_async_internal with a
        platform_registry.

    Args:
        task: Optional callable that processes inputs and returns TaskResult
              (sync or async). If None, examples must have pre-populated outputs
              (historical data pattern).
        dataset: List of ExampleData instances (with ExpectedResult for expected field)
        scorers: List of scorer callables
                 (sync or async, can mix). Scorers receive TaskResult as output
                 parameter and ExpectedResult as expected parameter.
        platform: Typed platform configuration (PlatformConfig subclass) selecting
                  the adapter. Defaults to the local filesystem platform when None.
        config: Optional ``EvalConfig`` governing how the eval runs; see
                ``EvalConfig`` for the available settings.

    Returns:
        EvalResult with aggregate scores and all example results

    Raises:
        ValueError: If dataset is empty or scorers is empty or if task is None and
                    examples don't have pre-populated outputs

    Examples:
        Basic async evaluation:
        >>> async def my_scorer(
        ...     output: TaskResult, expected: ExpectedResult
        ... ):
        ...     await asyncio.sleep(1.0)  # Simulate LLM call
        ...     return Score(name="Test", value=1.0, passed=True)
        >>>
        >>> async def main():
        ...     result = await run_eval_async(
        ...         task=lambda x: TaskResult(output=f\"Output: {x}\"),
        ...         dataset=[
        ...             ExampleData(
        ...                 input=\"test\",
        ...                 expected=ExpectedResult(expected=\"test\")
        ...             )
        ...         ],
        ...         scorers=[my_scorer],
        ...     )
        ...     print(result.scores)
        >>>
        >>> asyncio.run(main())
    """
    # This stub exists for type checking and documentation.
    # The real implementation is wired in _composition.py with the default registry.
    raise NotImplementedError(
        "run_eval_async must be imported from agent_evals package, "
        "not agent_evals.core.runner. "
        "Use: from agent_evals import run_eval_async"
    )
