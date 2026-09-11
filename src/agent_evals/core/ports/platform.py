# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Platform adapter protocol.

Defines:
- Platform: Protocol that all platform adapters must implement
"""

from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Protocol

from agent_evals.core.ports.scorer import Scorer
from agent_evals.core.types import (
    EvalConfig,
    EvalResult,
    ExampleData,
    PlatformConfig,
)


class Platform(Protocol):
    """Protocol defining the interface that all platform adapters must implement.

    Platform adapters handle running evaluations.

    The public contract is three methods:

    - ``evaluate`` — synchronous evaluation (backward compatibility)
    - ``aevaluate`` — asynchronous evaluation (async-first)
    - ``pull_traces`` — export historical traces as a dataset

    Evaluation data flows through ``evaluate`` / ``aevaluate``; the sync method
    typically wraps the async implementation with asyncio.run() to avoid code
    duplication. The remaining methods on this Protocol are private conversion
    helpers (leading underscore) that implementers use to translate between the
    unified types and a backend's native shapes.

    Task call contract:
        Adapters MUST invoke ``task(input_value)`` where ``input_value`` is the
        bare ``ExampleData.input`` from the dataset, not a wrapper dict. This
        contract is the seam that keeps a single user task body portable across
        platforms. Adapters whose SDK expects a different task-call shape MUST
        unwrap it internally before calling the user's task. Pinned by
        ``tests/validation/test_platform_contract.py``.
    """

    config_class: ClassVar[type[PlatformConfig]]
    """The PlatformConfig subclass this adapter accepts.

    Pairs each adapter with its concrete config type. Callers that build a
    config from an untyped source (e.g. a parsed YAML mapping) use this to
    construct the correct typed model; the adapter narrows the base
    PlatformConfig it receives to this type internally.
    """

    def evaluate(
        self,
        task: Callable,
        dataset: list[ExampleData],
        evaluators: Sequence[Scorer],
        platform: PlatformConfig | None = None,
        config: EvalConfig | None = None,
    ) -> EvalResult:
        """Run a synchronous evaluation.

        Args:
            task: The task function to evaluate
            dataset: The dataset to evaluate on
            evaluators: The evaluators to use for scoring
            platform: Typed platform configuration (adapter-specific subclass).
                Adapters narrow this to their own config subclass internally.
            config: Optional ``EvalConfig`` governing how the eval runs;
                see ``EvalConfig`` for the available settings. Adapters that
                defer execution to a platform SDK may accept and ignore it.

        Returns:
            EvalResult with all examples and aggregate scores
        """
        ...

    async def aevaluate(
        self,
        task: Callable | None,
        dataset: list[ExampleData],
        evaluators: Sequence[Scorer],
        platform: PlatformConfig | None = None,
        config: EvalConfig | None = None,
    ) -> EvalResult:
        """Run an asynchronous evaluation.

        Args:
            task: The task function to evaluate (optional for historical data)
            dataset: The dataset to evaluate on
            evaluators: The evaluators to use for scoring
            platform: Typed platform configuration (adapter-specific subclass).
                Adapters narrow this to their own config subclass internally.
            config: Optional ``EvalConfig`` governing how the eval runs;
                see ``EvalConfig`` for the available settings. Adapters that
                defer execution to a platform SDK may accept and ignore it.

        Returns:
            EvalResult with all examples and aggregate scores
        """
        ...

    def pull_traces(
        self,
        config: dict[str, Any] | None = None,
        filter: str | None = None,
    ) -> list[ExampleData]:
        """Export platform traces and convert them into ExampleData records.

        Adapters should fetch trace history using the provided configuration and
        optional filter, then transform each trace into an ExampleData entry
        containing the execution input, output, and any expected result stored
        in the trace payload.

        The first part of this method should handle exporting traces from the
        platform using the provided configuration and filter. After fetching the
        raw traces, delegate conversion to `self._convert_traces` to produce
        ExampleData entries.

        Args:
            config: Platform-specific configuration for trace export.
            filter: Optional filter clause to limit exported traces (e.g. by
                score thresholds, time ranges, or other criteria).
        Returns:
            List of ExampleData derived from the platform's execution history.
        Raises:
            ValueError: If required authentication or configuration is missing.
            RuntimeError: If trace export or conversion fails.
        Example:
            >>> dataset = platform.pull_traces(
            ...     config={"project_id": "my-project"},
            ... )
        """
        ...

    def _convert_traces(self, traces: Any) -> list[ExampleData]:
        """Convert raw platform traces into ExampleData records.

        This private helper encapsulates the transformation logic from the
        platform's trace format into the standardized ExampleData schema.
        """
        ...

    def _convert_to_platform_scorer(
        self, evaluators: Sequence[Scorer]
    ) -> list[Callable[..., Any]]:
        """Convert Agent Evals scorers into platform-specific scorer objects.

        Adapters are responsible for adapting both the scorer call signature and
        return type into the backend's expected interface (for example, a bare
        numeric score or the backend's own feedback object).
        """
        ...

    def _convert_to_platform_dataset(self, dataset: list[ExampleData]) -> Any:
        """Convert Agent Evals dataset into a platform-specific data structure.

        Concrete adapters typically implement this as a thin transformation that
        preserves input / expected / metadata while matching the backend's
        native dataset schema.
        """
        ...

    def _convert_from_platform_result(self, platform_result: Any) -> EvalResult:
        """Convert the platform-specific evaluation result into an EvalResult.

        This should encapsulate the logic to parse the raw platform result object
        and extract relevant metrics, pass rates, and other evaluation data into
        the standardized EvalResult format.
        """
        ...
