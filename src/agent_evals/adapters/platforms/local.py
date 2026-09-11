# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local filesystem platform adapter.

Implements Platform for local file storage.
"""

import asyncio
import inspect
import json
import logging
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar, Literal

from foundry_agent_core import redact_session_ids
from foundry_agent_core.encryption import encrypt, load_encryption_key
from pydantic import Field

from agent_evals.adapters.platforms.utils import (
    compute_aggregate_scores,
    compute_summary_counts,
    is_async_callable,
)
from agent_evals.core._registries import platform_registry
from agent_evals.core.ports import Scorer
from agent_evals.core.types import (
    Dataset,
    EvalConfig,
    EvalExample,
    EvalResult,
    ExampleData,
    ExpectedResult,
    PathSafeIdentifier,
    PlatformConfig,
    Score,
    TaskResult,
    task_result_to_example,
)

logger = logging.getLogger(__name__)

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MiB — symmetric with _readers.py
_MAX_LINE_SIZE = 10 * 1024 * 1024  # 10 MiB per JSONL record


class LocalConfig(PlatformConfig):
    """Config for the local (filesystem) platform adapter."""

    name: Literal["local"] = "local"
    # Path-safe constrained here and nowhere else among the platform configs.
    # This adapter's ``experiment`` becomes a directory name under
    # ``output_dir/experiments/``, so a value carrying a separator or a drive
    # prefix creates directories outside the configured root. The remote
    # adapters' ``experiment`` fields (mlflow, braintrust, langfuse) and the
    # base ``PlatformConfig.experiment`` are run labels transmitted to a remote
    # service with no local filesystem sink, so constraining them would reject
    # names those services accept while preventing nothing.
    experiment: PathSafeIdentifier | None = Field(
        default=None,
        description=(
            "Output folder name under output_dir/experiments/; "
            "auto-generated as eval-<timestamp> if omitted. You usually "
            "leave this unset."
        ),
    )
    output_dir: str | None = Field(
        default=None,
        description=(
            "Root directory for experiment output. Defaults to "
            "./local_eval_results when unset."
        ),
    )


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
                f"Dataset item {i} is not an ExampleData instance. "
                f"Got {type(example).__name__}: {example}"
            )


@platform_registry.register("local")  # ty: ignore[invalid-argument-type]
class LocalPlatform:
    """Local filesystem adapter for evaluation results.

    Stores evaluation results in JSONL format to local filesystem.
    Directory structure:
        {output_dir}/experiments/{experiment_id}/
        ├── metadata.json       # Experiment metadata
        ├── results.jsonl       # Per-example results (streaming)
        └── summary.json        # Aggregate scores and statistics

    Unlike cloud adapters (MLflow, Braintrust), the local adapter handles the
    complete evaluation pipeline internally:
    1. Task execution for each example
    2. Scorer evaluation in parallel
    3. Result aggregation and storage

    This makes it ideal for development, testing, and offline analysis.
    """

    config_class: ClassVar[type[PlatformConfig]] = LocalConfig

    def __init__(self):
        """Initialize local adapter."""
        self._encryption_key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        self.output_dir: Path | None = Path("./local_eval_results")
        self.experiment_id: str | None = None
        self.experiment_path: Path | None = None

    def evaluate(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        evaluators: Sequence[Scorer],
        platform: PlatformConfig | None = None,
        config: EvalConfig | None = None,
    ) -> EvalResult:
        """Run a synchronous evaluation.

        Args:
            task: Optional callable that processes inputs. If None, examples must have
                  pre-populated outputs (historical data pattern).
            dataset: The dataset to evaluate on
            evaluators: The evaluators to use for scoring
            platform: Typed platform configuration (LocalConfig or None for default).
            config: Concurrency controls; see EvalConfig.

        Returns:
            EvalResult: The evaluation results
        """
        return asyncio.run(self.aevaluate(task, dataset, evaluators, platform, config))

    async def aevaluate(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        evaluators: Sequence[Scorer],
        platform: PlatformConfig | None = None,
        config: EvalConfig | None = None,
    ) -> EvalResult:
        """Run an asynchronous evaluation.

        Args:
            task: Optional callable that processes inputs. If None, examples must have
                  pre-populated outputs (historical data pattern).
            dataset: The dataset to evaluate on
            evaluators: The evaluators to use for scoring
            platform: Typed platform configuration (LocalConfig or None for default).
            config: Concurrency controls; see EvalConfig.

        Returns:
            EvalResult: The converted evaluation result

        Raises:
            ValueError: If dataset is empty or evaluators is empty
        """
        start_time = time.perf_counter()

        # Validate inputs
        _validate_inputs(dataset, evaluators)

        # Narrow platform config to LocalConfig (or use defaults)
        if platform is None:
            cfg = LocalConfig()
        elif isinstance(platform, LocalConfig):
            cfg = platform
        else:
            raise ValueError(
                "LocalPlatform requires a LocalConfig platform configuration "
                f"(got {type(platform).__name__}). Pass platform=LocalConfig(...) "
                "or omit platform to use defaults."
            )
        if cfg.output_dir:
            self.output_dir = Path(cfg.output_dir)
        experiment_name = cfg.experiment or f"eval-{int(time.time())}"

        # Initialize experiment
        self._initialize_experiment(experiment_name, cfg.model_dump())

        exec_cfg = config or EvalConfig()
        semaphore = asyncio.Semaphore(exec_cfg.max_concurrent_tests)

        logger.info(
            "Starting evaluation: %d examples, %d evaluators, max_concurrent_tests=%d",
            len(dataset),
            len(evaluators),
            exec_cfg.max_concurrent_tests,
        )

        # Process examples concurrently, bounded by max_concurrent_tests.
        # Each task returns (index, EvalExample); we sort by index to preserve
        # dataset order regardless of completion order.
        async def _run_one(
            i: int, example_data: ExampleData
        ) -> tuple[int, EvalExample]:
            async with semaphore:
                try:
                    logger.debug("Processing example %d/%d", i + 1, len(dataset))
                    example = await self._process_example(
                        example_data, task, evaluators
                    )
                except Exception as e:
                    logger.error("Failed to process example %d: %s", i, e)
                    example = EvalExample(
                        input=example_data.input,
                        output="",
                        expected=example_data.expected.expected
                        if example_data.expected
                        else "",
                        scores={},
                        metadata=example_data.metadata,
                        duration=0.0,
                        error=str(e),
                    )
                self._log_example(example)
                return i, example

        async with asyncio.TaskGroup() as tg:
            tasks = [tg.create_task(_run_one(i, ex)) for i, ex in enumerate(dataset)]

        examples: list[EvalExample] = [
            t.result()[1] for t in sorted(tasks, key=lambda t: t.result()[0])
        ]

        # Aggregate through the shared helpers so the counts and the scores
        # come from one implementation and cannot disagree.
        #
        # Deliberately not passing expected scorer names derived from the
        # evaluator callables: `scores` is keyed by the name the scorer puts
        # in its own Score, which for a factory-built scorer is not
        # `evaluator.__name__` (that is the literal "scorer"). Deriving names
        # from the callables would invent a phantom failing key on healthy
        # runs. The helper instead marks a wholly-unscored run explicitly.
        scores, pass_rates = compute_aggregate_scores(examples)
        total, successful, failed = compute_summary_counts(examples)

        # Create result
        duration = time.perf_counter() - start_time
        result = EvalResult(
            experiment_id=self.experiment_id or "",
            experiment_url=(
                str(self.experiment_path.absolute()) if self.experiment_path else ""
            ),
            platform="local",
            scores=scores,
            pass_rates=pass_rates,
            examples=examples,
            summary={
                "total_examples": total,
                "successful_examples": successful,
                "failed_examples": failed,
            },
            duration=duration,
        )

        # Save final result
        self._save_summary(result)

        logger.info("Evaluation complete in %.2fs", duration)
        return result

    def _initialize_experiment(
        self, experiment_name: str, config: dict[str, Any]
    ) -> None:
        """Initialize experiment directory and metadata.

        Args:
            experiment_name: Name of the experiment
            config: Configuration dictionary
        """
        # Generate unique experiment ID
        self.experiment_id = str(uuid.uuid4())

        # Create experiment directory
        if self.output_dir:
            self.experiment_path = (
                self.output_dir
                / "experiments"
                / f"{experiment_name}-{self.experiment_id}"
            )
        else:
            self.experiment_path = None
        if self.experiment_path:
            self.experiment_path.mkdir(parents=True, exist_ok=True)

        # Save experiment metadata
        created_at = time.time()
        metadata = {
            "experiment_id": self.experiment_id,
            "experiment_name": experiment_name,
            "created_at": created_at,
            "created_at_iso": datetime.fromtimestamp(created_at, UTC).isoformat(),
            "config": config,
            "platform": "local",
        }

        if self.experiment_path:
            serialized = json.dumps(metadata)
            if len(serialized) > _MAX_FILE_SIZE:
                raise ValueError(
                    f"Serialized metadata exceeds maximum file size "
                    f"({_MAX_FILE_SIZE} bytes): {len(serialized)} bytes"
                )
            metadata_path = self.experiment_path / "metadata.json"
            with metadata_path.open("w") as f:
                json.dump(encrypt(metadata, self._encryption_key), f)

        logger.info("Experiment initialized: %s", self.experiment_path)

    async def _process_example(
        self,
        example_data: ExampleData,
        task: Callable[[Any], Any] | None,
        evaluators: Sequence[Scorer],
    ) -> EvalExample:
        """Process a single example: execute task and invoke evaluators.

        Args:
            example_data: Validated ExampleData instance
            task: Optional task function
            evaluators: List of evaluator callables

        Returns:
            EvalExample with task output and all evaluator results
        """
        input_value = example_data.input
        expected_value = example_data.expected
        metadata = example_data.metadata

        # Execute task with timing
        start_time = time.perf_counter()
        task_error = None
        output = None

        if example_data.output is not None:
            # Use pre-populated output (historical data pattern)
            output: TaskResult = example_data.output
            if task is not None:
                logger.warning(
                    "Pre-populated output found for input '%s'. "
                    "Skipping task execution. "
                    "Pre-populated output takes precedence over task execution.",
                    input_value,
                )
            logger.debug("Using pre-populated output for historical data")
        elif task is not None:
            # Execute task
            try:
                output: TaskResult = await self._execute_task_async(task, input_value)
            except Exception as e:
                task_error = str(e)
                output = TaskResult(output="")  # Set to empty string on task failure
                logger.error("Task failed for input %s: %s", input_value, e)
        else:
            raise ValueError(
                "Either task must be provided or examples must have pre-populated "
                "outputs (historical data pattern)"
            )

        duration = time.perf_counter() - start_time

        # Only score an output the task actually produced. On a crash `output`
        # is the empty-string placeholder above, and scoring against it
        # invites a genuine pass from any scorer that doesn't inspect the
        # output (or that happens to match ""), which then averages into the
        # headline aggregate as real performance.
        if task_error is not None:
            scores: dict[str, Score] = {}
        else:
            # Execute all evaluators in parallel
            scores = await self._execute_evaluators_parallel(
                evaluators, output, expected_value
            )

        # Create EvalExample via the trajectory-aware helper so
        # context["outputs"] survives onto EvalExample.trajectory.
        example = task_result_to_example(
            task_result=output,
            input=input_value,
            expected=expected_value,
            scores=scores,
            metadata=metadata,
            duration=duration,
            error=task_error,
        )

        return example

    async def _execute_task_async(
        self, task: Callable[[Any], Any], input_value: Any
    ) -> Any:
        """Execute task (async or sync).

        Args:
            task: Task callable (sync or async)
            input_value: Input to pass to task

        Returns:
            Task output
        """
        # Run sync tasks directly (they're typically fast).
        result = task(input_value)
        # Detection is best-effort, so a task that hides its asyncness still
        # lands here as an awaitable; awaiting keeps coroutines out of outputs.
        if inspect.isawaitable(result):
            return await result
        return result

    async def _execute_evaluators_parallel(
        self,
        evaluators: Sequence[Scorer],
        output: TaskResult,
        expected: ExpectedResult | None,
    ) -> dict[str, Score]:
        """Execute all evaluators in parallel using asyncio.TaskGroup.

        Args:
            evaluators: List of evaluator callables (sync or async)
            output: Task output to evaluate
            expected: Expected output for comparison
            context: Additional context dict

        Returns:
            Dict mapping evaluator names to Score objects
        """
        scores = {}

        # Create tasks for all evaluators using TaskGroup
        async with asyncio.TaskGroup() as tg:
            tasks = []
            for evaluator in evaluators:
                task = tg.create_task(
                    self._invoke_evaluator_safe(evaluator, output, expected)
                )
                tasks.append(task)

        # Collect results from all tasks
        for task in tasks:
            score = task.result()
            scores[score.name] = score

        return scores

    async def _invoke_evaluator_safe(
        self, evaluator: Scorer, output: TaskResult, expected: ExpectedResult | None
    ) -> Score:
        """Safely invoke an evaluator, catching exceptions and returning error scores.

        Args:
            evaluator: Evaluator callable
            output: Task output to evaluate
            expected: Expected output for comparison
            context: Additional context dict

        Returns:
            Score object (error score if evaluator failed)
        """
        evaluator_name = None
        try:
            score = await self._invoke_evaluator(evaluator, output, expected)
            return score
        except Exception as e:
            # Evaluator failure: create 0.0 error score
            if evaluator_name is None:
                evaluator_name = str(
                    evaluator.__name__
                    if hasattr(evaluator, "__name__")
                    else evaluator.__class__.__name__
                )
            logger.error("Evaluator %s failed: %s", evaluator_name, e)
            return Score(
                name=evaluator_name,
                value=0.0,
                passed=False,
                # `error` is what the summary counts and the aggregates read to
                # tell "the scorer failed" apart from "the scorer scored 0.0";
                # the two are otherwise numerically identical. Kept in
                # metadata as well for callers reading the older convention.
                error=str(e),
                metadata={"error": str(e)},
                reasoning=f"Evaluator failed with error: {e}",
            )

    async def _invoke_evaluator(
        self, evaluator: Scorer, output: TaskResult, expected: ExpectedResult | None
    ) -> Score:
        """Invoke an evaluator (sync or async).

        Args:
            evaluator: Evaluator callable
            output: Task output to evaluate
            expected: Expected output for comparison
            context: Additional context dict

        Returns:
            Score object
        """
        if is_async_callable(evaluator):
            result = evaluator(output, expected)
        else:
            # Sync scorers run on a thread so blocking calls (e.g. requests,
            # time.sleep) don't stall the event loop.
            result = await asyncio.to_thread(evaluator, output, expected)
        # Detection is best-effort, so a scorer that hides its asyncness still
        # lands here as an awaitable; awaiting keeps coroutines out of scores.
        if isinstance(result, Score):
            return result
        return await result

    def _log_example(self, example: EvalExample) -> None:
        """Log a single example result to results.jsonl.

        Session identifiers are masked on the way out (STIG V-222577). The
        redaction runs on the serialized string rather than excluding fields,
        because there is no field to exclude: a session identifier arrives
        inside the free-form ``metadata`` / ``context`` dicts, not as a
        declared attribute of ``EvalExample``. An example carrying no
        ``session_id`` key serializes byte-identically to before.

        The masking is deliberately irreversible, so a token in this file
        cannot be mapped back to a session during incident response.

        Args:
            example: EvalExample to log
        """
        if not self.experiment_path:
            return

        results_path = self.experiment_path / "results.jsonl"
        with results_path.open("a") as f:
            redacted = redact_session_ids(example.model_dump_json())
            if len(redacted) > _MAX_LINE_SIZE:
                raise ValueError(
                    f"Serialized example exceeds maximum record size "
                    f"({_MAX_LINE_SIZE} bytes): {len(redacted)} bytes"
                )
            envelope = encrypt(json.loads(redacted), self._encryption_key)
            f.write(json.dumps(envelope) + "\n")

    def _save_summary(self, result: EvalResult) -> None:
        """Save evaluation summary to summary.json.

        Args:
            result: Complete evaluation result
        """
        if not self.experiment_path:
            return

        summary_data = {
            "experiment_id": result.experiment_id,
            "platform": result.platform,
            "scores": result.scores,
            "pass_rates": result.pass_rates,
            "summary": result.summary,
            "duration": result.duration,
            "total_examples": len(result.examples),
        }

        serialized = json.dumps(summary_data)
        if len(serialized) > _MAX_FILE_SIZE:
            raise ValueError(
                f"Serialized summary exceeds maximum file size "
                f"({_MAX_FILE_SIZE} bytes): {len(serialized)} bytes"
            )
        summary_path = self.experiment_path / "summary.json"
        with summary_path.open("w") as f:
            json.dump(encrypt(summary_data, self._encryption_key), f)
