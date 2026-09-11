# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Braintrust platform adapter.

Implements ``Platform`` for Braintrust. The adapter validates
config, normalizes the dataset and scorers, then delegates the eval to a
``BraintrustClientPort`` (default: ``_RealBraintrustClient`` wrapping the
``braintrust`` SDK module). The wrapper builds ``EvalCase`` and calls
``braintrust.EvalAsync``; the adapter converts the result into an
``EvalResult``.

``pull_traces()`` exports traces via the Braintrust BTQL HTTP API
(separate from the SDK seam) and converts them into ``ExampleData``.

Agent Evals scorers (which return ``Score``) are wrapped to return
floats for the Braintrust SDK; threshold metadata flows through
``_scorer_thresholds`` to ``_convert_case_scores``.
"""

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal, cast

from pydantic import Field

from agent_evals.adapters.platforms.braintrust_client import (
    BraintrustClientPort,
    _RealBraintrustClient,
)
from agent_evals.adapters.platforms.transport_posture import log_transport_posture
from agent_evals.adapters.platforms.utils import (
    VerdictCache,
    compute_aggregate_scores,
    compute_summary_counts,
    infer_scorer_name,
    verdict_key,
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
    PlatformConfig,
    Score,
    TaskResult,
    task_result_to_example,
)

logger = logging.getLogger(__name__)

# Default threshold above which numeric scores are considered a "pass".
PASS_THRESHOLD = 0.5

# Braintrust BTQL API endpoint
BTQL_ENDPOINT = "https://api.braintrust.dev/btql"


def _normalize_expected(raw_expected: Any) -> ExpectedResult | None:
    """Normalize expected value to ExpectedResult.

    Args:
        raw_expected: Raw expected value (string, ExpectedResult, or None)

    Returns:
        ExpectedResult or None
    """
    if raw_expected is None:
        return None
    if isinstance(raw_expected, ExpectedResult):
        return raw_expected
    return ExpectedResult(expected=str(raw_expected))


# Import guard for optional dependency
try:
    import braintrust
except ImportError as e:
    raise ImportError(
        "Braintrust adapter requires the braintrust package. "
        "Install it with: uv add 'agent-evals[braintrust]' "
        "or: uv add braintrust"
    ) from e


class BraintrustConfig(PlatformConfig):
    """Config for the Braintrust platform adapter."""

    name: Literal["braintrust"] = "braintrust"
    experiment: str = Field(description="Experiment name shown in the Braintrust UI.")
    project: str = Field(description="Braintrust project the experiment lives under.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Freeform metadata attached to the experiment.",
    )


@platform_registry.register("braintrust")  # ty: ignore[invalid-argument-type]
class BraintrustPlatform:
    """Braintrust platform adapter using the `braintrust.Eval` API."""

    config_class: ClassVar[type[PlatformConfig]] = BraintrustConfig

    def __init__(self, *, client: BraintrustClientPort | None = None) -> None:
        """Initialize the adapter.

        Args:
            client: Optional client port. Tests inject a fake; production
                leaves this unset and the default is constructed lazily.
        """
        self._last_experiment_url: str | None = None
        self._last_experiment_id: str | None = None
        self._scorer_thresholds: dict[str, float] = {}
        self._verdicts = VerdictCache()
        self._client: BraintrustClientPort | None = client
        log_transport_posture(logger, "braintrust")

    def _get_client(self) -> BraintrustClientPort:
        """Return the client port, lazy-constructing the default on first call."""
        if self._client is None:
            self._client = _RealBraintrustClient(braintrust)
        return self._client

    # ------------------------------------------------------------------
    # Public Platform interface
    # ------------------------------------------------------------------
    def evaluate(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        evaluators: Sequence[Scorer],
        platform: PlatformConfig | None = None,
        config: EvalConfig | None = None,  # noqa: ARG002 - Braintrust SDK manages concurrency
    ) -> EvalResult:
        """Run evaluation synchronously (wraps `aevaluate` with asyncio.run)."""
        return asyncio.run(self.aevaluate(task, dataset, evaluators, platform, config))

    async def aevaluate(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        evaluators: Sequence[Scorer],
        platform: PlatformConfig | None = None,
        config: EvalConfig | None = None,  # noqa: ARG002 - Braintrust SDK manages concurrency
    ) -> EvalResult:
        """Run evaluation asynchronously using `braintrust.Eval`.

        Args:
            task: User task function to evaluate (sync or async, but typically sync).
            dataset: Agent Evals dataset (list[ExampleData]).
            evaluators: Agent Evals scorers (e.g., from `agent_evals.adapters.scorers`)
                       These are automatically converted to Braintrust-compatible format
            platform: Typed platform configuration (BraintrustConfig).
            config: Accepted for Protocol conformance; Braintrust SDK manages
                concurrency.

        Returns:
            EvalResult: Unified Agent Evals result.

        Raises:
            ValueError: If required config fields or API key are missing.
        """
        cfg = platform if isinstance(platform, BraintrustConfig) else None
        if cfg is None:
            raise ValueError(
                "Braintrust adapter requires a BraintrustConfig platform "
                "configuration. Pass platform=BraintrustConfig("
                "project='my-project', experiment='my-eval')."
            )

        project = cfg.project
        experiment = cfg.experiment
        metadata = cfg.metadata

        # Check for API key
        if not os.getenv("BRAINTRUST_API_KEY"):
            raise ValueError(
                "Braintrust adapter requires BRAINTRUST_API_KEY environment variable. "
                "Get your API key from https://www.braintrust.dev/app/settings"
            )

        # Verdicts are per-run: a reused adapter instance must not serve an
        # earlier run's verdict to a later example that hashes to the same key,
        # and two concurrent runs on one instance would fight over the cache.
        self._verdicts.begin_run(type(self).__name__)
        try:
            return await self._aevaluate_inner(
                task, dataset, evaluators, project, experiment, metadata
            )
        finally:
            self._verdicts.end_run()

    async def _aevaluate_inner(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        evaluators: Sequence[Scorer],
        project: str,
        experiment: str,
        metadata: dict[str, Any],
    ) -> EvalResult:
        """Run the evaluation body, with the verdict cache already open.

        Split out of :meth:`aevaluate` so the cache's run lifetime is a single
        ``try``/``finally`` there rather than wrapping this whole body inline.
        """
        # Convert Agent Evals dataset to Braintrust eval data
        bt_data = self._convert_to_platform_dataset(dataset)

        # If no task is provided, use a historical replay task that returns
        # the pre-populated output from each case. This enables historical
        # scoring without re-executing the original task.
        if task is None:
            logger.info(
                "No task provided; using historical replay task based on "
                "pre-populated outputs."
            )
            task = self._prepare_historical_replay(dataset)

        # Wrap task so Braintrust shows plain strings in the UI.
        task = self._wrap_task_for_braintrust(task)

        # Convert agent-evals scorers to Braintrust-compatible scorers
        bt_scorers = self._convert_to_platform_scorer(evaluators)

        logger.info(
            "Starting Braintrust Eval: project=%s, experiment=%s, examples=%d",
            project,
            experiment,
            len(bt_data),
        )

        client = self._get_client()
        eval_result = await client.run_eval(
            project=project,
            experiment=experiment,
            data=bt_data,
            task=task,
            scorers=bt_scorers,
            metadata=metadata,
        )

        # Convert Braintrust Eval result to Agent Evals EvalResult
        final = self._convert_from_platform_result(
            eval_result, dataset, project, experiment, metadata
        )
        return final

    # ------------------------------------------------------------------
    # Data conversion helpers
    # ------------------------------------------------------------------
    def _prepare_historical_replay(self, dataset: Dataset) -> Callable:
        """Build a replay task that returns each example's pre-populated output.

        Outputs are queued by input value (FIFO per input). Unhashable
        inputs fall back to a single-pass FIFO list. Missing outputs
        become ``TaskResult(output="")``.

        Returns:
            A replay task callable matching the ``task(input_value, *_, **__)``
            shape Braintrust's SDK invokes.
        """
        precomputed: dict[Any, list[Any]] = {}
        fallback: list[Any] = []
        for example in dataset:
            out: Any = (
                example.output if example.output is not None else TaskResult(output="")
            )
            try:
                precomputed.setdefault(example.input, []).append(out)
            except TypeError:
                # Unhashable input; fall back to FIFO ordering.
                fallback.append(out)

        fallback_iter = iter(fallback)

        def replay_task(input_value: Any, *_args: Any, **_kwargs: Any) -> Any:
            try:
                queue = precomputed.get(input_value)
            except TypeError:
                # Unhashable input; replay from the fallback queue below.
                queue = None
            if queue:
                return queue.pop(0)
            try:
                return next(fallback_iter)
            except StopIteration:
                return TaskResult(output="")

        return replay_task

    def _wrap_task_for_braintrust(self, task: Callable) -> Callable:
        """Wrap a task to return plain string outputs for Braintrust UI/scorers.

        Handles both sync/async tasks and TaskResult returns. Also handles
        Braintrust passing extra args by falling back to just the primary input.
        """
        if getattr(task, "_bt_wrapped", False):
            return task

        is_async_task = asyncio.iscoroutinefunction(task)

        if is_async_task:

            async def wrapped_task(*args, **kwargs):
                try:
                    result = await task(*args, **kwargs)
                except TypeError:
                    # Braintrust may pass extra args; retry with just the input
                    primary_arg = args[0] if args else None
                    result = await task(primary_arg)
                if isinstance(result, TaskResult):
                    return result
                return "" if result is None else result

        else:

            def wrapped_task(*args, **kwargs):
                try:
                    result = task(*args, **kwargs)
                except TypeError:
                    # Braintrust may pass extra args; retry with just the input
                    primary_arg = args[0] if args else None
                    result = task(primary_arg)
                if isinstance(result, TaskResult):
                    return result
                return "" if result is None else result

        marked_task = cast(Any, wrapped_task)
        marked_task._bt_wrapped = True
        return marked_task

    def _convert_to_platform_scorer(
        self, evaluators: Sequence[Scorer]
    ) -> list[Callable]:
        """Convert Agent Evals scorers to Braintrust-compatible scorers.

        Wraps each scorer to:
        - Convert Braintrust's (output, expected, **kwargs) to
          (TaskResult, ExpectedResult)
        - Preserve async/sync behavior
        - Extract float from Score.value

        Args:
            evaluators: Agent Evals scorer functions (sync or async)

        Returns:
            Braintrust-compatible scorer functions returning float
        """
        bt_scorers: list[Callable] = []

        for evaluator in evaluators:
            scorer_name = self._infer_scorer_name(evaluator)
            is_async = asyncio.iscoroutinefunction(evaluator)

            # Create wrapper with proper closure capture
            def _make_wrapper(eval_fn, name, async_fn):
                if async_fn:

                    async def wrapper(
                        output: Any,
                        expected: ExpectedResult | None = None,
                        **kwargs,
                    ):
                        """Async Braintrust scorer wrapper."""
                        normalized_expected = _normalize_expected(expected)
                        # Prepare TaskResult
                        if isinstance(output, TaskResult):
                            if kwargs:
                                context = {**(output.context or {}), **kwargs}
                                task_result = TaskResult(
                                    output=output.output, context=context
                                )
                            else:
                                task_result = output
                        else:
                            task_result = TaskResult(
                                output=str(output), context=kwargs or None
                            )

                        # Call scorer and extract value
                        score = await eval_fn(
                            result=task_result, expected=normalized_expected
                        )
                        if not isinstance(score, Score):
                            raise TypeError(
                                f"Scorer must return Score, got {type(score).__name__}"
                            )
                        if name not in self._scorer_thresholds:
                            self._scorer_thresholds[name] = score.metadata.get(
                                "_threshold", PASS_THRESHOLD
                            )
                        # Braintrust accepts only the float, so keep the
                        # verdict in-process; _convert_case_scores reads it
                        # back off the case. Both sides derive the key from
                        # (output, expected) — the case object carries both.
                        self._verdicts.record(
                            name,
                            verdict_key(task_result, normalized_expected),
                            score,
                        )
                        return float(score.value)

                    wrapper.__name__ = name
                    return wrapper

                else:

                    def wrapper(
                        output: Any,
                        expected: ExpectedResult | None = None,
                        **kwargs,
                    ):
                        """Sync Braintrust scorer wrapper."""
                        normalized_expected = _normalize_expected(expected)
                        # Prepare TaskResult
                        if isinstance(output, TaskResult):
                            if kwargs:
                                context = {**(output.context or {}), **kwargs}
                                task_result = TaskResult(
                                    output=output.output, context=context
                                )
                            else:
                                task_result = output
                        else:
                            task_result = TaskResult(
                                output=str(output), context=kwargs or None
                            )

                        # Call scorer and extract value
                        score = eval_fn(
                            result=task_result, expected=normalized_expected
                        )

                        # Handle case where sync scorer returns awaitable
                        if asyncio.iscoroutine(score):
                            score = asyncio.run(score)

                        if not isinstance(score, Score):
                            raise TypeError(
                                f"Scorer must return Score, got {type(score).__name__}"
                            )
                        if name not in self._scorer_thresholds:
                            self._scorer_thresholds[name] = score.metadata.get(
                                "_threshold", PASS_THRESHOLD
                            )
                        # See the async wrapper above: the verdict stays
                        # in-process rather than riding the float to Braintrust.
                        self._verdicts.record(
                            name,
                            verdict_key(task_result, normalized_expected),
                            score,
                        )
                        return float(score.value)

                    wrapper.__name__ = name
                    return wrapper

            bt_scorers.append(_make_wrapper(evaluator, scorer_name, is_async))

        return bt_scorers

    def _infer_scorer_name(self, evaluator: Callable) -> str:
        """Infer a human-friendly scorer name from a callable.

        Delegates to the shared derivation in ``platforms.utils`` so every
        adapter that keys per-scorer state by name derives that key identically.
        """
        return infer_scorer_name(evaluator)

    def _convert_to_platform_dataset(self, dataset: Dataset) -> list[dict[str, Any]]:
        """Convert Agent Evals Dataset to Braintrust EvalCase-shaped dicts.

        Each dict carries:
            - input: original input
            - expected: expected output (optional)
            - metadata: example-level metadata (optional)

        Pre-populated ``ExampleData.output`` is intentionally NOT
        included — it's read directly by ``_prepare_historical_replay``
        from the original ``Dataset`` when ``aevaluate`` runs in
        replay mode.
        """
        bt_data: list[dict[str, Any]] = []
        for example in dataset:
            case: dict[str, Any] = {"input": example.input}
            if example.expected is not None:
                case["expected"] = example.expected
            if example.metadata:
                case["metadata"] = example.metadata
            bt_data.append(case)
        return bt_data

    def _convert_from_platform_result(
        self,
        platform_result: Any,
        original_dataset: Dataset,
        project: str,
        experiment: str,
        metadata: dict[str, Any],
    ) -> EvalResult:
        """Convert Braintrust Eval result into Agent Evals EvalResult.

        This implementation uses a minimal, structure-tolerant approach:
        - If `platform_result` has a `results` attribute, we try to build EvalExample
          objects from it.
        - If not, we fall back to echoing the original dataset with no scores.
        - Aggregate scores and pass rates are computed from the per-example
          Score objects when available.
        """
        examples = self._build_examples_from_platform_result(
            platform_result, original_dataset
        )
        aggregate_scores, pass_rates = compute_aggregate_scores(examples)
        (
            total_examples,
            successful_examples,
            failed_examples,
        ) = compute_summary_counts(examples)

        return EvalResult(
            experiment_id=str(experiment),
            experiment_url=self._get_experiment_url(project, experiment),
            platform="braintrust",
            scores=aggregate_scores,
            pass_rates=pass_rates,
            examples=examples,
            summary={
                "total_examples": total_examples,
                "successful_examples": successful_examples,
                "failed_examples": failed_examples,
            },
            metadata=metadata,
        )

    def _build_examples_from_platform_result(
        self, platform_result: Any, original_dataset: Dataset
    ) -> list[EvalExample]:
        """Build per-example results from platform result."""
        results = getattr(platform_result, "results", None)

        if results is None:
            logger.warning(
                "Braintrust Eval result has no 'results' attribute; "
                "falling back to dataset-only examples."
            )
            return [
                EvalExample(
                    input=ex.input,
                    output="",
                    expected=ex.expected.expected if ex.expected else None,
                    scores={},
                    metadata=ex.metadata,
                    duration=0.0,
                    error=None,
                )
                for ex in original_dataset
            ]

        examples: list[EvalExample] = []
        for case in results:
            (
                c_input,
                c_output,
                c_expected,
                c_metadata,
                c_error,
                c_duration,
                original_task_result,
            ) = self._extract_case_fields(case)
            scores_dict = self._convert_case_scores(case)

            if original_task_result is not None:
                # The case carried a TaskResult; delegate to the helper
                # so context["outputs"] survives onto trajectory.
                examples.append(
                    task_result_to_example(
                        task_result=original_task_result,
                        input=c_input,
                        expected=ExpectedResult(expected=c_expected)
                        if c_expected is not None
                        else None,
                        scores=scores_dict,
                        metadata=c_metadata,
                        duration=float(c_duration),
                        error=c_error,
                    )
                )
            else:
                # No TaskResult to mine (dict-style case): preserve
                # previous shape; trajectory genuinely unknown.
                examples.append(
                    EvalExample(
                        input=c_input,
                        output=c_output,
                        expected=c_expected,
                        scores=scores_dict,
                        metadata=c_metadata,
                        duration=float(c_duration),
                        error=c_error,
                    )
                )

        return examples

    def _extract_case_fields(
        self, case: Any
    ) -> tuple[Any, Any, Any, dict[str, Any], Any, float, TaskResult | None]:
        """Extract common fields from a Braintrust case result.

        Returns the original ``TaskResult`` (if any) alongside the
        unwrapped string output so callers can read
        ``context['outputs']`` for trajectory persistence without
        re-implementing the unwrap dance.
        """
        # Handle both object-style and dict-style access.
        c_input = getattr(case, "input", None)
        if c_input is None and isinstance(case, dict):
            c_input = case.get("input")

        c_output = getattr(case, "output", None)
        if c_output is None and isinstance(case, dict):
            c_output = case.get("output")
        original_task_result: TaskResult | None = None
        if isinstance(c_output, TaskResult):
            original_task_result = c_output
            c_output = c_output.output

        c_expected = getattr(case, "expected", None)
        if c_expected is None and isinstance(case, dict):
            c_expected = case.get("expected")
        if isinstance(c_expected, ExpectedResult):
            c_expected = c_expected.expected

        c_metadata = getattr(case, "metadata", None)
        if c_metadata is None and isinstance(case, dict):
            c_metadata = case.get("metadata", {})
        if c_metadata is None:
            c_metadata = {}

        c_error = getattr(case, "error", None)
        if c_error is None and isinstance(case, dict):
            c_error = case.get("error")

        c_duration = getattr(case, "duration", None)
        if c_duration is None and isinstance(case, dict):
            c_duration = case.get("duration", 0.0)
        if c_duration is None:
            c_duration = 0.0

        return (
            c_input,
            c_output,
            c_expected,
            c_metadata,
            c_error,
            float(c_duration),
            original_task_result,
        )

    def _convert_case_scores(self, case: Any) -> dict[str, Score]:
        """Convert Braintrust case scores into Agent Evals Score objects."""
        scores_dict: dict[str, Score] = {}

        case_scores = getattr(case, "scores", None)
        if case_scores is None and isinstance(case, dict):
            case_scores = case.get("scores")

        if not case_scores:
            return scores_dict

        # Rebuild the key the scorer wrapper used, from the same two parts.
        # The case carries the raw output and expected it was scored with;
        # verdict_key unwraps TaskResult/ExpectedResult so the wrapper's
        # envelope and this bare form agree.
        case_key = verdict_key(
            self._case_field(case, "output"),
            self._case_field(case, "expected"),
        )

        items = (
            case_scores.items() if hasattr(case_scores, "items") else list(case_scores)
        )

        for name, raw in items:
            # Handle dict outputs from Braintrust scorers
            if isinstance(raw, dict):
                value = raw.get("score")
            else:
                # Try to extract a numeric value from objects with `.score`
                value = getattr(raw, "score", raw)

            if value is None:
                continue

            threshold = self._scorer_thresholds.get(name, PASS_THRESHOLD)

            passed = False
            if isinstance(value, bool):
                passed = value
                float_val = 1.0 if value else 0.0
            elif isinstance(value, int | float):
                float_val = float(value)
                passed = float_val >= threshold
            elif isinstance(value, str):
                lower = value.lower()
                if lower in {"yes", "true", "pass"}:
                    float_val = 1.0
                    passed = True
                elif lower in {"no", "false", "fail"}:
                    float_val = 0.0
                    passed = False
                else:
                    try:
                        float_val = float(value)
                        passed = float_val >= threshold
                    except ValueError:
                        # Non-numeric string; skip
                        continue
            else:
                # Unsupported type; skip
                continue

            # Prefer the verdict the scorer actually computed. The coercion
            # above still runs and still owns float_val; its `passed` is the
            # fallback for scores with no originating agent-evals Score
            # (platform-native scorers, historical results) and for a guarded
            # cache miss.
            preserved = self._verdicts.lookup(name, case_key, float_val)
            if preserved is not None:
                passed = preserved

            scores_dict[name] = Score(
                name=name,
                value=float_val,
                passed=passed,
            )

        return scores_dict

    @staticmethod
    def _case_field(case: Any, field: str) -> Any:
        """Read a field off a case that may be object- or dict-shaped.

        Mirrors the access pattern in ``_extract_case_fields``; kept separate so
        the verdict key can be derived without running the whole unwrap dance.
        """
        value = getattr(case, field, None)
        if value is None and isinstance(case, dict):
            value = case.get(field)
        return value

    def _get_experiment_url(self, project: str, experiment: str) -> str:
        """Construct and cache the Braintrust experiment URL.

        The organization segment is taken from the ``BRAINTRUST_ORG_NAME``
        environment variable, defaulting to ``"default"`` when unset.
        The resulting URL is also stored on the adapter for later inspection.
        """

        org = os.getenv("BRAINTRUST_ORG_NAME", "default")
        experiment_url = (
            f"https://www.braintrust.dev/app/{org}/p/{project}/experiments/{experiment}"
        )
        self._last_experiment_url = experiment_url
        self._last_experiment_id = str(experiment)
        return experiment_url

    # ------------------------------------------------------------------
    # Historical data utilities
    # ------------------------------------------------------------------
    def pull_traces(
        self,
        config: dict[str, Any] | None = None,
        filter: str | None = None,
    ) -> Dataset:
        """Export traces via BTQL and convert them into ExampleData records.

        Args:
            config: Configuration containing:
                - project_id (required): Braintrust project ID to export traces from.
                - api_key (optional): Braintrust API key. If omitted, falls back to
                  BRAINTRUST_API_KEY env var.
                - query (optional): Custom BTQL query to execute. If provided, the
                  ``filter`` argument is ignored.
            filter: Optional BTQL filter clause (e.g. ``"scores.Factuality > 0.8"``).
                Ignored if ``config['query']`` is provided.

        Returns:
            Dataset (list of ExampleData) derived from the platform's execution history.

        Raises:
            ValueError: If required configuration or API key is missing.
            RuntimeError: If BTQL API request fails or response is malformed.
        """
        cfg = config or {}
        project_id = cfg.get("project_id") or cfg.get("project")
        api_key = cfg.get("api_key") or os.environ.get("BRAINTRUST_API_KEY")
        query = cfg.get("query")

        if not project_id:
            raise ValueError(
                "Braintrust API project_id required. Pass config={'project_id': '...'}."
            )
        if not api_key:
            raise ValueError(
                "Braintrust API key required. Pass api_key argument or set "
                "BRAINTRUST_API_KEY environment variable."
            )

        if query is None:
            parts = ["select: *", f"from: project_logs('{project_id}')"]
            if filter:
                parts.append(f"filter: {filter}")
            query = " | ".join(parts)

        payload = {"query": query, "fmt": "json"}
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            BTQL_ENDPOINT,
            data=data,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                charset = response.headers.get_content_charset() or "utf-8"
                response_body = response.read().decode(charset)
        except urllib.error.HTTPError as exc:
            error_text = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(
                f"Braintrust API error {exc.code}: {error_text or exc.reason}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Unable to reach Braintrust API: {exc.reason}") from exc

        try:
            response_json = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Braintrust API returned invalid JSON.") from exc

        traces: list[dict[str, Any]]
        if isinstance(response_json, list):
            traces = response_json
        elif isinstance(response_json, dict):
            if "data" in response_json:
                data = response_json["data"]
                if not isinstance(data, list):
                    raise RuntimeError(
                        f"BTQL response has 'data' field but it is not a list "
                        f"(got {type(data).__name__})"
                    )
                traces = data
            else:
                for key in ("results", "rows", "items"):
                    value = response_json.get(key)
                    if isinstance(value, list):
                        traces = value
                        break
                else:
                    raise RuntimeError(
                        "Unexpected BTQL response format. Expected list or dict with "
                        "'data', 'results', 'rows', or 'items' array."
                    )
        else:
            raise RuntimeError(
                f"Unexpected BTQL response type: {type(response_json).__name__}. "
                "Expected list or dict."
            )

        return self._convert_traces(traces)

    def _convert_traces(self, traces: list[dict[str, Any]]) -> Dataset:
        """Convert Braintrust trace rows into ExampleData records.

        Braintrust's framework persists the task return value as the row's
        ``output`` field (``framework.py:1598``), serializing Pydantic v2
        instances via ``model_dump(exclude_none=True)`` (``bt_json.py:78``).
        When the user task returned a ``TaskResult`` carrying trajectory,
        the row therefore looks like::

            row["output"] == {
                "output": "<bare string output>",
                "context": {"outputs": [...messages...]},
            }

        This method unwraps that shape and places the messages back under
        ``TaskResult.context["outputs"]`` so re-evaluation finds trajectory
        where forward-path scorers always read it. String-returning tasks
        produce a plain ``row["output"] == str``; that case is preserved
        as-is with no context.
        """
        examples: list[ExampleData] = []

        for row in traces:
            if not isinstance(row, dict):
                continue

            is_root = row.get("is_root")
            root_span_id = row.get("root_span_id")
            span_id = row.get("span_id")
            if not (
                is_root is True
                or (isinstance(root_span_id, str) and span_id == root_span_id)
            ):
                continue

            metadata = {"id": row.get("id"), "root_span_id": root_span_id}

            output_result = self._row_output_to_task_result(row.get("output"))

            raw_expected = row.get("expected")
            expected_result: ExpectedResult | None
            if raw_expected is None:
                expected_result = None
            else:
                expected_result = ExpectedResult(expected=str(raw_expected))

            examples.append(
                ExampleData(
                    input=row.get("input"),
                    output=output_result,
                    expected=expected_result,
                    metadata=metadata,
                )
            )

        return examples

    @staticmethod
    def _row_output_to_task_result(raw_output: Any) -> TaskResult | None:
        """Reconstruct a ``TaskResult`` from a BTQL row's ``output`` field.

        Three shapes are recognized:

        1. ``None`` — task didn't log an output. Returns ``None``.
        2. ``dict`` with at least a ``"output"`` key — the model_dump'd
           ``TaskResult``. Unwraps ``output`` to the bare string and
           ``context["outputs"]`` (if present) into the new
           ``TaskResult.context``.
        3. anything else (string, number, list) — a non-TaskResult task
           return. Coerced to ``str`` and stored as the bare output with
           no context.

        See ``framework.py:1598`` (braintrust logs the task return as
        ``output=...``) and ``bt_json.py:78`` (``model_dump(exclude_none=True)``
        for the Pydantic instance) for the serialization path.
        """
        if raw_output is None:
            return None

        if isinstance(raw_output, dict) and "output" in raw_output:
            inner_output = raw_output.get("output", "")
            inner_context = raw_output.get("context")
            if isinstance(inner_context, dict):
                return TaskResult(output=str(inner_output), context=inner_context)
            return TaskResult(output=str(inner_output))

        return TaskResult(output=str(raw_output))
