# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""MLflow platform adapter.

Implements Platform for MLflow experiment tracking platform.

This adapter automatically converts Agent Evals scorer functions to MLflow scorers
by applying the @mlflow.genai.scorer decorator on-the-fly. This allows any function
following the Agent Evals Scorer protocol to work seamlessly with MLflow's
mlflow.genai.evaluate() function.
"""

import asyncio
import contextlib
import inspect
import json
import logging
import os
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal

from mlflow.genai.evaluation.entities import (
    EvaluationResult,
    Expectation,
    Feedback,
)
from pydantic import Field

from agent_evals.adapters.platforms.mlflow_client import (
    MlflowSessionPort,
    ScorerSpec,
    SessionInfo,
    TraceRecord,
    _RealMlflowSession,
)
from agent_evals.adapters.platforms.transport_posture import log_transport_posture
from agent_evals.adapters.platforms.utils import (
    MAX_KEY_PART_LEN,
    VerdictCache,
    compute_aggregate_scores,
    compute_summary_counts,
    infer_scorer_name,
    verdict_key,
)
from agent_evals.core._registries import platform_registry
from agent_evals.core.ports import Scorer
from agent_evals.core.transport import validate_secure_transport
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
    _derive_tool_calls,
)

logger = logging.getLogger(__name__)

# Default threshold above which numeric scores are considered a "pass".
PASS_THRESHOLD = 0.5


def _safe_str(value: Any) -> str | None:
    """``str(value)``, or ``None`` if the object's ``__str__`` fails.

    A rendering longer than :data:`MAX_KEY_PART_LEN` is also refused: these
    strings become verdict-key parts and dedup markers, so an unbounded one
    would be hashed and held for the length of a run. Refusing it costs one
    candidate rendering, never a failed run.
    """
    try:
        rendered = str(value)
    except Exception:  # noqa: BLE001 - a user __str__ may raise anything
        return None
    return rendered if len(rendered) <= MAX_KEY_PART_LEN else None


def _safe_repr(value: Any) -> str | None:
    """``repr(value)``, or ``None`` if the object's ``__repr__`` fails.

    Bounded by :data:`MAX_KEY_PART_LEN` for the same reason as
    :func:`_safe_str`.
    """
    try:
        rendered = repr(value)
    except Exception:  # noqa: BLE001 - a user __repr__ may raise anything
        return None
    return rendered if len(rendered) <= MAX_KEY_PART_LEN else None


def _input_renderings(inputs: Any) -> list[Any]:
    """Return every form of ``inputs`` the reconstruction path may see.

    The scorer wrapper is handed the inputs dict this adapter built
    (``{param_name: example.input}``), but the reconstruction path reads them
    back out of ``record.request`` after MLflow has serialized and reshaped
    them — see :meth:`MLflowPlatform._extract_request_response`. No single
    rendering round-trips on every MLflow return shape, so the recording side
    registers the same verdict under each candidate rendering and the
    reconstruction side looks up the one it actually holds.

    Registering several keys is safe: each maps to the same verdict, and a
    rendering that is too coarse to distinguish two examples poisons only its
    own slot (the cache marks conflicting verdicts ambiguous and falls back).
    """
    candidates: list[Any] = [inputs]

    # ``str`` invokes user code and may raise anything; a rendering that cannot
    # be produced is simply one fewer candidate, never a failed run.
    stringified = _safe_str(inputs)
    if stringified is not None:
        candidates.append(stringified)

    # Single-parameter tasks are the common case; the bare value is what a
    # trace records when MLflow unwraps the one-key dict.
    if isinstance(inputs, dict) and len(inputs) == 1:
        only_value = next(iter(inputs.values()))
        candidates.append(only_value)
        only_stringified = _safe_str(only_value)
        if only_stringified is not None:
            candidates.append(only_stringified)

    # Non-serializable inputs simply have one fewer candidate rendering.
    # RecursionError is neither TypeError nor ValueError, so it is named too:
    # a self-referential or very deeply nested input must not fail the run.
    # Bounded like the renderings above: ``json.dumps`` on platform-supplied
    # data has no inherent size limit, and an oversized encoding would be
    # hashed into a key and retained rather than discarded.
    with contextlib.suppress(TypeError, ValueError, RecursionError):
        encoded = json.dumps(inputs)
        if len(encoded) <= MAX_KEY_PART_LEN:
            candidates.append(encoded)

    # Preserve order, drop duplicates by rendered form. A candidate that cannot
    # be repr'd cannot be deduplicated or keyed, so it is dropped.
    seen: set[str] = set()
    unique: list[Any] = []
    for candidate in candidates:
        marker = _safe_repr(candidate)
        if marker is None:
            continue
        if marker not in seen:
            seen.add(marker)
            unique.append(candidate)
    return unique


# Import guard for optional dependency
try:
    import mlflow
except ImportError as e:
    raise ImportError(
        "MLflow adapter requires the mlflow package. "
        "Install it with: uv add 'agent-evals[mlflow]' "
        "or: uv add mlflow"
    ) from e


class MlflowConfig(PlatformConfig):
    """Config for the MLflow platform adapter.

    The tracking server URI is read from the ``MLFLOW_TRACKING_URI``
    environment variable, not config.
    """

    name: Literal["mlflow"] = "mlflow"
    experiment: str = Field(
        description="MLflow experiment name (mlflow.set_experiment); created if absent."
    )


@platform_registry.register("mlflow")  # ty: ignore[invalid-argument-type]
class MLflowPlatform:
    """MLflow platform adapter for evaluation results.

    This adapter integrates with MLflow's evaluation system using:
    - mlflow.set_tracking_uri() for tracking server configuration
    - mlflow.set_experiment() for experiment setup
    - mlflow.genai.evaluate() for running evaluations
    - mlflow.search_traces() for retrieving evaluation results

    The adapter provides both synchronous and asynchronous evaluation methods
    and automatically converts Agent Evals scorer functions to MLflow scorers.

    Async user tasks:
        ``mlflow.genai.evaluate`` is synchronous and drives an async
        ``predict_fn`` with its own ``asyncio.run``, which only works when
        no event loop is already running. ``aevaluate`` therefore offloads
        the blocking ``evaluate`` call to a worker thread via
        ``asyncio.to_thread`` — on that thread there is no running loop, so
        MLflow's async handling works and both sync and async user tasks are
        supported (no ``nest-asyncio`` required).

        Caveat (MLflow policy, not ours): MLflow applies a per-prediction
        timeout to async tasks via the ``MLFLOW_GENAI_EVAL_ASYNC_TIMEOUT``
        environment variable (default 300s). A single async case that runs
        longer raises ``asyncio.TimeoutError``; raise the env var for slow
        agents.
    """

    config_class: ClassVar[type[PlatformConfig]] = MlflowConfig

    def __init__(self, *, session: MlflowSessionPort | None = None) -> None:
        """Initialize MLflow adapter.

        Args:
            session: Optional MlflowSessionPort to inject (typically a
                FakeMlflowSession in tests). When None (production
                default), the first call lazy-creates a
                _RealMlflowSession wrapping the real ``mlflow`` module.
        """
        self.experiment_id: str | None = None
        self.run_id: str | None = None
        self._scorer_thresholds: dict[str, float] = {}
        self._verdicts = VerdictCache()
        self._session: MlflowSessionPort | None = session
        self._session_info: SessionInfo | None = None
        log_transport_posture(logger, "mlflow")

    def _get_session(self) -> MlflowSessionPort:
        """Return cached session, lazy-creating a _RealMlflowSession on first call.

        Unlike PR #202's Langfuse seam (which wraps a real ``get_client()``
        SDK call here), MLflow has no client-level constructor that can
        fail at this point — actual SDK errors surface from the first
        ``configure()`` call when ``mlflow.set_tracking_uri()`` runs. So
        this accessor is a plain lazy-initializer.
        """
        if self._session is None:
            self._session = _RealMlflowSession(mlflow)
        return self._session

    def _resolve_tracking_uri(self) -> str:
        """Return the MLflow tracking URI from the environment.

        Raises:
            ValueError: If MLFLOW_TRACKING_URI is not set.
        """
        uri = os.getenv("MLFLOW_TRACKING_URI", "")
        if not uri:
            raise ValueError(
                "MLflow adapter requires the MLFLOW_TRACKING_URI environment variable "
                "(e.g. https://mlflow.example.com, or a local store such as "
                "./mlruns)."
            )
        return uri

    def _configure_session(
        self,
        session: MlflowSessionPort,
        *,
        tracking_uri: str,
        experiment: str,
    ) -> SessionInfo:
        """Validate the tracking URI, then configure the session with it.

        Both resolution paths converge here — ``aevaluate`` (env var) and
        ``pull_traces`` (config dict) — so this is the single site where a
        caller-supplied tracking URI is validated.

        Validation lives in the adapter rather than in
        ``_RealMlflowSession.configure`` deliberately: ``configure`` sits
        behind the injectable ``MlflowSessionPort``, so a caller passing
        ``session=`` would supply their own ``configure`` and the check would
        go with it. A security check inside a collaborator the caller can
        replace is bypassable by construction. The adapter is also what
        resolves the caller's configuration, so the value is the adapter's to
        police.

        Args:
            session: The session port to configure.
            tracking_uri: Caller-supplied tracking URI.
            experiment: Experiment name to configure.

        Returns:
            SessionInfo: The configured session's info.

        Raises:
            ValueError: If the tracking URI is not an accepted transport — it
                must encrypt in transit, transmit nothing, or address only the
                machine this process runs on.
        """
        validate_secure_transport("MLFLOW_TRACKING_URI", tracking_uri)
        return session.configure(tracking_uri=tracking_uri, experiment=experiment)

    def evaluate(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        evaluators: Sequence[Scorer],
        platform: PlatformConfig | None = None,
        config: EvalConfig | None = None,  # noqa: ARG002 - MLflow SDK manages concurrency
    ) -> EvalResult:
        """Run a synchronous evaluation.

        Args:
            task: The task function to evaluate
            dataset: The dataset to evaluate on
            evaluators: The evaluators to use for scoring
            platform: Typed platform configuration (MlflowConfig).
            config: Accepted for Protocol conformance; MLflow's
                own evaluate() governs concurrency.

        Returns:
            EvalResult: The evaluation results
        """

        final = asyncio.run(self.aevaluate(task, dataset, evaluators, platform, config))

        return final

    async def aevaluate(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        evaluators: Sequence[Scorer],
        platform: PlatformConfig | None = None,
        config: EvalConfig | None = None,  # noqa: ARG002 - MLflow SDK manages concurrency
    ) -> EvalResult:
        """Run an asynchronous evaluation through the session port."""
        cfg = platform if isinstance(platform, MlflowConfig) else None
        if cfg is None:
            raise ValueError(
                "MLflow adapter requires a MlflowConfig platform configuration. "
                "Pass platform=MlflowConfig(experiment='my-experiment')."
            )

        session = self._get_session()
        session_info = self._configure_session(
            session,
            tracking_uri=self._resolve_tracking_uri(),
            experiment=cfg.experiment,
        )
        self._session_info = session_info
        self.experiment_id = session_info.experiment_id

        # A caller may reuse one adapter instance across runs; a verdict from
        # an earlier run must never be reported for this run's examples, and
        # two concurrent runs on one instance would fight over the cache.
        self._verdicts.begin_run(type(self).__name__)
        try:
            return await self._aevaluate_inner(task, dataset, evaluators, session)
        finally:
            self._verdicts.end_run()

    async def _aevaluate_inner(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        evaluators: Sequence[Scorer],
        session: Any,
    ) -> EvalResult:
        """Run the evaluation body, with the verdict cache already open.

        Split out of :meth:`aevaluate` so the cache's run lifetime is a single
        ``try``/``finally`` there rather than wrapping this whole body inline.
        """
        mlflow_dataset = self._convert_to_platform_dataset(task, dataset)
        scorer_specs = self._convert_to_platform_scorer(evaluators)

        # Offload MLflow's blocking, synchronous genai.evaluate to a worker
        # thread. MLflow drives async predict_fns with its own asyncio.run;
        # that only works when there is NO running loop. Running it inline on
        # our event-loop thread (aevaluate is async) makes MLflow's asyncio.run
        # nest and raise "Detected a running event loop". to_thread escapes the
        # loop so both sync and async user tasks work (issue #239).
        platform_result: EvaluationResult = await asyncio.to_thread(
            session.evaluate,
            data=mlflow_dataset,
            predict_fn=task,
            scorers=scorer_specs,
        )
        self.run_id = platform_result.run_id

        return self._convert_from_platform_result(platform_result)

    def pull_traces(
        self,
        config: dict[str, Any] | None = None,
        filter: str | None = None,
    ) -> list[ExampleData]:
        """Export platform traces and convert them into ExampleData records."""
        if not config:
            raise ValueError("MLflow adapter requires a config dictionary")
        if filter is not None:
            self._validate_filter_string(filter)

        session = self._get_session()
        session_info = self._configure_session(
            session,
            tracking_uri=config.get("tracking_uri", ""),
            experiment=config.get("experiment", ""),
        )
        self._session_info = session_info

        records = session.search_traces(
            experiment_id=session_info.experiment_id,
            run_id=None,
            filter=filter,
        )

        return self._convert_traces(records)

    def _convert_traces(self, records: list[TraceRecord]) -> list[ExampleData]:
        """Convert TraceRecord instances into ExampleData entries.

        When the trace carries a decodable ``messages`` array on either
        side of the request/response, the trajectory is placed under
        ``TaskResult.context["outputs"]`` per the existing forward-path
        convention (matches ``TaskResult.from_messages``). Re-evaluation
        then finds trajectory in the same place scorers always read it.
        """
        historical_data: list[ExampleData] = []
        for record in records:
            input_value = record.request

            expectations = [
                a
                for a in record.assessments
                if isinstance(a, dict) and "expectation" in a
            ]
            expected = (
                str(expectations[0]["expectation"]["value"]) if expectations else ""
            )
            output = str(record.response) if record.response else ""

            trajectory = self._trajectory_from_request_response(
                record.request, record.response
            )
            if trajectory:
                output_result = TaskResult(
                    output=output, context={"outputs": trajectory}
                )
            else:
                output_result = TaskResult(output=output)

            historical_data.append(
                ExampleData(
                    input=input_value,
                    expected=ExpectedResult(expected=expected),
                    output=output_result,
                )
            )

        return historical_data

    @staticmethod
    def _validate_filter_string(filter_str: str) -> None:
        """Validate MLflow filter string syntax.

        Performs basic validation of filter string format and field names to catch
        common errors before sending to MLflow. This is not exhaustive validation
        but helps catch obvious mistakes.

        Args:
            filter_str: The filter string to validate

        Raises:
            ValueError: If the filter string contains invalid syntax or
                unsupported fields
        """
        if not filter_str or not filter_str.strip():
            raise ValueError("Filter string cannot be empty")

        # Define valid field prefixes and operators
        valid_field_prefixes = {
            "trace.",
            "span.",
            "tag.",
            "metadata.",
            "feedback.",
            "expectation.",
        }

        valid_operators = {"=", "!=", ">", "<", ">=", "<=", "LIKE", "ILIKE", "RLIKE"}

        # Basic syntax checks
        if filter_str.count('"') % 2 != 0:
            raise ValueError(
                "Filter string has unmatched quotes. "
                "Ensure all string values are properly quoted."
            )

        # Check for at least one valid field prefix
        has_valid_field = any(prefix in filter_str for prefix in valid_field_prefixes)

        if not has_valid_field:
            raise ValueError(
                f"Filter string must contain at least one valid field prefix. "
                f"Valid prefixes: {', '.join(sorted(valid_field_prefixes))}\n"
                f'Example: trace.status = "OK"'
            )

        # Check for at least one valid operator
        has_valid_operator = any(op in filter_str for op in valid_operators)

        if not has_valid_operator:
            raise ValueError(
                f"Filter string must contain at least one valid operator. "
                f"Valid operators: {', '.join(sorted(valid_operators))}\n"
                f'Example: trace.status = "OK"'
            )

    def _convert_to_platform_scorer(
        self, evaluators: Sequence[Scorer]
    ) -> list[ScorerSpec]:
        """Convert Agent Evals scorer functions to ScorerSpec instances.

        The wrapper applies @mlflow.genai.scorer decoration before
        calling mlflow.genai.evaluate; this method only constructs the
        wrapper closure that maps Agent Evals's Score → MLflow's
        Feedback (and runs async scorers synchronously, since MLflow's
        evaluate harness is sync-only).
        """
        scorer_specs: list[ScorerSpec] = []

        for evaluator in evaluators:
            # Must match how the reconstruction path below derives the key it
            # looks up, or the declared threshold is never found. ``__name__``
            # alone does not: a factory-built scorer is a closure literally
            # named ``scorer``, so every one of them collapses onto one key.
            scorer_name = infer_scorer_name(evaluator)
            is_async = asyncio.iscoroutinefunction(evaluator)

            def make_wrapper(original_scorer, name, async_scorer):
                def mlflow_scorer_wrapper(outputs=None, inputs=None, expectations=None):
                    expected = None
                    if expectations:
                        expected = ExpectedResult(
                            expected=expectations.get("expected", ""),
                            context=expectations.get("context"),
                        )

                    raw: Any
                    if async_scorer:
                        raw = asyncio.run(
                            original_scorer(result=outputs, expected=expected)
                        )
                    else:
                        raw = original_scorer(result=outputs, expected=expected)
                        if asyncio.iscoroutine(raw):
                            raw = asyncio.run(raw)

                    if not isinstance(raw, Score):
                        raise TypeError(
                            f"Scorer must return Score, got {type(raw).__name__}"
                        )
                    score: Score = raw

                    threshold = score.metadata.get("_threshold", PASS_THRESHOLD)

                    # Key the cache by the name this score is reported under, so
                    # the reconstruction path's ``assessment.name`` lookup
                    # resolves the same string this wrote. Also key by the name
                    # the ScorerSpec was registered under: MLflow decorates the
                    # wrapper with that name, so the assessment can come back
                    # under either. Both map to the same threshold, so whichever
                    # name the platform reports, the lookup hits.
                    for key in (score.name, name):
                        if not key:
                            continue
                        existing = self._scorer_thresholds.get(key)
                        if existing is None:
                            self._scorer_thresholds[key] = threshold
                        elif existing != threshold:
                            # Two scorers reporting under one name cannot be
                            # told apart on the way back: the reconstruction
                            # path only has ``assessment.name``. First writer
                            # wins, so the other scorer is scored against a
                            # threshold it never declared. Nothing here can
                            # resolve that, so surface it instead of dropping
                            # it silently.
                            logger.warning(
                                "Scorer name %r already has threshold %s cached; "
                                "keeping it and ignoring %s. Two scorers reporting "
                                "under one name are indistinguishable when scores "
                                "come back, so one will be judged against a "
                                "threshold it did not declare. Give them distinct "
                                "Score.name values.",
                                key,
                                existing,
                                threshold,
                            )

                    # Keep the scorer's own verdict in-process; Feedback below
                    # carries only the numeric value, exactly as before.
                    # Recorded under score.name, not the wrapper's name: the
                    # Feedback is named from score.name, so that is the name the
                    # reconstruction path looks up. The two differ whenever a
                    # factory's closure name is not its Score name (StateMatch).
                    for rendering in _input_renderings(inputs):
                        self._verdicts.record(score.name, verdict_key(rendering), score)

                    return Feedback(
                        name=score.name,
                        value=score.value,
                        rationale=score.reasoning,
                    )

                return mlflow_scorer_wrapper

            wrapper = make_wrapper(evaluator, scorer_name, is_async)
            scorer_specs.append(ScorerSpec(name=scorer_name, fn=wrapper))

        return scorer_specs

    def _convert_from_platform_result(
        self,
        platform_result: EvaluationResult,
    ) -> EvalResult:
        """Convert MLflow EvaluationResult to EvalResult.

        Pulls run-scoped traces via the session port (run_id non-None →
        wrapper picks return_type='list' internally and produces
        TraceRecord instances). The conversion code below speaks
        TraceRecord; the adapter no longer touches Trace objects directly.
        """
        if self._session_info is None:
            raise RuntimeError(
                "_convert_from_platform_result called before configure(); "
                "this is an internal error in MLflowPlatform."
            )

        records = self._get_session().search_traces(
            experiment_id=self._session_info.experiment_id,
            run_id=platform_result.run_id,
            filter=None,
        )

        eval_examples: list[EvalExample] = []
        for record in records:
            # The verdict cache is keyed on the example's inputs, not its
            # output: _extract_request_response reshapes record.request, so the
            # output does not round-trip as an exact value. The recording side
            # registered several renderings of the same inputs (see
            # _input_renderings); this is the one reachable here.
            record_key = verdict_key(
                self._extract_request_response(
                    str(record.request) if record.request else "",
                    str(record.response) if record.response else "",
                )[0]
            )

            scores: dict[str, Score] = {}
            for assessment in record.assessments:
                if not isinstance(assessment, Feedback):
                    continue
                scorer_name = assessment.name
                if assessment.feedback is None:
                    logger.warning(
                        "Skipping scorer %s: Feedback.feedback is None", scorer_name
                    )
                    continue
                score_value = assessment.feedback.value
                if score_value is None:
                    logger.warning(
                        "Skipping scorer %s: Feedback.feedback.value is None",
                        scorer_name,
                    )
                    continue

                threshold = self._scorer_thresholds.get(scorer_name, PASS_THRESHOLD)
                match score_value:
                    case bool():
                        float_value = 1.0 if score_value else 0.0
                        passed = score_value
                    case int() | float():
                        float_value = float(score_value)
                        passed = float_value >= threshold
                    case str():
                        if score_value.lower() == "yes":
                            float_value = 1.0
                            passed = True
                        elif score_value.lower() == "no":
                            float_value = 0.0
                            passed = False
                        else:
                            try:
                                float_value = float(score_value)
                                if float_value != float_value:  # NaN check
                                    logger.warning(
                                        "Skipping scorer %s: NaN score value",
                                        scorer_name,
                                    )
                                    continue
                                passed = float_value >= threshold
                            except ValueError:
                                logger.warning(
                                    "Skipping scorer %s: non-numeric string value %r",
                                    scorer_name,
                                    score_value,
                                )
                                continue
                    case _:
                        logger.warning(
                            "Skipping scorer %s: unsupported value type %s",
                            scorer_name,
                            type(score_value).__name__,
                        )
                        continue

                # A scorer owns its verdict; prefer the one it returned over the
                # threshold comparison above, which is now the fallback for
                # platform-native scores and historical traces.
                preserved = self._verdicts.lookup(scorer_name, record_key, float_value)
                if preserved is not None:
                    passed = preserved

                scores[scorer_name] = Score(
                    name=scorer_name,
                    value=float_value,
                    passed=passed,
                )

            expected_values: dict[str, Any] = {}
            for assessment in record.assessments:
                if not isinstance(assessment, Expectation):
                    continue
                if assessment.expectation is None:
                    logger.warning(
                        "Skipping expectation %s: Expectation.expectation is None",
                        assessment.name,
                    )
                    continue
                expected_values[assessment.name] = assessment.expectation.value

            request, response = self._extract_request_response(
                str(record.request) if record.request else "",
                str(record.response) if record.response else "",
            )
            trajectory = self._trajectory_from_request_response(
                record.request, record.response
            )
            tool_calls = _derive_tool_calls(trajectory) if trajectory else None

            eval_example = EvalExample(
                input=request,
                output=response,
                expected=str(expected_values.get("expected", "")),
                scores=scores,
                metadata=record.trace_metadata,
                duration=(record.execution_duration_ms or 0) / 1000.0,
                error=None,
                trajectory=trajectory,
                tool_calls=tool_calls,
            )
            eval_examples.append(eval_example)

        aggregate_scores, pass_rates = compute_aggregate_scores(eval_examples)
        (
            total_examples,
            successful_examples,
            failed_examples,
        ) = compute_summary_counts(eval_examples)

        return EvalResult(
            experiment_id=platform_result.run_id,
            experiment_url=self._get_experiment_url(),
            platform="mlflow",
            scores=aggregate_scores,
            pass_rates=pass_rates,
            examples=eval_examples,
            summary={
                "total_examples": total_examples,
                "successful_examples": successful_examples,
                "failed_examples": failed_examples,
            },
        )

    def _convert_to_platform_dataset(
        self, task: Callable[[Any], Any] | None, dataset: Dataset
    ) -> list[dict[str, Any]]:
        """Convert Agent Evals Dataset to MLflow dataset format.

        When a task function is provided, it must have exactly one parameter
        (excluding 'self' for methods). This parameter will be used to create
        the MLflow input dictionary: {param_name: example.input}.

        Args:
            task: Task function to evaluate. If provided, must have exactly one
                parameter (excluding 'self').
            dataset: Agent Evals Dataset

        Returns:
            List of dicts representing MLflow dataset

        Raises:
            ValueError: If task function has incorrect number of parameters
        """
        mlflow_dataset = []

        # Validate task signature if provided
        if task:
            task_sig = inspect.signature(task)
            # Filter out 'self' parameter for methods
            non_self_params = [name for name in task_sig.parameters if name != "self"]

            if len(non_self_params) != 1:
                raise ValueError(
                    f"Task function must have exactly one parameter "
                    f"(excluding 'self'), but found {len(non_self_params)} "
                    f"parameters: {non_self_params}. MLflow expects input to be "
                    f"a dictionary where the single parameter name becomes the "
                    f"key and example.input becomes the value."
                )

        for _i, example in enumerate(dataset):
            expected_value = (
                example.expected.expected if example.expected is not None else None
            )
            expected_context = (
                example.expected.context
                if isinstance(example.expected, ExpectedResult)
                else None
            )
            inputs = {}
            if task:
                task_sig = inspect.signature(task)
                # Get the single non-self parameter name
                param_name = next(
                    name for name in task_sig.parameters if name != "self"
                )
                inputs = {param_name: example.input}
            else:
                inputs = example.input

            expectations_dict: dict[str, Any] = {"expected": expected_value}
            if expected_context:
                expectations_dict["context"] = expected_context

            record = {
                "inputs": inputs,
                "outputs": example.output,
                "expectations": expectations_dict,
                "tags": example.metadata,
            }
            mlflow_dataset.append(record)
        return mlflow_dataset

    def _extract_request_response(
        self, str_request: str, str_response: str
    ) -> tuple[str, str]:
        """Extract request and response from MLflow Trace object.

        Empty inputs are returned as empty strings; an empty string is the
        caller's safe-default when ``trace.data.request`` / ``.response`` is
        ``None`` on the SDK side.

        Args:
            str_request: Serialized JSON request payload, or empty string.
            str_response: Serialized JSON response payload, or empty string.

        Returns:
            Tuple of (request, response).
        """
        if not str_request and not str_response:
            return "", ""

        try:
            request_obj = json.loads(str_request) if str_request else {}
            response_obj = json.loads(str_response) if str_response else {}
            only_input = request_obj["messages"][-1]["content"]
            if "structured_response" in response_obj:
                only_output = str(response_obj["structured_response"])
            else:
                only_output = response_obj["messages"][-1]["content"]
        except KeyError, json.JSONDecodeError, IndexError:
            only_input = str_request
            only_output = str_response

        return only_input, only_output

    @staticmethod
    def _trajectory_from_request_response(
        request: dict[str, Any] | str | None,
        response: dict[str, Any] | str | None,
    ) -> list[dict[str, Any]] | None:
        """Reconstruct trajectory from a TraceRecord's request/response.

        Accepts the typed union ``dict | str | None`` directly off the
        TraceRecord (no caller-side coercion). Concatenates messages
        recovered from both sides; returns ``None`` if neither side
        yields a usable message array.

        Two shapes are recognized, in priority order:

        1. ``{"output": ..., "context": {"outputs": [...]}}`` — the
           canonical agent-evals shape produced when the user task
           returns a ``TaskResult``. MLflow's auto-tracing stores the
           Pydantic-dumped value verbatim; trajectory lives at
           ``context["outputs"]``. This is the path real production
           traces take.
        2. ``{"messages": [...]}`` — a legacy / chat-completion-style
           envelope. Used by the test fake and by traces that come
           from chat-completion logging rather than agent-evals.

        Callers needing ``tool_calls`` derive them by passing the
        returned trajectory through ``_derive_tool_calls`` from
        ``agent_evals.core.types``.
        """

        def _messages(payload: dict[str, Any] | str | None) -> list[dict[str, Any]]:
            if payload is None:
                return []
            if isinstance(payload, str):
                if not payload:
                    return []
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError as e:
                    # A non-empty payload that fails to decode is almost
                    # certainly a real shape mismatch (MLflow logged
                    # something other than JSON, or the TraceRecord
                    # contract was violated upstream). DEBUG so it's
                    # discoverable when investigating "why is trajectory
                    # missing on these traces" without spamming WARN
                    # for routine non-trajectory traces.
                    logger.debug(
                        "Could not decode TraceRecord payload as JSON (%s); "
                        "trajectory will not be reconstructed from this side.",
                        e,
                    )
                    return []
            if not isinstance(payload, dict):
                return []

            # Canonical agent-evals shape (TaskResult model_dump).
            ctx = payload.get("context")
            if isinstance(ctx, dict):
                outputs = ctx.get("outputs")
                if isinstance(outputs, list):
                    return outputs

            # Legacy chat-completion envelope.
            msgs = payload.get("messages")
            return msgs if isinstance(msgs, list) else []

        trajectory = [*_messages(request), *_messages(response)]
        return trajectory if trajectory else None

    def _get_experiment_url(self) -> str:
        """Build the experiment URL from cached session info."""
        if not self.run_id or not self.experiment_id or self._session_info is None:
            return "mlflow://localhost:5000"

        tracking_uri = self._session_info.tracking_uri
        if tracking_uri.startswith("file://"):
            return f"file://{tracking_uri[7:]}/{self.experiment_id}/{self.run_id}"
        elif tracking_uri.startswith("http"):
            return (
                f"{tracking_uri}/#/experiments/{self.experiment_id}/runs/{self.run_id}"
            )
        else:
            return f"{tracking_uri}/experiments/{self.experiment_id}/runs/{self.run_id}"
