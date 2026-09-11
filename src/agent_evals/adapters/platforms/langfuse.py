# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""LangFuse platform adapter.

Implements Platform for the LangFuse experiment tracking platform.

This adapter integrates with LangFuse's experiment system using:
- langfuse.get_client() for client initialization
- client.run_experiment() for running evaluations
- client.api.trace.get() for trace export
- Evaluation objects for scoring results

The adapter automatically converts Agent Evals scorer functions to LangFuse
evaluator functions that return Evaluation objects.
"""

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable, Sequence
from typing import Any, ClassVar, Literal

from foundry_agent_core import mask_session_id
from pydantic import Field

from agent_evals.adapters.platforms.langfuse_client import (
    DatasetItemRecord,
    LangfuseClientPort,
    TraceRecord,
    _RealLangfuseClient,
)
from agent_evals.adapters.platforms.langfuse_filter import (
    matches_filter,
    parse_filter,
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
    task_result_to_example,
)

logger = logging.getLogger(__name__)

# Default threshold above which numeric scores are considered a "pass".
PASS_THRESHOLD = 0.5


def _mask_in_text(text: str, raw: str | None) -> str:
    """Replace verbatim occurrences of ``raw`` in ``text`` with its masked token.

    STIG V-222577 (CCI-001184) — masking the values *this* module formats into
    a message is not sufficient. When an identifier reaches the Langfuse SDK,
    the SDK's own error message can echo it back, and interpolating that
    message carries the raw value into a log record or an exception traceback
    without any line here having formatted it. This scrubs by exact value,
    which does not depend on how the SDK chose to render it.

    Best-effort by construction: an SDK that truncates, case-folds, or
    URL-encodes the value before echoing it defeats the substitution. That is
    why it layers on top of masking our own format arguments rather than
    replacing it.

    ``raw`` being ``None`` or blank is the common case (no identifier in
    scope), and returns ``text`` untouched rather than injecting a placeholder
    into an unrelated message.
    """
    if not raw or not raw.strip():
        return text
    return text.replace(raw, mask_session_id(raw))


def _scrub_exception_args(exc: BaseException, raw: str | None) -> None:
    """Mask ``raw`` inside ``exc.args``, in place.

    Needed because ``raise ... from e`` is preserved for debuggability, and
    Python's default traceback renders the *whole* cause chain. Masking only
    the message we construct would still disclose the raw identifier via the
    chained exception's own message — an unmasked identifier in a traceback is
    as much a disclosure as one in a log line.

    Rewriting ``args`` keeps the exception's type, its traceback, and its place
    in the chain, so ``__cause__`` remains useful; only the rendered strings
    change. Best-effort in the same sense as :func:`_mask_in_text`: an
    exception whose ``__str__`` is not derived from ``args`` is not reached,
    and a few built-in types reject ``args`` assignment outright.
    """
    if not raw or not raw.strip():
        return
    try:
        exc.args = tuple(
            _mask_in_text(arg, raw) if isinstance(arg, str) else arg for arg in exc.args
        )
    except Exception as scrub_error:
        # Deliberately swallowed: failing to scrub must not replace the real
        # error with a masking error. Logged at debug (without the identifier)
        # so the gap is observable rather than silent.
        logger.debug(
            "Could not scrub args on %s: %s",
            type(exc).__name__,
            type(scrub_error).__name__,
        )


def _render_payload(value: Any) -> str:
    """Render an arbitrary trace payload as the string scorers read.

    ``TaskResult.output`` and ``ExpectedResult.expected`` are strings,
    but a trace's payload is arbitrary JSON. Mappings and sequences are
    serialized as JSON rather than ``str()``-ed, because ``str()`` of a
    dict emits Python repr — single quotes, ``None`` instead of ``null``
    — which is neither valid JSON nor what a scorer comparing against a
    recorded answer would expect.

    Returns ``""`` for ``None``: the absence of an *output* is not an
    error the way a missing input is, and an empty string is what the
    dataset path has always produced for it.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict | list | tuple):
        try:
            return json.dumps(value, default=str)
        except TypeError, ValueError:
            # Unserializable payload (a set, a custom object nested past
            # `default=str`'s reach). Falling back to repr keeps the
            # example rather than dropping a record over formatting.
            return str(value)
    return str(value)


def _extract_messages(payload: Any) -> list[dict[str, Any]] | None:
    """Recover an OpenAI-shaped message list from a trace payload, if present.

    Populates ``TaskResult.context["outputs"]`` so trajectory scorers
    find a trajectory where forward-path scorers always read it. Returns
    ``None`` when the payload carries no message sequence — inventing one
    from a plain string payload would fabricate a trajectory.
    """
    if isinstance(payload, dict):
        messages = payload.get("messages")
        if isinstance(messages, list) and all(isinstance(m, dict) for m in messages):
            return messages
        return None
    if payload and isinstance(payload, list):
        if all(isinstance(m, dict) and "role" in m for m in payload):
            return payload
        return None
    return None


class _MissingItem:
    """Per-invocation sentinel when the SDK calls a wrapped task without an item.

    Each instance gets a unique ``id()`` so concurrent contract violations
    don't share a sidecar key.
    """


# Import guard for optional dependency
try:
    from langfuse import Evaluation, get_client
    from langfuse.experiment import ExperimentResult
except ImportError as e:
    raise ImportError(
        "LangFuse adapter requires the langfuse package. "
        "Install it with: uv add 'agent-evals[langfuse]' "
        "or: uv add langfuse"
    ) from e


class LangfuseConfig(PlatformConfig):
    """Config for the Langfuse platform adapter.

    The adapter maps the canonical ``experiment`` to the Langfuse SDK's
    ``name`` parameter.
    """

    name: Literal["langfuse"] = "langfuse"
    experiment: str = Field(
        description="Run name in Langfuse. Mapped to the SDK's `name` by the adapter."
    )
    description: str = Field(
        default="", description="Human-readable description of the experiment run."
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Freeform metadata attached to the experiment run.",
    )


@platform_registry.register("langfuse")  # ty: ignore[invalid-argument-type]
class LangFusePlatform:
    """LangFuse platform adapter for evaluation results.

    This adapter integrates with LangFuse's experiment system and provides
    both synchronous and asynchronous evaluation methods. It automatically
    converts Agent Evals scorer functions to LangFuse evaluator functions.
    """

    config_class: ClassVar[type[PlatformConfig]] = LangfuseConfig

    def __init__(self, *, client: LangfuseClientPort | None = None) -> None:
        """Initialize the LangFuse adapter.

        Args:
            client: Optional ``LangfuseClientPort`` implementation. Tests
                pass a ``FakeLangfuseClient``; production callers leave
                this unset and the adapter lazily wraps
                ``langfuse.get_client()`` in ``_RealLangfuseClient`` on
                first use. Keyword-only to keep the existing
                ``LangFusePlatform()`` positional invocation
                byte-identical.
        """
        self._scorer_thresholds: dict[str, float] = {}
        self._verdicts = VerdictCache()
        self._client: LangfuseClientPort | None = client
        log_transport_posture(logger, "langfuse")

    def _get_client(self) -> LangfuseClientPort:
        """Return the cached client port, lazy-wrapping the SDK if needed.

        Falls back to ``_RealLangfuseClient(langfuse.get_client())`` on
        first call when no client was injected at construction time.

        This is the single seam through which the adapter obtains its
        SDK collaborator. Tests substitute a fake by passing ``client=``
        to ``__init__``; production callers go through the default
        ``get_client()`` path with the real wrapper.

        SDK initialization failures (bad credentials, missing env vars,
        unreachable host) are wrapped here so that ``aevaluate`` and
        ``pull_traces`` both surface the same actionable ``RuntimeError``
        message — without this wrap, ``aevaluate`` would raise the raw
        SDK exception while ``pull_traces``' outer ``except`` would
        re-wrap it as a "Failed to export traces" error, giving
        operators inconsistent diagnoses for the same root cause.
        """
        if self._client is None:
            # Validated here, and deliberately BEFORE the try below. The SDK
            # reads LANGFUSE_HOST itself inside get_client(), so this is the
            # only point where the value is observable to this adapter before a
            # connection is possible. Placement is load-bearing: the except
            # clause re-raises everything as a credentials RuntimeError, so a
            # ValueError raised inside would be swallowed and misreported,
            # sending an operator hunting for a bad API key. Do not move this
            # into the try.
            #
            # Inside the `self._client is None` branch: an injected client's
            # host was resolved by the caller, not by this library, so it is
            # not this library's to police.
            #
            # An unset or empty host is not an error — the SDK supplies its own
            # encrypted default.
            host = os.getenv("LANGFUSE_HOST", "")
            if host:
                validate_secure_transport("LANGFUSE_HOST", host)

            try:
                sdk_client = get_client()
            except Exception as e:
                logger.error(
                    "Failed to initialize Langfuse SDK client via get_client(): %s",
                    e,
                )
                raise RuntimeError(
                    "LangFuse platform could not obtain an SDK client. "
                    "Check LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and "
                    "LANGFUSE_HOST environment variables."
                ) from e
            self._client = _RealLangfuseClient(sdk_client)
        return self._client

    @staticmethod
    def _to_sdk_kwargs(cfg: LangfuseConfig) -> dict:
        """Map a LangfuseConfig to the kwargs Langfuse SDK's run_experiment expects."""
        return {
            "name": cfg.experiment,
            "description": cfg.description,
            "metadata": cfg.metadata,
        }

    def evaluate(
        self,
        task: Callable[[Any], Any],
        dataset: Dataset,
        evaluators: Sequence[Scorer],
        platform: PlatformConfig | None = None,
        config: EvalConfig | None = None,  # noqa: ARG002 - Langfuse SDK manages concurrency
    ) -> EvalResult:
        """Run a synchronous evaluation.

        Args:
            task: The task function to evaluate
            dataset: The dataset to evaluate on
            evaluators: The evaluators to use for scoring
            platform: Typed platform configuration (LangfuseConfig).
            config: Accepted for Protocol conformance; Langfuse
                defers to its SDK for concurrency handling.

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
        config: EvalConfig | None = None,  # noqa: ARG002 - Langfuse SDK manages concurrency
    ) -> EvalResult:
        """Run an asynchronous evaluation.

        Args:
            task: The task function to evaluate (optional for historical data)
            dataset: The dataset to evaluate on
            evaluators: The evaluators to use for scoring
            platform: Typed platform configuration (LangfuseConfig).
            config: Accepted for Protocol conformance; Langfuse SDK manages concurrency.

        Returns:
            EvalResult: The converted evaluation result

        Raises:
            ValueError: If platform config is missing or wrong type
        """
        cfg = platform if isinstance(platform, LangfuseConfig) else None
        if cfg is None:
            raise ValueError(
                "LangFuse adapter requires a LangfuseConfig platform configuration. "
                "Pass platform=LangfuseConfig(experiment='my-experiment')."
            )

        sdk_kwargs = self._to_sdk_kwargs(cfg)

        # A caller may reuse one adapter instance across runs; a verdict from
        # an earlier run must never be reported for this run's examples, and
        # two concurrent runs on one instance would fight over the cache.
        self._verdicts.begin_run(type(self).__name__)
        try:
            return await self._aevaluate_inner(task, dataset, evaluators, sdk_kwargs)
        finally:
            self._verdicts.end_run()

    async def _aevaluate_inner(
        self,
        task: Callable[[Any], Any] | None,
        dataset: Dataset,
        evaluators: Sequence[Scorer],
        sdk_kwargs: dict[str, Any],
    ) -> EvalResult:
        """Run the evaluation body, with the verdict cache already open.

        Split out of :meth:`aevaluate` so the cache's run lifetime is a single
        ``try``/``finally`` there rather than wrapping this whole body inline.
        """
        langfuse = self._get_client()
        if not langfuse.auth_check():
            raise RuntimeError("LangFuse client authentication failed.")

        if not task:

            def lf_passthrough_task(input_value):
                return input_value

            task = lf_passthrough_task
            langfuse_dataset = self._convert_to_langfuse_historical_dataset(dataset)
        else:
            langfuse_dataset = self._convert_to_platform_dataset(dataset)
        langfuse_evaluators = self._convert_to_platform_scorer(evaluators)

        # The SDK filters raised tasks out of item_results and never exposes
        # per-item duration; capture both at the boundary. Keys are id() of
        # the langfuse_dataset entries, which stay alive until reconciliation.
        item_errors: dict[int, str] = {}
        item_durations: dict[int, float] = {}
        task = self._make_safe_task(task, item_errors, item_durations)

        experiment_name = sdk_kwargs["name"]
        experiment_description = sdk_kwargs.get("description", "")
        metadata = sdk_kwargs.get("metadata", {})

        logger.info(
            "Starting LangFuse experiment: name=%s, examples=%d",
            experiment_name,
            len(langfuse_dataset),
        )

        langfuse_result = langfuse.run_experiment(
            name=experiment_name,
            description=experiment_description,
            data=langfuse_dataset,
            task=task,
            evaluators=langfuse_evaluators,
            metadata=metadata,
        )

        return self._convert_from_platform_result(
            langfuse_result,
            dataset,
            item_errors=item_errors,
            item_durations=item_durations,
            langfuse_dataset=langfuse_dataset,
        )

    def pull_traces(
        self,
        config: dict[str, Any] | None = None,
        filter: str | None = None,
    ) -> list[ExampleData]:
        """Export platform history and convert it into ExampleData records.

        Two sources, selected explicitly by ``config["source"]``:

        - ``"dataset"`` (the default): resolves the items of a named
          Langfuse dataset, reading each item's traced output. Curated
          records, each typically carrying an ``expected_output``.
        - ``"traces"``: lists raw execution traces, narrowed by session,
          user, and count. Production traffic, carrying no answer key.

        The source is never inferred. ``session_id`` was an accepted-and-
        ignored key before trace selection existed, so a caller may be
        passing one today and receiving dataset results; inferring the
        trace path from its presence would silently reroute them to an
        entirely different data source on upgrade.

        **Where each parameter is applied.** ``session_id``, ``user_id``,
        and ``limit`` are expressed to Langfuse as query parameters — the
        platform narrows before anything crosses the network. ``filter``
        is applied *here*, by this library, because Langfuse's own
        ``filter`` parameter takes precedence over its ``sessionId`` /
        ``userId`` parameters and forwarding a clause through it would
        silently discard the selection above. This diverges from the
        MLflow and Braintrust adapters, which filter server-side; see
        ``docs/platforms.md`` for the per-adapter table.

        **``limit`` is a total budget, not a page size.** It returns the
        *most recent* N matching records: every listing request is
        ordered ``timestamp.desc``, which also gives positional
        pagination the defined order it needs so a record cannot fall
        between two pages. Note this is the opposite of Langfuse's own
        ``limit``, which is per-page. Ordering is single-field, so
        records sharing an identical timestamp — plausible under batch
        ingestion — can reorder between requests; a caller who needs
        exactness should narrow the query rather than rely on a boundary.

        Args:
            config: Platform-specific configuration for trace export.
                ``source: "dataset"`` (default) requires:
                - dataset: Name of the Langfuse dataset to export
                and reads, but does not select on:
                - session_id: Used only to scrub the identifier out of an
                  SDK error message this path may raise (V-222577). It
                  narrows nothing here — dataset membership is fixed by
                  the dataset. Pass ``source: "traces"`` to select by
                  session. Documented rather than dropped because a
                  caller passing it today is relying on the current
                  behaviour, and documented rather than left silent
                  because a key that is read and unexplained reads as a
                  filter that does not work.
                ``source: "traces"`` accepts:
                - session_id: Restrict to one session (platform-side)
                - user_id: Restrict to one user (platform-side)
                - limit: Total most-recent records to return (platform-side)
                - expected_metadata_key: Metadata key on each trace to
                  read an expected value from. Absent by default — raw
                  traces carry no answer key, and a key is never
                  inferred, because a metadata field that resembles one
                  may be an unrelated annotation.
            filter: Optional clause narrowing the exported records, as
                one or more ``<field> <op> <value>`` comparisons joined
                by ``AND`` (e.g. ``"metadata.environment = prod"``).
                Applied by this library, to the ``"traces"`` source only.
        Returns:
            List of ExampleData derived from the platform's execution history.
        Raises:
            ValueError: If required authentication or configuration is
                missing, if ``filter`` cannot be interpreted, or if
                ``filter`` is supplied with the dataset source.
            RuntimeError: If trace export or conversion fails.
        Example:
            >>> dataset = adapter.pull_traces(
            ...     config={"dataset": "my-dataset"}
            ... )
            >>> recent = adapter.pull_traces(
            ...     config={"source": "traces", "session_id": "abc", "limit": 20}
            ... )
        """
        config = config or {}
        source = config.get("source", "dataset")
        if source not in ("dataset", "traces"):
            raise ValueError(
                f"LangFuse adapter received unknown source {source!r}. "
                "Supported: 'dataset' (default) or 'traces'."
            )

        if source == "traces":
            return self._pull_traces_from_listing(config, filter)

        if "dataset" not in config:
            raise ValueError(
                "LangFuse adapter requires config with 'dataset' key. "
                "Example: config={'dataset': 'my-dataset'}"
            )
        if filter is not None:
            # Not silently dropped. A clause the adapter cannot honour on
            # this path is the inert-parameter defect: the caller gets a
            # plausible result set that does not answer their question,
            # with nothing to indicate the difference.
            raise ValueError(
                "LangFuse adapter applies 'filter' to the trace-listing "
                "source only. Pass config={'source': 'traces', ...} to "
                "filter execution traces, or drop 'filter' to export a "
                "dataset unchanged."
            )

        langfuse = self._get_client()

        # Read for masking only on this path: the dataset source selects by
        # dataset name, and a caller who passed 'session_id' before trace
        # selection existed must keep receiving what they received. The raw
        # value still reaches any query that uses it — masking is an
        # output-boundary control.
        raw_session_id = config.get("session_id")
        session_id = raw_session_id if isinstance(raw_session_id, str) else None

        try:
            items = langfuse.get_dataset_items(config["dataset"])
            return self._convert_traces(items, langfuse)

        except Exception as e:
            # V-222577: an SDK error can echo a session identifier it was
            # queried with. Two scrubs, because a traceback renders both links
            # of the chain: the cause's own args (kept as __cause__ by
            # `from e`), and the message constructed here.
            _scrub_exception_args(e, session_id)
            detail = _mask_in_text(str(e), session_id)
            raise RuntimeError(
                f"Failed to export traces from LangFuse: {detail}"
            ) from e

    def _pull_traces_from_listing(
        self,
        config: dict[str, Any],
        filter: str | None,
    ) -> list[ExampleData]:
        """Export raw execution traces, narrowed by the caller's selection.

        Args:
            config: The ``source: "traces"`` configuration. See
                ``pull_traces`` for the accepted keys.
            filter: Optional library-form clause, applied here.

        Returns:
            One ExampleData per matching trace whose input could be
            resolved.
        """
        if "dataset" in config:
            # Ambiguous rather than merely redundant: the two sources
            # return different records from different endpoints, and
            # silently preferring one would give the caller a result set
            # they cannot attribute to what they asked for.
            raise ValueError(
                "LangFuse adapter received both 'source': 'traces' and a "
                "'dataset' key. These select different sources; pass one. "
                "Drop 'dataset' to list execution traces, or drop 'source' "
                "to export the dataset."
            )

        raw_session_id = config.get("session_id")
        session_id = raw_session_id if isinstance(raw_session_id, str) else None
        raw_user_id = config.get("user_id")
        user_id = raw_user_id if isinstance(raw_user_id, str) else None
        limit = self._read_trace_limit(config.get("limit"))
        expected_key = config.get("expected_metadata_key")
        if expected_key is not None and not isinstance(expected_key, str):
            raise ValueError(
                f"LangFuse adapter requires 'expected_metadata_key' to be a "
                f"string naming a metadata field; got "
                f"{type(expected_key).__name__}."
            )

        # Parsed before the query, and deliberately outside the try below:
        # an uninterpretable clause is the caller's error, reported as
        # such, not re-wrapped as an export failure. Failing before the
        # network call also means a typo costs nothing.
        clauses = parse_filter(filter) if filter is not None else None

        langfuse = self._get_client()

        try:
            records = langfuse.list_traces(
                session_id=session_id,
                user_id=user_id,
                limit=limit,
            )
            if clauses is not None:
                records = [r for r in records if matches_filter(r, clauses)]
            return self._convert_trace_records(records, expected_key)

        except Exception as e:
            # V-222577: an SDK error can echo the session identifier it was
            # queried with. Two scrubs, because a traceback renders both
            # links of the chain — the cause's own args (kept as __cause__
            # by `from e`) and the message constructed here.
            _scrub_exception_args(e, session_id)
            detail = _mask_in_text(str(e), session_id)
            raise RuntimeError(
                f"Failed to export traces from LangFuse: {detail}"
            ) from e

    @staticmethod
    def _read_trace_limit(raw: Any) -> int | None:
        """Validate the record budget, rejecting values that cannot bound a listing.

        A non-integer or negative ``limit`` is rejected rather than
        coerced: silently treating ``limit="20"`` or ``limit=-1`` as
        "no limit" would export an entire project when the caller asked
        for a bounded slice.
        """
        if raw is None:
            return None
        # bool is an int subclass, and `limit=True` is a mistake, not a budget.
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise ValueError(
                f"LangFuse adapter requires 'limit' to be an integer number of "
                f"records; got {type(raw).__name__}."
            )
        if raw < 0:
            raise ValueError(
                f"LangFuse adapter requires 'limit' to be non-negative; got {raw}."
            )
        return raw

    def _convert_trace_records(
        self,
        records: list[TraceRecord],
        expected_metadata_key: str | None = None,
    ) -> list[ExampleData]:
        """Convert listed traces into ExampleData records.

        A trace's payload is arbitrary JSON with no agreed shape, unlike a
        dataset item's. ``input`` is carried through unchanged — the field
        is typed ``Any`` precisely so a structured payload survives to the
        task that re-runs it, and stringifying a dict here would invent a
        shape the caller then has to parse back out.

        Args:
            records: Traces returned by ``LangfuseClientPort.list_traces``.
            expected_metadata_key: Optional metadata key to read an
                expected value from. When absent, examples carry no
                expected value: a raw trace has no answer key, and
                substituting an empty one is how scoring against nothing
                begins.

        Returns:
            One ExampleData per record whose input could be resolved.
        """
        historical_data: list[ExampleData] = []
        for record in records:
            if record.input is None:
                # Nothing to re-run. Skipped with a warning rather than
                # emitted as an empty example, and rather than aborting
                # the export — one unresolvable record must not cost the
                # caller the other 200.
                logger.warning(
                    "Skipping trace %s: no input payload to build an example from.",
                    mask_session_id(record.id),
                )
                continue

            metadata = dict(record.metadata)
            if record.session_id:
                # Raw, and under exactly this key spelling. `redact_session_ids`
                # (persistence) and the benchmark report's metadata renderer
                # both recognise `session_id` and nothing else, so `sessionId`
                # would reach results.jsonl verbatim while every masking test
                # still passed. Storing a pre-masked token here instead would
                # be double-masked at persistence — mask(mask(raw)) in the
                # file against mask(raw) in the logs — breaking the
                # cross-emission correlation the masking control guarantees.
                metadata["session_id"] = record.session_id

            expected_result = self._resolve_trace_expected(
                record, expected_metadata_key
            )

            messages = _extract_messages(record.output)
            if messages is None:
                messages = _extract_messages(record.input)
            rendered_output = _render_payload(record.output)
            if messages:
                output_result = TaskResult(
                    output=rendered_output, context={"outputs": messages}
                )
            else:
                output_result = TaskResult(output=rendered_output)

            historical_data.append(
                ExampleData(
                    input=record.input,
                    expected=expected_result,
                    output=output_result,
                    metadata=metadata,
                )
            )

        return historical_data

    @staticmethod
    def _resolve_trace_expected(
        record: TraceRecord,
        expected_metadata_key: str | None,
    ) -> ExpectedResult | None:
        """Read an expected value from the caller-nominated metadata key, if any.

        Returns ``None`` when no key was nominated, when the record does
        not carry it, or when its value is empty — absence yields absence.
        Tolerating a missing key matters: a batch where only some records
        were annotated exports in full rather than failing.
        """
        if not expected_metadata_key:
            return None
        raw = record.metadata.get(expected_metadata_key)
        if raw is None:
            return None
        rendered = _render_payload(raw)
        # Whitespace-only counts as empty. A blank annotation is no
        # annotation, and passing `"   "` through would be the
        # empty-expected defect wearing a space: a real comparison against
        # nothing, reported as a low score rather than as a missing
        # reference.
        return ExpectedResult(expected=rendered) if rendered.strip() else None

    def _convert_traces(
        self,
        items: list[DatasetItemRecord],
        langfuse: LangfuseClientPort,
    ) -> list[ExampleData]:
        """Convert dataset items + their traced outputs into ExampleData records.

        Resolves each item's ``source_trace_id`` via ``get_trace_output``,
        which the real wrapper implements as ``api.trace.get(id)``. This
        path fetches by id rather than by listing because it already knows
        the id it wants — a membership check against a listing would be
        strictly more work for the same answer.

        The port's ``list_traces`` exists for the other source, where no
        ids are known in advance, and it absorbs pagination internally: a
        single ``api.trace.list()`` call returns only one page (default
        50), so the earlier absence of a list-shaped method was a way of
        avoiding that trap rather than a reason it could not be solved.

        Args:
            items: Dataset items returned by ``LangfuseClientPort.get_dataset_items``.
            langfuse: The port used to resolve trace outputs.

        Returns:
            List of ExampleData derived from the platform's execution history.
        """
        historical_data: list[ExampleData] = []
        for item in items:
            source_trace_id = item.source_trace_id or ""
            try:
                output = langfuse.get_trace_output(source_trace_id)
            except Exception as e:
                # V-222577: the identifier is masked in both places it can
                # reach this record — our own format argument, and the SDK
                # error text, which quotes the ID it failed to find.
                logger.warning(
                    "Skipping dataset item: trace %s could not be fetched (%s: %s)",
                    mask_session_id(source_trace_id),
                    type(e).__name__,
                    _mask_in_text(str(e), source_trace_id),
                )
                continue

            try:
                messages = langfuse.get_trace_messages(source_trace_id)
            except Exception as e:
                # Logging the exception type makes wrapper bugs (TypeError,
                # AttributeError) visibly distinguishable from network /
                # SDK-API errors at log-grep time. The identifier itself is
                # masked per V-222577, in the format argument and in the SDK
                # error text alike.
                logger.warning(
                    "Trace %s output retrieved but messages unavailable "
                    "(%s: %s); example will still be emitted with no trajectory.",
                    mask_session_id(source_trace_id),
                    type(e).__name__,
                    _mask_in_text(str(e), source_trace_id),
                )
                messages = None

            input_data = item.input or ""
            output_data = output or ""
            metadata = item.metadata or {}
            expected = item.expected_output or ""
            expected_result = (
                ExpectedResult(expected=str(expected)) if expected else None
            )

            if isinstance(messages, list) and messages:
                output_result = TaskResult(
                    output=str(output_data), context={"outputs": messages}
                )
            else:
                output_result = TaskResult(output=str(output_data))

            historical_data.append(
                ExampleData(
                    input=input_data,
                    expected=expected_result,
                    output=output_result,
                    metadata=metadata,
                )
            )

        return historical_data

    def _make_safe_task(
        self,
        user_task: Callable[..., Any],
        errors: dict[int, str],
        durations: dict[int, float],
    ) -> Callable[..., Any]:
        """Wrap a user task so exceptions and durations are captured per-item.

        Bridges the langfuse SDK's ``task(item={"input": ..., ...})`` calling
        shape to the agent-evals contract ``task(input_value)``. The wrapper
        unwraps ``item["input"]`` before invoking the user task; the user
        never sees the SDK dict. See ``Platform`` Protocol docstring
        for the cross-adapter contract.

        On exception: record ``str(exc)`` in ``errors[id(item)]`` and return a
        sentinel ``TaskResult`` so the SDK keeps the item in ``item_results``.
        Always record wall time in ``durations[id(item)]``. Sidecars are keyed
        by ``id(item)`` — the SDK dict in the normal path, or a fresh
        ``_MissingItem`` sentinel on contract violations. Either way each
        case's key is unique even when two cases share an interned input
        value; keying by ``id(input_value)`` would collide on duplicates,
        regressing per-item attribution.
        """

        def _resolve_item(args, kwargs) -> Any:
            if "item" in kwargs:
                return kwargs["item"]
            if args:
                return args[0]
            # SDK contract violation: log once so it's visible, then use a
            # fresh sentinel as the sidecar key so concurrent None
            # invocations don't overwrite each other.
            logger.warning("safe_task wrapper invoked without an item")
            return _MissingItem()

        def _unwrap_input(item: Any) -> Any:
            """Extract the bare ``input`` value from the SDK dict.

            Non-dict items (e.g., ``_MissingItem`` sentinel from SDK
            contract violations) pass through unchanged so the user task
            still receives a value and the wrapper's exception path
            captures whatever the user code does with it.
            """
            if isinstance(item, dict):
                return item.get("input")
            return item

        async def safe_task_async(*args, **kwargs):
            item = _resolve_item(args, kwargs)
            key = id(item)
            start = time.perf_counter()
            try:
                result = user_task(_unwrap_input(item))
                # User tasks decorated in odd ways may pass `iscoroutinefunction`
                # while returning a non-coroutine; defend symmetrically with the
                # sync path's runtime check.
                if asyncio.iscoroutine(result):
                    result = await result
                return result
            except Exception as e:
                errors[key] = self._safe_str(e)
                return TaskResult(output=f"Error: {errors[key]}")
            finally:
                durations[key] = time.perf_counter() - start

        def safe_task_sync(*args, **kwargs):
            item = _resolve_item(args, kwargs)
            key = id(item)
            start = time.perf_counter()
            try:
                result = user_task(_unwrap_input(item))
            except Exception as e:
                errors[key] = self._safe_str(e)
                durations[key] = time.perf_counter() - start
                return TaskResult(output=f"Error: {errors[key]}")

            # Some user wrappers (classes with `async __call__`, custom
            # decorators) return a coroutine without being detectable as
            # async-functions; await the result directly so we don't hand
            # an unawaited coroutine back to the SDK.
            if asyncio.iscoroutine(result):

                async def _await_existing():
                    try:
                        return await result
                    except Exception as e:
                        errors[key] = self._safe_str(e)
                        return TaskResult(output=f"Error: {errors[key]}")
                    finally:
                        durations[key] = time.perf_counter() - start

                return _await_existing()

            durations[key] = time.perf_counter() - start
            return result

        if asyncio.iscoroutinefunction(user_task):
            return safe_task_async
        return safe_task_sync

    @staticmethod
    def _safe_str(exc: BaseException) -> str:
        """Stringify ``exc``, falling back to its type name if ``__str__`` raises."""
        try:
            return str(exc)
        except Exception:
            return f"<{type(exc).__name__}: unrepr-able>"

    def _infer_scorer_name(self, evaluator: Callable) -> str:
        """Infer a human-friendly scorer name from a callable.

        Delegates to the shared derivation in ``platforms.utils`` so every
        adapter that keys per-scorer state by name derives that key identically.
        """
        return infer_scorer_name(evaluator)

    def _convert_to_platform_scorer(
        self, evaluators: Sequence[Scorer]
    ) -> list[Callable]:
        """Convert Agent Evals scorer functions to LangFuse evaluator functions.

        LangFuse evaluators receive (input, output, expected_output, metadata) and
        return Evaluation objects. This method wraps Agent Evals scorers to match
        this interface.

        Args:
            evaluators: List of Agent Evals scorer functions (sync or async)

        Returns:
            List of LangFuse evaluator functions returning Evaluation objects
        """
        langfuse_evaluators = []

        for evaluator in evaluators:
            scorer_name = self._infer_scorer_name(evaluator)
            is_async = asyncio.iscoroutinefunction(evaluator)

            def create_langfuse_evaluator(original_scorer, name, async_scorer):
                if async_scorer:

                    async def langfuse_evaluator_wrapper(
                        *, input, output, expected_output=None, metadata=None, **kwargs
                    ):
                        # Convert to Agent Evals types, preserving context
                        if isinstance(expected_output, ExpectedResult):
                            expected_result = expected_output
                        elif expected_output:
                            expected_result = ExpectedResult(
                                expected=str(expected_output)
                            )
                        else:
                            expected_result = None

                        # Call original scorer
                        score = await original_scorer(
                            result=output, expected=expected_result
                        )

                        if not isinstance(score, Score):
                            raise TypeError(
                                f"Scorer must return Score, got {type(score).__name__}"
                            )

                        if name not in self._scorer_thresholds:
                            self._scorer_thresholds[name] = score.metadata.get(
                                "_threshold", PASS_THRESHOLD
                            )

                        # Keep the scorer's own verdict in-process. The comment
                        # below stays human-readable prose; it is not parsed.
                        # Keyed on the raw expected_output the item carried,
                        # not the converted ExpectedResult: the reconstruction
                        # path reads that same raw field back off the item.
                        self._verdicts.record(
                            name, verdict_key(output, expected_output), score
                        )

                        # Convert to LangFuse Evaluation
                        return Evaluation(
                            name=name,
                            value=score.value,
                            comment=f"Passed: {score.passed}",
                        )
                else:

                    def langfuse_evaluator_wrapper(
                        *, input, output, expected_output=None, metadata=None, **kwargs
                    ):
                        # Convert to Agent Evals types, preserving context
                        if isinstance(expected_output, ExpectedResult):
                            expected_result = expected_output
                        elif expected_output:
                            expected_result = ExpectedResult(
                                expected=str(expected_output)
                            )
                        else:
                            expected_result = None

                        # Call original scorer
                        score = original_scorer(result=output, expected=expected_result)

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

                        # Keep the scorer's own verdict in-process. The comment
                        # below stays human-readable prose; it is not parsed.
                        # Keyed on the raw expected_output the item carried,
                        # not the converted ExpectedResult: the reconstruction
                        # path reads that same raw field back off the item.
                        self._verdicts.record(
                            name, verdict_key(output, expected_output), score
                        )

                        # Convert to LangFuse Evaluation
                        return Evaluation(
                            name=name,
                            value=score.value,
                            comment=f"Passed: {score.passed}",
                        )

                return langfuse_evaluator_wrapper

            langfuse_evaluator = create_langfuse_evaluator(
                evaluator, scorer_name, is_async
            )
            langfuse_evaluators.append(langfuse_evaluator)

        return langfuse_evaluators

    def _convert_to_platform_dataset(self, dataset: Dataset) -> list[dict[str, Any]]:
        """Convert Agent Evals Dataset to LangFuse dataset format.

        Args:
            dataset: Agent Evals Dataset

        Returns:
            List of dicts representing LangFuse dataset format
        """
        langfuse_dataset = []

        for example in dataset:
            record = {
                "input": example.input,
            }

            if example.expected is not None:
                # Preserve full ExpectedResult so scorer wrappers receive
                # context (e.g. trajectory reference_outputs).
                record["expected_output"] = example.expected

            if example.metadata:
                record["metadata"] = example.metadata

            langfuse_dataset.append(record)

        return langfuse_dataset

    def _convert_to_langfuse_historical_dataset(
        self, dataset: Dataset
    ) -> list[dict[str, Any]]:
        """Convert Agent Evals Dataset to LangFuse dataset format for historical data.

        Args:
            dataset: Agent Evals Dataset

        Returns:
            List of dicts representing LangFuse dataset format for historical data
        """
        langfuse_historical_dataset = []

        for example in dataset:
            record: dict[str, Any] = {
                "input": example.output,
            }

            if example.expected is not None:
                # Preserve full ExpectedResult so scorer wrappers receive
                # context (e.g. trajectory reference_outputs).
                record["expected_output"] = example.expected

            if example.metadata:
                record["metadata"] = example.metadata

            langfuse_historical_dataset.append(record)

        return langfuse_historical_dataset

    def _convert_from_platform_result(
        self,
        platform_result: ExperimentResult,
        original_dataset: Dataset,
        item_errors: dict[int, str] | None = None,
        item_durations: dict[int, float] | None = None,
        langfuse_dataset: list[dict[str, Any]] | None = None,
    ) -> EvalResult:
        """Convert LangFuse experiment result to EvalResult.

        Args:
            platform_result: The LangFuse experiment result object
            original_dataset: The original dataset for preserving example data
            item_errors: id(item) -> error string for items whose task raised.
            item_durations: id(item) -> wall time in seconds.
            langfuse_dataset: The dict list passed to ``run_experiment``;
                used to reconcile dropped items by index.

        Returns:
            EvalResult: Converted evaluation result
        """
        eval_examples = self._build_examples_from_platform_result(
            platform_result,
            original_dataset,
            item_errors=item_errors or {},
            item_durations=item_durations or {},
            langfuse_dataset=langfuse_dataset or [],
        )

        aggregate_scores, pass_rates = compute_aggregate_scores(eval_examples)
        total, successful, failed = compute_summary_counts(eval_examples)

        return EvalResult(
            experiment_id=platform_result.experiment_id,
            experiment_url=platform_result.dataset_run_url or "",
            platform="langfuse",
            scores=aggregate_scores,
            pass_rates=pass_rates,
            examples=eval_examples,
            summary={
                "total_examples": total,
                "successful_examples": successful,
                "failed_examples": failed,
            },
        )

    def _build_examples_from_platform_result(
        self,
        platform_result: ExperimentResult,
        original_dataset: Dataset,
        item_errors: dict[int, str] | None = None,
        item_durations: dict[int, float] | None = None,
        langfuse_dataset: list[dict[str, Any]] | None = None,
    ) -> list[EvalExample]:
        """Build per-example results from LangFuse experiment result.

        ExperimentItemResult exposes only item/output/evaluations/trace_id/
        dataset_run_id. Input/expected/metadata come from result.item (see
        ``_read_item_field``); duration/error come from sidecars populated
        by ``_make_safe_task``. The trailing reconcile loop is defensive:
        the wrapper prevents the SDK from dropping items in production, but
        any path that bypasses it (e.g. a non-Exception failure) still
        produces a visible EvalExample.
        """
        examples: list[EvalExample] = []
        item_errors = item_errors or {}
        item_durations = item_durations or {}
        langfuse_dataset = langfuse_dataset or []

        item_id_to_index = {id(d): i for i, d in enumerate(langfuse_dataset)}
        seen_indices: set[int] = set()

        for result in platform_result.item_results:
            sdk_item = getattr(result, "item", None)
            sdk_item_id: int | None = id(sdk_item) if sdk_item is not None else None
            if sdk_item_id is not None and sdk_item_id in item_id_to_index:
                i = item_id_to_index[sdk_item_id]
            else:
                i = len(examples)
            seen_indices.add(i)

            original_example = (
                original_dataset[i] if i < len(original_dataset) else None
            )

            input_data = self._read_item_field(sdk_item, "input")
            if input_data is None and original_example:
                input_data = original_example.input

            raw_output = getattr(result, "output", "")
            original_task_result: TaskResult | None = None
            if isinstance(raw_output, TaskResult):
                original_task_result = raw_output
                output_data = raw_output.output
            else:
                output_data = str(raw_output) if raw_output else ""

            raw_expected = self._read_item_field(sdk_item, "expected_output")
            if isinstance(raw_expected, ExpectedResult):
                expected_data = raw_expected.expected
            elif raw_expected is not None:
                expected_data = str(raw_expected)
            elif original_example and original_example.expected:
                expected_data = original_example.expected.expected
            else:
                expected_data = None

            metadata = self._read_item_field(sdk_item, "metadata") or {}
            if not metadata and original_example:
                metadata = original_example.metadata or {}

            # The evaluator wrapper saw exactly these two objects — the raw task
            # output and the item's raw expected_output — so both sides derive
            # the same key. verdict_key unwraps the TaskResult / ExpectedResult
            # envelopes, so it does not matter which side still holds one.
            result_key = verdict_key(raw_output, raw_expected)

            scores_dict: dict[str, Score] = {}
            evaluations = getattr(result, "evaluations", []) or []
            for evaluation in evaluations:
                eval_name = getattr(evaluation, "name", "unknown")
                eval_value = getattr(evaluation, "value", 0.0)
                threshold = self._scorer_thresholds.get(eval_name, PASS_THRESHOLD)
                try:
                    float_value = float(eval_value)
                    passed = float_value >= threshold
                except ValueError, TypeError:
                    float_value = 0.0
                    passed = False

                # A scorer owns its verdict; prefer the one it returned over the
                # threshold comparison above, which is now the fallback for
                # platform-native scores and historical traces.
                preserved = self._verdicts.lookup(eval_name, result_key, float_value)
                if preserved is not None:
                    passed = preserved

                scores_dict[eval_name] = Score(
                    name=eval_name,
                    value=float_value,
                    passed=passed,
                )

            duration = (
                item_durations.get(sdk_item_id, 0.0) if sdk_item_id is not None else 0.0
            )
            error = item_errors.get(sdk_item_id) if sdk_item_id is not None else None

            if original_task_result is not None:
                examples.append(
                    task_result_to_example(
                        task_result=original_task_result,
                        input=input_data,
                        expected=ExpectedResult(expected=expected_data)
                        if expected_data is not None
                        else None,
                        scores=scores_dict,
                        metadata=metadata,
                        duration=float(duration),
                        error=error,
                    )
                )
            else:
                examples.append(
                    EvalExample(
                        input=input_data,
                        output=output_data,
                        expected=expected_data,
                        scores=scores_dict,
                        metadata=metadata,
                        duration=float(duration),
                        error=error,
                    )
                )

        # Defensive reconciliation: surface any item the SDK dropped, using
        # whatever the wrapper recorded in the sidecar.
        for i, lf_item in enumerate(langfuse_dataset):
            if i in seen_indices:
                continue
            sidecar_key = id(lf_item)
            error = item_errors.get(sidecar_key)
            duration = item_durations.get(sidecar_key, 0.0)
            if error is None:
                # Differentiate by whether the wrapper's `finally` ran:
                # presence of a duration means the wrapper executed but the
                # exception escaped `except Exception` (e.g. CancelledError);
                # absence means the SDK rejected the item before our wrapper
                # was invoked at all.
                if sidecar_key in item_durations:
                    error = "task interrupted (BaseException not caught)"
                else:
                    error = (
                        "task did not produce a result "
                        "(item dropped before wrapper ran)"
                    )

            original_example = (
                original_dataset[i] if i < len(original_dataset) else None
            )
            input_data = self._read_item_field(lf_item, "input")
            if input_data is None and original_example:
                input_data = original_example.input
            expected_data = None
            if original_example and original_example.expected:
                expected_data = original_example.expected.expected
            metadata = (original_example.metadata or {}) if original_example else {}

            examples.append(
                EvalExample(
                    input=input_data,
                    output="",
                    expected=expected_data,
                    scores={},
                    metadata=metadata,
                    duration=float(duration),
                    error=error,
                )
            )

        return examples

    @staticmethod
    def _read_item_field(item: Any, field: str) -> Any:
        """Read a field from a LocalExperimentItem dict or DatasetItem object."""
        if item is None:
            return None
        if isinstance(item, dict):
            return item.get(field)
        return getattr(item, field, None)
