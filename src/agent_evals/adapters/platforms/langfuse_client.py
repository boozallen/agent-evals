# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Langfuse SDK seam: port + production wrapper + domain record.

Module-local artifacts the LangFuse adapter (`langfuse.py`) uses to
talk to the langfuse SDK. Split into its own file as a progressive-
disclosure boundary: an engineer onboarding the adapter can read
`langfuse.py` to understand *what the adapter does*, and click into
this file only when they need to understand *what the adapter needs
from the SDK*.

This is an **adapter-internal seam**, not a hexagonal port. Hexagonal
ports (contracts that core depends on) live in `core/ports/`. Each
platform adapter has its own module-local seam — ports are NOT
unified across Langfuse, MLflow, and Braintrust. See
`docs/architecture.md` "Adapter Testability Pattern" for the
structural recipe future adapter PRs follow.

What lives here:

- ``LangfuseClientPort`` — Protocol describing the operations the
  adapter needs from the Langfuse SDK.
- ``_RealLangfuseClient`` — concrete implementation of the port that
  delegates to a real ``langfuse.Langfuse`` instance and absorbs the
  SDK's nested attribute access.
- ``DatasetItemRecord`` — flat dataclass mirroring the four SDK
  dataset-item fields the adapter actually reads.
- ``TraceRecord`` — flat dataclass mirroring the SDK trace-listing
  fields the adapter reads, for the trace-selection path.

What does NOT live here:

- The adapter itself (``LangFusePlatform``) — see ``langfuse.py``.
- The test fake (``FakeLangfuseClient``) — see
  ``tests/helpers/fake_langfuse.py``.
- The ``Evaluation`` and ``ExperimentResult`` SDK types — those are
  re-exported from the langfuse package and used as value types
  that pass through the adapter unwrapped.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

# Import guard for optional dependency. Mirrors the guard at the top
# of langfuse.py so importing this module produces the same actionable
# error message when the langfuse extra isn't installed.
try:
    from langfuse.experiment import ExperimentResult, LocalExperimentItem
except ImportError as e:
    raise ImportError(
        "LangFuse adapter requires the langfuse package. "
        "Install it with: uv add 'agent-evals[langfuse]' "
        "or: uv add langfuse"
    ) from e


@dataclass
class DatasetItemRecord:
    """Flat record extracted from a Langfuse dataset item.

    The Langfuse SDK's dataset items expose ``source_trace_id``, ``input``,
    ``expected_output``, and ``metadata`` as object attributes. This
    dataclass mirrors those fields as a flat record so the adapter (and
    test fakes) can construct and consume them without knowing about the
    SDK's underlying types.

    Used by ``LangfuseClientPort.get_dataset_items``.
    """

    source_trace_id: str | None
    input: Any
    expected_output: Any
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TraceRecord:
    """Flat record extracted from a Langfuse trace listing.

    Mirrors the fields of the SDK's ``TraceWithDetails`` that the adapter
    reads when building examples from raw execution history. Frozen
    because the listing is a read-only snapshot: nothing downstream of
    ``list_traces`` has a reason to mutate a record in place, and the
    page loop deduplicates by ``id``, which is only sound if the
    identity of a record cannot change after it is collected.

    Deliberately carries no expected-value field. A Langfuse *dataset
    item* has ``expected_output``; a raw *trace* does not, because
    nothing recorded an answer key at execution time. Adding a field
    here that is always ``None`` would invite the empty-substitute
    behaviour the trace-expected-value contract forbids — the absence
    is the information.

    ``input``, ``output``, and ``metadata`` are ``Any``: a trace's
    payload is arbitrary JSON with no agreed shape, unlike a dataset
    item's. Rendering them is the adapter's concern, not the seam's.

    Used by ``LangfuseClientPort.list_traces``.
    """

    id: str
    input: Any
    output: Any
    metadata: dict[str, Any] = field(default_factory=dict)
    session_id: str | None = None
    user_id: str | None = None


# Sort order requested on every trace-listing page.
#
# Not a caller-facing option. Two guarantees rest on it:
#
# 1. ``limit`` means "the most recent N matching records" rather than an
#    arbitrary slice, which is what makes a bounded export useful for
#    the "score how my agent handled recent traffic" case.
# 2. Pages are addressed positionally (``page=1,2,3...``), so an
#    undefined order lets a record fall *between* two pages and never
#    be returned. The SDK declares ``order_by`` as ``Optional[str] =
#    None`` and documents no fallback, so without this there is no
#    ordering guarantee to rest completeness on. Deduplication by
#    ``id`` catches the duplicate half of an unstable sort; nothing
#    catches the omission half.
#
# Accepted limitation: Langfuse's ``order_by`` is single-field, so there
# is no tiebreak. Records sharing a timestamp can reorder between
# requests, so one straddling a page boundary within a tie group can
# still be missed. ``id.asc`` would be fully deterministic but would
# make ``limit`` an arbitrary-though-stable slice, losing the recency
# semantics that are the point.
_TRACE_ORDER_BY = "timestamp.desc"

# Records requested per listing page when the caller sets no budget.
# Matches the SDK's own default page size.
DEFAULT_TRACE_PAGE_SIZE = 50


class LangfuseClientPort(Protocol):
    """Module-local port describing what ``LangFusePlatform`` needs from Langfuse.

    The port speaks the adapter's domain (returns scalars and dataclasses
    where possible). The concrete ``_RealLangfuseClient`` wrapper absorbs
    SDK quirks: nested attribute traversal (the SDK's
    ``client.api.trace.get(id).observations[-1].output`` chain becomes a
    single ``client.get_trace_output(id)`` call), object-vs-dataclass
    shape conversion, and any module-level state coupling the SDK exposes.

    Each platform adapter has its own port; ports are NOT unified across
    Langfuse, MLflow, and Braintrust. See ``docs/architecture.md``
    "Adapter Testability Pattern" for the structural recipe.
    """

    def auth_check(self) -> bool:
        """Return True if the Langfuse credentials are valid.

        Production: delegates to ``langfuse.Langfuse.auth_check()``.
        """
        ...

    def run_experiment(
        self,
        *,
        name: str,
        description: str,
        data: list[dict[str, Any]],
        task: Callable[..., Any],
        evaluators: list[Callable[..., Any]],
        metadata: dict[str, Any],
    ) -> ExperimentResult:
        """Run an evaluation experiment, returning the real SDK ``ExperimentResult``.

        ``ExperimentResult`` is intentionally *not* wrapped: it is a value
        type that the adapter's ``_build_examples_from_platform_result``
        consumes via defensive ``getattr`` access. SDK shape drift in this
        return type is supposed to surface in the live integration test
        (`tests/integration/test_langfuse_adapter_live.py`), not be hidden
        by the wrapper.
        """
        ...

    def get_dataset_items(self, name: str) -> list[DatasetItemRecord]:
        """Return the dataset's items as flat ``DatasetItemRecord`` instances.

        Production: looks up the dataset via ``client.get_dataset(name)``,
        iterates ``.items``, and extracts ``source_trace_id`` / ``input`` /
        ``expected_output`` / ``metadata`` from each.
        """
        ...

    def list_traces(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        limit: int | None = None,
        page_size: int = DEFAULT_TRACE_PAGE_SIZE,
    ) -> list[TraceRecord]:
        """Return the complete set of traces matching the given selection.

        Pagination is the implementation's concern, not the caller's. The
        returned list is the *whole* matching set, flattened across
        however many pages the platform needed — a caller cannot
        distinguish a project with 40 matching traces from one with
        4,000, so a first-page-only answer would report absence where
        the record exists.

        Args:
            session_id: Restrict to one session. Expressed to the
                platform as a query parameter, not applied afterwards.
            user_id: Restrict to one user. Also platform-side.
            limit: Total record budget across all pages — NOT a page
                size. This is the opposite of the SDK's own ``limit``,
                which is per-page, so the distinction is load-bearing:
                ``limit=10`` returns the 10 most recent matching records
                and issues one request, rather than fetching a full page
                and discarding most of it.
            page_size: Records requested per page. An efficiency knob
                only; it cannot change which records are returned.

        Returns:
            Matching traces as flat ``TraceRecord`` instances, most
            recent first, each appearing exactly once. An empty matching
            set returns ``[]`` rather than raising.

        Production: loops ``client.api.trace.list(...)`` over pages,
        ordering every request explicitly (see ``_TRACE_ORDER_BY``).
        """
        ...

    def get_trace_output(self, trace_id: str) -> Any:
        """Return the last-observation output from the named trace.

        Production: calls ``client.api.trace.get(trace_id)`` and reads
        ``trace.observations[-1].output`` (or ``None`` if no observations).
        Raises if the trace cannot be fetched (e.g., 404, retention
        expired, cross-project ID).
        """
        ...

    def get_trace_messages(self, trace_id: str) -> list[dict[str, Any]] | None:
        """Return the OpenAI-format message sequence from the named trace, or None.

        Used by the reverse path (``_convert_traces``) to populate
        ``TaskResult.context["outputs"]`` so re-evaluation finds
        trajectory where forward-path scorers always read it.

        Production: calls ``client.api.trace.get(trace_id)`` and walks
        observations whose ``input``/``output`` carry message arrays
        (langfuse logs LLM observations with ``input.messages`` /
        ``output.messages``). Returns None when no messages can be
        recovered.
        """
        ...


class _RealLangfuseClient(LangfuseClientPort):
    """Production wrapper around the real Langfuse SDK client.

    Implements ``LangfuseClientPort`` by delegating to a real
    ``langfuse.Langfuse`` instance and absorbing the SDK's nested attribute
    access. This is the only place in the adapter family that knows the
    SDK's wire format; the rest of ``LangFusePlatform`` speaks the port.

    Explicitly inherits from ``LangfuseClientPort`` so a method-signature
    drift fails type-check at this class instead of at every call site.
    """

    def __init__(self, sdk_client: Any) -> None:
        # Typed as Any so unit tests can pass duck-typed stubs (e.g.,
        # SimpleNamespace) without ceremonial casts. Production callers
        # always pass langfuse.get_client() → Langfuse. The wrapper
        # itself is the only place that touches the SDK's nested shape;
        # static checking inside this class still works because the
        # method bodies use specific attribute access patterns the
        # wrapper either documents or absorbs.
        self._sdk = sdk_client

    def auth_check(self) -> bool:
        return self._sdk.auth_check()

    def run_experiment(
        self,
        *,
        name: str,
        description: str,
        data: list[dict[str, Any]],
        task: Callable[..., Any],
        evaluators: list[Callable[..., Any]],
        metadata: dict[str, Any],
    ) -> ExperimentResult:
        # The SDK's signature wants list[LocalExperimentItem | DatasetItem]
        # (TypedDicts). At runtime the SDK accepts plain dicts; the dict
        # carries the same shape the TypedDict expects, so this is a safe
        # widening for the call.
        return self._sdk.run_experiment(
            name=name,
            description=description,
            data=cast(list[LocalExperimentItem], data),
            task=task,
            evaluators=evaluators,
            metadata=metadata,
        )

    def get_dataset_items(self, name: str) -> list[DatasetItemRecord]:
        dataset = self._sdk.get_dataset(name)
        return [
            DatasetItemRecord(
                source_trace_id=getattr(item, "source_trace_id", None),
                input=getattr(item, "input", None),
                expected_output=getattr(item, "expected_output", None),
                metadata=getattr(item, "metadata", None) or {},
            )
            for item in dataset.items
        ]

    def list_traces(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        limit: int | None = None,
        page_size: int = DEFAULT_TRACE_PAGE_SIZE,
    ) -> list[TraceRecord]:
        # The per-page request size is computed ONCE and held constant for
        # the whole listing. This looks like a missed optimisation on the
        # final page and is not: the SDK addresses pages positionally, so
        # the server derives the offset as (page - 1) * limit. Shrinking
        # `limit` to the remaining budget mid-loop would move the window
        # under the loop — with page_size=50 and limit=60, a second
        # request of limit=10 reads offset 10, re-reading records already
        # collected and never returning records 50-59. Bounding the total
        # is the accumulator's job; the request size is the SDK's cursor
        # arithmetic and must not vary.
        request_size = page_size if limit is None else min(page_size, limit)
        if request_size <= 0:
            return []

        records: list[TraceRecord] = []
        seen: set[str] = set()
        page = 1
        total_pages: int | None = None

        while True:
            response = self._sdk.api.trace.list(
                page=page,
                limit=request_size,
                session_id=session_id,
                user_id=user_id,
                # Explicit on every request, including the first. See
                # _TRACE_ORDER_BY: positional pagination over an
                # undefined order can drop a record silently.
                order_by=_TRACE_ORDER_BY,
                # `fields` is deliberately omitted. Naming field groups
                # would let a caller (or a future edit) request 'core'
                # only, which returns traces whose input/output/metadata
                # are empty — the conversion would then produce examples
                # with no content and no error. Omitting it returns
                # everything.
            )

            batch = list(getattr(response, "data", None) or [])
            budget_spent = False
            for trace in batch:
                trace_id = getattr(trace, "id", None) or ""
                # Deduplicate by id. A write landing in the project
                # mid-listing shifts records between pages and can
                # deliver the same trace twice; unguarded that becomes a
                # duplicated example, which is a scoring error rather
                # than a cosmetic one.
                if trace_id in seen:
                    continue
                seen.add(trace_id)
                raw_metadata = getattr(trace, "metadata", None)
                records.append(
                    TraceRecord(
                        id=trace_id,
                        input=getattr(trace, "input", None),
                        output=getattr(trace, "output", None),
                        metadata=raw_metadata if isinstance(raw_metadata, dict) else {},
                        session_id=getattr(trace, "session_id", None),
                        user_id=getattr(trace, "user_id", None),
                    )
                )
                if limit is not None and len(records) >= limit:
                    budget_spent = True
                    break

            if budget_spent:
                break

            if total_pages is None:
                meta = getattr(response, "meta", None)
                reported = getattr(meta, "total_pages", None)
                total_pages = reported if isinstance(reported, int) else None

            # `total_pages` gives exact termination. The short-page check
            # is a deliberate second condition rather than redundancy: a
            # project mutating during the listing can make the count
            # stale, and an empty or short page is the only signal left.
            if not batch or len(batch) < request_size:
                break
            if total_pages is not None and page >= total_pages:
                break

            page += 1

        return records

    def get_trace_output(self, trace_id: str) -> Any:
        trace = self._sdk.api.trace.get(trace_id)
        observations = getattr(trace, "observations", None) or []
        if not observations:
            return None
        return observations[-1].output

    def get_trace_messages(self, trace_id: str) -> list[dict[str, Any]] | None:
        """Walk observations and concatenate any ``messages`` arrays they carry.

        Langfuse logs LLM observations with ``input`` and ``output``
        payloads that carry an OpenAI-shaped ``messages`` list (the
        adapter writes ``messages``/``input``/``output`` consistently
        in forward-path traces). The wrapper returns the concatenated
        sequence in observation order; returns None if nothing usable
        was found.
        """
        trace = self._sdk.api.trace.get(trace_id)
        observations = getattr(trace, "observations", None) or []
        collected: list[dict[str, Any]] = []
        for obs in observations:
            for side in ("input", "output"):
                payload = getattr(obs, side, None)
                if isinstance(payload, dict):
                    msgs = payload.get("messages")
                    if isinstance(msgs, list):
                        collected.extend(msgs)
        return collected if collected else None
