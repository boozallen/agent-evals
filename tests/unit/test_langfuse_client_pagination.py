# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Wrapper-level tests for ``_RealLangfuseClient.list_traces`` pagination.

These drive the production wrapper against a **stub SDK**, not against
``FakeLangfuseClient``. That is deliberate and load-bearing: the fake
implements the port, and the port's contract is that the caller receives
the complete matching set with pagination already absorbed. A fake that
returns everything in one call therefore cannot fail the way production
fails, so adapter-level tests prove nothing about the page loop.

Pagination is the acceptance criterion most likely to be quietly wrong —
a first-page-only listing returns plausible records and reports absence
for everything past record 50 — so it is covered here, one page-loop
property per test, against real ``Traces`` / ``TraceWithDetails`` /
``MetaResponse`` instances so SDK shape drift surfaces rather than being
absorbed by a hand-rolled stand-in.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from langfuse.api import Traces, TraceWithDetails
from langfuse.api.utils.pagination.types.meta_response import MetaResponse

from agent_evals.adapters.platforms.langfuse_client import (
    DEFAULT_TRACE_PAGE_SIZE,
    _RealLangfuseClient,
)

_EPOCH = dt.datetime(2026, 1, 1, 12, 0, 0, tzinfo=dt.UTC)


def _trace(trace_id: str, **overrides: Any) -> TraceWithDetails:
    """Build a real ``TraceWithDetails`` with every required field populated."""
    payload: dict[str, Any] = {
        "id": trace_id,
        "timestamp": _EPOCH,
        "name": "run",
        "input": {"question": f"q-{trace_id}"},
        "output": f"answer-{trace_id}",
        "session_id": "session-a",
        "user_id": "user-a",
        "metadata": {},
        "tags": [],
        "public": False,
        "environment": "default",
        "html_path": f"/traces/{trace_id}",
        "latency": 1.0,
        "total_cost": 0.0,
        "observations": [],
        "scores": [],
        "release": None,
        "version": None,
    }
    payload.update(overrides)
    return TraceWithDetails(**payload)


class _StubTraceApi:
    """Stub for ``client.api.trace``, serving pre-built pages by page number.

    Records every call so a test can assert what the loop *asked for*,
    not merely what it returned. The distinction matters for the ordering
    argument: a listing can look correct on a stub that happens to return
    sorted data while sending no ``order_by`` at all.
    """

    def __init__(self, pages: dict[int, list[TraceWithDetails]], total_items: int):
        self._pages = pages
        self._total_items = total_items
        self.calls: list[dict[str, Any]] = []

    def list(self, **kwargs: Any) -> Traces:
        self.calls.append(kwargs)
        page = kwargs["page"]
        page_limit = kwargs["limit"]
        data = self._pages.get(page, [])
        total_pages = max(1, -(-self._total_items // page_limit))
        return Traces(
            data=data,
            meta=MetaResponse(
                page=page,
                limit=page_limit,
                total_items=self._total_items,
                total_pages=total_pages,
            ),
            deprecation=None,
        )


def _client(pages: dict[int, list[TraceWithDetails]], total_items: int | None = None):
    """Wire a ``_RealLangfuseClient`` over a stub SDK serving ``pages``."""
    if total_items is None:
        total_items = sum(len(p) for p in pages.values())
    stub_api = _StubTraceApi(pages, total_items)

    class _StubSdk:
        def __init__(self) -> None:
            self.api = type("_Api", (), {"trace": stub_api})()

    return _RealLangfuseClient(_StubSdk()), stub_api


class TestListTracesPagination:
    """The page loop returns the complete matching set, exactly once each."""

    def test_records_past_the_first_page_are_returned(self):
        """A matching set spanning two pages comes back whole.

        This is the defect the capability exists to prevent: a caller
        cannot distinguish a project with 3 matching traces from one with
        5 whose listing stopped at the page boundary, so a truncated
        listing reports absence where the record exists.
        """
        client, stub = _client(
            {
                1: [_trace("t1"), _trace("t2"), _trace("t3")],
                2: [_trace("t4"), _trace("t5")],
            },
            total_items=5,
        )

        records = client.list_traces(page_size=3)

        assert [r.id for r in records] == ["t1", "t2", "t3", "t4", "t5"]
        assert len(records) == 5, "returned count must equal meta.total_items"
        assert [c["page"] for c in stub.calls] == [1, 2]

    def test_every_request_specifies_the_sort_order(self):
        """Ordering is sent on every page, not merely on the first.

        Nothing else in the suite would catch its removal: a stub (and a
        real project) can return records in a plausible order while the
        request specifies none, and the failure mode of an unordered
        positional listing is an *omission* — no duplicate, no gap, no
        count to reconcile against.
        """
        client, stub = _client(
            {1: [_trace("t1"), _trace("t2")], 2: [_trace("t3")]},
            total_items=3,
        )

        client.list_traces(page_size=2)

        assert len(stub.calls) == 2
        for call in stub.calls:
            assert call["order_by"] == "timestamp.desc", (
                "every listing request must specify an explicit order; "
                f"got {call.get('order_by')!r}"
            )

    def test_fields_is_not_narrowed(self):
        """``fields`` stays unsent so input/output/metadata come back populated.

        Requesting a narrower field group is a performance optimisation
        that silently empties the payload the conversion depends on —
        examples with no content and no error.
        """
        client, stub = _client({1: [_trace("t1")]}, total_items=1)

        client.list_traces()

        assert "fields" not in stub.calls[0]

    def test_selection_is_expressed_in_the_query(self):
        """Session and user constraints reach the SDK call, unmasked.

        Masking is an output-boundary control; a masked identifier in the
        query would match nothing.
        """
        client, stub = _client({1: [_trace("t1")]}, total_items=1)

        client.list_traces(session_id="session-raw-42", user_id="user-raw-7")

        assert stub.calls[0]["session_id"] == "session-raw-42"
        assert stub.calls[0]["user_id"] == "user-raw-7"


class TestListTracesTermination:
    """Retrieval stops, and stops for the right reason."""

    def test_single_page_issues_one_request(self):
        client, stub = _client({1: [_trace("t1"), _trace("t2")]}, total_items=2)

        records = client.list_traces(page_size=DEFAULT_TRACE_PAGE_SIZE)

        assert len(records) == 2
        assert len(stub.calls) == 1

    def test_empty_matching_set_returns_empty_list(self):
        """No match is not an error — an empty set is the answer."""
        client, stub = _client({}, total_items=0)

        records = client.list_traces()

        assert records == []
        assert len(stub.calls) == 1

    def test_request_count_is_bounded_by_the_matching_set(self):
        """The loop cannot outrun the data it is walking."""
        client, stub = _client(
            {1: [_trace("t1")], 2: [_trace("t2")], 3: [_trace("t3")]},
            total_items=3,
        )

        records = client.list_traces(page_size=1)

        assert len(records) == 3
        assert len(stub.calls) == 3, "one request per page, no re-reads"

    def test_stale_total_pages_does_not_loop_forever(self):
        """A short page terminates even when ``total_pages`` over-reports.

        A project mutating mid-listing can make the reported count stale.
        ``total_pages`` gives exact termination in the quiet case; the
        short-page check is the second condition that keeps a stale count
        from becoming an unbounded loop.
        """
        # total_items=100 over page_size=2 claims 50 pages; only page 1
        # actually has data, and it is short.
        client, stub = _client({1: [_trace("t1")]}, total_items=100)

        records = client.list_traces(page_size=2)

        assert [r.id for r in records] == ["t1"]
        assert len(stub.calls) == 1


class TestListTracesLimit:
    """``limit`` is a total record budget, not a page size."""

    def test_limit_bounds_the_total_across_pages(self):
        """The budget bounds the whole listing, not each page.

        Langfuse's own ``limit`` is per-page, so a reader who knows the
        SDK will assume the wrong one; this pins ours.
        """
        client, _ = _client(
            {
                1: [_trace("t1"), _trace("t2")],
                2: [_trace("t3"), _trace("t4")],
                3: [_trace("t5")],
            },
            total_items=5,
        )

        records = client.list_traces(limit=3, page_size=2)

        assert [r.id for r in records] == ["t1", "t2", "t3"]

    def test_small_limit_issues_one_small_request(self):
        """``limit=1`` fetches one record, not a page it then discards."""
        client, stub = _client({1: [_trace("t1")]}, total_items=50)

        records = client.list_traces(limit=1)

        assert len(records) == 1
        assert len(stub.calls) == 1
        assert stub.calls[0]["limit"] == 1

    def test_request_size_is_constant_across_pages(self):
        """Every request carries the SAME page size, even on the last page.

        Regression guard for a subtle cursor bug. The SDK addresses pages
        positionally, so the server derives the offset as
        ``(page - 1) * limit``. Shrinking the request to the remaining
        budget mid-loop moves the window: with ``page_size=2`` and
        ``limit=3``, a second request of ``limit=1`` would read offset 1
        — re-reading a record already collected and never returning the
        third. Bounding the total is the accumulator's job; the request
        size is cursor arithmetic and must not vary.
        """
        client, stub = _client(
            {1: [_trace("t1"), _trace("t2")], 2: [_trace("t3"), _trace("t4")]},
            total_items=4,
        )

        client.list_traces(limit=3, page_size=2)

        sizes = {call["limit"] for call in stub.calls}
        assert sizes == {2}, f"page size must not vary mid-listing; got {sizes}"

    def test_zero_limit_requests_nothing(self):
        """A zero budget short-circuits rather than fetching a page."""
        client, stub = _client({1: [_trace("t1")]}, total_items=1)

        assert client.list_traces(limit=0) == []
        assert stub.calls == []


class TestListTracesDeduplication:
    """A record delivered twice is returned once."""

    def test_record_shifted_between_pages_appears_once(self):
        """Concurrent writes can deliver the same trace on two pages.

        An offset listing over a mutating project re-reads records when a
        new write shifts them forward. Unguarded this becomes a
        duplicated example, which is a scoring error rather than a
        cosmetic one — the duplicate is scored twice and skews the
        aggregate.
        """
        client, _ = _client(
            {
                1: [_trace("t1"), _trace("t2")],
                2: [_trace("t2"), _trace("t3")],
            },
            total_items=4,
        )

        records = client.list_traces(page_size=2)

        ids = [r.id for r in records]
        assert ids == ["t1", "t2", "t3"]
        assert len(ids) == len(set(ids)), "each matching record exactly once"


class TestTraceRecordShape:
    """The flat record carries what the adapter reads, and nothing invented."""

    def test_sdk_fields_are_mapped_onto_the_record(self):
        client, _ = _client(
            {
                1: [
                    _trace(
                        "t1",
                        input={"question": "why"},
                        output="because",
                        session_id="s-9",
                        user_id="u-9",
                        metadata={"env": "prod"},
                    )
                ]
            },
            total_items=1,
        )

        (record,) = client.list_traces()

        assert record.id == "t1"
        assert record.input == {"question": "why"}
        assert record.output == "because"
        assert record.session_id == "s-9"
        assert record.user_id == "u-9"
        assert record.metadata == {"env": "prod"}

    def test_non_dict_metadata_becomes_an_empty_dict(self):
        """A trace's metadata is arbitrary JSON; the record promises a dict.

        Langfuse permits a scalar or list there. Coercing at the seam
        keeps every consumer from re-checking the type, and an empty dict
        is the honest reading of "carries no metadata keys".
        """
        client, _ = _client(
            {1: [_trace("t1", metadata="not-a-mapping")]}, total_items=1
        )

        (record,) = client.list_traces()

        assert record.metadata == {}
