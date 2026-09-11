# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for LangFusePlatform.pull_traces resolution strategy.

**Pagination is now solved rather than avoided.** The port previously had
no list-shaped method at all, and this file's contract was that it must
stay that way: a single ``api.trace.list()`` call returns only the first
page (default 50 items), so any list-based membership pattern produced
false negatives once a project had more than one page of traces. Removing
the method removed the trap, at the cost of making session, user, and
count selection unimplementable.

``LangfuseClientPort.list_traces`` now exists and absorbs pagination
inside the wrapper — it walks pages under a pinned ``timestamp.desc``
order and returns the complete matching set. The page loop is tested
where it lives, against a stub SDK, in
``tests/unit/test_langfuse_client_pagination.py``; a fake that returns
everything in one call cannot fail the way production fails, so
adapter-level tests would prove nothing about it.

What this file still pins is the **dataset** path, which does not list at
all:

- The adapter resolves each dataset item's ``source_trace_id`` via
  ``get_trace_output`` (the real wrapper's
  ``client.api.trace.get(id).observations[-1].output``). Fetching by a
  known id is not a pagination problem, and a listing would be strictly
  more work for the same answer.
- Adding the trace source must not change this path. Both tests below
  are pre-existing and unmodified; they are the regression evidence for
  that, and a third test asserts the dataset path issues no listing call.

Trace-source coverage lives in
``tests/unit/test_langfuse_trace_selection.py`` (source dispatch,
platform-side selection, the filter clause) and
``tests/unit/test_langfuse_trace_conversion.py`` (trace-to-example
conversion and session attribution).
"""

import logging

from foundry_agent_core import mask_session_id
from helpers.fake_langfuse import FakeLangfuseClient

from agent_evals.adapters.platforms.langfuse import LangFusePlatform
from agent_evals.adapters.platforms.langfuse_client import DatasetItemRecord


class TestPullTracesResolutionContract:
    """Structural contracts for how pull_traces resolves dataset items."""

    def test_pull_traces_resolves_each_item_via_get_trace_output(self):
        """Contract: pull_traces calls ``get_trace_output`` per dataset item.

        The flat port has only ``get_dataset_items`` and
        ``get_trace_output``; there is no ``list`` method. A successful
        resolution proves the adapter walked the dataset items and
        looked up each trace's already-extracted output.
        """
        fake = FakeLangfuseClient(
            dataset_items={
                "d": [
                    DatasetItemRecord(
                        source_trace_id="some-trace-id",
                        input="input text",
                        expected_output="expected text",
                        metadata={},
                    )
                ]
            },
            trace_outputs={"some-trace-id": "output text"},
        )

        adapter = LangFusePlatform(client=fake)
        examples = adapter.pull_traces(config={"dataset": "d"})

        assert len(examples) == 1
        assert examples[0].input == "input text"
        assert examples[0].output is not None
        assert examples[0].output.output == "output text"

    def test_missing_trace_is_skipped_with_warning(self, caplog):
        """Contract: a dataset item whose trace lookup raises is logged and skipped.

        Handles retention-expired traces, cross-project IDs, and any
        other single-item failure without aborting the whole
        ``pull_traces`` call. The fake raises ``FakeTraceNotFoundError``
        for unseeded trace IDs (an ``Exception`` subclass), which the
        adapter's bare ``except Exception`` catches exactly like the
        real SDK case.
        """
        fake = FakeLangfuseClient(
            dataset_items={
                "mixed": [
                    DatasetItemRecord(
                        source_trace_id="trace-good",
                        input="Good input",
                        expected_output="Good expected",
                        metadata={},
                    ),
                    DatasetItemRecord(
                        source_trace_id="trace-missing",
                        input="Missing input",
                        expected_output="Irrelevant",
                        metadata={},
                    ),
                ]
            },
            # trace-missing intentionally not seeded → fake raises on lookup
            trace_outputs={"trace-good": "Good output"},
        )

        adapter = LangFusePlatform(client=fake)
        with caplog.at_level(
            logging.WARNING, logger="agent_evals.adapters.platforms.langfuse"
        ):
            examples = adapter.pull_traces(config={"dataset": "mixed"})

        assert len(examples) == 1
        assert examples[0].input == "Good input"
        assert examples[0].output is not None
        assert examples[0].output.output == "Good output"

        # The warning must still identify WHICH item was skipped, so an
        # operator can diagnose — but by masked token, not by raw ID
        # (STIG V-222577). The token is deterministic per identifier, so it
        # preserves the diagnostic value the raw ID used to carry, including
        # correlating the same item across log lines.
        #
        # Two-sided on purpose. Asserting only that the token is present would
        # pass if the raw ID were emitted beside it; asserting only that the
        # raw ID is absent would pass if the line dropped the identifier
        # altogether, silently destroying what this test exists to protect.
        warnings = [
            r.getMessage() for r in caplog.records if r.levelno == logging.WARNING
        ]
        emitted = "\n".join(warnings)
        assert "trace-missing" not in emitted, (
            f"Raw trace ID must not be logged; got: {warnings}"
        )
        assert mask_session_id("trace-missing") in emitted, (
            f"Expected a warning carrying the masked token for 'trace-missing'; "
            f"got: {warnings}"
        )

    def test_dataset_path_issues_no_trace_listing(self):
        """Contract: the dataset path resolves by id and never lists.

        Now that ``list_traces`` exists, "the port has no list method" no
        longer enforces this structurally, so it is asserted directly.
        A dataset export that started listing would silently change what
        it fetched — every trace in the project instead of the handful the
        dataset names — and still return plausible examples.
        """
        fake = FakeLangfuseClient(
            dataset_items={
                "d": [
                    DatasetItemRecord(
                        source_trace_id="tr-1",
                        input="i",
                        expected_output="e",
                        metadata={},
                    )
                ]
            },
            trace_outputs={"tr-1": "o"},
        )

        adapter = LangFusePlatform(client=fake)
        examples = adapter.pull_traces(config={"dataset": "d"})

        assert len(examples) == 1
        assert fake.list_traces_calls == []
