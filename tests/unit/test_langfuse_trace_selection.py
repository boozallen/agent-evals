# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace-selection contracts for ``LangFusePlatform.pull_traces``.

Covers *which* records the adapter returns and *where* each selection is
applied: session / user / count expressed to the platform as query
parameters, the library-form ``filter`` clause applied here, and the
source dispatch that keeps the pre-existing dataset path untouched.

The central guarantee is that a documented parameter is never inert.
Accepting a parameter and ignoring it is worse than rejecting it — the
caller receives a plausible result set that does not answer the question
they asked, with nothing in the return value or the logs to indicate the
difference. These tests are written to fail on that state specifically,
which is why several assert on what the *client received* rather than
only on what came back.
"""

from __future__ import annotations

from typing import Any

import pytest
from helpers.fake_langfuse import FakeLangfuseClient

from agent_evals.adapters.platforms.langfuse import LangFusePlatform
from agent_evals.adapters.platforms.langfuse_client import (
    DatasetItemRecord,
    TraceRecord,
)


def _trace(
    trace_id: str,
    *,
    session_id: str | None = None,
    user_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    input: Any = None,
    output: Any = "answer",
) -> TraceRecord:
    return TraceRecord(
        id=trace_id,
        input=input if input is not None else f"question-{trace_id}",
        output=output,
        metadata=metadata or {},
        session_id=session_id,
        user_id=user_id,
    )


_MIXED_PROJECT = [
    _trace("t1", session_id="s-alpha", user_id="u-1", metadata={"env": "prod"}),
    _trace("t2", session_id="s-alpha", user_id="u-2", metadata={"env": "dev"}),
    _trace("t3", session_id="s-beta", user_id="u-1", metadata={"env": "prod"}),
    _trace("t4", session_id="s-beta", user_id="u-2", metadata={"env": "dev"}),
]


class TestSourceDispatch:
    """The trace path is opt-in, never inferred."""

    def test_traces_source_selects_the_listing_path(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        examples = adapter.pull_traces(config={"source": "traces"})

        assert len(examples) == 4

    def test_session_id_alone_does_not_reroute_to_the_trace_path(self):
        """``session_id`` was an accepted-and-ignored key before this change.

        A caller may be passing one today and receiving dataset results.
        Inferring the trace source from its presence would silently
        reroute them to an entirely different data source on upgrade —
        the reason the selector is explicit rather than inferred.
        """
        fake = FakeLangfuseClient(
            dataset_items={
                "d": [
                    DatasetItemRecord(
                        source_trace_id="tr-1",
                        input="dataset input",
                        expected_output="dataset expected",
                        metadata={},
                    )
                ]
            },
            trace_outputs={"tr-1": "dataset output"},
            traces=_MIXED_PROJECT,
        )
        adapter = LangFusePlatform(client=fake)

        examples = adapter.pull_traces(config={"dataset": "d", "session_id": "s-alpha"})

        assert [e.input for e in examples] == ["dataset input"]
        assert fake.list_traces_calls == [], (
            "presence of session_id must not trigger a trace listing"
        )

    def test_both_sources_named_is_rejected(self):
        """Ambiguous rather than redundant: two sources, two endpoints."""
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        with pytest.raises(ValueError, match="both") as excinfo:
            adapter.pull_traces(config={"source": "traces", "dataset": "d"})

        message = str(excinfo.value)
        assert "source" in message and "dataset" in message, (
            f"error must name both keys so the caller knows which to drop: {message}"
        )

    def test_unknown_source_is_rejected(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient())

        with pytest.raises(ValueError, match="traces"):
            adapter.pull_traces(config={"source": "sessions"})

    def test_dataset_source_still_requires_a_dataset(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient())

        with pytest.raises(ValueError, match="'dataset' key"):
            adapter.pull_traces(config={})

        with pytest.raises(ValueError, match="'dataset' key"):
            adapter.pull_traces(config=None)


class TestPlatformSideSelection:
    """Session, user, and count are expressed to the platform, not post-filtered."""

    def test_session_selection_narrows_the_result_set(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        examples = adapter.pull_traces(
            config={"source": "traces", "session_id": "s-alpha"}
        )

        assert {e.metadata["session_id"] for e in examples} == {"s-alpha"}
        assert len(examples) == 2

    def test_user_selection_narrows_the_result_set(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        examples = adapter.pull_traces(config={"source": "traces", "user_id": "u-1"})

        assert [e.input for e in examples] == ["question-t1", "question-t3"]

    def test_session_and_user_are_conjunctive(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        examples = adapter.pull_traces(
            config={"source": "traces", "session_id": "s-beta", "user_id": "u-2"}
        )

        assert [e.input for e in examples] == ["question-t4"]

    def test_count_bounds_the_result_set(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        examples = adapter.pull_traces(config={"source": "traces", "limit": 2})

        assert len(examples) == 2

    def test_each_constraint_is_expressed_in_the_query(self):
        """The constraint must reach the client, not be applied afterwards.

        Post-hoc discarding is not equivalent: it transfers every record
        in the project across the network, and a count applied after
        retrieval bounds what the caller sees without bounding what was
        fetched. Asserted on the recorded call, because a result set
        filtered either way looks identical.
        """
        fake = FakeLangfuseClient(traces=_MIXED_PROJECT)
        adapter = LangFusePlatform(client=fake)

        adapter.pull_traces(
            config={
                "source": "traces",
                "session_id": "s-alpha",
                "user_id": "u-1",
                "limit": 3,
            }
        )

        assert fake.list_traces_calls == [
            {
                "session_id": "s-alpha",
                "user_id": "u-1",
                "limit": 3,
                "page_size": 50,
            }
        ]

    def test_raw_session_identifier_reaches_the_query(self):
        """Masking is an output-boundary control; a masked query matches nothing."""
        fake = FakeLangfuseClient(traces=[_trace("t1", session_id="s-raw-value-42")])
        adapter = LangFusePlatform(client=fake)

        examples = adapter.pull_traces(
            config={"source": "traces", "session_id": "s-raw-value-42"}
        )

        assert fake.list_traces_calls[0]["session_id"] == "s-raw-value-42"
        assert len(examples) == 1

    def test_empty_matching_set_is_not_an_error(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        assert (
            adapter.pull_traces(config={"source": "traces", "session_id": "s-none"})
            == []
        )

    @pytest.mark.parametrize("bad_limit", ["20", 2.5, True, -1])
    def test_unusable_limit_is_rejected(self, bad_limit: Any):
        """A limit that cannot bound a listing is an error, not "no limit".

        Coercing ``limit="20"`` or ``limit=-1`` to unbounded would export
        an entire project when the caller asked for a bounded slice.
        """
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        with pytest.raises(ValueError, match="limit"):
            adapter.pull_traces(config={"source": "traces", "limit": bad_limit})


class TestFilterClause:
    """The library-form clause is applied here, and never widens selection."""

    def test_clause_alone_selects_exactly_the_satisfying_records(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        examples = adapter.pull_traces(
            config={"source": "traces"}, filter="metadata.env = prod"
        )

        assert [e.input for e in examples] == ["question-t1", "question-t3"]

    def test_clause_and_session_selection_are_both_applied(self):
        """The case the SDK's own ``filter`` would break.

        Langfuse's ``filter`` parameter takes precedence over its
        ``sessionId`` / ``userId`` parameters, so forwarding the caller's
        clause through it would silently discard the session selection
        and return records from every session that satisfied the clause.
        """
        fake = FakeLangfuseClient(traces=_MIXED_PROJECT)
        adapter = LangFusePlatform(client=fake)

        examples = adapter.pull_traces(
            config={"source": "traces", "session_id": "s-beta"},
            filter="metadata.env = prod",
        )

        assert [e.input for e in examples] == ["question-t3"]
        assert {e.metadata["session_id"] for e in examples} == {"s-beta"}
        # Session selection still went to the platform; the clause did not
        # take its place.
        assert fake.list_traces_calls[0]["session_id"] == "s-beta"

    def test_clauses_are_conjunctive(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        examples = adapter.pull_traces(
            config={"source": "traces"},
            filter="metadata.env = prod AND user_id = u-2",
        )

        assert examples == []

    def test_clause_matching_nothing_yields_an_empty_set(self):
        """Not the unfiltered set in its place — that is the defect."""
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        examples = adapter.pull_traces(
            config={"source": "traces"}, filter="metadata.env = staging"
        )

        assert examples == []

    @pytest.mark.parametrize(
        ("clause", "expected_token"),
        [
            ("metadata.env LIKE 'prod'", "metadata.env LIKE 'prod'"),
            ("env = prod", "env"),
            ("metadata.env = 'unbalanced", "unbalanced"),
            ("scores.Factuality > 0.8", "scores.Factuality"),
            ("metadata = prod", "metadata"),
        ],
    )
    def test_uninterpretable_clause_is_reported_naming_the_token(
        self, clause: str, expected_token: str
    ):
        """An error at the call, not a wrong result set in the numbers.

        Silently returning the unfiltered set would be the inert-parameter
        defect relocated, so each rejection must name what it could not
        interpret.
        """
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        with pytest.raises(ValueError) as excinfo:
            adapter.pull_traces(config={"source": "traces"}, filter=clause)

        assert expected_token in str(excinfo.value)

    def test_score_fields_are_rejected_with_the_reason(self):
        """Score *values* are not in a trace listing — only score IDs are.

        Comparing against an ID string and reporting the result as a
        score threshold would be a wrong answer dressed as a right one,
        so the rejection explains why rather than just refusing.
        """
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        with pytest.raises(ValueError, match="score IDs, not values"):
            adapter.pull_traces(
                config={"source": "traces"}, filter="scores.Factuality > 0.8"
            )

    def test_unparseable_clause_fails_before_any_query(self):
        """A typo costs nothing: rejected before the network call."""
        fake = FakeLangfuseClient(traces=_MIXED_PROJECT)
        adapter = LangFusePlatform(client=fake)

        with pytest.raises(ValueError):
            adapter.pull_traces(config={"source": "traces"}, filter="nonsense!!")

        assert fake.list_traces_calls == []

    def test_clause_error_is_not_rewrapped_as_an_export_failure(self):
        """The caller's clause error stays a ValueError, not a RuntimeError.

        Wrapping it would tell an operator the export failed when the
        input was simply wrong.
        """
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        with pytest.raises(ValueError):
            adapter.pull_traces(config={"source": "traces"}, filter="= 5")

    def test_filter_on_the_dataset_source_is_rejected(self):
        """Not silently dropped, which is what it was before this change."""
        fake = FakeLangfuseClient(
            dataset_items={"d": []},
            traces=_MIXED_PROJECT,
        )
        adapter = LangFusePlatform(client=fake)

        with pytest.raises(ValueError, match="trace-listing source only"):
            adapter.pull_traces(config={"dataset": "d"}, filter="metadata.env = prod")

    def test_operators_are_supported_on_metadata_values(self):
        adapter = LangFusePlatform(
            client=FakeLangfuseClient(
                traces=[
                    _trace("t1", metadata={"latency": 10}),
                    _trace("t2", metadata={"latency": 200}),
                ]
            )
        )

        slow = adapter.pull_traces(
            config={"source": "traces"}, filter="metadata.latency >= 100"
        )
        assert [e.input for e in slow] == ["question-t2"]

        fast = adapter.pull_traces(
            config={"source": "traces"}, filter="metadata.latency < 100"
        )
        assert [e.input for e in fast] == ["question-t1"]

    def test_inequality_operator_excludes_matching_records(self):
        adapter = LangFusePlatform(client=FakeLangfuseClient(traces=_MIXED_PROJECT))

        examples = adapter.pull_traces(
            config={"source": "traces"}, filter="metadata.env != prod"
        )

        assert [e.input for e in examples] == ["question-t2", "question-t4"]


class TestDatasetPathPreserved:
    """Adding selection is additive. The old path is byte-identical."""

    def test_dataset_only_export_is_unchanged(self):
        """A caller naming only a dataset receives exactly what they did before.

        The two pre-existing tests in
        ``test_langfuse_pull_traces_pagination.py`` pin the same contract
        unmodified; this one states it in the new file so the guarantee is
        visible beside the change that could break it.
        """
        fake = FakeLangfuseClient(
            dataset_items={
                "d": [
                    DatasetItemRecord(
                        source_trace_id="tr-1",
                        input="the input",
                        expected_output="the expected",
                        metadata={"k": "v"},
                    )
                ]
            },
            trace_outputs={"tr-1": "the output"},
        )
        adapter = LangFusePlatform(client=fake)

        examples = adapter.pull_traces(config={"dataset": "d"})

        assert len(examples) == 1
        example = examples[0]
        assert example.input == "the input"
        assert example.output is not None
        assert example.output.output == "the output"
        assert example.expected is not None
        assert example.expected.expected == "the expected"
        assert example.metadata == {"k": "v"}
        assert fake.list_traces_calls == [], "dataset path must not list traces"

    def test_dataset_expected_value_survives(self):
        fake = FakeLangfuseClient(
            dataset_items={
                "d": [
                    DatasetItemRecord(
                        source_trace_id="tr-1",
                        input="q",
                        expected_output="the answer key",
                        metadata={},
                    )
                ]
            },
            trace_outputs={"tr-1": "o"},
        )
        adapter = LangFusePlatform(client=fake)

        (example,) = adapter.pull_traces(config={"dataset": "d"})

        assert example.expected is not None
        assert example.expected.expected == "the answer key"

    def test_explicit_dataset_source_behaves_as_the_default(self):
        fake = FakeLangfuseClient(
            dataset_items={
                "d": [
                    DatasetItemRecord(
                        source_trace_id="tr-1",
                        input="q",
                        expected_output="a",
                        metadata={},
                    )
                ]
            },
            trace_outputs={"tr-1": "o"},
        )
        adapter = LangFusePlatform(client=fake)

        explicit = adapter.pull_traces(config={"source": "dataset", "dataset": "d"})
        implicit = adapter.pull_traces(config={"dataset": "d"})

        assert explicit == implicit
