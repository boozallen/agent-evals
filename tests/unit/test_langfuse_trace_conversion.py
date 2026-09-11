# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Trace-to-example conversion for the Langfuse trace-listing source.

A dataset item arrives with an agreed shape and an answer key. A raw
execution trace arrives with neither: ``input`` and ``output`` are
arbitrary JSON, and there is no expected value anywhere in the record.
These tests pin what the adapter does with that — carry the payload
through without inventing structure, and represent a missing answer key
as *missing* rather than as an empty string.

The empty-string substitution is the failure worth naming. An
``expected=""`` reaches a scorer as a real comparison against nothing, so
a caller reads a low similarity score and concludes their agent
regressed, when in fact no reference existed to compare against. Absence
must stay absent so the scorer layer (see ``requires_expected``) can say
so.

Session attribution is also covered here, two-sided per V-222577: the
raw identifier must never reach a persisted boundary, and the masked
token must be present and *correlatable* — traces sharing a session must
share a token, or masking has destroyed the grouping the attribution
exists to provide.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from foundry_agent_core import mask_session_id
from foundry_agent_core.encryption import load_encryption_key
from helpers.fake_langfuse import FakeLangfuseClient

from agent_evals import read_encrypted_jsonl
from agent_evals.adapters.platforms.langfuse import LangFusePlatform
from agent_evals.adapters.platforms.langfuse_client import TraceRecord
from agent_evals.adapters.platforms.local import LocalConfig, LocalPlatform
from agent_evals.core.types import ExpectedResult, Score, TaskResult

LANGFUSE_LOGGER = "agent_evals.adapters.platforms.langfuse"

SENTINEL_SESSION_ID = "sess-DO-NOT-LOG-trace-path"
OTHER_SESSION_ID = "sess-DO-NOT-LOG-different"


def _record(
    trace_id: str = "t1",
    *,
    input: Any = "the question",
    output: Any = "the answer",
    metadata: dict[str, Any] | None = None,
    session_id: str | None = None,
    user_id: str | None = None,
) -> TraceRecord:
    return TraceRecord(
        id=trace_id,
        input=input,
        output=output,
        metadata=metadata or {},
        session_id=session_id,
        user_id=user_id,
    )


def _pull(records: list[TraceRecord], **config: Any):
    adapter = LangFusePlatform(client=FakeLangfuseClient(traces=records))
    return adapter.pull_traces(config={"source": "traces", **config})


class TestExpectedValueAbsence:
    """A raw trace has no answer key, and absence is represented as absence."""

    def test_raw_trace_yields_no_expected_value(self):
        """``None``, never ``""``.

        An empty string is a comparison against nothing that reports as a
        low score, which reads to the caller as an agent regression rather
        than as a missing reference.
        """
        (example,) = _pull([_record()])

        assert example.expected is None

    def test_no_nomination_consults_no_metadata_field(self):
        """A field that *looks* like an answer key is not treated as one.

        Inferring it would silently score a whole export against whatever
        happened to be annotated, and the caller would have no way to
        tell that from a genuine reference set.
        """
        (example,) = _pull(
            [
                _record(
                    metadata={
                        "expected": "looks like an answer key",
                        "expected_output": "so does this",
                        "ground_truth": "and this",
                    }
                )
            ]
        )

        assert example.expected is None

    def test_nominated_field_supplies_the_expected_value(self):
        (example,) = _pull(
            [_record(metadata={"gold": "the reference answer"})],
            expected_metadata_key="gold",
        )

        assert example.expected == ExpectedResult(expected="the reference answer")

    @pytest.mark.parametrize("empty", ["", "   ", "\n\t"])
    def test_empty_nominated_value_is_treated_as_absent(self, empty: str):
        """An annotation present but empty is no annotation at all.

        Including the whitespace-only forms deliberately: ``"   "`` is
        truthy, so a bare falsy check would pass it through as a real
        reference and produce exactly the empty-comparison score this
        change exists to eliminate.
        """
        (example,) = _pull(
            [_record(metadata={"gold": empty})],
            expected_metadata_key="gold",
        )

        assert example.expected is None

    def test_nominated_field_no_record_carries_still_exports(self):
        """A nomination nothing satisfies completes the export.

        Raising would make the caller's annotation typo cost them the
        whole batch; the examples are still useful to a reference-free
        scorer.
        """
        examples = _pull(
            [_record("t1"), _record("t2")],
            expected_metadata_key="never-annotated",
        )

        assert len(examples) == 2
        assert all(e.expected is None for e in examples)

    def test_structured_nominated_value_is_rendered(self):
        (example,) = _pull(
            [_record(metadata={"gold": {"answer": 42}})],
            expected_metadata_key="gold",
        )

        assert example.expected is not None
        assert json.loads(example.expected.expected) == {"answer": 42}

    def test_partially_annotated_batch_exports_in_full(self):
        """Record count preserved; only the annotated records carry a key."""
        examples = _pull(
            [
                _record("t1", metadata={"gold": "a"}),
                _record("t2", metadata={}),
                _record("t3", metadata={"gold": "c"}),
            ],
            expected_metadata_key="gold",
        )

        assert len(examples) == 3
        assert [e.expected.expected if e.expected else None for e in examples] == [
            "a",
            None,
            "c",
        ]


class TestPayloadConversion:
    """Arbitrary JSON in, no invented structure out."""

    def test_scalar_payload_survives(self):
        (example,) = _pull([_record(input="why", output="because")])

        assert example.input == "why"
        assert example.output is not None
        assert example.output.output == "because"

    def test_structured_input_is_carried_through_unchanged(self):
        """``ExampleData.input`` is ``Any`` so the task can re-run the payload.

        Stringifying here would invent a shape the caller has to parse
        back out, and a task replaying a structured request would receive
        a JSON blob where it expected a mapping.
        """
        payload = {"question": "why", "context": ["a", "b"]}

        (example,) = _pull([_record(input=payload)])

        assert example.input == payload

    def test_structured_output_is_rendered_as_text(self):
        """The scored output is a string; the structure is not lost, only rendered."""
        (example,) = _pull([_record(output={"answer": "because", "score": 1})])

        assert example.output is not None
        assert json.loads(example.output.output) == {"answer": "because", "score": 1}

    def test_message_list_output_populates_the_trajectory_context(self):
        """Trajectory scorers read ``context["outputs"]``, the existing convention."""
        messages = [
            {"role": "user", "content": "why"},
            {"role": "assistant", "content": "because"},
        ]

        (example,) = _pull([_record(output=messages)])

        assert example.output is not None
        assert example.output.context["outputs"] == messages

    def test_missing_output_becomes_an_empty_rendering_not_a_skip(self):
        """A trace with an input but no output is still a re-runnable case."""
        (example,) = _pull([_record(input="why", output=None)])

        assert example.input == "why"
        assert example.output is not None
        assert example.output.output == ""

    def test_record_without_input_is_skipped_with_a_warning(self, caplog):
        """One unresolvable record must not cost the caller the other 200."""
        with caplog.at_level(logging.WARNING, logger=LANGFUSE_LOGGER):
            examples = _pull([_record("t1", input=None), _record("t2", input="ok")])

        assert [e.input for e in examples] == ["ok"]
        assert any("Skipping trace" in r.message for r in caplog.records)

    def test_trace_metadata_reaches_example_metadata(self):
        (example,) = _pull([_record(metadata={"env": "prod", "run": 3})])

        assert example.metadata["env"] == "prod"
        assert example.metadata["run"] == 3


class TestSessionAttribution:
    """V-222577: attribution is carried, under the key the redaction matches."""

    def test_session_id_is_carried_under_the_exact_key_spelling(self):
        """``session_id``, not ``sessionId``.

        ``redact_session_ids`` and the report's key set match only this
        spelling, so a camelCase key would reach ``results.jsonl``
        verbatim while every masking test still passed — a control that
        looks green and is not.
        """
        (example,) = _pull([_record(session_id=SENTINEL_SESSION_ID)])

        assert example.metadata["session_id"] == SENTINEL_SESSION_ID
        assert "sessionId" not in example.metadata

    def test_attribution_is_stored_raw_not_pre_masked(self):
        """Masking happens at the boundary, once.

        Pre-masking here would be masked again at persistence —
        ``mask(mask(raw))`` in the file against ``mask(raw)`` in the logs
        — so the two emissions could no longer be correlated, which is
        the whole point of a deterministic token.
        """
        (example,) = _pull([_record(session_id=SENTINEL_SESSION_ID)])

        assert example.metadata["session_id"] != mask_session_id(SENTINEL_SESSION_ID)

    def test_user_id_is_not_carried_into_example_metadata(self):
        """No redaction recognises a ``user_id`` key.

        Carrying it would open a disclosure path with no control behind
        it, and the caller already knows the user they selected on.
        """
        (example,) = _pull([_record(session_id="s-1", user_id="u-secret")])

        assert "user_id" not in example.metadata
        assert "userId" not in example.metadata

    def test_absent_session_id_adds_no_key(self):
        (example,) = _pull([_record(session_id=None)])

        assert "session_id" not in example.metadata

    async def test_persisted_results_carry_the_masked_token_not_the_raw_value(
        self, tmp_path
    ):
        """The two-sided end-to-end assertion: raw absent, masked present.

        Neither half suffices. Asserting only that the token appears
        passes when the raw value is written beside it; asserting only
        absence passes when the line drops the attribution entirely,
        destroying the diagnostic value it exists for.
        """
        dataset = _pull([_record(session_id=SENTINEL_SESSION_ID)])

        def _scorer(output: TaskResult, expected: Any, **_: Any) -> Score:
            return Score(name="s", value=1.0, passed=True, reasoning=None, metadata={})

        local = LocalPlatform()
        await local.aevaluate(
            task=lambda value: TaskResult(output="replayed"),
            dataset=dataset,
            evaluators=[_scorer],
            platform=LocalConfig(experiment="trace-masking", output_dir=str(tmp_path)),
        )

        raw_text = (local.experiment_path / "results.jsonl").read_text()
        assert SENTINEL_SESSION_ID not in raw_text
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        record = next(
            read_encrypted_jsonl(local.experiment_path / "results.jsonl", key)
        )
        assert record["metadata"]["session_id"] == mask_session_id(SENTINEL_SESSION_ID)

    async def test_masked_attribution_stays_correlatable(self, tmp_path):
        """Masking hides the value; it must not destroy the grouping.

        Traces from one session must share a token and traces from
        another must not, or an operator can no longer tell "these three
        failures came from one conversation" from "these three failures
        are unrelated" — the question session attribution is carried to
        answer.
        """
        dataset = _pull(
            [
                _record("t1", session_id=SENTINEL_SESSION_ID),
                _record("t2", session_id=SENTINEL_SESSION_ID),
                _record("t3", session_id=OTHER_SESSION_ID),
            ]
        )

        def _scorer(output: TaskResult, expected: Any, **_: Any) -> Score:
            return Score(name="s", value=1.0, passed=True, reasoning=None, metadata={})

        local = LocalPlatform()
        await local.aevaluate(
            task=lambda value: TaskResult(output="replayed"),
            dataset=dataset,
            evaluators=[_scorer],
            platform=LocalConfig(
                experiment="trace-correlation", output_dir=str(tmp_path)
            ),
        )

        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        records = list(
            read_encrypted_jsonl(local.experiment_path / "results.jsonl", key)
        )
        tokens = [r["metadata"]["session_id"] for r in records]

        assert len(tokens) == 3
        assert tokens[0] == tokens[1], "same session must share one masked token"
        assert tokens[2] != tokens[0], "different sessions must not collide"
        for token in tokens:
            assert SENTINEL_SESSION_ID not in token
            assert OTHER_SESSION_ID not in token
