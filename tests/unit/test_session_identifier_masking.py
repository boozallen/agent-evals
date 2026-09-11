# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""STIG V-222577 — session and trace identifiers must not reach an output boundary.

Every assertion here is deliberately **two-sided**: the raw identifier is
absent AND the masked token is present. Neither half suffices alone. Asserting
only that the token appears passes even when the raw value is emitted beside
it; asserting only absence passes when the line drops the identifier
altogether, destroying the diagnostic value the emission exists for.

Sentinels are distinctive (``DO-NOT-LOG``) so a verbatim-absence assertion
cannot pass by accident against an unrelated substring.

The three boundary kinds covered:

1. **Log records** — ``langfuse.py`` interpolates ``source_trace_id`` into two
   warnings.
2. **Exception text** — ``pull_traces`` re-raises as
   ``RuntimeError(f"...: {e}")``. Once a ``session_id`` reaches the Langfuse
   query, the SDK can echo that value back in its own message, so the raw
   identifier lands in our exception text without any line of our code having
   formatted it. Masking our own format arguments cannot catch that.
3. **Persisted output** — ``local.py._log_example`` writes
   ``EvalExample.model_dump_json()`` to ``results.jsonl``, and a session
   identifier arrives inside the free-form ``metadata`` dict where there is no
   declared field to exclude.
4. **Rendered report** — ``benchmark/report.py`` renders every user-defined
   metadata key as a slicing dimension. ``ScenarioSpec.dimensions`` is
   free-form by key, so a benchmark YAML may declare ``session_id``; the value
   then reaches both a Breakdown group label and a Tasks table cell, and
   ``BenchmarkResult.write_report`` persists the result.
"""

import json
import logging
import re
from typing import Any

import pytest
from foundry_agent_core import mask_session_id
from foundry_agent_core.encryption import load_encryption_key
from helpers.fake_langfuse import FakeLangfuseClient

from agent_evals._readers import read_encrypted_jsonl
from agent_evals.adapters.platforms.langfuse import LangFusePlatform
from agent_evals.adapters.platforms.langfuse_client import DatasetItemRecord
from agent_evals.adapters.platforms.local import LocalConfig, LocalPlatform
from agent_evals.benchmark.types import BenchmarkResult
from agent_evals.core.types import (
    EvalExample,
    EvalResult,
    ExampleData,
    ExpectedResult,
    Score,
    TaskResult,
)

LANGFUSE_LOGGER = "agent_evals.adapters.platforms.langfuse"

SENTINEL_TRACE_ID = "trace-DO-NOT-LOG-a1b2c3"
SENTINEL_SESSION_ID = "sess-DO-NOT-LOG-a1b2c3"


class _EchoingLangfuseClient(FakeLangfuseClient):
    """Fake whose ``get_dataset_items`` echoes a value back in its error.

    Stands in for an SDK that renders a query parameter into its own error
    message — a 400 quoting the filter it rejected, a 404 naming what it
    looked for. That is the disclosure Decision 3 targets: the identifier
    reaches our ``RuntimeError`` text through ``{e}``, not through anything
    we interpolated.
    """

    def __init__(self, *, echo: str) -> None:
        super().__init__()
        self._echo = echo

    def get_dataset_items(self, name: str) -> list[DatasetItemRecord]:
        raise RuntimeError(f"400 Bad Request: no traces for session {self._echo!r}")


class TestLogEmissionsAreMasked:
    """The two ``langfuse.py`` warnings that interpolate a trace identifier."""

    def test_unfetchable_trace_id_is_masked_in_warning(self, caplog):
        """The trace-fetch-failure warning must not disclose the raw ID.

        Diagnostic intent is preserved by the masked token, which is stable
        per identifier — an operator can still tell which item was skipped,
        and can still correlate the same item across log lines.
        """
        fake = FakeLangfuseClient(
            dataset_items={
                "d": [
                    DatasetItemRecord(
                        source_trace_id=SENTINEL_TRACE_ID,
                        input="in",
                        expected_output="exp",
                        metadata={},
                    )
                ]
            },
            # Deliberately unseeded → the fake raises on lookup.
            trace_outputs={},
        )

        adapter = LangFusePlatform(client=fake)
        with caplog.at_level(logging.WARNING, logger=LANGFUSE_LOGGER):
            examples = adapter.pull_traces(config={"dataset": "d"})

        assert examples == []
        messages = [r.getMessage() for r in caplog.records]
        assert messages, "expected a warning for the skipped item"
        emitted = "\n".join(messages)
        assert SENTINEL_TRACE_ID not in emitted, (
            f"raw trace ID disclosed in log output: {messages}"
        )
        assert mask_session_id(SENTINEL_TRACE_ID) in emitted, (
            f"masked token missing, so the item is no longer diagnosable: {messages}"
        )

    def test_messages_unavailable_warning_masks_trace_id(self, caplog):
        """The messages-unavailable warning must not disclose the raw ID either.

        Distinct path from the one above: the trace output resolved, so the
        item IS emitted, and only the trajectory is missing. The fake's
        ``get_trace_messages`` is made to raise to reach it.
        """
        fake = FakeLangfuseClient(
            dataset_items={
                "d": [
                    DatasetItemRecord(
                        source_trace_id=SENTINEL_TRACE_ID,
                        input="in",
                        expected_output="exp",
                        metadata={},
                    )
                ]
            },
            trace_outputs={SENTINEL_TRACE_ID: "the output"},
        )

        def _raise(trace_id: str) -> Any:
            raise RuntimeError("messages endpoint unavailable")

        fake.get_trace_messages = _raise  # type: ignore[method-assign]

        adapter = LangFusePlatform(client=fake)
        with caplog.at_level(logging.WARNING, logger=LANGFUSE_LOGGER):
            examples = adapter.pull_traces(config={"dataset": "d"})

        # The example still comes through; only the trajectory is absent.
        assert len(examples) == 1
        messages = [r.getMessage() for r in caplog.records]
        assert messages, "expected a warning for the unavailable messages"
        emitted = "\n".join(messages)
        assert SENTINEL_TRACE_ID not in emitted, (
            f"raw trace ID disclosed in log output: {messages}"
        )
        assert mask_session_id(SENTINEL_TRACE_ID) in emitted, (
            f"masked token missing, so the item is no longer diagnosable: {messages}"
        )


class TestExceptionBoundaryIsScrubbed:
    """The ``RuntimeError`` ``pull_traces`` raises on any export failure."""

    def test_session_id_echoed_by_sdk_is_masked_in_raised_error(self):
        """A session ID the SDK echoed back must not survive into our error text.

        This is the acceptance criterion the ticket flags as most often
        missed. An unmasked identifier in a traceback is as much a disclosure
        as one in a log line, so the assertion covers the rendered exception
        text — which is what a traceback prints.
        """
        adapter = LangFusePlatform(
            client=_EchoingLangfuseClient(echo=SENTINEL_SESSION_ID)
        )

        with pytest.raises(RuntimeError) as exc_info:
            adapter.pull_traces(
                config={"dataset": "d", "session_id": SENTINEL_SESSION_ID}
            )

        rendered = str(exc_info.value)
        assert SENTINEL_SESSION_ID not in rendered, (
            f"raw session ID disclosed in exception text: {rendered!r}"
        )
        assert mask_session_id(SENTINEL_SESSION_ID) in rendered, (
            f"masked token missing from exception text: {rendered!r}"
        )

    def test_chained_cause_is_preserved_and_also_scrubbed(self):
        """``from e`` chaining survives, and the cause discloses nothing either.

        Both halves matter and they pull against each other. Chaining is what
        makes the failure debuggable — the original SDK exception type and
        traceback stay reachable. But Python's default traceback renders the
        *whole* chain, so a cause whose own message still held the raw value
        would disclose in a traceback exactly what masking removed from the
        message above it.
        """
        adapter = LangFusePlatform(
            client=_EchoingLangfuseClient(echo=SENTINEL_SESSION_ID)
        )

        with pytest.raises(RuntimeError) as exc_info:
            adapter.pull_traces(
                config={"dataset": "d", "session_id": SENTINEL_SESSION_ID}
            )

        cause = exc_info.value.__cause__
        assert cause is not None, "chaining lost; the original error is unreachable"
        assert isinstance(cause, RuntimeError)

        rendered_chain = f"{exc_info.value}\n{cause}"
        assert SENTINEL_SESSION_ID not in rendered_chain, (
            f"raw session ID disclosed via the chained cause: {rendered_chain!r}"
        )
        assert mask_session_id(SENTINEL_SESSION_ID) in str(cause), (
            f"cause lost the identifier entirely, not just masked it: {cause!r}"
        )

    def test_no_session_id_in_config_leaves_the_error_untouched(self):
        """The common path today: ``config`` carries no ``session_id``.

        Nothing may change — in particular no placeholder token may be
        injected into an unrelated error message. ``mask_session_id(None)``
        returns ``sid:<none>``, so a naive implementation that masked
        unconditionally would stamp that string into every export failure.
        """
        adapter = LangFusePlatform(client=_EchoingLangfuseClient(echo="some-dataset"))

        with pytest.raises(RuntimeError) as exc_info:
            adapter.pull_traces(config={"dataset": "d"})

        rendered = str(exc_info.value)
        assert rendered == (
            "Failed to export traces from LangFuse: "
            "400 Bad Request: no traces for session 'some-dataset'"
        )
        assert "sid:" not in rendered, (
            f"a masked token was injected into an unrelated error: {rendered!r}"
        )

    def test_raw_session_id_still_reaches_the_client(self):
        """Masking is an output-boundary control, not an input filter.

        The raw value must keep reaching the platform, or session-scoped
        querying is broken by the control meant to protect its output.
        Asserted on what the client actually received, not merely on the
        absence of an exception.
        """
        received: list[str] = []

        class _RecordingClient(FakeLangfuseClient):
            def get_dataset_items(self, name: str) -> list[DatasetItemRecord]:
                received.append(name)
                return []

        adapter = LangFusePlatform(client=_RecordingClient())
        config: dict[str, Any] = {
            "dataset": "my-dataset",
            "session_id": SENTINEL_SESSION_ID,
        }
        assert adapter.pull_traces(config=config) == []

        assert received == ["my-dataset"]
        # The adapter must not have masked, popped, or otherwise rewritten the
        # caller's config; pull_traces reads session_id from it to filter.
        assert config["session_id"] == SENTINEL_SESSION_ID


class TestPersistedOutputIsRedacted:
    """``local.py._log_example`` writing ``results.jsonl``."""

    @pytest.mark.asyncio
    async def test_session_id_in_metadata_is_masked_in_results_jsonl(self, tmp_path):
        """A session ID in free-form metadata must not reach disk raw.

        Driven end-to-end through ``aevaluate`` rather than by calling
        ``_log_example`` directly, so the test also proves
        ``ExampleData.metadata`` actually reaches the persisted line — the
        reachability claim the redaction rests on.
        """
        dataset = [
            ExampleData(
                input="q",
                expected=ExpectedResult(expected="a"),
                metadata={"session_id": SENTINEL_SESSION_ID},
            )
        ]

        def _scorer(output: TaskResult, expected: ExpectedResult | None) -> Score:
            return Score(
                name="Stub", value=1.0, passed=True, reasoning=None, metadata={}
            )

        adapter = LocalPlatform()
        await adapter.aevaluate(
            task=lambda value: TaskResult(output="a"),
            dataset=dataset,
            evaluators=[_scorer],
            platform=LocalConfig(experiment="masking", output_dir=str(tmp_path)),
        )

        assert adapter.experiment_path is not None
        results_path = adapter.experiment_path / "results.jsonl"
        raw_text = results_path.read_text()

        assert SENTINEL_SESSION_ID not in raw_text, (
            "raw session ID persisted to results.jsonl"
        )
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        records = list(read_encrypted_jsonl(results_path, key))
        assert len(records) == 1
        record = records[0]
        assert record["metadata"]["session_id"] == mask_session_id(SENTINEL_SESSION_ID)

    @pytest.mark.asyncio
    async def test_example_without_session_id_serializes_unchanged(self, tmp_path):
        """Redaction is a no-op on the overwhelmingly common path.

        Compared against ``model_dump_json()`` on the same example rather than
        against a hand-written literal, so the assertion stays true as
        ``EvalExample`` gains fields.
        """
        dataset = [
            ExampleData(
                input="q",
                expected=ExpectedResult(expected="a"),
                metadata={"tenant": "acme", "notes": "no identifiers here"},
            )
        ]

        def _scorer(output: TaskResult, expected: ExpectedResult | None) -> Score:
            return Score(
                name="Stub", value=1.0, passed=True, reasoning=None, metadata={}
            )

        adapter = LocalPlatform()
        result = await adapter.aevaluate(
            task=lambda value: TaskResult(output="a"),
            dataset=dataset,
            evaluators=[_scorer],
            platform=LocalConfig(experiment="no-masking", output_dir=str(tmp_path)),
        )

        assert adapter.experiment_path is not None
        results_path = adapter.experiment_path / "results.jsonl"
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        records = list(read_encrypted_jsonl(results_path, key))
        assert len(records) == 1
        assert records[0] == json.loads(result.examples[0].model_dump_json())


def _report_example(metadata: dict[str, Any]) -> EvalExample:
    """An all-passing example carrying the given metadata."""
    return EvalExample(
        input={"prompt": "p"},
        output="",
        expected="",
        scores={
            "StateMatch": Score(
                name="StateMatch", value=1.0, passed=True, reasoning=None, metadata={}
            )
        },
        metadata=metadata,
        duration=0.1,
    )


def _report_result(*examples: EvalExample) -> BenchmarkResult:
    """A single-capability BenchmarkResult wrapping the given examples."""
    return BenchmarkResult(
        benchmark="masking",
        eval_results={
            "cap": EvalResult(
                experiment_id="exp-1",
                experiment_url="file:///tmp/exp-1",
                platform="local",
                scores={"StateMatch": 1.0},
                pass_rates={"StateMatch": 1.0},
                examples=list(examples),
                summary={
                    "total_examples": len(examples),
                    "successful_examples": len(examples),
                    "failed_examples": 0,
                },
            )
        },
    )


class TestBenchmarkReportIsMasked:
    """``benchmark/report.py`` rendering a ``session_id`` slicing dimension.

    This boundary was an open question in the design, which guessed the report
    renders "scores and capability names, not trace or session metadata". The
    read during implementation found otherwise: ``_discover_dimensions``
    collects *every* metadata key that is not library-added, and
    ``ScenarioSpec.dimensions`` accepts any key. So the disclosure is reachable
    from benchmark YAML alone, with no library change.
    """

    def test_session_id_dimension_is_masked_in_both_render_sites(self):
        """Masked in the Breakdown group label AND the Tasks table cell.

        Both are asserted because they are separate call sites reached by
        separate code paths — masking one and not the other still discloses.
        """
        result = _report_result(
            _report_example(
                {
                    "scenario_id": "s1",
                    "scenario_name": "First scenario",
                    "session_id": SENTINEL_SESSION_ID,
                }
            )
        )

        md = result.to_markdown()
        masked = mask_session_id(SENTINEL_SESSION_ID)

        assert SENTINEL_SESSION_ID not in md, (
            "raw session ID rendered into the benchmark report"
        )
        # Breakdown rollup: `### By session_id` with one group per value.
        assert "### By session_id" in md, (
            "the dimension stopped being rendered at all; masking must not "
            "silently drop the slicing axis"
        )
        assert md.count(masked) >= 2, (
            f"expected the masked token at both the Breakdown label and the "
            f"Tasks cell; found {md.count(masked)} occurrence(s)"
        )

    def test_other_dimensions_and_the_missing_value_placeholder_are_untouched(self):
        """Masking is keyed on the metadata key, so nothing else changes.

        Also pins the ``-`` placeholder: a scenario that omits a dimension the
        report renders must still show ``-``, not a masked token derived from
        the placeholder.
        """
        result = _report_result(
            _report_example(
                {
                    "scenario_id": "s1",
                    "scenario_name": "Has both",
                    "tenant": "acme",
                    "session_id": SENTINEL_SESSION_ID,
                }
            ),
            _report_example(
                {
                    "scenario_id": "s2",
                    "scenario_name": "Tenant only",
                    "tenant": "acme",
                }
            ),
        )

        md = result.to_markdown()

        assert "acme" in md, "a non-identifier dimension value was masked"
        assert "### By tenant" in md

        # s2 declares no session_id → its Tasks row shows the placeholder.
        # The Tasks section is a fixed-width code block, not a pipe table, so
        # the column is matched as a whitespace-delimited standalone token.
        s2_row = next((line for line in md.splitlines() if line.startswith("s2")), None)
        assert s2_row is not None, f"no Tasks row for scenario s2:\n{md}"
        assert re.search(r"\s-\s", s2_row), (
            f"missing-value placeholder absent from the s2 row: {s2_row!r}"
        )
        assert mask_session_id("-") not in md, "the `-` placeholder was itself masked"
