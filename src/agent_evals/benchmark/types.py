# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark result type -- aggregates per-capability EvalResult objects.

Aggregation rule: macro-average across capabilities (each capability
contributes equally to benchmark-level scores regardless of test count).
"""

from collections.abc import Callable
from statistics import fmean
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

from agent_evals import EvalResult
from agent_evals.core.types import PathSafeIdentifier

if TYPE_CHECKING:
    from pathlib import Path

    from agent_evals.benchmark.config import BenchmarkConfig


class BenchmarkResult(BaseModel):
    """Aggregate benchmark result combining per-capability EvalResults.

    aggregate_scores and aggregate_pass_rates are macro-averaged across
    capabilities. Per-capability EvalResult objects are retained under
    eval_results for drill-down.
    """

    benchmark: PathSafeIdentifier
    """Benchmark name. Path-safe constrained because ``write_report`` derives
    the report filename from this field, so a value carrying separators or a
    drive prefix would redirect the write outside the requested directory.
    """
    agent_name: str | None = None
    """Agent identity, populated by the runner from the `Agent` instance's
    `name` class attribute. Surfaced in the Run info section of the report
    so a reader can tell which agent produced the result.
    """
    agent_version: str | None = None
    """Agent version, populated by the runner from the `Agent` instance's
    `version` class attribute. Surfaced alongside `agent_name`.
    """
    config: BenchmarkConfig | None = None
    """The BenchmarkConfig the runner used, or None if this result was
    constructed outside the runner (e.g., tests, historical-data ingestion,
    hand-built fixtures). The reporter uses `config.description` and
    other fields when available to enrich the Run info section.
    """
    config_path: str | None = None
    """The YAML path string the caller passed to `run_benchmark_async`,
    preserved as-typed (relative or absolute). Shown in the Run info
    section so a reader can find the config that produced the report.
    """
    eval_results: dict[str, EvalResult] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def aggregate_scores(self) -> dict[str, float]:
        """Macro-average scores across capabilities, per scorer name."""
        return self._macro_average(lambda er: er.scores)

    @property
    def aggregate_pass_rates(self) -> dict[str, float]:
        """Macro-average pass rates across capabilities, per scorer name."""
        return self._macro_average(lambda er: er.pass_rates)

    def to_markdown(self) -> str:
        """Render a Markdown report for this benchmark run.

        Returns a self-contained Markdown string suitable for printing, writing
        to disk, or pasting into a PR / Slack. Includes score, per-dimension
        rollups, failing tests with reasoning, and per-capability task tables.

        See `docs/benchmarks.md` for sample output and customization guidance.
        """
        from agent_evals.benchmark.report import build_markdown_report

        return build_markdown_report(self)

    def write_report(
        self,
        results_dir: str | Path = "benchmarks/results",
        filename: str | None = None,
    ) -> Path:
        """Write this benchmark's Markdown report to disk; return the path.

        The persist sibling of `to_markdown()`: `to_markdown()` returns the
        report as a string, `write_report()` puts it on disk. Creates
        `results_dir` if missing; default filename is
        `{YYYY-MM-DD_HH-MM-SS}_{benchmark}.md` (pass `filename` to override,
        useful for deterministic tests or custom naming). Performs no printing.

        See `docs/benchmarks.md` for the end-to-end run/report flow.
        """
        from agent_evals.benchmark.artifacts import _write_report

        return _write_report(self, results_dir, filename)

    def _macro_average(
        self, select: Callable[[EvalResult], dict[str, float]]
    ) -> dict[str, float]:
        """Macro-average a per-scorer dict across capabilities.

        A capability that does not declare a given scorer is skipped for that
        key rather than counted as 0.0: scorers are bound per capability by
        its ``success:`` block, so absence normally means "not applicable
        here", not "failed here". Averaging in a zero would penalise every
        capability that legitimately does not use a scorer.

        Total failure is therefore represented upstream instead — a
        capability whose examples all failed carries an explicit failing key
        (see ``compute_aggregate_scores``), which does roll up through here.
        """
        all_keys: set[str] = set()
        for er in self.eval_results.values():
            all_keys.update(select(er).keys())

        out: dict[str, float] = {}
        for key in all_keys:
            values = [
                select(er)[key]
                for er in self.eval_results.values()
                if key in select(er)
            ]
            if values:
                out[key] = fmean(values)
        return out


# Resolve the forward reference to BenchmarkConfig so Pydantic can validate
# the `config` field at runtime. Import here (not at module top) to avoid
# the circular dependency: config.py doesn't need BenchmarkResult but this
# module needs the BenchmarkConfig type for validation.
from agent_evals.benchmark.config import BenchmarkConfig  # noqa: E402

BenchmarkResult.model_rebuild()
