# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Markdown report generator for BenchmarkResult.

Pure function: BenchmarkResult in, Markdown string out. No disk writes,
no side effects, deterministic output for identical input.

Structure (Liquid-inspired, adapted for multi-capability):
    # Benchmark: <name>                    (H1 title)
    ## Run info                            (benchmark / description / config path)
    ## Score                               (headline pass rate + counts)
    ## Breakdown                           (per-dimension rollups)
    ### By capability
    ### By <dimension>                      (one per scenario dimension)
    ## Tasks                                (per-capability task tables, PASS/FAIL)
"""

import statistics
from datetime import UTC, datetime

from foundry_agent_core import mask_session_id

from agent_evals.benchmark.types import BenchmarkResult

# Keys the benchmark loader adds to every example — exclude from rollup.
_LIBRARY_METADATA_KEYS = {"scenario_id", "scenario_name", "capability", "run_index"}

# Metadata keys whose values are session identifiers (STIG V-222577).
#
# `ScenarioSpec.dimensions` is free-form by key, so a benchmark YAML may
# declare `session_id` as a slicing dimension. Every dimension value is
# rendered into this report, and `BenchmarkResult.write_report` persists the
# rendered Markdown — so the report is an output boundary in the same sense
# `results.jsonl` is.
#
# The spelling matches the key `foundry_agent_core.redact_session_ids`
# recognises, deliberately: the report's coverage and the `results.jsonl`
# redaction's coverage stay identical, including their shared limitation that
# a differently-spelled key is not caught. Inventing a broader key-matching
# policy here would drift from the canonical implementation.
_SESSION_IDENTIFIER_KEYS = frozenset({"session_id"})


def _render_metadata_value(key: str, value: object) -> str:
    """Stringify a metadata value, masking it when the key names a session ID.

    Masking is deterministic per identifier, so it is safe to mask *before*
    grouping: distinct raw values still map to distinct tokens, and the same
    raw value still lands in the same group. The rollup's shape is unchanged;
    only the label is.
    """
    text = str(value)
    return mask_session_id(text) if key in _SESSION_IDENTIFIER_KEYS else text


def _er_total_and_passed(er) -> tuple[int, int]:
    """Count total + all-scorers-passing for a single EvalResult.

    We do NOT use summary["successful_examples"] because that counts
    examples that ran without an exception, not examples where every scorer
    returned passed=True. The whole report (Score, Breakdown, Tasks) uses
    the same "all scorers passed" definition for consistency.

    Falls back to summary values ONLY when the EvalResult has no examples
    list (e.g., tests that construct results from minimal fixtures).
    """
    if er.examples:
        total = len(er.examples)
        passed = sum(
            1
            for ex in er.examples
            if ex.scores and all(s.passed for s in ex.scores.values())
        )
        return total, passed
    return (
        er.summary.get("total_examples", 0),
        er.summary.get("successful_examples", 0),
    )


def _total_and_passed(result: BenchmarkResult) -> tuple[int, int]:
    """Sum total + passed across all capabilities (all-scorers-passing)."""
    total = 0
    passed = 0
    for er in result.eval_results.values():
        t, p = _er_total_and_passed(er)
        total += t
        passed += p
    return total, passed


def _score_section(result: BenchmarkResult) -> list[str]:
    total, passed = _total_and_passed(result)
    pct = (100 * passed / total) if total else 0
    return [
        "## Score",
        "",
        f"{passed}/{total} ({pct:.0f}%)",
        "",
    ]


def _runs_per_scenario(result: BenchmarkResult) -> int:
    """Infer runs_per_scenario from the dataset's run_index metadata."""
    max_run_index = 0
    for er in result.eval_results.values():
        for ex in er.examples:
            if ex.metadata and "run_index" in ex.metadata:
                max_run_index = max(max_run_index, ex.metadata["run_index"])
    return max_run_index + 1


def _run_info_section(result: BenchmarkResult) -> list[str]:
    """Render the Run info table.

    Shows benchmark name, description, and config file path. All three
    come from BenchmarkResult.config / config_path (populated by the
    runner). Falls back to just `benchmark` when the result was
    constructed outside the runner (tests, historical data, fixtures).
    """
    params: list[tuple[str, str]] = [("benchmark", result.benchmark)]

    if result.agent_name:
        agent_label = result.agent_name
        if result.agent_version:
            agent_label = f"{agent_label} {result.agent_version}"
        params.append(("agent", agent_label))

    if result.config is not None and result.config.description:
        params.append(("description", result.config.description))

    if result.config_path is not None:
        params.append(("config", result.config_path))

    rows = "\n".join(f"| {k} | {v} |" for k, v in params)
    return [
        "## Run info",
        "",
        "| Parameter | Value |",
        "|---|---|",
        rows,
        "",
    ]


def _breakdown_by_capability(result: BenchmarkResult) -> list[str]:
    lines = ["### By capability", ""]
    for cap in sorted(result.eval_results.keys()):
        total, passed = _er_total_and_passed(result.eval_results[cap])
        pct = (100 * passed / total) if total else 0
        lines.append(f"- `{cap}` — {pct:.0f}% ({passed}/{total} examples)")
    lines.append("")
    return lines


def _discover_dimensions(result: BenchmarkResult) -> list[str]:
    """Return sorted list of unique metadata keys that are user-defined dimensions."""
    keys: set[str] = set()
    for er in result.eval_results.values():
        for ex in er.examples:
            if ex.metadata:
                keys.update(ex.metadata.keys())
    return sorted(keys - _LIBRARY_METADATA_KEYS)


def _breakdown_by_dimension(result: BenchmarkResult, dimension: str) -> list[str]:
    """Group examples by metadata[dimension], compute pass rate per group."""
    groups: dict[str, tuple[int, int]] = {}  # value -> (passed, total)
    for er in result.eval_results.values():
        for ex in er.examples:
            if not ex.metadata or dimension not in ex.metadata:
                continue
            value = _render_metadata_value(dimension, ex.metadata[dimension])
            all_passed = (
                all(score.passed for score in ex.scores.values())
                if ex.scores
                else False
            )
            passed, total = groups.get(value, (0, 0))
            groups[value] = (passed + (1 if all_passed else 0), total + 1)

    if not groups:
        return []

    lines = [f"### By {dimension}", ""]
    for value in sorted(groups.keys()):
        passed, total = groups[value]
        pct = (100 * passed / total) if total else 0
        lines.append(f"- `{value}` — {pct:.0f}% ({passed}/{total} examples)")
    lines.append("")
    return lines


def _breakdown_section(result: BenchmarkResult) -> list[str]:
    lines = ["## Breakdown", ""]
    lines += _breakdown_by_capability(result)
    for dimension in _discover_dimensions(result):
        lines += _breakdown_by_dimension(result, dimension)
    return lines


def _group_examples_by_scenario_id(examples):
    """Collapse multi-run examples to one row per scenario_id."""
    grouped: dict = {}
    for ex in examples:
        scenario_id = (ex.metadata or {}).get("scenario_id", "?")
        grouped.setdefault(scenario_id, []).append(ex)
    return grouped


def _tasks_section(result: BenchmarkResult) -> list[str]:
    dimensions = _discover_dimensions(result)
    multi_run = _runs_per_scenario(result) > 1

    lines = ["## Tasks", ""]
    for cap in sorted(result.eval_results.keys()):
        er = result.eval_results[cap]
        lines.append(f"### {cap}")
        lines.append("")
        if not er.examples:
            continue

        if multi_run:
            header_cols = ["#", "Name"] + dimensions + ["Pass%", "Std", "Time"]
        else:
            header_cols = ["#", "Name"] + dimensions + ["Result", "Time"]

        rows = []
        grouped = _group_examples_by_scenario_id(er.examples)
        # Preserve insertion order (dicts keep it on modern Python).
        for scenario_id, runs in grouped.items():
            first = runs[0]
            scenario_name = (first.metadata or {}).get("scenario_name", "<unnamed>")
            # `-` marks a dimension this scenario does not declare. Kept out of
            # _render_metadata_value so the placeholder is never itself masked.
            metadata = first.metadata or {}
            dimension_values = [
                _render_metadata_value(d, metadata[d]) if d in metadata else "-"
                for d in dimensions
            ]
            avg_duration = sum(r.duration for r in runs) / len(runs)

            if multi_run:
                pass_flags = [
                    1 if (r.scores and all(s.passed for s in r.scores.values())) else 0
                    for r in runs
                ]
                pct = 100 * sum(pass_flags) / len(pass_flags)
                std = (
                    statistics.stdev([float(f) for f in pass_flags])
                    if len(pass_flags) > 1
                    else 0.0
                )
                row = [
                    str(scenario_id),
                    scenario_name,
                    *[str(v) for v in dimension_values],
                    f"{pct:.0f}%",
                    f"{std:.2f}",
                    f"{avg_duration:.1f}s",
                ]
            else:
                passed = first.scores and all(s.passed for s in first.scores.values())
                status = "PASS" if passed else "FAIL"
                row = [
                    str(scenario_id),
                    scenario_name,
                    *[str(v) for v in dimension_values],
                    status,
                    f"{avg_duration:.1f}s",
                ]
            rows.append(row)

        # Column-width calculation and row rendering (same pattern as Task 9).
        widths = [len(col) for col in header_cols]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(val))

        def _render_row(row, widths=widths):
            return "  ".join(val.ljust(widths[i]) for i, val in enumerate(row))

        lines.append("```")
        lines.append(_render_row(header_cols))
        lines.append("  ".join("-" * w for w in widths))
        for row in rows:
            lines.append(_render_row(row))
        lines.append("```")
        lines.append("")
    return lines


def build_markdown_report(result: BenchmarkResult) -> str:
    """Return a Markdown-formatted report for a BenchmarkResult."""
    timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [f"# Benchmark: {result.benchmark}", "", f"_Generated {timestamp}_", ""]
    lines += _run_info_section(result)
    lines += _score_section(result)
    lines += _breakdown_section(result)
    lines += _tasks_section(result)
    return "\n".join(lines)
