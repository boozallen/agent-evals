# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for BenchmarkResult.to_markdown() reporter."""


def _fake_eval_result(
    total: int,
    passed: int,
    scorer_name: str = "StateMatch",
    experiment_id: str = "exp-1",
    examples: list | None = None,
):
    """Build an EvalResult with the right aggregate shape for reporter tests."""
    from agent_evals.core.types import EvalResult

    return EvalResult(
        experiment_id=experiment_id,
        experiment_url=f"file:///tmp/{experiment_id}",
        platform="local",
        scores={scorer_name: passed / total if total else 0.0},
        pass_rates={scorer_name: passed / total if total else 0.0},
        examples=examples or [],
        summary={
            "total_examples": total,
            "successful_examples": passed,
            "failed_examples": total - passed,
        },
    )


class TestToMarkdownExists:
    def test_method_returns_string(self):
        from agent_evals.benchmark.types import BenchmarkResult

        result = BenchmarkResult(benchmark="empty-test", eval_results={})
        out = result.to_markdown()
        assert isinstance(out, str)
        assert len(out) > 0


class TestScoreSection:
    def test_score_has_passed_over_total(self):
        from agent_evals.benchmark.types import BenchmarkResult

        result = BenchmarkResult(
            benchmark="b",
            eval_results={"cap1": _fake_eval_result(total=10, passed=8)},
        )
        md = result.to_markdown()
        assert "## Score" in md
        # Should show 8/10 passed, since 80% of capability examples passed.
        assert "8/10" in md or "8 / 10" in md
        assert "80" in md  # percentage appears

    def test_score_across_multiple_capabilities(self):
        from agent_evals.benchmark.types import BenchmarkResult

        result = BenchmarkResult(
            benchmark="b",
            eval_results={
                "cap1": _fake_eval_result(total=10, passed=8),
                "cap2": _fake_eval_result(total=5, passed=5, experiment_id="exp-2"),
            },
        )
        md = result.to_markdown()
        # 13 passed of 15 total across both capabilities.
        assert "13/15" in md or "13 / 15" in md

    def test_score_counts_only_examples_with_all_scorers_passing(self):
        """Score must agree with Failures section: an example with ANY failing
        scorer does NOT count as passed, even if it didn't crash.

        Regression guard: earlier version of _total_and_passed sourced
        `passed` from summary["successful_examples"], which counts
        examples that ran without exception — so a test that crashed
        zero scorers but got passed=False on one of them was wrongly
        reported as a pass.
        """
        from agent_evals.benchmark.types import BenchmarkResult
        from agent_evals.core.types import EvalExample, EvalResult, Score

        def _example(all_scorers_pass: bool):
            return EvalExample(
                input={"prompt": "p"},
                output="",
                expected="",
                scores={
                    "StateMatch": Score(
                        name="StateMatch",
                        value=1.0,
                        passed=True,
                        reasoning=None,
                        metadata={},
                    ),
                    "TrajectorySubsetMatch": Score(
                        name="TrajectorySubsetMatch",
                        value=1.0 if all_scorers_pass else 0.0,
                        passed=all_scorers_pass,
                        reasoning=None,
                        metadata={},
                    ),
                },
                metadata={"scenario_id": 1, "scenario_name": "t", "capability": "c"},
                duration=0.1,
            )

        # 2 examples; 1 had a failing trajectory scorer but didn't crash.
        # summary.successful_examples is 2 (nothing crashed) but passed=1.
        er = EvalResult(
            experiment_id="exp",
            experiment_url="file:///tmp/exp",
            platform="local",
            scores={"StateMatch": 1.0, "TrajectorySubsetMatch": 0.5},
            pass_rates={"StateMatch": 1.0, "TrajectorySubsetMatch": 0.5},
            examples=[_example(True), _example(False)],
            summary={
                "total_examples": 2,
                "successful_examples": 2,
                "failed_examples": 0,
            },
        )
        result = BenchmarkResult(benchmark="b", eval_results={"c": er})
        md = result.to_markdown()
        # Score should report 1/2 (one example had a failing scorer), not 2/2.
        assert "1/2" in md


class TestRunInfoSection:
    def test_minimal_run_info_shows_benchmark_name(self):
        """When BenchmarkResult has no config attached (tests, historical
        data), Run info still shows at least the benchmark name."""
        from agent_evals.benchmark.types import BenchmarkResult

        result = BenchmarkResult(
            benchmark="home-v1",
            eval_results={"lights": _fake_eval_result(total=4, passed=4)},
        )
        md = result.to_markdown()
        assert "## Run info" in md
        assert "home-v1" in md
        # No description or config rows when they aren't available.
        assert "description" not in md.lower() or "description" in "".join(
            line for line in md.split("\n") if "|" not in line
        )

    def test_run_info_shows_agent_identity_when_present(self):
        """agent_name + agent_version surface in the Run info table."""
        from agent_evals.benchmark.types import BenchmarkResult

        result = BenchmarkResult(
            benchmark="home-v1",
            agent_name="home-langgraph",
            agent_version="0.1.0",
            eval_results={"lights": _fake_eval_result(total=1, passed=1)},
        )
        md = result.to_markdown()
        assert "home-langgraph 0.1.0" in md

    def test_run_info_handles_agent_name_without_version(self):
        """agent_name alone shows without a trailing version."""
        from agent_evals.benchmark.types import BenchmarkResult

        result = BenchmarkResult(
            benchmark="home-v1",
            agent_name="home-agent",
            eval_results={"lights": _fake_eval_result(total=1, passed=1)},
        )
        md = result.to_markdown()
        assert "home-agent" in md
        # No trailing space / None in the agent row
        agent_line = next(
            line for line in md.split("\n") if "agent" in line and "home-agent" in line
        )
        assert "None" not in agent_line

    def test_run_info_omits_agent_row_when_name_is_empty(self):
        """Default `Agent.name = ""` should not produce an agent row."""
        from agent_evals.benchmark.types import BenchmarkResult

        result = BenchmarkResult(
            benchmark="home-v1",
            agent_name="",
            agent_version=None,
            eval_results={"lights": _fake_eval_result(total=1, passed=1)},
        )
        md = result.to_markdown()
        # The Run info table should not have an agent row when name is empty.
        run_info_section = md.split("## Score")[0]
        assert "| agent |" not in run_info_section

    def test_full_run_info_with_config_and_path(self, default_benchmark_registries):
        """With a BenchmarkConfig + config_path, the table shows benchmark,
        description, and the config file path — all three sourced from
        what the consumer actually wrote."""
        from agent_evals.adapters.platforms.local import LocalConfig
        from agent_evals.benchmark.config import (
            BenchmarkConfig,
        )
        from agent_evals.benchmark.types import BenchmarkResult

        config = BenchmarkConfig.model_validate(
            {
                "benchmark": "home-v1",
                "description": "Home-automation regression suite",
                "platform": LocalConfig(),
                "capabilities": [
                    {
                        "name": "lights",
                        "scenarios": [
                            {
                                "id": 1,
                                "name": "t",
                                "prompt": "p",
                                "success": {"state": {"x": 1}},
                            }
                        ],
                    }
                ],
            },
            context=default_benchmark_registries,
        )
        result = BenchmarkResult(
            benchmark="home-v1",
            config=config,
            config_path="benchmark/home-v1.yaml",
            eval_results={"lights": _fake_eval_result(total=1, passed=1)},
        )
        md = result.to_markdown()
        assert "home-v1" in md
        assert "Home-automation regression suite" in md
        assert "benchmark/home-v1.yaml" in md


class TestBreakdownByCapability:
    def test_breakdown_shows_each_capability_pass_rate(self):
        from agent_evals.benchmark.types import BenchmarkResult

        result = BenchmarkResult(
            benchmark="b",
            eval_results={
                "lights": _fake_eval_result(total=4, passed=4),
                "doors": _fake_eval_result(total=3, passed=2),
            },
        )
        md = result.to_markdown()
        assert "## Breakdown" in md
        assert "### By capability" in md
        # Both capabilities appear with pass rate + count
        assert "lights" in md
        assert "doors" in md
        assert "100" in md  # lights pass rate
        assert "67" in md  # doors pass rate (2/3)

    def test_breakdown_sorts_capabilities_alphabetically(self):
        from agent_evals.benchmark.types import BenchmarkResult

        result = BenchmarkResult(
            benchmark="b",
            eval_results={
                "zeta": _fake_eval_result(total=1, passed=1, experiment_id="z"),
                "alpha": _fake_eval_result(total=1, passed=1, experiment_id="a"),
            },
        )
        md = result.to_markdown()
        assert md.index("alpha") < md.index("zeta")

    def test_breakdown_counts_all_scorers_passing_not_successful_examples(self):
        """Breakdown must agree with Score on what 'passed' means: all
        scorers passed, not 'example didn't crash'.

        Regression: dogfood caught that By capability showed 100% while
        Score showed 90% — _breakdown_by_capability was reading
        summary['successful_examples'] (ran without error) instead of
        walking examples to check all-scorers-passed, same bug we
        already fixed in _total_and_passed.
        """
        from agent_evals.benchmark.types import BenchmarkResult
        from agent_evals.core.types import EvalExample, EvalResult, Score

        def _example(all_scorers_pass: bool):
            return EvalExample(
                input={"prompt": "p"},
                output="",
                expected="",
                scores={
                    "StateMatch": Score(
                        name="StateMatch",
                        value=1.0,
                        passed=True,
                        reasoning=None,
                        metadata={},
                    ),
                    "TrajectorySubsetMatch": Score(
                        name="TrajectorySubsetMatch",
                        value=1.0 if all_scorers_pass else 0.0,
                        passed=all_scorers_pass,
                        reasoning=None,
                        metadata={},
                    ),
                },
                metadata={
                    "scenario_id": 1,
                    "scenario_name": "t",
                    "capability": "doors",
                },
                duration=0.1,
            )

        # 3 examples, 1 has a failing scorer but ran cleanly.
        # successful_examples (runtime success) = 3; all-scorers-pass = 2.
        er = EvalResult(
            experiment_id="exp",
            experiment_url="file:///tmp/exp",
            platform="local",
            scores={"StateMatch": 1.0, "TrajectorySubsetMatch": 0.67},
            pass_rates={"StateMatch": 1.0, "TrajectorySubsetMatch": 0.67},
            examples=[_example(True), _example(False), _example(True)],
            summary={
                "total_examples": 3,
                "successful_examples": 3,  # all ran without error
                "failed_examples": 0,
            },
        )
        result = BenchmarkResult(benchmark="b", eval_results={"doors": er})
        md = result.to_markdown()
        # Breakdown should show 2/3 (67%), NOT 3/3 (100%).
        assert "2/3" in md
        assert "67%" in md
        # Explicitly assert 3/3 is NOT in the capability breakdown line.
        # (It may appear elsewhere — e.g., in a task table row with duration —
        # so restrict to the breakdown block.)
        breakdown_start = md.index("### By capability")
        breakdown_end = md.index("##", breakdown_start + 1)
        breakdown_block = md[breakdown_start:breakdown_end]
        assert "3/3" not in breakdown_block


class TestBreakdownByDimensionAxis:
    def _example(
        self,
        passed: bool,
        metadata: dict,
        scorer_name: str = "StateMatch",
    ):
        """Build an EvalExample with the right shape for reporter tests."""
        from agent_evals.core.types import EvalExample, Score

        return EvalExample(
            input={"prompt": "p"},
            output="",
            expected="",
            scores={
                scorer_name: Score(
                    name=scorer_name,
                    value=1.0 if passed else 0.0,
                    passed=passed,
                    reasoning=None,
                    metadata={},
                )
            },
            metadata=metadata,
            duration=0.1,
        )

    def test_user_dimension_produces_axis_rollup(self):
        from agent_evals.benchmark.types import BenchmarkResult

        examples = [
            self._example(
                True, {"scenario_id": 1, "capability": "c", "phrasing": "imperative"}
            ),
            self._example(
                False, {"scenario_id": 2, "capability": "c", "phrasing": "indirect"}
            ),
            self._example(
                True, {"scenario_id": 3, "capability": "c", "phrasing": "imperative"}
            ),
        ]
        er = _fake_eval_result(total=3, passed=2, examples=examples)
        result = BenchmarkResult(benchmark="b", eval_results={"c": er})
        md = result.to_markdown()

        assert "### By phrasing" in md
        # imperative: 2/2 passed = 100%; indirect: 0/1 passed = 0%
        assert "imperative" in md
        assert "indirect" in md
        assert "100" in md
        # 0% present — could be 0 or "0%"

    def test_multiple_dimension_axes(self):
        # Cover phrasing + depth — dynamic discovery should catch both.
        from agent_evals.benchmark.types import BenchmarkResult

        examples = [
            self._example(
                True,
                {
                    "scenario_id": 1,
                    "capability": "c",
                    "phrasing": "imp",
                    "depth": "lit",
                },
            ),
            self._example(
                False,
                {
                    "scenario_id": 2,
                    "capability": "c",
                    "phrasing": "ind",
                    "depth": "sem",
                },
            ),
        ]
        er = _fake_eval_result(total=2, passed=1, examples=examples)
        result = BenchmarkResult(benchmark="b", eval_results={"c": er})
        md = result.to_markdown()

        assert "### By phrasing" in md
        assert "### By depth" in md

    def test_library_added_keys_not_treated_as_axes(self):
        """scenario_id, scenario_name, capability, run_index should NOT appear as rollup axes."""
        from agent_evals.benchmark.types import BenchmarkResult

        examples = [
            self._example(
                True,
                {
                    "scenario_id": 1,
                    "scenario_name": "n",
                    "capability": "c",
                    "run_index": 0,
                },
            ),
        ]
        er = _fake_eval_result(total=1, passed=1, examples=examples)
        result = BenchmarkResult(benchmark="b", eval_results={"c": er})
        md = result.to_markdown()

        # Library-added keys should not produce rollup sections.
        assert "### By scenario_id" not in md
        assert "### By scenario_name" not in md
        assert "### By run_index" not in md


class TestTasksSectionSingleRun:
    def _example(self, scenario_id, scenario_name, passed, dimensions=None):
        from agent_evals.core.types import EvalExample, Score

        metadata = {
            "scenario_id": scenario_id,
            "scenario_name": scenario_name,
            "capability": "c",
            "run_index": 0,
        }
        if dimensions:
            metadata.update(dimensions)
        return EvalExample(
            input={"prompt": f"prompt-{scenario_id}"},
            output="",
            expected="",
            scores={
                "StateMatch": Score(
                    name="StateMatch",
                    value=1.0 if passed else 0.0,
                    passed=passed,
                    reasoning=None,
                    metadata={},
                )
            },
            metadata=metadata,
            duration=0.5,
        )

    def test_per_capability_task_sections(self):
        from agent_evals.benchmark.types import BenchmarkResult

        ex_lights = [
            self._example(1, "kitchen on", True),
            self._example(2, "bedroom off", False),
        ]
        ex_doors = [self._example(5, "lock front", True)]
        result = BenchmarkResult(
            benchmark="b",
            eval_results={
                "lights": _fake_eval_result(total=2, passed=1, examples=ex_lights),
                "doors": _fake_eval_result(
                    total=1, passed=1, examples=ex_doors, experiment_id="exp2"
                ),
            },
        )
        md = result.to_markdown()

        assert "## Tasks" in md
        assert "### lights" in md
        assert "### doors" in md

    def test_task_table_shows_pass_fail(self):
        from agent_evals.benchmark.types import BenchmarkResult

        examples = [
            self._example(1, "k on", True),
            self._example(2, "b off", False),
        ]
        result = BenchmarkResult(
            benchmark="b",
            eval_results={"c": _fake_eval_result(total=2, passed=1, examples=examples)},
        )
        md = result.to_markdown()

        assert "PASS" in md
        assert "FAIL" in md

    def test_task_table_includes_user_dimensions_as_columns(self):
        from agent_evals.benchmark.types import BenchmarkResult

        examples = [
            self._example(
                1,
                "k on",
                True,
                dimensions={"phrasing": "imperative", "depth": "literal"},
            ),
        ]
        result = BenchmarkResult(
            benchmark="b",
            eval_results={"c": _fake_eval_result(total=1, passed=1, examples=examples)},
        )
        md = result.to_markdown()

        assert "imperative" in md
        assert "literal" in md


class TestTasksSectionMultiRun:
    def test_multi_run_collapses_to_pass_pct_and_std(self):
        from agent_evals.benchmark.types import BenchmarkResult
        from agent_evals.core.types import EvalExample, Score

        # Three runs of the same test: 2 pass, 1 fail -> 67% pass rate
        def _ex(run_idx, passed):
            return EvalExample(
                input={"prompt": "p"},
                output="",
                expected="",
                scores={
                    "StateMatch": Score(
                        name="StateMatch",
                        value=1.0 if passed else 0.0,
                        passed=passed,
                        reasoning=None,
                        metadata={},
                    )
                },
                metadata={
                    "scenario_id": 1,
                    "scenario_name": "flaky",
                    "capability": "c",
                    "run_index": run_idx,
                },
                duration=0.1,
            )

        examples = [_ex(0, True), _ex(1, False), _ex(2, True)]
        er = _fake_eval_result(total=3, passed=2, examples=examples)
        result = BenchmarkResult(benchmark="b", eval_results={"c": er})
        md = result.to_markdown()

        # One row per test (not three).
        tasks_idx = md.index("## Tasks")
        tasks_section = md[tasks_idx:]
        assert tasks_section.count("flaky") == 1

        # Pass% and Std columns should be present.
        assert "Pass%" in tasks_section or "Pass %" in tasks_section
        assert "Std" in tasks_section
        # 2/3 = 67%
        assert "67" in tasks_section


class TestDeterministicOutput:
    def test_same_input_same_output_except_timestamp(self):
        import re

        from agent_evals.benchmark.types import BenchmarkResult

        result = BenchmarkResult(
            benchmark="b",
            eval_results={"c": _fake_eval_result(total=2, passed=1)},
        )
        a = result.to_markdown()
        b = result.to_markdown()

        # Strip the timestamp line from both.
        def strip_ts(s: str) -> str:
            return re.sub(r"_Generated [^_]+_", "_Generated ..._", s)

        assert strip_ts(a) == strip_ts(b)


class TestGeneratedTimestamp:
    def test_timestamp_is_utc_and_labeled(self):
        """The Generated line must be UTC and say so."""
        import re
        from datetime import UTC, datetime, timedelta

        from agent_evals.benchmark.types import BenchmarkResult

        before = datetime.now(UTC)
        result = BenchmarkResult(benchmark="b", eval_results={})
        out = result.to_markdown()
        after = datetime.now(UTC)

        match = re.search(r"_Generated (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) UTC_", out)
        assert match is not None, out

        parsed = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S").replace(
            tzinfo=UTC
        )
        assert before - timedelta(seconds=1) <= parsed <= after + timedelta(seconds=1)
