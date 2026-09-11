# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests locking the loader's scenario-grouping contract.

Two scenarios with the same success-checker class but different
scorer-affecting configuration must land in separate groups, each with
its own correctly-configured scorer. Two scenarios with equivalent
configuration must share a group.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from agent_evals.benchmark.loader import compile_capability, load_benchmark
from agent_evals.core.types import Score, TaskResult


def test_tool_use_scenarios_with_different_match_modes_split_into_groups(
    tmp_path: Path, default_benchmark_registries
) -> None:
    """Two tool_use scenarios with different match values produce 2 groups,
    and the second scenario's scorer behaves under its own match mode."""
    yaml_text = textwrap.dedent(
        """\
        benchmark: match-modes-v1
        platform: local
        capabilities:
          - name: doors
            scenarios:
              - id: 1
                name: lock front (superset)
                prompt: Lock the front door
                success:
                  tool_use:
                    match_args: superset
                    calls:
                      - name: lock_door
                        args: { door: front }
              - id: 2
                name: lock back (exact)
                prompt: Lock the back door
                success:
                  tool_use:
                    match_args: exact
                    calls:
                      - name: lock_door
                        args: { door: back }
        """
    )
    p = tmp_path / "mixed_match.yaml"
    p.write_text(yaml_text)
    config = load_benchmark(p, **default_benchmark_registries)

    groups = compile_capability(config.capabilities[0], config.execution)

    assert len(groups) == 2, (
        f"Two scenarios with different match modes should produce 2 groups; "
        f"got {len(groups)}"
    )

    # Locate scenario 2's group and exercise its scorer with extra args.
    # Under match_args=exact this must fail; under match_args=superset
    # (scenario 1's mode) it would pass — the assertion catches the leak.
    scenario_2_group = None
    for dataset, scorers in groups:
        if dataset[0].metadata["scenario_id"] == 2:
            scenario_2_group = (dataset, scorers)
            break

    assert scenario_2_group is not None, "Scenario 2's group not found"

    dataset, scorers = scenario_2_group
    scorer = scorers[0]
    actual = TaskResult(
        output="",
        context={
            "outputs": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "name": "lock_door",
                            "args": {"door": "back", "force": True},
                        },
                    ],
                }
            ]
        },
    )
    score = scorer(actual, dataset[0].expected)
    assert isinstance(score, Score)
    assert score.passed is False, (
        "Scenario 2 declared match_args=exact; an extra arg must fail the score. "
        "Passing means scenario 2's group is using scenario 1's match_args=superset scorer."
    )


def test_tool_use_scenarios_with_same_match_share_a_group(
    tmp_path: Path, default_benchmark_registries
) -> None:
    """Equal scorer configuration consolidates into one group.

    Guards against over-splitting: only scenarios with *different*
    scorer configs go into separate groups.
    """
    yaml_text = textwrap.dedent(
        """\
        benchmark: same-mode-v1
        platform: local
        capabilities:
          - name: doors
            scenarios:
              - id: 1
                name: lock front (subset)
                prompt: Lock the front door
                success:
                  tool_use:
                    match_args: subset
                    calls:
                      - name: lock_door
                        args: { door: front }
              - id: 2
                name: lock back (subset)
                prompt: Lock the back door
                success:
                  tool_use:
                    match_args: subset
                    calls:
                      - name: lock_door
                        args: { door: back }
        """
    )
    p = tmp_path / "same_match.yaml"
    p.write_text(yaml_text)
    config = load_benchmark(p, **default_benchmark_registries)

    groups = compile_capability(config.capabilities[0], config.execution)

    assert len(groups) == 1, (
        f"Two scenarios with the same match mode should share one group; "
        f"got {len(groups)}"
    )
    dataset, scorers = groups[0]
    assert len(dataset) == 2
    assert len(scorers) == 1


def test_three_scenarios_mixed_match_modes_split_correctly(
    tmp_path: Path, default_benchmark_registries
) -> None:
    """Interleaved scenarios accumulate by key, not by adjacency.

    Order subset/exact/subset; the two subset scenarios share a group
    even though an exact scenario sits between them in YAML order.
    """
    yaml_text = textwrap.dedent(
        """\
        benchmark: mixed-modes-v1
        platform: local
        capabilities:
          - name: doors
            scenarios:
              - id: 1
                name: lock front (subset)
                prompt: Lock the front door
                success:
                  tool_use:
                    match_args: subset
                    calls:
                      - name: lock_door
                        args: { door: front }
              - id: 2
                name: lock back (exact)
                prompt: Lock the back door
                success:
                  tool_use:
                    match_args: exact
                    calls:
                      - name: lock_door
                        args: { door: back }
              - id: 3
                name: lock side (subset)
                prompt: Lock the side door
                success:
                  tool_use:
                    match_args: subset
                    calls:
                      - name: lock_door
                        args: { door: side }
        """
    )
    p = tmp_path / "interleaved.yaml"
    p.write_text(yaml_text)
    config = load_benchmark(p, **default_benchmark_registries)

    groups = compile_capability(config.capabilities[0], config.execution)

    assert len(groups) == 2, f"Expected 2 groups (subset×2, exact×1); got {len(groups)}"

    scenario_ids_per_group = sorted(
        sorted(ex.metadata["scenario_id"] for ex in dataset) for dataset, _ in groups
    )
    assert scenario_ids_per_group == [[1, 3], [2]], (
        f"Expected scenarios 1+3 in one group (both subset) and scenario 2 alone "
        f"(exact); got {scenario_ids_per_group}"
    )


def test_split_groups_carry_correctly_configured_scorers(
    tmp_path: Path, default_benchmark_registries
) -> None:
    """Each split group's scorer behaves per its own configuration.

    Exercises both halves of the split: the superset group accepts extra
    args (reference ⊆ actual), the exact group rejects them.
    """
    yaml_text = textwrap.dedent(
        """\
        benchmark: split-groups-v1
        platform: local
        capabilities:
          - name: doors
            scenarios:
              - id: 1
                name: lock front (superset)
                prompt: Lock the front door
                success:
                  tool_use:
                    match_args: superset
                    calls:
                      - name: lock_door
                        args: { door: front }
              - id: 2
                name: lock back (exact)
                prompt: Lock the back door
                success:
                  tool_use:
                    match_args: exact
                    calls:
                      - name: lock_door
                        args: { door: back }
        """
    )
    p = tmp_path / "split_groups.yaml"
    p.write_text(yaml_text)
    config = load_benchmark(p, **default_benchmark_registries)

    groups = compile_capability(config.capabilities[0], config.execution)

    groups_by_scenario_id = {
        dataset[0].metadata["scenario_id"]: (dataset, scorers)
        for dataset, scorers in groups
    }

    # Group 1 (superset): an actual call with extra args must PASS.
    dataset_1, scorers_1 = groups_by_scenario_id[1]
    actual_with_extra = TaskResult(
        output="",
        context={
            "outputs": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": "lock_door", "args": {"door": "front", "force": True}}
                    ],
                }
            ]
        },
    )
    score_superset = scorers_1[0](actual_with_extra, dataset_1[0].expected)
    assert isinstance(score_superset, Score)
    assert score_superset.passed is True, (
        "Group 1 declared match_args=superset; extra args should pass."
    )

    # Group 2 (exact): an actual call with extra args must FAIL.
    dataset_2, scorers_2 = groups_by_scenario_id[2]
    actual_with_extra_back = TaskResult(
        output="",
        context={
            "outputs": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {"name": "lock_door", "args": {"door": "back", "force": True}}
                    ],
                }
            ]
        },
    )
    score_exact = scorers_2[0](actual_with_extra_back, dataset_2[0].expected)
    assert isinstance(score_exact, Score)
    assert score_exact.passed is False, (
        "Group 2 declared match_args=exact; extra args should fail."
    )


def test_different_checker_classes_with_equal_scorer_config_stay_separate(
    tmp_path: Path, default_benchmark_registries
) -> None:
    """Group key includes CheckSpec type name, not just scorer_config.

    ``StateCheck`` and ``ToolUseCheck`` DTOs both could plausibly produce
    ``{}`` from ``compile_check(spec).scorer_config`` (``StateCheck`` does
    today). Two scenarios using different spec types must land in different
    groups even if their ``scorer_config`` happens to be equal, because the
    spec type name disambiguates the key.
    """
    yaml_text = textwrap.dedent(
        """\
        benchmark: cross-class-v1
        platform: local
        capabilities:
          - name: mixed
            scenarios:
              - id: 1
                name: state check
                prompt: Turn on the kitchen light
                success:
                  state:
                    lights.kitchen.state: "on"
              - id: 2
                name: trajectory check
                prompt: Turn on the kitchen light
                success:
                  tool_use:
                    match_args: exact
                    calls:
                      - name: set_light
                        args: { room: kitchen, state: "on" }
        """
    )
    p = tmp_path / "cross_class.yaml"
    p.write_text(yaml_text)
    config = load_benchmark(p, **default_benchmark_registries)

    groups = compile_capability(config.capabilities[0], config.execution)

    assert len(groups) == 2, (
        f"Different checker classes must produce different groups even with "
        f"equal scorer_config(); got {len(groups)}"
    )


def test_class_name_disambiguates_when_scorer_configs_are_equal() -> None:
    """Group key splits on class name even when ``scorer_config`` matches.

    StateCheck and a hypothetical second CheckSpec type both could produce
    ``{}`` from scorer_config. Two scenarios using different spec types must
    land in different groups even if their scorer_config happens to be equal,
    because the class name disambiguates the key.
    """
    from typing import cast

    from agent_evals.benchmark.config import ScenarioSpec
    from agent_evals.benchmark.loader import _scenario_group_key
    from agent_evals.core.checks import StateCheck, ToolCall, ToolUseCheck

    class _ScenarioStub:
        def __init__(self, parsed_check_specs: list) -> None:
            self.parsed_check_specs = parsed_check_specs

    state_scenario = _ScenarioStub([StateCheck(expected_state={})])
    tool_use_scenario = _ScenarioStub(
        [
            ToolUseCheck(
                calls=(ToolCall(name="x", args={}),),
                match_calls="exact",
                match_args="exact",
            )
        ]
    )

    state_key = _scenario_group_key(cast(ScenarioSpec, state_scenario))
    tool_use_key = _scenario_group_key(cast(ScenarioSpec, tool_use_scenario))

    assert state_key != tool_use_key, (
        "Two specs of different types must produce different keys; "
        "the class name is the tiebreaker."
    )
