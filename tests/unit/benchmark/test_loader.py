# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark YAML loader and capability compilation."""

import textwrap
from pathlib import Path

import pytest

SAMPLE_YAML = textwrap.dedent(
    """
    benchmark: home-v1
    platform:
      name: local
      experiment: t
    execution:
      runs_per_scenario: 2
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: kitchen on
            prompt: Turn on the kitchen light
            dimensions: { phrasing: imperative, depth: literal }
            success:
              state:
                lights.kitchen.state: "on"
      - name: status
        scenarios:
          - id: 57
            name: all status
            prompt: Check all devices
            dimensions: { phrasing: imperative, depth: literal }
            success:
              tool_use:
                calls:
                  - name: get_device_status
                    args: { device_type: all }
    """
).strip()


# A heterogeneous capability with three distinct success-block shapes:
# state-only, tool_use-only, and both. Used by the grouping tests.
HETEROGENEOUS_YAML = textwrap.dedent(
    """
    benchmark: mixed-v1
    platform:
      name: local
      experiment: t
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: state only
            prompt: Turn on the kitchen light
            success:
              state:
                lights.kitchen.state: "on"
          - id: 2
            name: trajectory only
            prompt: Check status
            success:
              tool_use:
                calls:
                  - name: get_status
                    args: {}
          - id: 3
            name: both
            prompt: Turn on and verify
            success:
              state:
                lights.kitchen.state: "on"
              tool_use:
                calls:
                  - name: turn_on
                    args: { device: kitchen }
    """
).strip()


HOMOGENEOUS_YAML = textwrap.dedent(
    """
    benchmark: homogeneous-v1
    platform:
      name: local
      experiment: t
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: a
            prompt: Turn on kitchen
            success:
              state: { lights.kitchen.state: "on" }
          - id: 2
            name: b
            prompt: Turn off kitchen
            success:
              state: { lights.kitchen.state: "off" }
          - id: 3
            name: c
            prompt: Turn on bedroom
            success:
              state: { lights.bedroom.state: "on" }
    """
).strip()


class TestLoadBenchmark:
    def test_load_benchmark_from_path(
        self, tmp_path: Path, default_benchmark_registries
    ):
        from agent_evals.benchmark.loader import load_benchmark

        p = tmp_path / "b.yaml"
        p.write_text(SAMPLE_YAML)
        config = load_benchmark(p, **default_benchmark_registries)
        assert config.benchmark == "home-v1"
        assert len(config.capabilities) == 2
        assert config.execution.runs_per_scenario == 2


# ---------------------------------------------------------------------------
# Multi-file composition (#177)
#
# These tests cover the load-time URI-resolution pass added by spec 021.
# Single-file YAML continues to parse byte-identically; the new path
# only fires when a list-of-string entry uses a registered URI scheme.
# ---------------------------------------------------------------------------


_PARENT_PROLOGUE = textwrap.dedent(
    """
    benchmark: multi-file-v1
    platform:
      name: local
      experiment: t
    execution:
      runs_per_scenario: 1
    capabilities:
    """
).strip()


_LIGHTS_CHILD = textwrap.dedent(
    """
    name: lights
    scenarios:
      - id: 1
        name: kitchen on
        prompt: Turn on the kitchen light
        success:
          state:
            lights.kitchen.state: "on"
    """
).strip()


_DOORS_CHILD = textwrap.dedent(
    """
    name: doors
    scenarios:
      - id: 1
        name: lock front
        prompt: Lock the front door
        success:
          state:
            doors.front: locked
    """
).strip()


class TestLoadBenchmarkMultiFile:
    def test_single_file_unchanged(self, tmp_path: Path, default_benchmark_registries):
        """Existing single-file YAML continues to parse byte-identically."""
        from agent_evals.benchmark.loader import load_benchmark

        p = tmp_path / "b.yaml"
        p.write_text(SAMPLE_YAML)
        config = load_benchmark(p, **default_benchmark_registries)
        # Same assertions as the legacy TestLoadBenchmark — regression guard.
        assert config.benchmark == "home-v1"
        assert len(config.capabilities) == 2
        assert config.capabilities[0].name == "lights"
        assert config.capabilities[1].name == "status"

    def test_resolves_capability_include(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """A `file://` reference under capabilities: resolves to the file's content."""
        from agent_evals.benchmark.loader import load_benchmark

        (tmp_path / "lights.yaml").write_text(_LIGHTS_CHILD)
        parent = tmp_path / "parent.yaml"
        parent.write_text(f"{_PARENT_PROLOGUE}\n  - file://lights.yaml\n")

        config = load_benchmark(parent, **default_benchmark_registries)
        assert len(config.capabilities) == 1
        assert config.capabilities[0].name == "lights"
        assert len(config.capabilities[0].scenarios) == 1
        assert config.capabilities[0].scenarios[0].name == "kitchen on"

    def test_resolves_multiple_includes(
        self, tmp_path: Path, default_benchmark_registries
    ):
        from agent_evals.benchmark.loader import load_benchmark

        (tmp_path / "lights.yaml").write_text(_LIGHTS_CHILD)
        (tmp_path / "doors.yaml").write_text(_DOORS_CHILD)
        parent = tmp_path / "parent.yaml"
        parent.write_text(
            f"{_PARENT_PROLOGUE}\n  - file://lights.yaml\n  - file://doors.yaml\n"
        )

        config = load_benchmark(parent, **default_benchmark_registries)
        assert [cap.name for cap in config.capabilities] == ["lights", "doors"]

    def test_mixed_inline_and_include_preserves_order(
        self, tmp_path: Path, default_benchmark_registries
    ):
        from agent_evals.benchmark.loader import load_benchmark

        (tmp_path / "lights.yaml").write_text(_LIGHTS_CHILD)
        parent = tmp_path / "parent.yaml"
        # Inline capability appears first, file include second.
        parent.write_text(
            f"{_PARENT_PROLOGUE}\n"
            "  - name: inline-cap\n"
            "    scenarios:\n"
            "      - id: 99\n"
            "        name: inline-scenario\n"
            "        prompt: p\n"
            "        success:\n"
            "          state:\n"
            "            x.y: z\n"
            "  - file://lights.yaml\n"
        )

        config = load_benchmark(parent, **default_benchmark_registries)
        assert [cap.name for cap in config.capabilities] == ["inline-cap", "lights"]

    def test_relative_path_resolves_against_parent_directory(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """`file://child.yaml` resolves against the parent file's directory, not CWD."""
        from agent_evals.benchmark.loader import load_benchmark

        subdir = tmp_path / "nested"
        subdir.mkdir()
        (subdir / "lights.yaml").write_text(_LIGHTS_CHILD)
        parent = subdir / "parent.yaml"
        parent.write_text(f"{_PARENT_PROLOGUE}\n  - file://lights.yaml\n")

        # CWD is unrelated to tmp_path; resolution must use the parent's dir.
        config = load_benchmark(parent, **default_benchmark_registries)
        assert config.capabilities[0].name == "lights"

    def test_attributes_child_read_errors_to_uri(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """A missing child file raises with BOTH child and parent URIs."""
        from agent_evals.benchmark.includes import BenchmarkConfigReadError
        from agent_evals.benchmark.loader import load_benchmark

        parent = tmp_path / "parent.yaml"
        parent.write_text(f"{_PARENT_PROLOGUE}\n  - file://nonexistent.yaml\n")

        with pytest.raises(BenchmarkConfigReadError) as exc:
            load_benchmark(parent, **default_benchmark_registries)
        msg = str(exc.value)
        assert "nonexistent.yaml" in msg
        # Parent URI must appear so users see both ends of the broken link.
        assert "parent.yaml" in msg
        assert isinstance(exc.value.__cause__, FileNotFoundError)

    def test_attributes_child_yaml_errors_to_uri(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """Malformed child YAML raises with BOTH child and parent URIs."""
        from agent_evals.benchmark.includes import BenchmarkConfigReadError
        from agent_evals.benchmark.loader import load_benchmark

        (tmp_path / "broken.yaml").write_text("name: lights\nscenarios: [oops:\n")
        parent = tmp_path / "parent.yaml"
        parent.write_text(f"{_PARENT_PROLOGUE}\n  - file://broken.yaml\n")

        with pytest.raises(BenchmarkConfigReadError) as exc:
            load_benchmark(parent, **default_benchmark_registries)
        msg = str(exc.value)
        assert "broken.yaml" in msg
        assert "parent.yaml" in msg

    def test_attributes_parent_yaml_errors_to_parent_uri(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """Malformed parent YAML raises with the parent URI in the message.

        Symmetric with child-side wrapping — parent and child errors
        attribute the same way regardless of which file is wrong.
        """
        from agent_evals.benchmark.loader import load_benchmark

        parent = tmp_path / "broken-parent.yaml"
        parent.write_text("benchmark: x\ncapabilities: [oops:\n")

        with pytest.raises(ValueError) as exc:
            load_benchmark(parent, **default_benchmark_registries)
        assert "broken-parent.yaml" in str(exc.value)

    def test_uri_string_in_non_capabilities_field_passes_through(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """Resolution walks ONLY the top-level capabilities: list.

        URI strings inside scenario fields (prompt:, dimensions:,
        success: payloads, etc.) must NOT be fetched — they're string
        values, not include directives. A future spec that adds
        scenario-level includes would change this; this test pins
        the v1 contract so the change is forced through review.
        """
        from pydantic import ValidationError

        from agent_evals.benchmark.loader import load_benchmark

        # An inline capability with a scenario whose `prompt:` contains
        # a `file://` literal. The resolver must NOT try to fetch it.
        parent = tmp_path / "parent.yaml"
        parent.write_text(
            f"{_PARENT_PROLOGUE}\n"
            "  - name: literal-uri-prompt\n"
            "    scenarios:\n"
            "      - id: 1\n"
            "        name: a scenario whose prompt mentions file://\n"
            "        prompt: please read file:///etc/passwd\n"
            "        success:\n"
            "          state:\n"
            "            x.y: z\n"
        )
        # Resolution must complete (validation succeeds); the prompt
        # arrives at the agent verbatim, never fetched.
        try:
            config = load_benchmark(parent, **default_benchmark_registries)
        except ValidationError as exc:
            pytest.fail(f"valid YAML failed validation: {exc}")
        cap = config.capabilities[0]
        assert cap.scenarios[0].prompt == "please read file:///etc/passwd"

    def test_resolution_happens_before_validation(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """URI strings never reach Pydantic.

        If validation ran before resolution, a `file://...` string in the
        capabilities list would surface as a Pydantic type error
        ("expected dict, got str") instead of resolving to file content.
        Catching the right kind of error proves the order.
        """
        from pydantic import ValidationError

        from agent_evals.benchmark.loader import load_benchmark

        (tmp_path / "lights.yaml").write_text(_LIGHTS_CHILD)
        parent = tmp_path / "parent.yaml"
        parent.write_text(f"{_PARENT_PROLOGUE}\n  - file://lights.yaml\n")

        # Should NOT raise ValidationError — the URI is resolved first.
        try:
            load_benchmark(parent, **default_benchmark_registries)
        except ValidationError as exc:
            pytest.fail(
                f"URI strings reached Pydantic; resolution must happen "
                f"before validation. Got: {exc}"
            )


class TestCompileCapability:
    def test_state_capability_replicates_each_scenario_by_runs_per_scenario(
        self, tmp_path: Path, default_benchmark_registries
    ):
        from agent_evals.benchmark.loader import (
            compile_capability,
            load_benchmark,
        )

        p = tmp_path / "b.yaml"
        p.write_text(SAMPLE_YAML)
        config = load_benchmark(p, **default_benchmark_registries)
        lights = config.capabilities[0]

        groups = compile_capability(lights, config.execution)
        # Homogeneous capability -> single group.
        assert len(groups) == 1
        dataset, scorers = groups[0]

        # runs_per_scenario=2, one scenario -> dataset length 2
        assert len(dataset) == 2
        first = dataset[0]
        assert first.input["prompt"] == "Turn on the kitchen light"
        assert first.expected is not None and first.expected.context is not None
        assert first.expected.context["expected_state"] == {
            "lights.kitchen.state": "on"
        }
        assert first.metadata["capability"] == "lights"
        assert first.metadata["phrasing"] == "imperative"
        assert first.metadata["depth"] == "literal"
        assert first.metadata["scenario_id"] == 1
        assert first.metadata["run_index"] in {0, 1}
        assert {
            dataset[0].metadata["run_index"],
            dataset[1].metadata["run_index"],
        } == {0, 1}
        assert len(scorers) == 1

    def test_trajectory_capability_compiles_reference_outputs(
        self, tmp_path: Path, default_benchmark_registries
    ):
        from agent_evals.benchmark.loader import (
            compile_capability,
            load_benchmark,
        )

        p = tmp_path / "b.yaml"
        p.write_text(SAMPLE_YAML)
        config = load_benchmark(p, **default_benchmark_registries)
        status = config.capabilities[1]

        groups = compile_capability(status, config.execution)
        assert len(groups) == 1
        dataset, _scorers = groups[0]
        assert len(dataset) == 2
        first = dataset[0]
        assert first.expected is not None and first.expected.context is not None
        assert first.expected.context["reference_outputs"] == [
            {
                "role": "assistant",
                "tool_calls": [
                    {"name": "get_device_status", "args": {"device_type": "all"}}
                ],
            }
        ]

    def test_compile_capability_returns_groups(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """A heterogeneous capability compiles into one group per scorer-set."""
        from agent_evals.benchmark.loader import (
            compile_capability,
            load_benchmark,
        )

        p = tmp_path / "mix.yaml"
        p.write_text(HETEROGENEOUS_YAML)
        config = load_benchmark(p, **default_benchmark_registries)
        lights = config.capabilities[0]

        groups = compile_capability(lights, config.execution)
        # Three distinct success-block shapes -> three groups.
        assert len(groups) == 3

        # Map each group to the set of scorer-factory names it carries.
        # Scorer factories return an inner closure with __qualname__
        # like "StateMatch.<locals>.scorer"; the prefix before "." is
        # the factory's name and uniquely identifies the scorer.
        def _factory_name(scorer):
            qn = getattr(scorer, "__qualname__", "")
            return qn.split(".", 1)[0] if qn else repr(scorer)

        scorer_name_sets: list[set[str]] = []
        for dataset, scorers in groups:
            assert len(dataset) == 1, (
                "Each scenario in this fixture is unique-shape, so each group "
                "should carry exactly one scenario."
            )
            scorer_name_sets.append({_factory_name(s) for s in scorers})

        # We don't lock the order of groups, but we lock the multiset of
        # scorer-set shapes. There must be exactly one group with just
        # StateMatch, one with just ToolCallExactMatch, and one with both.
        assert {"StateMatch"} in scorer_name_sets
        assert {"ToolCallExactMatch"} in scorer_name_sets
        assert {"StateMatch", "ToolCallExactMatch"} in scorer_name_sets

    def test_expected_context_from_success_checkers(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """Each scenario's ExpectedResult.context only carries its checkers' keys."""
        from agent_evals.benchmark.loader import (
            compile_capability,
            load_benchmark,
        )

        p = tmp_path / "mix.yaml"
        p.write_text(HETEROGENEOUS_YAML)
        config = load_benchmark(p, **default_benchmark_registries)
        lights = config.capabilities[0]

        groups = compile_capability(lights, config.execution)

        # Build a {scenario_id: context} map across all groups.
        ctx_by_id: dict[int, dict] = {}
        for dataset, _scorers in groups:
            for ex in dataset:
                assert ex.expected is not None and ex.expected.context is not None
                ctx_by_id[ex.metadata["scenario_id"]] = ex.expected.context

        # Scenario 1: state checker only -> only `expected_state` context key.
        assert set(ctx_by_id[1].keys()) == {"expected_state"}
        # Scenario 2: tool_use only -> only `reference_outputs` key.
        assert set(ctx_by_id[2].keys()) == {"reference_outputs"}
        # Scenario 3: both -> both keys.
        assert set(ctx_by_id[3].keys()) == {"expected_state", "reference_outputs"}

    def test_homogeneous_capability_single_group(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """All scenarios with identical success-block shape compile into one group."""
        from agent_evals.benchmark.loader import (
            compile_capability,
            load_benchmark,
        )

        p = tmp_path / "homo.yaml"
        p.write_text(HOMOGENEOUS_YAML)
        config = load_benchmark(p, **default_benchmark_registries)
        lights = config.capabilities[0]

        groups = compile_capability(lights, config.execution)
        assert len(groups) == 1
        dataset, scorers = groups[0]
        assert len(dataset) == 3
        # Single success-checker class -> single scorer.
        assert len(scorers) == 1


# ---------------------------------------------------------------------------
# Per-scenario preconditions: carried through compile_capability into
# ExampleData.input["preconditions"], orthogonal to scorer-set grouping.
# ---------------------------------------------------------------------------


PRECONDITION_YAML = textwrap.dedent(
    """
    benchmark: precond-v1
    platform:
      name: local
      experiment: t
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: kitchen off (precondition seeded on)
            prompt: Turn off the kitchen light
            precondition:
              state:
                lights.kitchen.state: "on"
            success:
              state:
                lights.kitchen.state: "off"
    """
).strip()


PRECONDITION_GROUPING_YAML = textwrap.dedent(
    """
    benchmark: precond-grouping-v1
    platform:
      name: local
      experiment: t
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: same success, precondition A
            prompt: Turn off the kitchen light
            precondition:
              state:
                lights.kitchen.state: "on"
            success:
              state:
                lights.kitchen.state: "off"
          - id: 2
            name: same success, precondition B
            prompt: Turn off the kitchen light again
            precondition:
              state:
                lights.kitchen.brightness: 80
            success:
              state:
                lights.kitchen.state: "off"
    """
).strip()


class TestCompileCapabilityPreconditions:
    def test_compile_capability_carries_preconditions(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """A scenario's parsed preconditions land on every replicated ExampleData.input."""
        from agent_evals.adapters.preconditions.state import StatePreconditionApplier
        from agent_evals.benchmark.loader import (
            compile_capability,
            load_benchmark,
        )

        p = tmp_path / "precond.yaml"
        p.write_text(PRECONDITION_YAML)
        config = load_benchmark(p, **default_benchmark_registries)
        lights = config.capabilities[0]

        groups = compile_capability(lights, config.execution)
        assert len(groups) == 1
        dataset, _scorers = groups[0]
        assert len(dataset) >= 1

        first = dataset[0]
        assert "preconditions" in first.input
        preconds = first.input["preconditions"]
        assert isinstance(preconds, list)
        assert len(preconds) == 1
        assert isinstance(preconds[0], StatePreconditionApplier)

    def test_compile_capability_empty_preconditions_default(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """A scenario without ``precondition:`` produces an empty preconditions list."""
        from agent_evals.benchmark.loader import (
            compile_capability,
            load_benchmark,
        )

        p = tmp_path / "no_precond.yaml"
        p.write_text(SAMPLE_YAML)
        config = load_benchmark(p, **default_benchmark_registries)
        lights = config.capabilities[0]

        groups = compile_capability(lights, config.execution)
        dataset, _scorers = groups[0]
        for ex in dataset:
            assert ex.input.get("preconditions") == []

    def test_preconditions_do_not_affect_grouping(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """Two scenarios with identical ``success:`` but different ``precondition:``
        share a single scorer-set group. Preconditions are per-example data, not
        part of the group key."""
        from agent_evals.benchmark.loader import (
            compile_capability,
            load_benchmark,
        )

        p = tmp_path / "precond_grouping.yaml"
        p.write_text(PRECONDITION_GROUPING_YAML)
        config = load_benchmark(p, **default_benchmark_registries)
        lights = config.capabilities[0]

        groups = compile_capability(lights, config.execution)
        # Identical success blocks -> single group, even though preconditions differ.
        assert len(groups) == 1
        dataset, _scorers = groups[0]
        # Both scenarios live in the same group.
        scenario_ids = {ex.metadata["scenario_id"] for ex in dataset}
        assert scenario_ids == {1, 2}

    def test_multiple_preconditions_apply_in_yaml_order(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """When a scenario declares multiple precondition keys, they reach
        the agent in YAML declaration order so any author-visible "this
        applier runs after that one" intent is honored.
        """
        from typing import Any

        from pydantic import BaseModel, ConfigDict

        from agent_evals.benchmark.loader import (
            compile_capability,
            load_benchmark,
        )
        from agent_evals.core._registries import _PreconditionRegistry

        class _RecordingApplier(BaseModel):
            model_config = ConfigDict(extra="forbid")
            tag: str

            def apply(self, target: Any) -> None: ...

        # Fresh-instance isolation: build a private registry and inject
        # it via load_benchmark's precondition_registry kwarg. No module
        # global is touched, so the test cannot leak state into others.
        fresh = _PreconditionRegistry()
        fresh.register("first")(_RecordingApplier)
        fresh.register("second")(_RecordingApplier)

        yaml_text = textwrap.dedent(
            """
            benchmark: precond-order-v1
            platform:
              name: local
              experiment: t
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: ordered
                    prompt: noop
                    precondition:
                      first: { tag: a }
                      second: { tag: b }
                    success:
                      state:
                        lights.kitchen.state: "off"
            """
        ).strip()
        p = tmp_path / "ordered.yaml"
        p.write_text(yaml_text)
        config = load_benchmark(
            p,
            platform_registry=default_benchmark_registries["platform_registry"],
            precondition_registry=fresh,
            success_checker_registry=default_benchmark_registries[
                "success_checker_registry"
            ],
            config_reader_registry=default_benchmark_registries[
                "config_reader_registry"
            ],
        )
        groups = compile_capability(config.capabilities[0], config.execution)
        preconds = groups[0][0][0].input["preconditions"]
        assert [p.tag for p in preconds] == ["a", "b"]


class TestLoadBenchmarkRegistryInjection:
    """Symmetric injection coverage for both registry kwargs."""

    def test_load_benchmark_uses_injected_success_checker_registry(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """``load_benchmark(..., success_checker_registry=fresh)`` threads
        the fresh registry into ScenarioSpec validation.

        Mirrors ``test_multiple_preconditions_apply_in_yaml_order`` for
        the ``success_checker_registry`` kwarg: a fresh registry with a
        recording fake checker is injected, and the YAML's ``success:``
        block resolves through the fake — proving the kwarg actually
        reaches the validator's context.
        """
        from typing import Any

        from pydantic import BaseModel, ConfigDict

        from agent_evals.benchmark.loader import load_benchmark
        from agent_evals.core._registries import (
            _SuccessCheckerRegistry,
        )
        from agent_evals.core.checks import StateCheck

        instantiated_payloads: list[dict[str, Any]] = []

        class _RecordingChecker(BaseModel):
            model_config = ConfigDict(extra="forbid")
            tag: str

            def __init__(self, **data: Any) -> None:
                super().__init__(**data)
                instantiated_payloads.append(dict(data))

            def to_spec(self) -> StateCheck:
                return StateCheck(expected_state={"tag": self.tag})

        fresh = _SuccessCheckerRegistry()
        fresh.register("fake_check")(_RecordingChecker)

        yaml_text = textwrap.dedent(
            """
            benchmark: success-injection-v1
            platform:
              name: local
              experiment: t
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: scenario
                    prompt: noop
                    success:
                      fake_check: { tag: alpha }
            """
        ).strip()
        p = tmp_path / "success_injected.yaml"
        p.write_text(yaml_text)

        config = load_benchmark(
            p,
            platform_registry=default_benchmark_registries["platform_registry"],
            success_checker_registry=fresh,
            precondition_registry=default_benchmark_registries["precondition_registry"],
            config_reader_registry=default_benchmark_registries[
                "config_reader_registry"
            ],
        )

        # The fake checker class was instantiated with the YAML payload.
        assert instantiated_payloads == [{"tag": "alpha"}]
        scenario = config.capabilities[0].scenarios[0]
        parsed = scenario.parsed_check_specs
        assert len(parsed) == 1
        assert isinstance(parsed[0], StateCheck)
        assert parsed[0].expected_state == {"tag": "alpha"}

    def test_load_benchmark_uses_injected_config_reader_registry(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """``load_benchmark(..., config_reader_registry=fresh)`` threads
        the fresh registry into the URI-resolution composition layer.

        Symmetric with ``test_load_benchmark_uses_injected_success_checker_registry``
        but for the config-reader family: a fresh registry with a
        recording fake reader is injected, and a ``mem://`` URI inside
        ``capabilities:`` resolves through the fake — proving the
        kwarg actually reaches ``_resolve_list_entries``.
        """
        from agent_evals.benchmark.loader import load_benchmark
        from agent_evals.core._registries import _BenchmarkConfigReaderRegistry

        read_uris: list[str] = []

        class _RecordingReader:
            def read(self, uri: str) -> str:
                read_uris.append(uri)
                return _LIGHTS_CHILD

        fresh = _BenchmarkConfigReaderRegistry()
        fresh.register("mem")(_RecordingReader)

        parent = tmp_path / "parent.yaml"
        parent.write_text(f"{_PARENT_PROLOGUE}\n  - mem:///lights.yaml\n")

        config = load_benchmark(
            parent,
            platform_registry=default_benchmark_registries["platform_registry"],
            config_reader_registry=fresh,
            precondition_registry=default_benchmark_registries["precondition_registry"],
            success_checker_registry=default_benchmark_registries[
                "success_checker_registry"
            ],
        )

        # The fake reader was invoked with the URI from the parent.
        assert read_uris == ["mem:///lights.yaml"]
        # And the resolved content was spliced into capabilities.
        assert len(config.capabilities) == 1
        assert config.capabilities[0].name == "lights"

    def test_load_benchmark_threads_both_registries_simultaneously(
        self, tmp_path: Path, default_benchmark_registries
    ):
        """``load_benchmark`` accepts ``precondition_registry`` AND
        ``success_checker_registry`` together.

        Exercises the dual-injection path: a fresh registry for each
        family is constructed, populated with a recording fake, and
        injected via both kwargs at once. The YAML declares one
        precondition and one success-checker entry; both fakes must be
        instantiated with their respective payloads.
        """
        from typing import Any

        from pydantic import BaseModel, ConfigDict

        from agent_evals.benchmark.loader import load_benchmark
        from agent_evals.core._registries import (
            _PreconditionRegistry,
            _SuccessCheckerRegistry,
        )
        from agent_evals.core.checks import StateCheck

        applier_payloads: list[dict[str, Any]] = []
        checker_payloads: list[dict[str, Any]] = []

        class _RecordingApplier(BaseModel):
            model_config = ConfigDict(extra="forbid")
            tag: str

            def __init__(self, **data: Any) -> None:
                super().__init__(**data)
                applier_payloads.append(dict(data))

            def apply(self, target: Any) -> None: ...

        class _RecordingChecker(BaseModel):
            model_config = ConfigDict(extra="forbid")
            tag: str

            def __init__(self, **data: Any) -> None:
                super().__init__(**data)
                checker_payloads.append(dict(data))

            def to_spec(self) -> StateCheck:
                return StateCheck(expected_state={"tag": self.tag})

        fresh_pre = _PreconditionRegistry()
        fresh_pre.register("fake_pre")(_RecordingApplier)
        fresh_check = _SuccessCheckerRegistry()
        fresh_check.register("fake_check")(_RecordingChecker)

        yaml_text = textwrap.dedent(
            """
            benchmark: dual-injection-v1
            platform:
              name: local
              experiment: t
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: dual
                    prompt: noop
                    precondition:
                      fake_pre: { tag: pre-payload }
                    success:
                      fake_check: { tag: check-payload }
            """
        ).strip()
        p = tmp_path / "dual_injected.yaml"
        p.write_text(yaml_text)

        config = load_benchmark(
            p,
            platform_registry=default_benchmark_registries["platform_registry"],
            precondition_registry=fresh_pre,
            success_checker_registry=fresh_check,
            config_reader_registry=default_benchmark_registries[
                "config_reader_registry"
            ],
        )

        # Both fakes were instantiated with their respective payloads.
        assert applier_payloads == [{"tag": "pre-payload"}]
        assert checker_payloads == [{"tag": "check-payload"}]
        scenario = config.capabilities[0].scenarios[0]
        assert len(scenario.parsed_preconditions) == 1
        assert isinstance(scenario.parsed_preconditions[0], _RecordingApplier)
        assert scenario.parsed_preconditions[0].tag == "pre-payload"
        assert len(scenario.parsed_check_specs) == 1
        assert isinstance(scenario.parsed_check_specs[0], StateCheck)
        assert scenario.parsed_check_specs[0].expected_state == {"tag": "check-payload"}


def test_load_benchmark_without_registry_kwargs_raises(tmp_path):
    """``load_benchmark(path)`` without kwargs raises TypeError.

    All kwargs are required; ``run_benchmark_async`` wires the
    defaults for the typical caller.
    """
    from agent_evals.benchmark.loader import load_benchmark

    p = tmp_path / "b.yaml"
    p.write_text("benchmark: x\nplatform:\n  name: local\ncapabilities: []\n")
    with pytest.raises(TypeError) as exc_info:
        load_benchmark(p)  # ty: ignore[missing-argument]
    msg = str(exc_info.value)
    assert "missing" in msg
    assert "platform_registry" in msg
    assert "precondition_registry" in msg
    assert "success_checker_registry" in msg
    assert "config_reader_registry" in msg


def test_compile_capability_attributes_compile_check_failures_to_scenario() -> None:
    """When compile_check raises (e.g., assert_never on a synthetic novel spec),
    compile_capability re-raises with capability + scenario breadcrumb so users
    can locate the offending YAML.
    """
    from typing import cast

    from agent_evals.benchmark.config import (
        CapabilitySpec,
        ExecutionBlock,
    )
    from agent_evals.benchmark.loader import compile_capability
    from agent_evals.core.checks import CheckSpec

    class _NotACheck:
        """Structurally not a CheckSpec member — assert_never will fire."""

    class _ScenarioStub:
        """Duck-typed scenario stub — compile_capability reads attributes only."""

        id = 99
        name = "boom_scenario"
        prompt = "test"
        dimensions: dict = {}
        parsed_check_specs = [cast(CheckSpec, _NotACheck())]
        parsed_preconditions: list = []

    capability = CapabilitySpec.model_construct(
        name="boom_capability",
        scenarios=[_ScenarioStub()],  # type: ignore[list-item]
    )
    execution = ExecutionBlock(runs_per_scenario=1, n_parallel_runs=1)

    with pytest.raises(AssertionError) as excinfo:
        compile_capability(capability, execution)

    msg = str(excinfo.value)
    assert "boom_capability" in msg, f"Capability name missing from error: {msg!r}"
    assert "99" in msg, f"Scenario id missing from error: {msg!r}"
    assert "boom_scenario" in msg, f"Scenario name missing from error: {msg!r}"
