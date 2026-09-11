# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for benchmark Pydantic config models."""

import pytest
from pydantic import ValidationError


class TestScenarioSpec:
    def test_minimal_scenario_requires_id_name_prompt_success(
        self, default_benchmark_registries
    ):
        from agent_evals.benchmark.config import ScenarioSpec

        s = ScenarioSpec.model_validate(
            {
                "id": 1,
                "name": "turn on light",
                "prompt": "Turn on kitchen light",
                "success": {"state": {"lights.kitchen.state": "on"}},
            },
            context=default_benchmark_registries,
        )
        assert s.id == 1
        assert s.dimensions == {}
        # The post-validator parses the success block into CheckSpec DTOs.
        assert len(s.parsed_check_specs) == 1
        assert s.parsed_check_specs[0].__class__.__name__ == "StateCheck"

    def test_missing_required_raises(self):
        from agent_evals.benchmark.config import ScenarioSpec

        with pytest.raises(ValidationError):
            ScenarioSpec.model_validate({"id": 1})  # missing name, prompt, success

    def test_old_tags_field_rejected(self):
        """Pre-#162 YAML using `tags:` must fail loudly so users see the rename.

        Locks the deliberate decision to ship the rename without a transitional
        alias: `extra="forbid"` is what makes that decision safe to ship, so a
        future relaxation (adding `extra="allow"`, an alias, or a deprecation
        shim) should break this test and force the trade-off back into review.
        """
        from agent_evals.benchmark.config import ScenarioSpec

        with pytest.raises(ValidationError) as exc:
            ScenarioSpec.model_validate(
                {
                    "id": 1,
                    "name": "k",
                    "prompt": "p",
                    "tags": {"phrasing": "imperative"},
                    "success": {"state": {"x": 1}},
                }
            )
        errors = exc.value.errors()
        assert any(
            e["type"] == "extra_forbidden" and e["loc"] == ("tags",) for e in errors
        )


class TestCapabilitySpec:
    def test_capability_requires_name_and_scenarios(self, default_benchmark_registries):
        from agent_evals.benchmark.config import CapabilitySpec

        cap = CapabilitySpec.model_validate(
            {
                "name": "lights",
                "scenarios": [
                    {
                        "id": 1,
                        "name": "kitchen on",
                        "prompt": "on",
                        "success": {"state": {"x": 1}},
                    }
                ],
            },
            context=default_benchmark_registries,
        )
        assert cap.name == "lights"
        assert len(cap.scenarios) == 1


class TestBenchmarkConfig:
    def test_minimal_benchmark_config(self, tmp_path, default_benchmark_registries):
        """BenchmarkConfig loaded via load_benchmark gets a typed LocalConfig."""
        import textwrap

        from agent_evals.adapters.platforms.local import LocalConfig
        from agent_evals.benchmark.loader import load_benchmark

        yaml_text = textwrap.dedent(
            """
            benchmark: home-v1
            platform: local
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: k
                    prompt: p
                    success:
                      state:
                        x: 1
            """
        ).strip()
        p = tmp_path / "b.yaml"
        p.write_text(yaml_text)
        config = load_benchmark(p, **default_benchmark_registries)
        assert config.benchmark == "home-v1"
        assert isinstance(config.platform, LocalConfig)
        assert config.platform.name == "local"
        # ExecutionBlock defaults
        assert config.execution.runs_per_scenario == 1
        assert config.execution.n_parallel_runs == 1

    @pytest.mark.requires_braintrust
    def test_platform_object_form(self, tmp_path, default_benchmark_registries):
        """BenchmarkConfig with flat platform dict loads typed BraintrustConfig."""
        import textwrap

        from agent_evals.adapters.platforms.braintrust import BraintrustConfig
        from agent_evals.benchmark.loader import load_benchmark

        yaml_text = textwrap.dedent(
            """
            benchmark: b
            platform:
              name: braintrust
              project: p
              experiment: v1
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: k
                    prompt: p
                    success:
                      state:
                        x: 1
            """
        ).strip()
        p = tmp_path / "b.yaml"
        p.write_text(yaml_text)
        config = load_benchmark(p, **default_benchmark_registries)
        assert isinstance(config.platform, BraintrustConfig)
        assert config.platform.name == "braintrust"
        assert config.platform.project == "p"
        assert config.platform.experiment == "v1"

    def test_explicit_execution_block_overrides_defaults(
        self, default_benchmark_registries, tmp_path
    ):
        import textwrap

        from agent_evals.benchmark.loader import load_benchmark

        yaml_text = textwrap.dedent(
            """
            benchmark: b
            platform: local
            execution:
              runs_per_scenario: 3
              n_parallel_runs: 5
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: k
                    prompt: p
                    success:
                      state:
                        x: 1
            """
        ).strip()
        p = tmp_path / "b.yaml"
        p.write_text(yaml_text)
        config = load_benchmark(p, **default_benchmark_registries)
        assert config.execution.runs_per_scenario == 3
        assert config.execution.n_parallel_runs == 5

    def test_explicit_execution_block_overrides_defaults_direct(
        self, default_benchmark_registries
    ):
        """Same as above but exercising BenchmarkConfig.model_validate directly
        with a pre-built platform instance.
        """
        from agent_evals.adapters.platforms.local import LocalConfig
        from agent_evals.benchmark.config import BenchmarkConfig

        config = BenchmarkConfig.model_validate(
            {
                "benchmark": "b",
                "platform": LocalConfig(name="local"),
                "execution": {
                    "runs_per_scenario": 3,
                    "n_parallel_runs": 5,
                },
                "capabilities": [
                    {
                        "name": "lights",
                        "scenarios": [
                            {
                                "id": 1,
                                "name": "k",
                                "prompt": "p",
                                "success": {"state": {"x": 1}},
                            }
                        ],
                    }
                ],
            },
            context=default_benchmark_registries,
        )
        assert config.execution.runs_per_scenario == 3
        assert config.execution.n_parallel_runs == 5

    def test_benchmark_result_aggregates_scores(self):
        from agent_evals.benchmark.types import BenchmarkResult
        from agent_evals.core.types import EvalResult

        eval_lights = EvalResult(
            experiment_id="e1",
            experiment_url="file:///tmp/e1",
            platform="local",
            scores={"StateMatch": 0.9},
            pass_rates={"StateMatch": 0.9},
            examples=[],
            summary={"total_examples": 10, "successful_examples": 9},
        )
        eval_doors = EvalResult(
            experiment_id="e2",
            experiment_url="file:///tmp/e2",
            platform="local",
            scores={"StateMatch": 0.7},
            pass_rates={"StateMatch": 0.7},
            examples=[],
            summary={"total_examples": 10, "successful_examples": 7},
        )

        result = BenchmarkResult(
            benchmark="home-v1",
            eval_results={"lights": eval_lights, "doors": eval_doors},
        )

        assert result.aggregate_scores["StateMatch"] == pytest.approx(0.8)
        assert result.aggregate_pass_rates["StateMatch"] == pytest.approx(0.8)

    def test_max_parallel_scorers_not_accepted_in_yaml(self):
        """Confirm the field is intentionally not exposed; raises on unknown key."""
        from agent_evals.adapters.platforms.local import LocalConfig
        from agent_evals.benchmark.config import BenchmarkConfig

        with pytest.raises(ValidationError):
            BenchmarkConfig.model_validate(
                {
                    "benchmark": "b",
                    "platform": LocalConfig(),
                    "execution": {"max_parallel_scorers": 8},
                    "capabilities": [
                        {
                            "name": "lights",
                            "scenarios": [
                                {
                                    "id": 1,
                                    "name": "k",
                                    "prompt": "p",
                                    "success": {"state": {"x": 1}},
                                }
                            ],
                        }
                    ],
                }
            )

    @pytest.mark.parametrize("old_key", ["runs_per_test", "max_parallel_tests"])
    def test_old_execution_keys_rejected(self, old_key):
        """Pre-#230 YAML using the old execution keys must fail loudly.

        Locks the deliberate decision to ship the runs_per_test ->
        runs_per_scenario / max_parallel_tests -> n_parallel_runs rename
        without a transitional alias: ``extra="forbid"`` is what makes that
        decision safe, so a future relaxation (adding a ``validation_alias``
        or loosening ``extra``) should break this test and force the
        trade-off back into review. Mirrors ``test_old_tags_field_rejected``.
        """
        from agent_evals.adapters.platforms.local import LocalConfig
        from agent_evals.benchmark.config import BenchmarkConfig

        with pytest.raises(ValidationError) as exc:
            BenchmarkConfig.model_validate(
                {
                    "benchmark": "b",
                    "platform": LocalConfig(),
                    "execution": {old_key: 1},
                    "capabilities": [
                        {
                            "name": "lights",
                            "scenarios": [
                                {
                                    "id": 1,
                                    "name": "k",
                                    "prompt": "p",
                                    "success": {"state": {"x": 1}},
                                }
                            ],
                        }
                    ],
                }
            )
        errors = exc.value.errors()
        assert any(
            e["type"] == "extra_forbidden" and e["loc"] == ("execution", old_key)
            for e in errors
        )


def test_scenario_validators_raise_without_registry_context():
    """Scenario validators raise when validation context lacks registries.

    The success and precondition validators require registries in the
    validation context; a missing context (or missing key) raises a
    Pydantic ``ValidationError`` wrapping the actionable message.
    """
    from pydantic import ValidationError

    from agent_evals.adapters.platforms.local import LocalConfig
    from agent_evals.benchmark.config import BenchmarkConfig

    # Minimal raw config exercising the success validator path.
    raw_with_success = {
        "benchmark": "test",
        "platform": LocalConfig(),
        "capabilities": [
            {
                "name": "lights",
                "scenarios": [
                    {
                        "id": 1,
                        "name": "kitchen",
                        "prompt": "turn on light",
                        "success": {"state": {"lights.kitchen.state": "on"}},
                    }
                ],
            }
        ],
    }
    with pytest.raises(ValidationError) as exc_info:
        BenchmarkConfig.model_validate(raw_with_success)  # context omitted
    msg = str(exc_info.value)
    assert "must be provided in validation context" in msg
    assert "success_checker_registry" in msg

    # And the precondition path.
    raw_with_precondition = {
        "benchmark": "test",
        "platform": LocalConfig(),
        "capabilities": [
            {
                "name": "lights",
                "scenarios": [
                    {
                        "id": 1,
                        "name": "kitchen",
                        "prompt": "turn on light",
                        "precondition": {"state": {"lights.kitchen.state": "off"}},
                        "success": {"state": {"lights.kitchen.state": "on"}},
                    }
                ],
            }
        ],
    }
    # Precondition path — provide a real success_checker_registry so the
    # success validator passes through; only precondition_registry is
    # missing. Without this scoping, the success validator short-circuits
    # and we never exercise _validate_precondition_block's missing-context
    # check.
    import agent_evals.adapters.success_checkers  # noqa: F401
    from agent_evals.core._registries import success_checker_registry

    with pytest.raises(ValidationError) as exc_info:
        BenchmarkConfig.model_validate(
            raw_with_precondition,
            context={"success_checker_registry": success_checker_registry},
        )
    msg = str(exc_info.value)
    assert "must be provided in validation context" in msg
    assert "precondition_registry" in msg


# ---------------------------------------------------------------------------
# Typed PlatformConfig round-trip (spec 033)
#
# Verifies that:
# 1. load_benchmark produces a concrete PlatformConfig subclass on
#    BenchmarkConfig.platform (not coerced to the base class).
# 2. Subclass fields survive (no coercion to base).
# 3. The old `id:`/`config:` nesting is rejected.
# 4. project: on local (which has no project field) is rejected.
# ---------------------------------------------------------------------------


class TestTypedPlatformConfig:
    def test_load_benchmark_produces_local_config_subclass(
        self, tmp_path, default_benchmark_registries
    ):
        """load_benchmark with platform: {name: local, experiment: v1} yields
        a LocalConfig instance with the experiment field populated.
        """
        import textwrap

        from agent_evals.adapters.platforms.local import LocalConfig
        from agent_evals.benchmark.loader import load_benchmark

        yaml_text = textwrap.dedent(
            """
            benchmark: typed-config-v1
            platform:
              name: local
              experiment: v1
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: k
                    prompt: p
                    success:
                      state:
                        x: 1
            """
        ).strip()
        p = tmp_path / "b.yaml"
        p.write_text(yaml_text)
        config = load_benchmark(p, **default_benchmark_registries)

        # Subclass is preserved — NOT coerced to base PlatformConfig.
        assert type(config.platform) is LocalConfig, (
            f"expected LocalConfig, got {type(config.platform).__name__}"
        )
        assert isinstance(config.platform, LocalConfig)
        # Subclass field survives.
        assert config.platform.experiment == "v1"
        assert config.platform.name == "local"

    def test_scalar_shorthand_produces_local_config(
        self, tmp_path, default_benchmark_registries
    ):
        """platform: local (scalar) expands to LocalConfig with no experiment."""
        import textwrap

        from agent_evals.adapters.platforms.local import LocalConfig
        from agent_evals.benchmark.loader import load_benchmark

        yaml_text = textwrap.dedent(
            """
            benchmark: shorthand-v1
            platform: local
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: k
                    prompt: p
                    success:
                      state:
                        x: 1
            """
        ).strip()
        p = tmp_path / "b.yaml"
        p.write_text(yaml_text)
        config = load_benchmark(p, **default_benchmark_registries)
        assert isinstance(config.platform, LocalConfig)
        assert config.platform.experiment is None

    def test_old_id_key_rejected(self, tmp_path, default_benchmark_registries):
        """platform: {id: local, config: {}} (old shape) must raise ValueError.

        The old shape had `id:` + `config:` nesting and no `name:` field;
        the resolver raises ValueError because `name:` is required.
        """
        import textwrap

        from agent_evals.benchmark.loader import load_benchmark

        yaml_text = textwrap.dedent(
            """
            benchmark: old-id-v1
            platform:
              id: local
              config: {}
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: k
                    prompt: p
                    success:
                      state:
                        x: 1
            """
        ).strip()
        p = tmp_path / "b.yaml"
        p.write_text(yaml_text)
        with pytest.raises(ValueError):
            load_benchmark(p, **default_benchmark_registries)

    def test_project_on_local_rejected(self, tmp_path, default_benchmark_registries):
        """platform: {name: local, project: x} must raise because LocalConfig
        has no project field (extra='forbid').
        """
        import textwrap

        from pydantic import ValidationError

        from agent_evals.benchmark.loader import load_benchmark

        yaml_text = textwrap.dedent(
            """
            benchmark: local-with-project-v1
            platform:
              name: local
              project: x
            capabilities:
              - name: lights
                scenarios:
                  - id: 1
                    name: k
                    prompt: p
                    success:
                      state:
                        x: 1
            """
        ).strip()
        p = tmp_path / "b.yaml"
        p.write_text(yaml_text)
        with pytest.raises(ValidationError):
            load_benchmark(p, **default_benchmark_registries)
