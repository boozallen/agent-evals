# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``scenarios:`` + ``success:`` YAML schema validators.

Three behaviors:

1. New shape parses cleanly.
2. ``success: {}`` is rejected with a clear validation error.
3. Unknown success-checker keys are rejected with the registry's
   "available success checkers" diagnostic.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agent_evals.adapters.platforms.local import LocalConfig

_MIN_NEW_SHAPE: dict = {
    "benchmark": "schema-v1",
    "platform": LocalConfig(),
    "capabilities": [
        {
            "name": "lights",
            "scenarios": [
                {
                    "id": 1,
                    "name": "kitchen on",
                    "prompt": "Turn on the kitchen light",
                    "success": {"state": {"lights.kitchen.state": "on"}},
                }
            ],
        }
    ],
}


class TestNewShapeParses:
    def test_scenarios_with_success_block_parses(self, default_benchmark_registries):
        from agent_evals.benchmark.config import BenchmarkConfig

        config = BenchmarkConfig.model_validate(
            _MIN_NEW_SHAPE, context=default_benchmark_registries
        )
        assert config.benchmark == "schema-v1"
        cap = config.capabilities[0]
        assert cap.name == "lights"
        assert len(cap.scenarios) == 1
        scenario = cap.scenarios[0]
        assert scenario.id == 1
        # The validator parsed the success block; both the raw dict and the
        # parsed-success-checkers list should be accessible.
        assert "state" in scenario.success
        parsed = scenario.parsed_check_specs
        assert len(parsed) == 1
        assert parsed[0].__class__.__name__ == "StateCheck"

    def test_scenarios_with_multiple_success_checkers_parses(
        self, default_benchmark_registries
    ):
        from agent_evals.benchmark.config import BenchmarkConfig

        payload = {
            "benchmark": "multi",
            "platform": LocalConfig(),
            "capabilities": [
                {
                    "name": "doors",
                    "scenarios": [
                        {
                            "id": 1,
                            "name": "lock front",
                            "prompt": "Lock the front door",
                            "success": {
                                "state": {"doors.front": "locked"},
                                "tool_use": {
                                    "calls": [
                                        {"name": "lock_door", "args": {"door": "front"}}
                                    ]
                                },
                            },
                        }
                    ],
                }
            ],
        }

        config = BenchmarkConfig.model_validate(
            payload, context=default_benchmark_registries
        )
        scenario = config.capabilities[0].scenarios[0]
        names = {c.__class__.__name__ for c in scenario.parsed_check_specs}
        assert names == {"StateCheck", "ToolUseCheck"}


class TestSuccessValidation:
    def test_empty_success_block_rejected(self):
        from agent_evals.benchmark.config import BenchmarkConfig

        bad = {
            "benchmark": "x",
            "platform": LocalConfig(),
            "capabilities": [
                {
                    "name": "c",
                    "scenarios": [
                        {
                            "id": 1,
                            "name": "n",
                            "prompt": "p",
                            "success": {},
                        }
                    ],
                }
            ],
        }
        with pytest.raises(ValidationError) as exc:
            BenchmarkConfig.model_validate(bad)
        msg = str(exc.value)
        assert "success" in msg
        # error message must explain that at least one success checker is required
        assert "empty" in msg.lower() or "at least one" in msg.lower()

    def test_unknown_success_checker_key_rejected(self, default_benchmark_registries):
        from agent_evals.benchmark.config import BenchmarkConfig

        bad = {
            "benchmark": "x",
            "platform": LocalConfig(),
            "capabilities": [
                {
                    "name": "c",
                    "scenarios": [
                        {
                            "id": 1,
                            "name": "n",
                            "prompt": "p",
                            "success": {"foo": {}},
                        }
                    ],
                }
            ],
        }
        with pytest.raises(ValidationError) as exc:
            BenchmarkConfig.model_validate(bad, context=default_benchmark_registries)
        msg = str(exc.value)
        # Registry error message — see adapters/success_checkers/registry.py.
        assert "Unknown success checker" in msg or "Available success checkers" in msg
        # Should list the available registered checkers so users can fix typos.
        assert "state" in msg
        assert "tool_use" in msg


def _scenario_with(**overrides) -> dict:
    """Build a benchmark config payload with a single scenario, overriding fields."""
    base_scenario = {
        "id": 1,
        "name": "kitchen on",
        "prompt": "Turn on the kitchen light",
        "success": {"state": {"lights.kitchen.state": "on"}},
    }
    base_scenario.update(overrides)
    return {
        "benchmark": "schema-v1",
        "platform": LocalConfig(),
        "capabilities": [
            {
                "name": "lights",
                "scenarios": [base_scenario],
            }
        ],
    }


class TestPreconditionParsing:
    def test_scenario_with_precondition_parses(self, default_benchmark_registries):
        from agent_evals.adapters.preconditions.state import StatePreconditionApplier
        from agent_evals.benchmark.config import BenchmarkConfig

        payload = _scenario_with(precondition={"state": {"lights.kitchen.state": "on"}})
        config = BenchmarkConfig.model_validate(
            payload, context=default_benchmark_registries
        )
        scenario = config.capabilities[0].scenarios[0]
        parsed = scenario.parsed_preconditions
        assert len(parsed) == 1
        assert isinstance(parsed[0], StatePreconditionApplier)

    def test_scenario_without_precondition_uses_empty_default(
        self, default_benchmark_registries
    ):
        from agent_evals.benchmark.config import BenchmarkConfig

        payload = _scenario_with()
        config = BenchmarkConfig.model_validate(
            payload, context=default_benchmark_registries
        )
        scenario = config.capabilities[0].scenarios[0]
        assert scenario.parsed_preconditions == []

    def test_unknown_precondition_key_rejected(self, default_benchmark_registries):
        from agent_evals.benchmark.config import BenchmarkConfig

        payload = _scenario_with(precondition={"foo": {}})
        with pytest.raises(ValidationError) as exc:
            BenchmarkConfig.model_validate(
                payload, context=default_benchmark_registries
            )
        msg = str(exc.value)
        # Registry error message — see adapters/preconditions/registry.py.
        assert "Unknown precondition" in msg or "Available preconditions" in msg
        # Should list the available registered preconditions so users can fix typos.
        assert "state" in msg

    def test_precondition_bad_payload_rejected(self, default_benchmark_registries):
        from agent_evals.benchmark.config import BenchmarkConfig

        payload = _scenario_with(precondition={"state": "not a dict"})
        with pytest.raises(ValidationError):
            BenchmarkConfig.model_validate(
                payload, context=default_benchmark_registries
            )

    def test_empty_precondition_block_allowed(self, default_benchmark_registries):
        from agent_evals.benchmark.config import BenchmarkConfig

        payload = _scenario_with(precondition={})
        config = BenchmarkConfig.model_validate(
            payload, context=default_benchmark_registries
        )
        scenario = config.capabilities[0].scenarios[0]
        assert scenario.parsed_preconditions == []


class TestCapabilityIdUniqueness:
    """Within-capability `id:` uniqueness validator.

    Closes a silent-failure bug: two scenarios sharing `id:` inside one
    capability collapse into a single row in the markdown report
    (``report.py:_group_examples_by_scenario_id``). Cross-capability
    duplicates remain allowed because each capability has its own
    ``EvalResult`` and the report iterates per-capability.
    """

    @staticmethod
    def _payload(capabilities: list[dict]) -> dict:
        return {
            "benchmark": "id-uniqueness",
            "platform": LocalConfig(),
            "capabilities": capabilities,
        }

    def test_duplicate_id_within_capability_rejected(
        self, default_benchmark_registries
    ):
        from agent_evals.benchmark.config import BenchmarkConfig

        payload = self._payload(
            [
                {
                    "name": "lights",
                    "scenarios": [
                        {
                            "id": 1,
                            "name": "kitchen on",
                            "prompt": "Turn on the kitchen light",
                            "success": {"state": {"lights.kitchen.state": "on"}},
                        },
                        {
                            "id": 1,
                            "name": "kitchen off",
                            "prompt": "Turn off the kitchen light",
                            "success": {"state": {"lights.kitchen.state": "off"}},
                        },
                    ],
                }
            ]
        )
        with pytest.raises(ValidationError) as exc:
            BenchmarkConfig.model_validate(
                payload, context=default_benchmark_registries
            )
        msg = str(exc.value)
        assert "lights" in msg
        assert "1" in msg
        # Both offending scenario names should appear so users can find them.
        assert "kitchen on" in msg
        assert "kitchen off" in msg

    def test_duplicate_id_across_capabilities_allowed(
        self, default_benchmark_registries
    ):
        from agent_evals.benchmark.config import BenchmarkConfig

        payload = self._payload(
            [
                {
                    "name": "lights",
                    "scenarios": [
                        {
                            "id": 1,
                            "name": "kitchen on",
                            "prompt": "p",
                            "success": {"state": {"lights.kitchen.state": "on"}},
                        }
                    ],
                },
                {
                    "name": "doors",
                    "scenarios": [
                        {
                            "id": 1,
                            "name": "lock front",
                            "prompt": "p",
                            "success": {"state": {"doors.front": "locked"}},
                        }
                    ],
                },
            ]
        )
        config = BenchmarkConfig.model_validate(
            payload, context=default_benchmark_registries
        )
        assert config.capabilities[0].scenarios[0].id == 1
        assert config.capabilities[1].scenarios[0].id == 1

    def test_unique_ids_within_capability_pass(self, default_benchmark_registries):
        from agent_evals.benchmark.config import BenchmarkConfig

        payload = self._payload(
            [
                {
                    "name": "lights",
                    "scenarios": [
                        {
                            "id": i,
                            "name": f"scenario-{i}",
                            "prompt": "p",
                            "success": {"state": {"lights.kitchen.state": "on"}},
                        }
                        for i in (1, 2, 3)
                    ],
                }
            ]
        )
        config = BenchmarkConfig.model_validate(
            payload, context=default_benchmark_registries
        )
        ids = [s.id for s in config.capabilities[0].scenarios]
        assert ids == [1, 2, 3]
