# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for BenchmarkConfig platform resolution via the injected registry.

Pins that the ``_resolve_platform`` before-validator maps a platform name +
flat dict (or scalar shorthand) into the adapter's typed ``PlatformConfig``,
that ``extra="forbid"`` on the config class surfaces typo'd YAML keys loudly,
and that missing or invalid inputs raise the expected errors.

Resolution previously happened in ``_composition._build_platform_config``;
it now lives inside the ``BenchmarkConfig`` Pydantic model_validator so that
platform joins the same "registry per family, resolve in a validator" pattern
as ``precondition_registry`` and ``success_checker_registry``.
"""

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Happy-path tests — go through load_benchmark + default_benchmark_registries
# so the full validator chain runs just like production.
# ---------------------------------------------------------------------------


def test_returns_typed_config_for_known_fields(
    tmp_path: Path, default_benchmark_registries
):
    from agent_evals.adapters.platforms.local import LocalConfig
    from agent_evals.benchmark.loader import load_benchmark

    yaml_text = textwrap.dedent(
        """
        benchmark: b
        platform:
          name: local
          experiment: e
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
    assert config.platform.experiment == "e"


def test_scalar_shorthand_expands_to_default_config(
    tmp_path: Path, default_benchmark_registries
):
    from agent_evals.adapters.platforms.local import LocalConfig
    from agent_evals.benchmark.loader import load_benchmark

    yaml_text = textwrap.dedent(
        """
        benchmark: b
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
    assert config.platform.name == "local"
    assert config.platform.experiment is None


# ---------------------------------------------------------------------------
# Error-path tests — use BenchmarkConfig.model_validate directly with a
# minimal valid dict for cleaner error-message assertions.
# ---------------------------------------------------------------------------

_MINIMAL_CAPABILITIES = [
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
]


def _context(default_benchmark_registries):
    """Return a validation context with all registries populated."""
    return {
        "platform_registry": default_benchmark_registries["platform_registry"],
        "precondition_registry": default_benchmark_registries["precondition_registry"],
        "success_checker_registry": default_benchmark_registries[
            "success_checker_registry"
        ],
    }


def test_unknown_key_raises_validation_error(default_benchmark_registries):
    from agent_evals.benchmark.config import BenchmarkConfig

    with pytest.raises(ValidationError):
        BenchmarkConfig.model_validate(
            {
                "benchmark": "b",
                "platform": {"name": "local", "experiment": "e", "bogus_key": 1},
                "capabilities": _MINIMAL_CAPABILITIES,
            },
            context=_context(default_benchmark_registries),
        )


def test_missing_name_raises_value_error(default_benchmark_registries):
    from agent_evals.benchmark.config import BenchmarkConfig

    with pytest.raises((ValueError, ValidationError), match="name"):
        BenchmarkConfig.model_validate(
            {
                "benchmark": "b",
                "platform": {},
                "capabilities": _MINIMAL_CAPABILITIES,
            },
            context=_context(default_benchmark_registries),
        )


def test_unknown_platform_name_raises_value_error(default_benchmark_registries):
    from agent_evals.benchmark.config import BenchmarkConfig

    with pytest.raises((ValueError, ValidationError), match="benchmark platform block"):
        BenchmarkConfig.model_validate(
            {
                "benchmark": "b",
                "platform": {"name": "nonexistent_platform"},
                "capabilities": _MINIMAL_CAPABILITIES,
            },
            context=_context(default_benchmark_registries),
        )


def test_none_platform_raises_value_error(default_benchmark_registries):
    """`platform: null` (or absent value) is neither a string nor a mapping."""
    from agent_evals.benchmark.config import BenchmarkConfig

    with pytest.raises(
        (ValueError, ValidationError), match="must be a string or mapping"
    ):
        BenchmarkConfig.model_validate(
            {
                "benchmark": "b",
                "platform": None,
                "capabilities": _MINIMAL_CAPABILITIES,
            },
            context=_context(default_benchmark_registries),
        )


def test_missing_registry_in_context_raises_value_error():
    """``_resolve_platform`` raises when ``platform_registry`` is absent from context."""
    from agent_evals.benchmark.config import BenchmarkConfig

    with pytest.raises((ValueError, ValidationError), match="platform_registry"):
        BenchmarkConfig.model_validate(
            {
                "benchmark": "b",
                "platform": "local",
                "capabilities": _MINIMAL_CAPABILITIES,
            },
            # Deliberately omit platform_registry from context.
            context={},
        )
