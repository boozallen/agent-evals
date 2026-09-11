# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Contract: every registered platform adapter declares a config_class
that subclasses PlatformConfig. Pins the dynamic name->config-class
resolution the composition root relies on (LSP/lookup guarantee)."""

import pytest

import agent_evals.adapters.platforms  # noqa: F401  (registers adapters)
from agent_evals.core._registries import platform_registry
from agent_evals.core.types import PlatformConfig

# Hardcoded to pin the expected adapter set — add new adapters here explicitly.
PLATFORMS = ["local", "mlflow", "braintrust", "langfuse"]


@pytest.mark.parametrize("name", PLATFORMS)
def test_adapter_declares_config_class(name):
    adapter_cls = platform_registry.get(name)
    config_cls = getattr(adapter_cls, "config_class", None)
    assert config_cls is not None, f"{name} adapter missing config_class"
    assert issubclass(config_cls, PlatformConfig)


@pytest.mark.parametrize("name", PLATFORMS)
def test_config_class_name_matches_registry_key(name):
    adapter_cls = platform_registry.get(name)
    # The config's default `name` must equal the registry key.
    # Read the field default rather than instantiate: mlflow/braintrust/langfuse
    # configs have required fields (e.g. `experiment`, braintrust `project`) that
    # block bare construction, so config_class().name would raise.
    assert adapter_cls.config_class.model_fields["name"].default == name
