# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pytest configuration and fixtures."""

from __future__ import annotations

import importlib.util
import os

import pytest

_HAS_BRAINTRUST = importlib.util.find_spec("braintrust") is not None


def pytest_collection_modifyitems(config, items):
    if _HAS_BRAINTRUST:
        return
    skip = pytest.mark.skip(reason="braintrust extra not installed")
    for item in items:
        if "requires_braintrust" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _encryption_key_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject a random encryption key for all tests."""
    monkeypatch.setenv("FOUNDRY_EVALS_ENCRYPTION_KEY", os.urandom(32).hex())


@pytest.fixture
def default_benchmark_registries() -> dict:
    """Production-default registries for tests calling ``load_benchmark`` directly.

    Splat into the call site as ``**default_benchmark_registries``. Tests
    that go through ``run_benchmark_async`` get these wired automatically
    and do not need the fixture.

    All four adapter-family registries are returned; ``load_benchmark``
    accepts them as keyword arguments and forwards them into the
    corresponding Pydantic validation contexts.
    """
    import agent_evals.adapters.benchmark_config_readers  # noqa: F401
    import agent_evals.adapters.platforms  # noqa: F401
    import agent_evals.adapters.preconditions  # noqa: F401
    import agent_evals.adapters.success_checkers  # noqa: F401
    from agent_evals.core._registries import (
        config_reader_registry,
        platform_registry,
        precondition_registry,
        success_checker_registry,
    )

    return {
        "platform_registry": platform_registry,
        "precondition_registry": precondition_registry,
        "success_checker_registry": success_checker_registry,
        "config_reader_registry": config_reader_registry,
    }
