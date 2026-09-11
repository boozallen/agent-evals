# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LocalPlatform with typed PlatformConfig.

Verifies that ``LocalPlatform.aevaluate`` accepts a typed ``LocalConfig``
and produces a valid ``EvalResult``, and raises on wrong config types.
"""

import asyncio

import pytest

from agent_evals.adapters.platforms.local import LocalConfig, LocalPlatform
from agent_evals.adapters.platforms.mlflow import MlflowConfig
from agent_evals.core.types import (
    EvalConfig,
    ExampleData,
    ExpectedResult,
    Score,
    TaskResult,
)


def _task(x):
    return TaskResult(output="ok")


def _scorer(
    result: TaskResult, expected: ExpectedResult | None = None, **context
) -> Score:
    return Score(name="S", value=1.0, passed=True, reasoning=None, metadata={})


def test_local_aevaluate_with_typed_config(tmp_path):
    result = asyncio.run(
        LocalPlatform().aevaluate(
            _task,
            [ExampleData(input="x", expected=ExpectedResult(expected="ok"))],
            [_scorer],
            platform=LocalConfig(experiment="run-1", output_dir=str(tmp_path)),
            config=EvalConfig(max_concurrent_tests=1),
        )
    )
    assert result.scores["S"] == 1.0


def test_local_rejects_wrong_config_type(tmp_path):
    """LocalPlatform.aevaluate must raise ValueError when given a non-LocalConfig."""
    with pytest.raises(ValueError, match="LocalConfig"):
        asyncio.run(
            LocalPlatform().aevaluate(
                task=lambda x: TaskResult(output="ok"),
                dataset=[ExampleData(input="x", expected=ExpectedResult(expected="x"))],
                evaluators=[_scorer],
                platform=MlflowConfig(experiment="ignored"),
            )
        )
