# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for typed PlatformConfig on remote adapters.

Verifies:
- MLflowPlatform._resolve_tracking_uri reads from MLFLOW_TRACKING_URI env var.
- LangFusePlatform._to_sdk_kwargs maps experiment → name correctly.
- Remote adapters raise ValueError when given a wrong config type.
"""

import asyncio

import pytest
from helpers.scorers import passing_scorer

from agent_evals.adapters.platforms.langfuse import LangfuseConfig, LangFusePlatform
from agent_evals.adapters.platforms.local import LocalConfig
from agent_evals.adapters.platforms.mlflow import MLflowPlatform
from agent_evals.core.types import ExampleData, TaskResult


def _task(x):
    return TaskResult(output="ok")


def test_mlflow_reads_tracking_uri_from_env(monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://localhost:5000")
    assert MLflowPlatform()._resolve_tracking_uri() == "https://localhost:5000"


def test_mlflow_missing_tracking_uri_raises(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        MLflowPlatform()._resolve_tracking_uri()


def test_langfuse_maps_experiment_to_name():
    kwargs = LangFusePlatform._to_sdk_kwargs(
        LangfuseConfig(experiment="math-eval", description="d")
    )
    assert kwargs["name"] == "math-eval"
    assert kwargs["description"] == "d"


def test_mlflow_rejects_wrong_config_type(monkeypatch):
    """MLflowPlatform.aevaluate must raise ValueError when given a non-MlflowConfig."""
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://localhost:5000")
    with pytest.raises(ValueError, match="MlflowConfig"):
        asyncio.run(
            MLflowPlatform().aevaluate(
                task=_task,
                dataset=[ExampleData(input="x")],
                evaluators=[passing_scorer],
                platform=LocalConfig(experiment="e"),
            )
        )
