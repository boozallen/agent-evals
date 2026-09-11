# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Per-adapter PlatformConfig subclass validation (pure, no SDK, no IO)."""

import pytest
from pydantic import ValidationError

from agent_evals.adapters.platforms.langfuse import LangfuseConfig
from agent_evals.adapters.platforms.local import LocalConfig
from agent_evals.adapters.platforms.mlflow import MlflowConfig


def test_local_name_defaults_and_is_literal():
    assert LocalConfig().name == "local"


def test_local_experiment_optional():
    assert LocalConfig().experiment is None


def test_local_accepts_output_dir():
    assert LocalConfig().output_dir is None
    cfg = LocalConfig(experiment="e", output_dir="/tmp/x")
    assert cfg.output_dir == "/tmp/x"


def test_local_rejects_project():
    # local never used `project`; it must not be a field (was silently dropped today).
    with pytest.raises(ValidationError):
        LocalConfig(project="home-bench")  # intentional - verifies runtime rejection


def test_mlflow_requires_experiment():
    with pytest.raises(ValidationError):
        MlflowConfig()  # ty: ignore[missing-argument]  # intentional - experiment required
    assert MlflowConfig(experiment="e").name == "mlflow"


def test_mlflow_rejects_tracking_uri():
    # tracking_uri now comes from MLFLOW_TRACKING_URI env, not config.
    with pytest.raises(ValidationError):
        MlflowConfig(
            experiment="e", tracking_uri="https://x"
        )  # intentional - verifies runtime rejection


@pytest.mark.requires_braintrust
def test_braintrust_requires_experiment_and_project():
    from agent_evals.adapters.platforms.braintrust import BraintrustConfig

    with pytest.raises(ValidationError):
        BraintrustConfig(project="p")  # ty: ignore[missing-argument]  # intentional - experiment required
    with pytest.raises(ValidationError):
        BraintrustConfig(experiment="e")  # ty: ignore[missing-argument]  # intentional - project required
    cfg = BraintrustConfig(experiment="e", project="p")
    assert cfg.name == "braintrust"
    assert cfg.metadata == {}


def test_langfuse_requires_experiment_optional_extras():
    with pytest.raises(ValidationError):
        LangfuseConfig()  # ty: ignore[missing-argument]  # intentional - experiment required
    cfg = LangfuseConfig(experiment="e", description="d", metadata={"k": "v"})
    assert cfg.name == "langfuse"
    assert cfg.description == "d"
    assert cfg.metadata == {"k": "v"}
