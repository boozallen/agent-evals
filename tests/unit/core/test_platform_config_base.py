# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the PlatformConfig base model."""

import pytest
from pydantic import ValidationError

from agent_evals.core.types import PlatformConfig


def test_base_requires_name():
    with pytest.raises(ValidationError):
        PlatformConfig()  # ty: ignore[missing-argument]  # name is required


def test_base_experiment_optional_defaults_none():
    cfg = PlatformConfig(name="x")
    assert cfg.experiment is None


def test_base_forbids_extra_keys():
    with pytest.raises(ValidationError):
        PlatformConfig(
            name="x", made_up_key="nope"
        )  # intentional - verifies runtime rejection


def test_base_accepts_experiment():
    cfg = PlatformConfig(name="x", experiment="run-1")
    assert cfg.experiment == "run-1"
