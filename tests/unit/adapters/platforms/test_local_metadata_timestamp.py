# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LocalPlatform's metadata.json created_at/created_at_iso fields.

created_at is a bare epoch float with no disambiguating sibling field.
created_at_iso adds a human-readable ISO-8601 UTC string derived from the
same instant, without changing created_at.
"""

import asyncio
from datetime import UTC, datetime

from foundry_agent_core.encryption import load_encryption_key

from agent_evals import read_encrypted_file
from agent_evals.adapters.platforms.local import LocalConfig, LocalPlatform
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


def test_metadata_json_has_iso_sibling_matching_epoch_float(tmp_path):
    before = datetime.now(UTC)

    result = asyncio.run(
        LocalPlatform().aevaluate(
            _task,
            [ExampleData(input="x", expected=ExpectedResult(expected="ok"))],
            [_scorer],
            platform=LocalConfig(experiment="ts-test", output_dir=str(tmp_path)),
            config=EvalConfig(max_concurrent_tests=1),
        )
    )
    after = datetime.now(UTC)

    exp_dir = tmp_path / "experiments" / f"ts-test-{result.experiment_id}"
    key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
    metadata = read_encrypted_file(exp_dir / "metadata.json", key)

    assert "created_at" in metadata
    assert isinstance(metadata["created_at"], float)

    assert "created_at_iso" in metadata
    parsed = datetime.fromisoformat(metadata["created_at_iso"])
    assert parsed.tzinfo is not None
    assert before <= parsed <= after

    # Both fields derive from the same instant, modulo the sub-microsecond
    # precision datetime's isoformat() can't round-trip exactly.
    assert abs(parsed.timestamp() - metadata["created_at"]) < 1e-6
