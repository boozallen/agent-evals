# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for encryption at rest and the _readers module."""

import json
from pathlib import Path

import pytest
from foundry_agent_core.encryption import encrypt, load_encryption_key

from agent_evals import (
    ExpectedResult,
    IntegrityError,
    Score,
    TaskResult,
    read_encrypted_file,
    read_encrypted_jsonl,
    run_eval,
)
from agent_evals.adapters.platforms.local import LocalConfig
from agent_evals.core.types import ExampleData


class TestSensitiveContentNotOnDisk:
    """AC #10: known sensitive content must not appear verbatim in results.jsonl."""

    def test_sensitive_input_not_in_plaintext(self, tmp_path: Path):
        sensitive_input = "TOP_SECRET_CREDENTIAL_XYZ_12345"
        sensitive_trajectory = "agent called tool with password=hunter2"

        def task(input_value):
            return TaskResult(
                output="redacted",
                context={
                    "outputs": [{"role": "assistant", "content": sensitive_trajectory}]
                },
            )

        def scorer(output: TaskResult, expected: ExpectedResult, **kwargs) -> Score:
            return Score(
                name="always_pass", value=1.0, passed=True, reasoning=None, metadata={}
            )

        dataset = [
            ExampleData(
                input=sensitive_input,
                expected=ExpectedResult(expected="redacted"),
            )
        ]

        run_eval(
            task=task,
            dataset=dataset,
            scorers=[scorer],
            platform=LocalConfig(experiment="sensitive_test", output_dir=str(tmp_path)),
        )

        results_file = next(tmp_path.rglob("results.jsonl"))
        raw_content = results_file.read_text()

        assert sensitive_input not in raw_content
        assert sensitive_trajectory not in raw_content


class TestReadEncryptedFile:
    """Unit tests for read_encrypted_file."""

    def test_roundtrip(self, tmp_path: Path):
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        data = {"hello": "world", "nested": {"a": 1}}
        envelope = encrypt(data, key)

        path = tmp_path / "test.json"
        path.write_text(json.dumps(envelope))

        result = read_encrypted_file(path, key)
        assert result == data

    def test_plaintext_returns_none(self, tmp_path: Path):
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        path = tmp_path / "plain.json"
        path.write_text(json.dumps({"hello": "world"}))

        result = read_encrypted_file(path, key)
        assert result is None

    def test_tampered_ciphertext_raises_integrity_error(self, tmp_path: Path):
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        data = {"secret": "value"}
        envelope = encrypt(data, key)
        envelope["ciphertext"] = "AAAA" + envelope["ciphertext"][4:]

        path = tmp_path / "tampered.json"
        path.write_text(json.dumps(envelope))

        with pytest.raises(IntegrityError):
            read_encrypted_file(path, key)


class TestReadEncryptedJsonl:
    """Unit tests for read_encrypted_jsonl."""

    def test_roundtrip_multiple_lines(self, tmp_path: Path):
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        records = [{"id": i, "data": f"record_{i}"} for i in range(5)]

        path = tmp_path / "results.jsonl"
        with path.open("w") as f:
            for record in records:
                f.write(json.dumps(encrypt(record, key)) + "\n")

        result = list(read_encrypted_jsonl(path, key))
        assert result == records

    def test_skips_unencrypted_lines(self, tmp_path: Path):
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        path = tmp_path / "mixed.jsonl"
        with path.open("w") as f:
            f.write(json.dumps({"plain": True}) + "\n")
            f.write(json.dumps(encrypt({"encrypted": True}, key)) + "\n")

        result = list(read_encrypted_jsonl(path, key))
        assert len(result) == 1
        assert result[0] == {"encrypted": True}

    def test_tampered_line_raises_integrity_error(self, tmp_path: Path):
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        envelope = encrypt({"data": "value"}, key)
        envelope["ciphertext"] = "AAAA" + envelope["ciphertext"][4:]

        path = tmp_path / "bad.jsonl"
        path.write_text(json.dumps(envelope) + "\n")

        with pytest.raises(IntegrityError):
            list(read_encrypted_jsonl(path, key))
