# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for integrity verification (V-222588).

Verifies that tampered encrypted artifacts raise IntegrityError, and that
unmodified artifacts round-trip without false positives.
"""

import json
from pathlib import Path

import pytest
from foundry_agent_core.encryption import encrypt, load_encryption_key

from agent_evals import IntegrityError, read_encrypted_file, read_encrypted_jsonl


@pytest.fixture()
def encryption_key() -> bytes:
    return load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")


def _flip_byte(envelope: dict) -> dict:
    """Corrupt the ciphertext to simulate tampering while keeping valid base64."""
    import base64

    raw = base64.b64decode(envelope["ciphertext"])
    corrupted = bytes([raw[0] ^ 0x01]) + raw[1:]
    return {**envelope, "ciphertext": base64.b64encode(corrupted).decode()}


class TestTamperDetection:
    """AC #6: modifying a byte in each artifact type raises IntegrityError."""

    def test_tampered_metadata_json(self, tmp_path: Path, encryption_key: bytes):
        data = {"experiment_id": "exp-1", "config": {"model": "gpt-4"}}
        envelope = encrypt(data, encryption_key)
        tampered = _flip_byte(envelope)

        path = tmp_path / "metadata.json"
        path.write_text(json.dumps(tampered))

        with pytest.raises(IntegrityError) as exc_info:
            read_encrypted_file(path, encryption_key)

        assert str(path) in str(exc_info.value)
        assert exc_info.value.path == path
        assert exc_info.value.line_number is None

    def test_tampered_results_jsonl(self, tmp_path: Path, encryption_key: bytes):
        records = [{"id": i, "score": 0.9} for i in range(3)]
        path = tmp_path / "results.jsonl"
        with path.open("w") as f:
            for i, record in enumerate(records):
                envelope = encrypt(record, encryption_key)
                if i == 1:
                    envelope = _flip_byte(envelope)
                f.write(json.dumps(envelope) + "\n")

        results = []
        with pytest.raises(IntegrityError) as exc_info:
            for record in read_encrypted_jsonl(path, encryption_key):
                results.append(record)

        assert len(results) == 1
        assert results[0] == records[0]
        assert exc_info.value.line_number == 2
        assert str(path) in str(exc_info.value)

    def test_tampered_summary_json(self, tmp_path: Path, encryption_key: bytes):
        data = {"pass_rate": 0.85, "total": 20, "passed": 17}
        envelope = encrypt(data, encryption_key)
        tampered = _flip_byte(envelope)

        path = tmp_path / "summary.json"
        path.write_text(json.dumps(tampered))

        with pytest.raises(IntegrityError) as exc_info:
            read_encrypted_file(path, encryption_key)

        assert exc_info.value.path == path

    def test_tampered_markdown_report(self, tmp_path: Path, encryption_key: bytes):
        data = {
            "content": "# Benchmark Report\n\n| Test | Pass |\n|---|---|\n| A | Yes |"
        }
        envelope = encrypt(data, encryption_key)
        tampered = _flip_byte(envelope)

        path = tmp_path / "2026-08-11_benchmark.md"
        path.write_text(json.dumps(tampered))

        with pytest.raises(IntegrityError) as exc_info:
            read_encrypted_file(path, encryption_key)

        assert exc_info.value.path == path


class TestRoundTrip:
    """AC #7: unmodified artifacts verify and read back successfully."""

    def test_metadata_json_roundtrip(self, tmp_path: Path, encryption_key: bytes):
        data = {"experiment_id": "exp-1", "config": {"model": "gpt-4"}}
        envelope = encrypt(data, encryption_key)

        path = tmp_path / "metadata.json"
        path.write_text(json.dumps(envelope))

        result = read_encrypted_file(path, encryption_key)
        assert result == data

    def test_results_jsonl_roundtrip(self, tmp_path: Path, encryption_key: bytes):
        records = [{"id": i, "score": 0.9 + i * 0.01} for i in range(5)]

        path = tmp_path / "results.jsonl"
        with path.open("w") as f:
            for record in records:
                f.write(json.dumps(encrypt(record, encryption_key)) + "\n")

        result = list(read_encrypted_jsonl(path, encryption_key))
        assert result == records

    def test_summary_json_roundtrip(self, tmp_path: Path, encryption_key: bytes):
        data = {"pass_rate": 0.85, "total": 20, "passed": 17}
        envelope = encrypt(data, encryption_key)

        path = tmp_path / "summary.json"
        path.write_text(json.dumps(envelope))

        result = read_encrypted_file(path, encryption_key)
        assert result == data

    def test_markdown_report_roundtrip(self, tmp_path: Path, encryption_key: bytes):
        data = {"content": "# Benchmark Report\n\nAll tests passed."}
        envelope = encrypt(data, encryption_key)

        path = tmp_path / "2026-08-11_benchmark.md"
        path.write_text(json.dumps(envelope))

        result = read_encrypted_file(path, encryption_key)
        assert result == data


class TestSizeGuards:
    """V-222612: reject oversized inputs before parsing."""

    def test_file_exceeding_max_size_raises(self, tmp_path: Path, monkeypatch):
        import agent_evals._readers as readers

        monkeypatch.setattr(readers, "_MAX_FILE_SIZE", 100)
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")

        path = tmp_path / "huge.json"
        path.write_text("x" * 200)

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            read_encrypted_file(path, key)

    def test_jsonl_line_exceeding_max_size_raises(self, tmp_path: Path, monkeypatch):
        import agent_evals._readers as readers

        monkeypatch.setattr(readers, "_MAX_LINE_SIZE", 50)
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")

        path = tmp_path / "huge.jsonl"
        path.write_text("x" * 100 + "\n")

        with pytest.raises(ValueError, match="exceeds maximum allowed size"):
            list(read_encrypted_jsonl(path, key))

    def test_file_within_limit_is_accepted(self, tmp_path: Path, encryption_key: bytes):
        data = {"small": "payload"}
        envelope = encrypt(data, encryption_key)

        path = tmp_path / "ok.json"
        path.write_text(json.dumps(envelope))

        result = read_encrypted_file(path, encryption_key)
        assert result == data


class TestIntegrityErrorType:
    """AC #5 + 4.7: IntegrityError is distinct from parse/IO errors."""

    def test_not_subclass_of_value_error(self):
        assert not issubclass(IntegrityError, ValueError)

    def test_not_subclass_of_json_decode_error(self):
        assert not issubclass(IntegrityError, json.JSONDecodeError)

    def test_not_subclass_of_os_error(self):
        assert not issubclass(IntegrityError, OSError)

    def test_is_subclass_of_exception(self):
        assert issubclass(IntegrityError, Exception)
