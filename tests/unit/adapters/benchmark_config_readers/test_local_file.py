# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for LocalFileConfigReader."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_evals.adapters.benchmark_config_readers import (
    LocalFileConfigReader,
    config_reader_registry,
)


def test_reads_text_from_file_uri(tmp_path: Path) -> None:
    target = tmp_path / "sample.yaml"
    target.write_text("hello: world\n", encoding="utf-8")

    reader = LocalFileConfigReader()
    content = reader.read(target.as_uri())

    assert content == "hello: world\n"


def test_reads_unicode_content(tmp_path: Path) -> None:
    target = tmp_path / "u.yaml"
    target.write_text("name: 日本語\n", encoding="utf-8")

    reader = LocalFileConfigReader()
    assert reader.read(target.as_uri()) == "name: 日本語\n"


def test_missing_file_raises_with_uri(tmp_path: Path) -> None:
    missing = tmp_path / "nonexistent.yaml"

    reader = LocalFileConfigReader()
    with pytest.raises(FileNotFoundError) as exc:
        reader.read(missing.as_uri())

    assert missing.as_uri() in str(exc.value)


def test_non_file_scheme_raises() -> None:
    reader = LocalFileConfigReader()
    with pytest.raises(ValueError) as exc:
        reader.read("https://example.com/x.yaml")

    msg = str(exc.value)
    assert "https" in msg
    assert "file" in msg


def test_relative_uri_rejected(tmp_path: Path) -> None:
    """Reader requires absolute URIs.

    Relative-path resolution is the composition layer's responsibility:
    it joins relative URIs against the parent URI *before* calling
    ``read()``. Keeping the reader parameterless preserves SRP and
    matches the precondition/applier "stateless adapter" pattern.
    """
    reader = LocalFileConfigReader()
    with pytest.raises(ValueError) as exc:
        reader.read("file://relative/path.yaml")

    msg = str(exc.value)
    assert "absolute" in msg.lower()


def test_registered_under_file_scheme() -> None:
    cls = config_reader_registry.get("file")
    assert cls is LocalFileConfigReader
    assert isinstance(cls(), LocalFileConfigReader)
