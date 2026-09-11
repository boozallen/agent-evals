# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Decrypting read utilities for encrypted eval artifacts.

Integrity model
---------------
Each record is independently protected by an AES-256-GCM authentication tag
(generated at write time by ``foundry_agent_core.encryption.encrypt``). Any
modification to a record's ciphertext, nonce, or tag is detected on read and
surfaces as ``IntegrityError``.

Threat scope:
- **Detected:** per-record modification (any byte change within a single
  encrypted envelope).
- **NOT detected:** record deletion, file truncation, and record reordering.
  These attacks alter the *set* of records without modifying any individual
  envelope. Detecting them would require a chained MAC or record-count manifest,
  which is not implemented.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from cryptography.exceptions import InvalidTag
from foundry_agent_core.encryption import decrypt, is_encrypted

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MiB per file
_MAX_LINE_SIZE = 10 * 1024 * 1024  # 10 MiB per JSONL record


class IntegrityError(Exception):
    """Raised when an encrypted artifact fails authentication tag verification.

    Indicates that the artifact has been modified since it was written.
    Distinct from parse errors (``json.JSONDecodeError``) and I/O errors
    (``OSError``) so callers can differentiate tampering from malformed data.
    """

    def __init__(self, path: str | Path, *, line_number: int | None = None) -> None:
        self.path = Path(path)
        self.line_number = line_number
        if line_number is not None:
            msg = f"Integrity verification failed: {self.path} line {line_number}"
        else:
            msg = f"Integrity verification failed: {self.path}"
        super().__init__(msg)


def read_encrypted_file(path: str | Path, key: bytes) -> dict[str, Any] | None:
    """Read and decrypt a single-record encrypted JSON file.

    Returns the decrypted dict, or None if the file does not contain
    an encrypted envelope (legacy/plaintext data).

    Raises ``ValueError`` if the file exceeds the size limit.
    """
    path = Path(path)
    size = path.stat().st_size
    if size > _MAX_FILE_SIZE:
        raise ValueError(
            f"File exceeds maximum allowed size ({_MAX_FILE_SIZE} bytes): {path}"
        )
    with path.open() as f:
        payload = json.load(f)
    if not is_encrypted(payload):
        return None
    try:
        return decrypt(payload, key)
    except InvalidTag:
        raise IntegrityError(path) from None


def read_encrypted_jsonl(path: str | Path, key: bytes) -> Iterator[dict[str, Any]]:
    """Read and decrypt an encrypted JSONL file, yielding one dict per line.

    Lines without the encrypted marker are skipped.

    Raises ``ValueError`` if any single line exceeds the per-record size limit.
    """
    path = Path(path)
    with path.open() as f:
        for line_number, line in enumerate(f, start=1):
            if len(line) > _MAX_LINE_SIZE:
                raise ValueError(
                    f"Line {line_number} exceeds maximum allowed size "
                    f"({_MAX_LINE_SIZE} bytes): {path}"
                )
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not is_encrypted(payload):
                continue
            try:
                yield decrypt(payload, key)
            except InvalidTag:
                raise IntegrityError(path, line_number=line_number) from None
