# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Disk-sink primitive for benchmark reports.

``_write_report`` is pure I/O; it does no registry resolution and has no
benchmark-runner coupling. ``BenchmarkResult.write_report`` (in
``benchmark/types.py``) is the public method that delegates here.

The ``filename`` override is validated before anything touches the
filesystem, because it is the one input to this module that can redirect
the write away from the requested directory.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from foundry_agent_core.encryption import encrypt, load_encryption_key

from agent_evals.benchmark.types import BenchmarkResult

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MiB — symmetric with _readers.py

# Windows treats these as device names in any directory, with or without an
# extension (``NUL``, ``nul.md``). Writing to one appears to succeed while the
# content goes to the device instead of a file.
_RESERVED_DEVICE_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)}
)

# Characters Win32 strips from the end of a path component before resolving it,
# which makes ``"NUL "`` and ``"NUL."`` reach the same device as ``"NUL"``.
# Verified on Windows 11: writing to ``NUL `` leaves nothing on disk, while
# ``COM1 `` creates a file named ``COM1``. Both must be folded away before the
# reserved-name comparison, or a trailing space walks straight past it.
_WIN32_TRAILING_TRIM = " ."

# Matches a leading drive specifier, as in ``C:evil.md``. This form carries no
# separator, so a separator-only guard admits it -- yet joining it to the
# output directory discards that directory entirely through drive anchoring:
# ``str(Path('benchmarks/results') / 'C:evil.md') == 'C:evil.md'``.
_DRIVE_PREFIX = re.compile(r"^[A-Za-z]:")


def _validate_filename(filename: str) -> str:
    """Return ``filename`` if it cannot redirect the write, else raise.

    Every rule here is enforced regardless of the host operating system.
    Deferring to the host would make a developer's Windows machine and the
    Linux CI runner disagree about which values are safe, so the rejected
    set is identical everywhere and CI asserts the same vectors a Windows
    developer does.

    The value is rejected rather than rewritten to its final component:
    silently turning ``sub/report.md`` into ``report.md`` hides a caller
    mistake and writes somewhere the caller did not ask for.

    Raises:
        ValueError: ``filename`` could redirect the write. The message names
            the offending value and the rule it broke.
    """
    if not filename or not filename.strip():
        raise ValueError("report filename must not be empty")
    if "/" in filename or "\\" in filename:
        raise ValueError(
            f"report filename {filename!r} contains a path separator. "
            f"Pass a bare filename; use results_dir to choose the directory."
        )
    if filename in (".", ".."):
        raise ValueError(
            f"report filename {filename!r} is a directory reference, not a name."
        )
    if _DRIVE_PREFIX.match(filename):
        raise ValueError(
            f"report filename {filename!r} starts with a drive specifier, "
            f"which would discard the output directory when joined to it."
        )
    if "\x00" in filename:
        raise ValueError(f"report filename {filename!r} contains a null byte.")
    # Compare on the name Win32 would actually resolve, not the name as
    # written: surrounding whitespace and trailing dots are stripped before
    # resolution, so ``"NUL "`` and ``"NUL."`` are the device too. Strip first,
    # then take the stem, so ``"nul.md."`` is caught as well.
    resolved = filename.strip().rstrip(_WIN32_TRAILING_TRIM)
    stem = resolved.split(".", 1)[0].upper()
    if stem in _RESERVED_DEVICE_NAMES:
        raise ValueError(
            f"report filename {filename!r} is a reserved device name on "
            f"Windows; writing to it would discard the report."
        )
    return filename


def _write_report(
    result: BenchmarkResult,
    results_dir: str | Path = "benchmarks/results",
    filename: str | None = None,
) -> Path:
    """Write the markdown report for `result` into `results_dir`.

    Creates `results_dir` if missing. Default filename is
    `{YYYY-MM-DD_HH-MM-SS}_{benchmark}.md`; pass `filename` to override
    (useful for deterministic tests or custom naming). An override that
    could redirect the write is rejected, before the directory is created,
    so a rejected call leaves no trace on disk.

    Returns the path that was written.

    Raises:
        ValueError: `filename` was supplied and could redirect the write.
    """
    key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
    name = (
        _validate_filename(filename)
        if filename is not None
        else f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{result.benchmark}.md"
    )
    content = result.to_markdown()
    if len(content) > _MAX_FILE_SIZE:
        raise ValueError(
            f"Report content exceeds maximum file size "
            f"({_MAX_FILE_SIZE} bytes): {len(content)} bytes"
        )
    out_dir = Path(results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    envelope = encrypt({"content": content}, key)
    path.write_text(json.dumps(envelope))
    return path
