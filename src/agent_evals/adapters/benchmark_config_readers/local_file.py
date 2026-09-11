# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Local-filesystem benchmark-config reader.

Registered under URI scheme ``file``. Reads UTF-8 text from absolute
``file://...`` URIs (``file:///abs/path`` or ``file://localhost/abs/path``;
both forms are RFC 8089 absolute, distinguished only by an optional
host component).

The reader requires **absolute** URIs. Relative-path resolution is the
composition layer's job: the loader joins a relative URI against the
parent URI *before* calling ``read()``. Keeping the reader stateless
and parameterless preserves SRP and matches the pattern other adapter
families use (the class IS the adapter; no per-call configuration).
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import url2pathname

from agent_evals.core._registries import config_reader_registry


@config_reader_registry.register("file")
class LocalFileConfigReader:
    """Read text content from absolute ``file://`` URIs.

    Accepts both ``file:///abs/path`` and ``file://localhost/abs/path``;
    both are RFC 8089 absolute. Rejects the relative shorthand form
    ``file://name/path`` — the composition layer must absolutize first.
    """

    def read(self, uri: str) -> str:
        """Return the UTF-8 text of the file referenced by ``uri``.

        ``uri`` must be an absolute ``file://...`` URI. The composition
        layer is responsible for resolving relative URIs against the
        parent file's URI before calling this method.

        Raises:
            ValueError: ``uri`` is not a ``file://`` URI, or is
                relative (a non-absolute path after the scheme).
            FileNotFoundError: The file does not exist. The error
                message includes the URI for attribution.
        """
        path = _uri_to_absolute_path(uri)
        try:
            return path.read_text(encoding="utf-8")
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"File not found: {uri}") from exc


def _uri_to_absolute_path(uri: str) -> Path:
    """Convert an absolute ``file://`` URI to a ``Path``.

    Accepts ``file:///abs/path`` and ``file://localhost/abs/path``
    (RFC 8089 absolute forms — empty or ``localhost`` netloc). The
    ambiguous ``file://something/path`` form — which promptfoo-style
    configs use as shorthand for "relative to the parent" — is
    rejected; the composition layer is responsible for expanding
    relative URIs to absolute before calling read().

    Uses ``urllib.request.url2pathname`` for cross-platform correctness
    (handles the Windows ``file:///C:/...`` form and percent-decoding).
    Python 3.14 deprecated the stdlib ``nturl2path`` module that
    ``url2pathname`` reaches for on Windows, not the public name used
    here, so ty reports a ``deprecated`` warning for this module on
    Windows and nothing on Linux. A ``[[tool.ty.overrides]]`` entry
    scoped to this file turns that rule off rather than an inline
    suppression comment, which would itself be flagged as unused on
    Linux; every other file still fails the gate on any deprecation.
    """
    parts = urlsplit(uri)
    if parts.scheme != "file":
        raise ValueError(
            f"LocalFileConfigReader requires a file:// URI; got "
            f"scheme {parts.scheme!r} in {uri!r}."
        )
    if parts.netloc not in ("", "localhost"):
        raise ValueError(
            f"LocalFileConfigReader requires an absolute URI "
            f"(file:///abs/path form); got relative {uri!r}. The "
            f"composition layer must resolve relative URIs against "
            f"the parent file's URI before calling read()."
        )
    candidate = Path(url2pathname(parts.path))
    if not candidate.is_absolute():
        raise ValueError(
            f"LocalFileConfigReader requires an absolute path; got {uri!r}."
        )
    return candidate


__all__ = ["LocalFileConfigReader"]
