# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""URI-resolution helpers for multi-file benchmark composition.

The benchmark loader uses these helpers to walk a parsed YAML dict
*before* validation, expanding ``file://...`` (or future ``s3://...``)
URI strings into the content of the referenced files. Resolution is
structural — only list-of-string positions are inspected — so string
values inside scenario fields are never accidentally fetched.

Composition happens before ``BenchmarkConfig.model_validate`` so the
Pydantic model never sees URI strings; ``extra="forbid"`` stays honest
and validation errors point at structural mistakes, not URI ambiguity.

``file://`` resolution is *contained*: an include may only name a file
inside the directory holding the config that references it. Containment
is evaluated on the decoded, resolved filesystem path rather than on the
URI text, because percent-encoded and backslash-bearing spellings look
contained as URIs while decoding to a path outside the parent. Accepted
candidates are returned in canonical form so cycle detection and the
read-once memo key on one spelling per target.

These functions are internal to ``benchmark/`` — not part of the
public API. See ``docs/architecture.md`` "Benchmark Config Readers"
for the design rationale.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import yaml

from agent_evals.core.ports import (
    AdapterRegistry,
    BenchmarkConfigReader,
)


class BenchmarkConfigReadError(RuntimeError):
    """Raised when reading or parsing an included benchmark-config URI fails.

    The original exception is preserved via ``__cause__`` (PEP 3134); the
    message names the URI that failed *and* the parent URI that referenced
    it, so users see both ends of a broken include link. We don't try to
    preserve the original exception's class — many stdlib exception
    classes (``UnicodeDecodeError``, ``urllib.error.HTTPError``) have
    multi-arg ``__init__`` signatures that can't be reconstructed from a
    single string, and downstream callers should ``except`` on this
    dedicated type rather than the leaf I/O exception anyway.
    """


def _is_uri_string(
    value: Any,
    *,
    registry: AdapterRegistry[type[BenchmarkConfigReader]],
) -> bool:
    """Return True iff ``value`` is a string whose scheme is registered.

    Scheme matching is case-insensitive (RFC 3986 §3.1). Schemes that are
    not registered (e.g. ``s3://`` when ``s3`` has no reader installed)
    return False so the value falls through to Pydantic validation, which
    will produce a structural error rather than the resolver mysteriously
    not fetching it.
    """
    if not isinstance(value, str):
        return False
    if "://" not in value:
        return False
    scheme = value.split("://", 1)[0].lower()
    return scheme in registry


def _split_uri(uri: str) -> tuple[str, str]:
    """Return ``(scheme, rest)`` for ``uri``."""
    scheme, _, rest = uri.partition("://")
    return scheme, rest


# Matches a leading ``/`` that precedes a Windows drive letter, as in the
# ``/C:/dir/file`` path component of ``file:///C:/dir/file``.
_LEADING_SLASH_BEFORE_DRIVE = re.compile(r"^/([A-Za-z]:)")

# Matches an authority that is nothing but a drive letter, as ``urlsplit``
# reports for the malformed-but-common ``file://C:/dir/file`` spelling.
_DRIVE_ONLY = re.compile(r"^[A-Za-z]:$")


def _uri_path_to_filesystem_path(uri: str) -> Path:
    """Decode the path component of a ``file://`` URI into a ``Path``.

    Percent-decoding happens here, which is the whole point: a check that
    reads the raw URI text is blind to ``%2e%2e`` and ``..%2f``, both of
    which decode into dot segments that escape the parent directory.
    Callers must therefore decode *before* resolving and comparing.

    ``urllib.request.url2pathname`` would do this, but it is deprecated —
    it delegates to ``nturl2path``, deprecated since Python 3.14, and this
    package supports 3.11 through 3.14. ``Path.from_uri`` is 3.13+, above
    our floor. So the two transformations that matter are done explicitly:
    percent-decoding, then stripping the leading slash that precedes a
    Windows drive letter.

    The returned path is *not* resolved; the caller resolves it, so that
    the decode step stays independently testable.
    """
    decoded = unquote(urlsplit(uri).path)
    return Path(_LEADING_SLASH_BEFORE_DRIVE.sub(r"\1", decoded))


def _reject_backslash(decoded: str, *, uri: str, base_uri: str) -> None:
    """Reject a decoded include path containing a backslash, on every platform.

    A backslash is a separator on Windows and an ordinary filename character
    on POSIX. Left to the host to interpret, ``..%5c..%5cevil.yaml`` and
    ``..\\..\\evil.yaml`` are traversals on Windows and inert single-segment
    names on Linux — so the containment check below would pass them in CI,
    which runs Linux only, and the vectors would go unasserted on the only
    platform the suite actually exercises.

    Rejecting outright rather than folding to ``/`` keeps the rule uniform:
    a URI is platform-independent text, no legitimate include filename in
    this feature contains a backslash, and a developer on Windows and CI on
    Linux then see identical behavior.

    Raises:
        BenchmarkConfigReadError: ``decoded`` contains a backslash.
    """
    if "\\" in decoded:
        raise BenchmarkConfigReadError(
            f"benchmark-config URI {uri!r} (referenced from {base_uri!r}) "
            f"contains a backslash, which is a path separator on some "
            f"platforms and an ordinary character on others. Use forward "
            f"slashes in include URIs."
        )


def _contain_within_parent(
    candidate_path: Path, parent_dir: Path, *, uri: str, base_uri: str
) -> Path:
    """Return the resolved ``candidate_path``, or raise if it escapes ``parent_dir``.

    Both sides are resolved before comparison, so a symlinked parent
    cannot be straddled: the link and its real target both collapse to the
    same real path, and a candidate outside that real target is rejected.

    The comparison runs on decoded filesystem paths, never on URI text.
    URI-level normalization reports ``%2e%2e/%2e%2e``, ``..%2f..%2f`` and
    ``\\..\\..\\`` as contained while all three decode to paths outside the
    parent, so a check written against the URI is a no-op against exactly
    the inputs an attacker supplies.

    Raises:
        BenchmarkConfigReadError: The candidate resolves outside
            ``parent_dir``. The message names both ends of the include
            link so the offending file is identifiable.
    """
    resolved = candidate_path.resolve()
    resolved_parent = parent_dir.resolve()
    if not resolved.is_relative_to(resolved_parent):
        raise BenchmarkConfigReadError(
            f"benchmark-config URI {uri!r} (referenced from {base_uri!r}) "
            f"resolves to {str(resolved)!r}, which is outside the "
            f"referencing file's directory {str(resolved_parent)!r}. "
            f"Includes may only name files at or below the directory of the "
            f"config that references them."
        )
    return resolved


def _resolve_relative(uri: str, base_uri: str) -> str:
    """Resolve a ``file://`` URI against ``base_uri``, contained to its directory.

    For schemes other than ``file``, returns ``uri`` unchanged — remote
    schemes (``s3``, ``https``) don't have a clean "relative" notion in
    v1, and absolute URIs for those schemes are required by convention.
    Containment is a filesystem notion, so it does not apply to them.

    Both ``file://`` forms are contained: the RFC 8089 absolute form
    (empty/``localhost`` netloc with a leading-slash path) and the
    promptfoo-style relative shorthand (``file://name/path``) that joins
    against the parent's directory. Absolute URIs were previously returned
    unchanged; they are now checked, which is a breaking change for
    absolute includes pointing outside the parent directory.

    Returns the *canonical* URI of the accepted target — the caller keys
    cycle detection and the read-once memo on it, so two spellings of one
    file collapse to a single key.

    Raises:
        BenchmarkConfigReadError: The target resolves outside the
            referencing file's directory, or the URI is a form whose
            filesystem meaning cannot be determined.
    """
    parts = urlsplit(uri)
    if parts.scheme != "file":
        return uri
    parent_dir = _parent_directory(base_uri)
    if parts.netloc in ("", "localhost") and parts.path.startswith("/"):
        decoded = unquote(parts.path)
        _reject_backslash(decoded, uri=uri, base_uri=base_uri)
        candidate = _uri_path_to_filesystem_path(uri)
    elif _DRIVE_ONLY.match(parts.netloc):
        # ``file://C:/dir/file`` puts the drive letter in the netloc. Left
        # alone this reaches the reader as a path with an embedded colon and
        # surfaces a bare ``OSError: Bad URL``, escaping this module's
        # documented error type. Reject it here, in the contract's terms.
        raise BenchmarkConfigReadError(
            f"benchmark-config URI {uri!r} (referenced from {base_uri!r}) "
            f"puts a drive letter in the URI authority. Use the RFC 8089 "
            f"absolute form (file:///C:/dir/file) or a relative include."
        )
    else:
        # Relative shorthand: the netloc is really the first path segment.
        # Percent-escapes are decoded here so ``..%2f`` becomes a separator
        # before containment looks at the result, not after.
        relative = unquote(parts.netloc + parts.path)
        _reject_backslash(relative, uri=uri, base_uri=base_uri)
        candidate = parent_dir / relative
    resolved = _contain_within_parent(candidate, parent_dir, uri=uri, base_uri=base_uri)
    return resolved.as_uri()


def _parent_directory(base_uri: str) -> Path:
    """Return the directory holding the config file named by ``base_uri``.

    Handles the UNC form (``file://server/share/dir/file``), which
    ``urlsplit`` reports with the host in ``netloc`` and only ``/share/...``
    in ``path``. Reading the path alone would compute containment against
    the wrong directory *and* break legitimate sibling includes, and
    ``benchmark/loader.py`` can itself produce this form from a UNC path.
    """
    parts = urlsplit(base_uri)
    if parts.netloc and parts.netloc.lower() != "localhost":
        # Reassemble the UNC root that urlsplit took apart.
        return Path(f"//{parts.netloc}{unquote(parts.path)}").parent
    return _uri_path_to_filesystem_path(base_uri).parent


def _resolve_list_entries(
    entries: list,
    *,
    base_uri: str,
    registry: AdapterRegistry[type[BenchmarkConfigReader]],
    cache: dict[str, Any] | None = None,
    in_flight: frozenset[str] = frozenset(),
    in_flight_chain: tuple[str, ...] = (),
) -> list:
    """Resolve URI strings in a list, parsing each into YAML content.

    For each entry:
      * **Inline value** (dict, scalar, etc.) — passes through unchanged.
      * **Registered URI string** — resolved via the matching reader,
        parsed as YAML, recursed into. If the parsed content is a
        ``list``, it splices into the parent (each child entry becomes
        one parent entry); otherwise it substitutes (one URI string →
        one parent entry).

    Args:
        entries: The list to walk.
        base_uri: The URI of the parent file. Used to resolve relative
            URIs and to attribute errors.
        registry: Reader registry to dispatch URI schemes through.
        cache: Per-pass URI → resolved-YAML memo. The loader passes a
            fresh ``dict()`` for each top-level call.
        in_flight: URIs currently mid-resolution. Used for cycle
            detection (different from the cache, which holds completed
            results). Immutable so each recursive call extends a new
            frozenset rather than mutating shared state.
        in_flight_chain: Ordered tuple shadowing ``in_flight`` so the
            cycle-detection error message can reproduce the actual
            include chain (sorting the frozenset would lose edge order).
    """
    if cache is None:
        cache = {}
    if registry is None:
        raise TypeError("_resolve_list_entries requires a non-None registry")
    out: list = []
    for entry in entries:
        if not _is_uri_string(entry, registry=registry):
            out.append(entry)
            continue
        absolute_uri = _resolve_relative(entry, base_uri)
        if absolute_uri in in_flight:
            chain = " -> ".join(in_flight_chain + (absolute_uri,))
            raise ValueError(f"Recursive include cycle detected: {chain}")
        if absolute_uri in cache:
            resolved = cache[absolute_uri]
        else:
            resolved = _read_and_parse(
                absolute_uri,
                parent_uri=base_uri,
                cache=cache,
                in_flight=in_flight | {absolute_uri},
                in_flight_chain=in_flight_chain + (absolute_uri,),
                registry=registry,
            )
            cache[absolute_uri] = resolved
        if isinstance(resolved, list):
            out.extend(resolved)
        else:
            out.append(resolved)
    return out


def _read_and_parse(
    uri: str,
    *,
    parent_uri: str,
    cache: dict[str, Any],
    in_flight: frozenset[str],
    in_flight_chain: tuple[str, ...],
    registry: AdapterRegistry[type[BenchmarkConfigReader]],
) -> Any:
    """Fetch ``uri`` content via the registry, parse YAML, recurse on lists.

    Errors from the registry lookup (unknown scheme) propagate verbatim;
    only the I/O leg (``reader.read(uri)``) and the YAML parse step are
    wrapped in ``BenchmarkConfigReadError`` with both the failed URI and
    the parent URI that referenced it.
    """
    scheme, _ = _split_uri(uri)
    reader = registry.get(scheme)()
    try:
        text = reader.read(uri)
    except (OSError, UnicodeError, ValueError) as exc:
        raise BenchmarkConfigReadError(
            f"failed to read benchmark-config URI {uri!r} "
            f"(referenced from {parent_uri!r}): {exc}"
        ) from exc
    try:
        parsed = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise BenchmarkConfigReadError(
            f"failed to parse YAML from {uri!r} (referenced from {parent_uri!r}): {exc}"
        ) from exc
    _validate_resolved(parsed, uri=uri, parent_uri=parent_uri)
    # Recurse only when the resolved content is a list; recursion on
    # nested fields inside a dict is out of scope for v1 (the loader
    # only recurse-walks ``capabilities:`` at the top level).
    if isinstance(parsed, list):
        return _resolve_list_entries(
            parsed,
            base_uri=uri,
            cache=cache,
            in_flight=in_flight,
            in_flight_chain=in_flight_chain,
            registry=registry,
        )
    return parsed


def _validate_resolved(parsed: Any, *, uri: str, parent_uri: str) -> None:
    """Reject empty / scalar child YAML with URI-attributed errors.

    The composition layer expects each included file to be either a
    ``CapabilitySpec`` mapping (one capability per file) or a list of
    them. Empty files (``yaml.safe_load("")`` returns ``None``) and
    bare scalars would otherwise silently substitute into the parent
    list and surface as cryptic Pydantic errors with no URI attribution.
    Empty list (``[]``) and empty dict (``{}``) are likewise rejected:
    the resolver would splice nothing into the parent (for ``[]``) or
    yield a structurally-invalid scenario (for ``{}``), in both cases
    silently — see the matching tests in test_composition.py.
    """
    if parsed is None:
        raise BenchmarkConfigReadError(
            f"benchmark-config URI {uri!r} resolved to empty content "
            f"(no YAML documents; referenced from {parent_uri!r}). "
            f"Expected a CapabilitySpec mapping or a list of them."
        )
    if not isinstance(parsed, (dict, list)):
        raise BenchmarkConfigReadError(
            f"benchmark-config URI {uri!r} resolved to a "
            f"{type(parsed).__name__} (referenced from {parent_uri!r}). "
            f"Expected a CapabilitySpec mapping or a list of them."
        )
    if not parsed:
        raise BenchmarkConfigReadError(
            f"benchmark-config URI {uri!r} resolved to empty content "
            f"(referenced from {parent_uri!r}). Expected a CapabilitySpec "
            f"mapping or a non-empty list of them."
        )
