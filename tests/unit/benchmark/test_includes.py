# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for benchmark URI-resolution helpers (composition layer)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent_evals.benchmark.includes import (
    BenchmarkConfigReadError,
    _is_uri_string,
    _resolve_list_entries,
    _resolve_relative,
    _split_uri,
    _uri_path_to_filesystem_path,
)
from agent_evals.core._registries import _BenchmarkConfigReaderRegistry


@pytest.fixture
def fresh_registry():
    """Return a fresh registry; tests register fakes against it directly."""
    return _BenchmarkConfigReaderRegistry()


class _FakeReader:
    """In-memory reader for tests; route URIs to canned content."""

    content: dict[str, str] = {}

    def read(self, uri: str) -> str:
        if uri not in self.content:
            raise FileNotFoundError(f"File not found: {uri}")
        return self.content[uri]


@pytest.fixture
def fake_reader(fresh_registry):
    fresh_registry.register("fake")(_FakeReader)
    _FakeReader.content = {}
    return _FakeReader


# ---------- _is_uri_string ----------


class TestIsUriString:
    def test_recognizes_registered_scheme(self, fresh_registry, fake_reader) -> None:
        assert _is_uri_string("fake://x.yaml", registry=fresh_registry) is True

    def test_rejects_unregistered_scheme(self, fresh_registry, fake_reader) -> None:
        # ``s3`` is not registered in this test context.
        assert _is_uri_string("s3://bucket/x.yaml", registry=fresh_registry) is False

    def test_rejects_plain_string(self, fresh_registry, fake_reader) -> None:
        assert _is_uri_string("not a uri", registry=fresh_registry) is False

    def test_rejects_non_string(self, fresh_registry, fake_reader) -> None:
        assert _is_uri_string({}, registry=fresh_registry) is False
        assert _is_uri_string(["fake://x.yaml"], registry=fresh_registry) is False
        assert _is_uri_string(42, registry=fresh_registry) is False
        assert _is_uri_string(None, registry=fresh_registry) is False


# ---------- _split_uri ----------


class TestSplitUri:
    def test_splits_scheme_and_rest(self) -> None:
        scheme, rest = _split_uri("file:///abs/path.yaml")
        assert scheme == "file"
        assert rest == "/abs/path.yaml"

    def test_splits_relative_uri(self) -> None:
        scheme, rest = _split_uri("file://relative/path.yaml")
        assert scheme == "file"
        # ``rest`` includes the netloc segment because the URI is
        # relative; absolute resolution is the caller's job.
        assert "relative" in rest


# ---------- _uri_path_to_filesystem_path ----------


class TestUriPathToFilesystemPath:
    """Decode-level tests, exercised directly rather than through the resolver.

    The decode step is hand-rolled (``url2pathname`` is deprecated on 3.14),
    so the drive-prefix strip and the percent-decode are asserted here
    independently. A bug in either would otherwise only surface as a
    confusing containment result several layers up.
    """

    def test_posix_absolute_path(self) -> None:
        assert _uri_path_to_filesystem_path("file:///proj/bench/child.yaml") == Path(
            "/proj/bench/child.yaml"
        )

    def test_windows_drive_prefix_slash_is_stripped(self) -> None:
        # ``file:///C:/x`` has path ``/C:/x``; the leading slash precedes a
        # drive letter and must go, or the path is nonsense on Windows and
        # anchors oddly elsewhere.
        assert _uri_path_to_filesystem_path("file:///C:/proj/bench/child.yaml") == Path(
            "C:/proj/bench/child.yaml"
        )

    def test_drive_prefix_strip_is_only_for_drive_letters(self) -> None:
        # A leading slash NOT followed by ``X:`` is a genuine POSIX root and
        # must survive, otherwise every absolute POSIX path becomes relative.
        assert str(_uri_path_to_filesystem_path("file:///proj/child.yaml")).startswith(
            ("/", "\\")
        )

    def test_percent_encoded_dot_segments_are_decoded(self) -> None:
        # ``%2e%2e`` is ``..``. Undecoded, a containment check sees an opaque
        # segment and reports "contained"; decoded, the escape is visible.
        decoded = _uri_path_to_filesystem_path(
            "file:///proj/bench/%2e%2e/%2e%2e/etc/passwd"
        )
        assert ".." in decoded.parts

    def test_percent_encoded_forward_slash_is_decoded(self) -> None:
        # ``..%2f..%2f`` decodes to ``../../`` — separators appear only after
        # decoding, so the segment count changes.
        decoded = _uri_path_to_filesystem_path(
            "file:///proj/bench/..%2f..%2fetc/passwd"
        )
        assert ".." in decoded.parts

    def test_encoded_vectors_escape_the_parent_once_resolved(self) -> None:
        """Both encoded forms resolve outside the parent directory.

        This is the assertion that justifies decoding before comparing:
        the same URIs report as contained under URI-level normalization.
        """
        parent_dir = Path("/proj/bench").resolve()
        for uri in (
            "file:///proj/bench/%2e%2e/%2e%2e/etc/passwd",
            "file:///proj/bench/..%2f..%2fetc/passwd",
        ):
            resolved = _uri_path_to_filesystem_path(uri).resolve()
            assert not resolved.is_relative_to(parent_dir), (
                f"{uri} decoded to {resolved}, which should be outside {parent_dir}"
            )

    def test_unencoded_child_stays_inside_the_parent(self) -> None:
        # Control case: a plain child must remain contained, or the check
        # would reject everything.
        parent_dir = Path("/proj/bench").resolve()
        resolved = _uri_path_to_filesystem_path(
            "file:///proj/bench/child.yaml"
        ).resolve()
        assert resolved.is_relative_to(parent_dir)

    def test_returns_unresolved_path(self) -> None:
        # The decode step deliberately does not resolve; the caller does.
        # Keeping them separate is what makes this class testable.
        assert ".." in _uri_path_to_filesystem_path("file:///a/b/../c.yaml").parts


# ---------- containment (_resolve_relative) ----------


class TestContainmentRejects:
    """One assertion per attack vector, each named in a comment.

    Every vector here is asserted on **all platforms**, including the
    ``ubuntu-latest`` runner that is CI's only OS. That uniformity is
    deliberate and is the reason the backslash forms are rejected outright
    rather than platform-guarded: a guarded test would skip in CI, leaving
    the vectors named but unproven on the one platform that runs them.
    See ``_reject_backslash`` for the rationale.
    """

    @staticmethod
    def _base(tmp_path: Path) -> str:
        return (tmp_path / "bench" / "main.yaml").as_uri()

    def test_relative_dotdot_escape(self, tmp_path: Path) -> None:
        # `../` — the plain relative escape. Resolves above the parent dir.
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_relative("file://../../evil.yaml", self._base(tmp_path))
        assert "outside" in str(exc.value)

    def test_absolute_path_outside_parent(self, tmp_path: Path) -> None:
        # Absolute URI pointing elsewhere on the filesystem. Previously
        # returned unchanged; now checked. This is the breaking change.
        outsider = (tmp_path / "elsewhere" / "evil.yaml").as_uri()
        with pytest.raises(BenchmarkConfigReadError):
            _resolve_relative(outsider, self._base(tmp_path))

    def test_windows_backslash_separators(self, tmp_path: Path) -> None:
        # `..\..\` — a traversal on Windows, an inert filename on POSIX.
        # Rejected on both so CI asserts it.
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_relative("file://..\\..\\evil.yaml", self._base(tmp_path))
        assert "backslash" in str(exc.value)

    def test_backslash_folded_absolute(self, tmp_path: Path) -> None:
        # Backslash-folding inside an otherwise absolute URI.
        base = self._base(tmp_path)
        parent_dir = tmp_path / "bench"
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_relative(f"{parent_dir.as_uri()}/\\..\\..\\evil.yaml", base)
        assert "backslash" in str(exc.value)

    def test_percent_encoded_dotdot(self, tmp_path: Path) -> None:
        # `%2e%2e` — `..` percent-encoded. Invisible to a URI-text check.
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_relative("file://%2e%2e/%2e%2e/evil.yaml", self._base(tmp_path))
        assert "outside" in str(exc.value)

    def test_percent_encoded_forward_slash(self, tmp_path: Path) -> None:
        # `..%2f` — the separator itself encoded, so the segment count only
        # changes after decoding.
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_relative("file://..%2f..%2fevil.yaml", self._base(tmp_path))
        assert "outside" in str(exc.value)

    def test_percent_encoded_backslash(self, tmp_path: Path) -> None:
        # `..%5c` — encoded backslash. Traversal on Windows only, so it is
        # caught by the platform-independent backslash rule instead.
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_relative("file://..%5c..%5cevil.yaml", self._base(tmp_path))
        assert "backslash" in str(exc.value)

    def test_error_names_both_ends_of_the_link(self, tmp_path: Path) -> None:
        base = self._base(tmp_path)
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_relative("file://../../evil.yaml", base)
        msg = str(exc.value)
        assert "../../evil.yaml" in msg, "candidate URI must appear"
        assert base in msg, "parent URI must appear so the link is traceable"


class TestContainmentAccepts:
    """Safe cases must keep working, each with why it is safe."""

    @staticmethod
    def _base(tmp_path: Path) -> str:
        return (tmp_path / "bench" / "main.yaml").as_uri()

    def test_plain_relative_child(self, tmp_path: Path) -> None:
        # Sibling of the parent config — the documented ordinary case.
        out = _resolve_relative("file://child.yaml", self._base(tmp_path))
        assert out == (tmp_path / "bench" / "child.yaml").resolve().as_uri()

    def test_child_in_subdirectory(self, tmp_path: Path) -> None:
        # Below the parent directory, so still contained.
        out = _resolve_relative("file://caps/lights.yaml", self._base(tmp_path))
        assert out == (tmp_path / "bench" / "caps" / "lights.yaml").resolve().as_uri()

    def test_dotdot_that_stays_inside(self, tmp_path: Path) -> None:
        # `sub/../child.yaml` normalizes back inside the parent. Containment
        # is about where a path lands, not whether it contains `..`.
        out = _resolve_relative("file://sub/../child.yaml", self._base(tmp_path))
        assert out == (tmp_path / "bench" / "child.yaml").resolve().as_uri()

    def test_absolute_uri_inside_parent(self, tmp_path: Path) -> None:
        # The absolute form is not banned — only escape is. AC 11.
        target = tmp_path / "bench" / "child.yaml"
        out = _resolve_relative(target.as_uri(), self._base(tmp_path))
        assert out == target.resolve().as_uri()

    def test_non_file_scheme_passes_through(self, tmp_path: Path) -> None:
        # Containment is a filesystem notion; remote schemes are untouched.
        uri = "s3://bucket/caps/lights.yaml"
        assert _resolve_relative(uri, self._base(tmp_path)) == uri

    def test_returns_canonical_uri_not_a_bool(self, tmp_path: Path) -> None:
        # The helper hands back the checked value, so a caller cannot use
        # the unchecked input by forgetting to branch on a boolean.
        out = _resolve_relative("file://sub/../child.yaml", self._base(tmp_path))
        assert isinstance(out, str)
        assert out.startswith("file:")
        assert ".." not in out


def test_symlinked_parent_cannot_be_straddled(tmp_path: Path) -> None:
    """A symlink to the parent does not widen containment.

    Both sides are resolved before comparison, so the link and its target
    collapse to one real path and a candidate outside that real path is
    rejected — the link cannot be used to reach a sibling directory.
    """
    real_parent = tmp_path / "real_bench"
    real_parent.mkdir()
    secret = tmp_path / "secret"
    secret.mkdir()
    (secret / "evil.yaml").write_text("name: x\n", encoding="utf-8")
    link = tmp_path / "linked_bench"
    try:
        link.symlink_to(real_parent, target_is_directory=True)
    except OSError, NotImplementedError:
        pytest.skip("symlink creation unavailable on this platform/account")

    base = (link / "main.yaml").as_uri()
    with pytest.raises(BenchmarkConfigReadError):
        _resolve_relative("file://../secret/evil.yaml", base)


# ---------- unusual file-URI forms ----------


class TestUnusualFileUriForms:
    def test_drive_in_netloc_raises_the_documented_type(self, tmp_path: Path) -> None:
        """``file://C:/...`` raises BenchmarkConfigReadError, not bare OSError.

        Previously this spelling reached the reader as a path with an
        embedded colon and surfaced ``OSError: Bad URL``, escaping the
        documented contract. AC 12.
        """
        base = (tmp_path / "bench" / "main.yaml").as_uri()
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_relative("file://C:/Windows/win.ini", base)
        assert "authority" in str(exc.value)

    def test_unc_base_uri_keeps_sibling_includes_working(self) -> None:
        """A UNC base URI resolves siblings against the right directory.

        ``urlsplit`` puts the host in ``netloc`` and only ``/share/...`` in
        ``path``; reading the path alone would compute containment against
        the wrong directory and break this legitimate include. AC 12.
        """
        base = "file://server/share/bench/main.yaml"
        out = _resolve_relative("file://child.yaml", base)
        # The server segment must survive rather than being silently dropped.
        assert "server" in out
        assert out.endswith("child.yaml")

    def test_unc_escape_is_still_rejected(self) -> None:
        base = "file://server/share/bench/main.yaml"
        with pytest.raises(BenchmarkConfigReadError):
            _resolve_relative("file://../../evil.yaml", base)


# ---------- _resolve_list_entries ----------


class TestResolveListEntries:
    def test_inline_dicts_pass_through(self, fresh_registry, fake_reader) -> None:
        entries = [{"name": "a"}, {"name": "b"}]
        assert (
            _resolve_list_entries(
                entries, base_uri="file:///parent.yaml", registry=fresh_registry
            )
            == entries
        )

    def test_resolves_uri_string_to_parsed_yaml(
        self, fresh_registry, fake_reader
    ) -> None:
        fake_reader.content = {
            "fake:///child.yaml": "name: lights\nscenarios: []\n",
        }
        entries = [{"name": "first"}, "fake:///child.yaml"]
        out = _resolve_list_entries(
            entries, base_uri="fake:///parent.yaml", registry=fresh_registry
        )
        assert out == [
            {"name": "first"},
            {"name": "lights", "scenarios": []},
        ]

    def test_splices_when_child_is_a_list(self, fresh_registry, fake_reader) -> None:
        """A child YAML may itself be a list — splice into the parent list."""
        fake_reader.content = {
            "fake:///many.yaml": (
                "- {name: a, scenarios: []}\n- {name: b, scenarios: []}\n"
            ),
        }
        out = _resolve_list_entries(
            [{"name": "first"}, "fake:///many.yaml", {"name": "last"}],
            base_uri="fake:///parent.yaml",
            registry=fresh_registry,
        )
        assert out == [
            {"name": "first"},
            {"name": "a", "scenarios": []},
            {"name": "b", "scenarios": []},
            {"name": "last"},
        ]

    def test_resolves_relative_uri_against_parent(
        self, fresh_registry, tmp_path: Path
    ) -> None:
        """Relative file:// URIs resolve against the parent file's directory."""
        # Use the real LocalFileConfigReader for this test so the
        # cross-platform path-joining logic is exercised end-to-end.
        from agent_evals.adapters.benchmark_config_readers.local_file import (
            LocalFileConfigReader,
        )

        fresh_registry.register("file")(LocalFileConfigReader)
        child = tmp_path / "child.yaml"
        child.write_text("name: lights\nscenarios: []\n", encoding="utf-8")
        parent = tmp_path / "parent.yaml"

        out = _resolve_list_entries(
            ["file://child.yaml"],
            base_uri=parent.as_uri(),
            registry=fresh_registry,
        )
        assert out == [{"name": "lights", "scenarios": []}]

    def test_recurses_through_chained_includes(
        self, fresh_registry, fake_reader
    ) -> None:
        """A child file may itself reference another URI in its content."""
        # Note: recursion only kicks in if the child is a list whose
        # entries include URI strings — we re-walk the spliced result.
        # Single-dict children don't trigger recursion in v1 (the only
        # recurse-walked field is the parent's list, not nested fields
        # inside a returned dict).
        fake_reader.content = {
            "fake:///top.yaml": "- fake:///middle.yaml\n",
            "fake:///middle.yaml": "name: deepest\nscenarios: []\n",
        }
        out = _resolve_list_entries(
            ["fake:///top.yaml"],
            base_uri="fake:///parent.yaml",
            registry=fresh_registry,
        )
        assert out == [{"name": "deepest", "scenarios": []}]

    def test_attributes_errors_to_uri(self, fresh_registry, fake_reader) -> None:
        """Errors during child read/parse name BOTH parent and child URIs.

        The wrapper catches concrete I/O errors (FileNotFoundError,
        OSError, UnicodeError, ValueError) and re-raises a single
        BenchmarkConfigReadError so callers can ``except`` on one type.
        Original is preserved via ``__cause__``.
        """
        # Deliberately don't add to fake_reader.content — read raises.
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_list_entries(
                ["fake:///missing.yaml"],
                base_uri="fake:///parent.yaml",
                registry=fresh_registry,
            )
        msg = str(exc.value)
        assert "fake:///missing.yaml" in msg, "child URI must appear in error message"
        assert "fake:///parent.yaml" in msg, (
            "parent URI must appear so users see both ends of broken link"
        )
        # Original FileNotFoundError must be reachable via __cause__.
        assert isinstance(exc.value.__cause__, FileNotFoundError)

    def test_in_pass_memoization(self, fresh_registry, fake_reader) -> None:
        """Same URI referenced twice in one resolution pass is read once."""
        call_count = 0

        class _CountingReader:
            def read(self, uri: str) -> str:
                nonlocal call_count
                call_count += 1
                return "name: shared\nscenarios: []\n"

        fresh_registry.register("counting")(_CountingReader)

        out = _resolve_list_entries(
            ["counting:///x.yaml", "counting:///x.yaml"],
            base_uri="counting:///parent.yaml",
            registry=fresh_registry,
        )
        assert len(out) == 2
        assert out[0] == out[1] == {"name": "shared", "scenarios": []}
        assert call_count == 1, (
            f"expected 1 read for the duplicated URI; got {call_count}"
        )

    def test_empty_child_yaml_rejected(self, fresh_registry, fake_reader) -> None:
        """An empty child file raises rather than silently substituting None.

        Without this guard, ``yaml.safe_load("")`` returns None and the
        resolver appends None to the parent list, producing a
        misleading Pydantic error downstream that doesn't name the
        offending URI.
        """
        fake_reader.content = {"fake:///empty.yaml": ""}
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_list_entries(
                ["fake:///empty.yaml"],
                base_uri="fake:///parent.yaml",
                registry=fresh_registry,
            )
        assert "fake:///empty.yaml" in str(exc.value)
        assert "empty" in str(exc.value).lower()

    def test_scalar_child_yaml_rejected(self, fresh_registry, fake_reader) -> None:
        """A child file containing a bare scalar raises with URI attribution."""
        fake_reader.content = {"fake:///scalar.yaml": "just text\n"}
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_list_entries(
                ["fake:///scalar.yaml"],
                base_uri="fake:///parent.yaml",
                registry=fresh_registry,
            )
        msg = str(exc.value)
        assert "fake:///scalar.yaml" in msg
        # Should name the type the user got and what they need.
        assert "str" in msg

    def test_empty_list_child_yaml_rejected(self, fresh_registry, fake_reader) -> None:
        """A child file containing an empty list (``[]``) raises.

        Without this guard, the resolver would splice nothing into the
        parent capabilities list — a silent no-op the user can't easily
        trace. The error must name the URI so the empty file is
        discoverable.
        """
        fake_reader.content = {"fake:///empty_list.yaml": "[]\n"}
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_list_entries(
                ["fake:///empty_list.yaml"],
                base_uri="fake:///parent.yaml",
                registry=fresh_registry,
            )
        msg = str(exc.value)
        assert "fake:///empty_list.yaml" in msg
        assert "empty" in msg.lower()

    def test_empty_dict_child_yaml_rejected(self, fresh_registry, fake_reader) -> None:
        """A child file containing an empty mapping (``{}``) raises.

        ``{}`` would substitute as a structurally-invalid scenario into
        the parent and surface downstream as a Pydantic error with no
        URI attribution. Reject at the URI boundary instead.
        """
        fake_reader.content = {"fake:///empty_dict.yaml": "{}\n"}
        with pytest.raises(BenchmarkConfigReadError) as exc:
            _resolve_list_entries(
                ["fake:///empty_dict.yaml"],
                base_uri="fake:///parent.yaml",
                registry=fresh_registry,
            )
        msg = str(exc.value)
        assert "fake:///empty_dict.yaml" in msg
        assert "empty" in msg.lower()

    def test_uppercase_scheme_recognized(self, fresh_registry, fake_reader) -> None:
        """RFC 3986: schemes are case-insensitive.

        ``FILE://...`` should be detected as a URI string and dispatched
        to the same reader as ``file://...``.
        """
        assert _is_uri_string("FAKE:///x.yaml", registry=fresh_registry) is True
        assert _is_uri_string("Fake:///x.yaml", registry=fresh_registry) is True

    def test_cycle_detected_preserves_chain_order(
        self, fresh_registry, fake_reader
    ) -> None:
        """A cycle raises and the error reproduces the include chain.

        Detection is via "in-flight" tracking: while resolving URI X, if
        a recursive call asks to resolve X again, we raise. The cache
        alone (which keys by URI and reuses results) wouldn't catch
        this — nothing is cached for X yet at the moment of detection.

        The error preserves edge order via an in_flight_chain tuple
        (sorting the frozenset would lose it). Test threads three nodes
        so order matters: A -> B -> C -> A must show that exact chain.
        """
        fake_reader.content = {
            "fake:///a.yaml": "- fake:///b.yaml\n",
            "fake:///b.yaml": "- fake:///c.yaml\n",
            "fake:///c.yaml": "- fake:///a.yaml\n",
        }
        with pytest.raises(ValueError) as exc:
            _resolve_list_entries(
                ["fake:///a.yaml"],
                base_uri="fake:///parent.yaml",
                registry=fresh_registry,
            )
        msg = str(exc.value)
        assert "cycle" in msg.lower() or "recursive" in msg.lower()
        expected_chain = (
            "fake:///a.yaml -> fake:///b.yaml -> fake:///c.yaml -> fake:///a.yaml"
        )
        assert expected_chain in msg, (
            f"cycle message must reproduce edge order; got: {msg}"
        )


class TestCanonicalKeying:
    """Equivalent spellings of one target collapse to a single key.

    Before containment, ``_resolve_relative`` returned the URI as written,
    so ``file://sub/../child.yaml`` and ``file://child.yaml`` were two keys
    in both ``cache`` and ``in_flight`` — which made the read-once memo and
    the cycle detector evadable by respelling. The resolver now returns the
    canonical URI and the walker keys on that. AC 8.
    """

    def test_two_spellings_are_one_cache_key(self, tmp_path: Path) -> None:
        reads: list[str] = []

        class _CountingFileReader:
            def read(self, uri: str) -> str:
                reads.append(uri)
                return "name: shared\nscenarios: []\n"

        registry = _BenchmarkConfigReaderRegistry()
        registry.register("file")(_CountingFileReader)

        bench = tmp_path / "bench"
        bench.mkdir()
        (bench / "sub").mkdir()
        base = (bench / "main.yaml").as_uri()

        out = _resolve_list_entries(
            ["file://child.yaml", "file://sub/../child.yaml"],
            base_uri=base,
            registry=registry,
        )
        assert len(out) == 2
        assert out[0] == out[1] == {"name": "shared", "scenarios": []}
        assert len(reads) == 1, (
            f"equivalent spellings must share one cache key; got reads {reads}"
        )

    def test_cycle_via_equivalent_spelling_still_raises(self, tmp_path: Path) -> None:
        bench = tmp_path / "bench"
        bench.mkdir()
        (bench / "sub").mkdir()
        a = bench / "a.yaml"
        b = bench / "b.yaml"
        # b points back at a under a different spelling of the same file.
        a.write_text("- file://b.yaml\n", encoding="utf-8")
        b.write_text("- file://sub/../a.yaml\n", encoding="utf-8")

        from agent_evals.adapters.benchmark_config_readers.local_file import (
            LocalFileConfigReader,
        )

        registry = _BenchmarkConfigReaderRegistry()
        registry.register("file")(LocalFileConfigReader)

        with pytest.raises(ValueError) as exc:
            _resolve_list_entries(
                ["file://a.yaml"],
                base_uri=(bench / "main.yaml").as_uri(),
                registry=registry,
            )
        msg = str(exc.value)
        assert "cycle" in msg.lower() or "recursive" in msg.lower()
        # Chain must be in edge order: a -> b -> a, on canonical URIs.
        canonical_a = a.resolve().as_uri()
        canonical_b = b.resolve().as_uri()
        assert msg.count(canonical_a) == 2, f"a must appear at both ends; got {msg}"
        assert canonical_b in msg


def test_rejection_happens_before_the_reader_is_called(tmp_path: Path) -> None:
    """An out-of-bounds include produces zero reads.

    Ordering is the point: if containment ran after the fetch, the file
    would already have been read (and any read side effect already taken)
    by the time the error was raised. AC 5.
    """
    reads: list[str] = []

    class _SpyReader:
        def read(self, uri: str) -> str:
            reads.append(uri)
            return "name: leaked\nscenarios: []\n"

    registry = _BenchmarkConfigReaderRegistry()
    registry.register("file")(_SpyReader)

    base = (tmp_path / "bench" / "main.yaml").as_uri()
    with pytest.raises(BenchmarkConfigReadError):
        _resolve_list_entries(
            ["file://../../evil.yaml"], base_uri=base, registry=registry
        )
    assert reads == [], f"reader must not be called for a rejected include; got {reads}"


def test_resolver_no_longer_launders_a_reader_rejectable_uri(tmp_path: Path) -> None:
    """The resolver does not convert a reader-rejected form into an accepted one.

    ``LocalFileConfigReader`` rejects relative include shorthand, but the
    resolver used to absolutize the ``..`` sequences first, so the reader
    saw a well-formed absolute path and accepted it — the resolver defeated
    the reader's own check. Containment now stops it at the resolver.
    """
    from agent_evals.adapters.benchmark_config_readers.local_file import (
        LocalFileConfigReader,
    )

    registry = _BenchmarkConfigReaderRegistry()
    registry.register("file")(LocalFileConfigReader)

    bench = tmp_path / "bench"
    bench.mkdir()
    (tmp_path / "outside.yaml").write_text(
        "name: leaked\nscenarios: []\n", encoding="utf-8"
    )
    base = (bench / "main.yaml").as_uri()

    with pytest.raises(BenchmarkConfigReadError) as exc:
        _resolve_list_entries(
            ["file://../outside.yaml"], base_uri=base, registry=registry
        )
    # Must fail on containment, not on a downstream read error.
    assert "outside the referencing file's directory" in str(exc.value)


def test_resolve_list_entries_without_registry_kwarg_raises():
    """``_resolve_list_entries`` without ``registry=`` raises TypeError."""
    from agent_evals.benchmark.includes import _resolve_list_entries

    with pytest.raises(TypeError) as exc_info:
        _resolve_list_entries(  # ty: ignore[missing-argument]
            ["inline-string-entry"],
            base_uri="file:///tmp/parent.yaml",
        )
    msg = str(exc_info.value)
    assert "missing" in msg
    assert "registry" in msg
