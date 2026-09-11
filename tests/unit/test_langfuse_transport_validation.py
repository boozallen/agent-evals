# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Transport validation of LANGFUSE_HOST.

The SDK reads ``LANGFUSE_HOST`` itself inside ``get_client()``, so the adapter
validates it in ``_get_client`` before that call. Placement matters: the
surrounding ``except Exception`` re-raises everything as a credentials
``RuntimeError``, so these tests assert the failure is a ``ValueError`` and
specifically *not* that ``RuntimeError`` — a regression that moved validation
inside the ``try`` would otherwise still "raise" and look green.
"""

from __future__ import annotations

import pytest

from agent_evals.adapters.platforms.langfuse import LangFusePlatform


class _StubClient:
    """Minimal stand-in for an injected LangfuseClientPort."""


def test_cleartext_host_raises_value_error_not_the_credentials_error(monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse.internal:3000")
    adapter = LangFusePlatform()

    with pytest.raises(ValueError, match="LANGFUSE_HOST") as exc_info:
        adapter._get_client()

    # Pins the before-the-try placement. RuntimeError is not a ValueError, so
    # this would fail if validation were moved inside the wrapped block.
    assert not isinstance(exc_info.value, RuntimeError)


def test_no_environment_variable_accepts_a_cleartext_host(monkeypatch):
    """There is no opt-out; a plausible bypass name changes nothing."""
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse.internal:3000")
    monkeypatch.setenv("FOUNDRY_AGENT_EVALS_ALLOW_HTTP", "1")
    adapter = LangFusePlatform()

    with pytest.raises(ValueError, match="LANGFUSE_HOST"):
        adapter._get_client()


def test_https_host_passes_validation(monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "https://cloud.langfuse.com")
    adapter = LangFusePlatform()

    try:
        adapter._get_client()
    except ValueError as e:  # pragma: no cover - fails the assertion below
        pytest.fail(f"an https host must pass validation: {e}")
    except RuntimeError:
        pass


def test_unset_host_is_not_an_error(monkeypatch):
    """The SDK supplies its own encrypted default when the host is unset."""
    monkeypatch.delenv("LANGFUSE_HOST", raising=False)
    adapter = LangFusePlatform()

    try:
        adapter._get_client()
    except ValueError as e:  # pragma: no cover - fails the assertion below
        pytest.fail(f"an unset host must not be a transport error: {e}")
    except RuntimeError:
        pass


def test_empty_host_is_not_an_error(monkeypatch):
    monkeypatch.setenv("LANGFUSE_HOST", "")
    adapter = LangFusePlatform()

    try:
        adapter._get_client()
    except ValueError as e:  # pragma: no cover - fails the assertion below
        pytest.fail(f"an empty host must not be a transport error: {e}")
    except RuntimeError:
        pass


def test_injected_client_skips_validation_entirely(monkeypatch):
    """A caller who injects a client owns that collaborator's transport.

    Validation sits inside the ``self._client is None`` branch, so a cleartext
    ambient host must not affect an injected client.
    """
    monkeypatch.setenv("LANGFUSE_HOST", "http://langfuse.internal:3000")
    stub = _StubClient()
    adapter = LangFusePlatform(client=stub)  # type: ignore[arg-type]

    assert adapter._get_client() is stub
