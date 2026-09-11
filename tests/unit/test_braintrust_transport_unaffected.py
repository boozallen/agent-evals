# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""The Braintrust trace-query endpoint is unaffected by transport enforcement.

``BTQL_ENDPOINT`` is a module-level ``https`` literal assigned once and never
reassigned, so no caller can downgrade it and it needs no validation. That makes
it the one path worth a *negative* regression test: enforcement must not break a
trace query that was already encrypted.
"""

from __future__ import annotations

import json
from urllib.parse import urlparse

import pytest

from agent_evals.adapters.platforms.braintrust import BTQL_ENDPOINT, BraintrustPlatform

pytestmark = pytest.mark.requires_braintrust


class _FakeResponse:
    """Minimal stand-in for the urlopen context manager."""

    def __init__(self, body: bytes) -> None:
        self._body = body
        self.headers = _FakeHeaders()

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc_info: object) -> bool:
        return False

    def read(self) -> bytes:
        return self._body


class _FakeHeaders:
    def get_content_charset(self) -> str:
        return "utf-8"


@pytest.fixture
def captured_requests(monkeypatch):
    """Capture the request without opening a socket.

    ``timeout`` is asserted rather than ignored: the adapter passes a bounded
    timeout today, and a stub that accepted and discarded it would let that
    behaviour regress silently (and vulture would flag the unused parameter).
    """
    requests = []

    def _fake_urlopen(request, timeout=None):
        assert timeout is not None, (
            "the adapter must pass a bounded timeout; an unbounded request can "
            "hang a run indefinitely"
        )
        requests.append(request)
        return _FakeResponse(json.dumps({"data": []}).encode("utf-8"))

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    return requests


def test_btql_endpoint_is_still_https():
    """The literal itself, in case someone edits the constant."""
    assert urlparse(BTQL_ENDPOINT).scheme == "https"


def test_trace_query_proceeds_with_enforcement_active(captured_requests):
    adapter = BraintrustPlatform()

    adapter.pull_traces(config={"project_id": "p-1", "api_key": "k-1"})

    assert len(captured_requests) == 1
    assert captured_requests[0].full_url == BTQL_ENDPOINT


def test_trace_query_endpoint_is_not_environment_dependent(
    captured_requests, monkeypatch
):
    """A hardcoded endpoint stays hardcoded whatever the environment says.

    Enforcement covers *caller-supplied* values. No variable rewrites an endpoint
    this library controls, so setting a plausible bypass name changes nothing —
    if it ever did, the constant would have stopped being constant.
    """
    monkeypatch.setenv("FOUNDRY_AGENT_EVALS_ALLOW_HTTP", "1")
    adapter = BraintrustPlatform()

    adapter.pull_traces(config={"project_id": "p-1", "api_key": "k-1"})

    assert len(captured_requests) == 1
    assert captured_requests[0].full_url == BTQL_ENDPOINT
    assert urlparse(captured_requests[0].full_url).scheme == "https"
