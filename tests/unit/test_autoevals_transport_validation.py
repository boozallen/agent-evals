# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Transport validation of the caller-supplied ``base_url`` on autoevals factories.

The 12 public factories reach ``base_url`` by three different routes, so this
file covers one factory from each rather than all twelve — the exhaustive check
across the whole registry is
``tests/validation/test_transport_enforcement.py``.

  - via ``_build_scorer_kwargs``: Factuality (9 factories share this route)
  - direct to the creator: EmbeddingSimilarity
  - fully inline in the factory body: LLMClassifier, Moderation

Rejection must happen at factory-call time, not at scoring time: a
configuration error surfacing mid-run, once per example, is the failure mode
these tests pin against.
"""

from __future__ import annotations

import pytest

from agent_evals.adapters.scorers.autoevals import (
    EmbeddingSimilarity,
    Factuality,
    LLMClassifier,
    Moderation,
)

# One representative per route. Ids make a failure name the route, not an index.
FACTORIES = [
    pytest.param(Factuality, id="helper-route"),
    pytest.param(EmbeddingSimilarity, id="direct-to-creator"),
    pytest.param(LLMClassifier, id="inline-classifier"),
    pytest.param(Moderation, id="inline-moderation"),
]


@pytest.mark.parametrize("factory", FACTORIES)
def test_cleartext_base_url_rejected_at_factory_call_time(factory):
    """No scorer callable is returned; the error is a configuration error."""
    with pytest.raises(ValueError, match="base_url"):
        factory(base_url="http://ollama.example.com:11434/v1")


@pytest.mark.parametrize("factory", FACTORIES)
def test_https_base_url_accepted(factory):
    scorer = factory(base_url="https://ollama.example.com/v1")
    assert callable(scorer)


@pytest.mark.parametrize("factory", FACTORIES)
def test_loopback_base_url_accepted(factory):
    """A model server on the developer's own machine must keep working.

    A locally run inference server serves plaintext by default, and loopback
    traffic reaches no network, so admitting it exposes nothing.
    """
    scorer = factory(base_url="http://localhost:11434/v1")
    assert callable(scorer)


@pytest.mark.parametrize("factory", FACTORIES)
def test_no_environment_variable_permits_cleartext(factory, monkeypatch):
    """There is no opt-out; a plausible bypass name changes nothing."""
    monkeypatch.setenv("FOUNDRY_AGENT_EVALS_ALLOW_HTTP", "1")

    with pytest.raises(ValueError, match="base_url"):
        factory(base_url="http://ollama.example.com:11434/v1")


@pytest.mark.parametrize("factory", FACTORIES)
def test_default_none_base_url_accepted(factory):
    """The documented default path must be unaffected."""
    scorer = factory()
    assert callable(scorer)
