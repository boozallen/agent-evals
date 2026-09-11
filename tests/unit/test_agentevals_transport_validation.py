# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Transport validation of the judge endpoint on the agentevals LLM-judge factories.

These two factories are shaped differently from every other endpoint in this
change. Their ``model`` parameter is a provider-prefixed *nickname*
(``"ollama:gpt-oss:20b"``), not a URL, and the address behind it is resolved
lazily by the provider SDK inside its own scorer closure. A nickname alone
therefore leaves this library with no address to check at the moment prompts and
trajectories are sent, which is why ``judge_base_url`` is required rather than
optional: the caller names the endpoint, it is validated at construction like
every other caller-supplied endpoint, and it is bound to the provider through the
SDK's ``judge=`` seam.

The registry guard in ``tests/validation/test_transport_enforcement.py`` selects
factories by a ``base_url`` parameter and therefore skips both of these. This
file is the only coverage of the judge path.
"""

from __future__ import annotations

import pytest

from agent_evals.adapters.scorers.agentevals import (
    GraphTrajectoryLLMAsJudge,
    TrajectoryLLMAsJudge,
)

# Both LLM-judge factories. Ids name the factory, not an index.
FACTORIES = [
    pytest.param(TrajectoryLLMAsJudge, id="TrajectoryLLMAsJudge"),
    pytest.param(GraphTrajectoryLLMAsJudge, id="GraphTrajectoryLLMAsJudge"),
]

# A model whose provider package ships with the SDK chain, so building a judge
# succeeds without a local model server. The default "ollama:..." nickname needs
# langchain-ollama, which is not a dependency of this package.
BUILDABLE_MODEL = "openai:gpt-4o-mini"


@pytest.fixture(autouse=True)
def _api_key(monkeypatch):
    """Building an OpenAI-family judge requires a key; the value is never used."""
    monkeypatch.setenv("OPENAI_API_KEY", "test-key-not-used")


@pytest.mark.parametrize("factory", FACTORIES)
def test_cleartext_judge_endpoint_rejected_at_factory_call_time(factory):
    """No scorer callable is returned; the error is a configuration error."""
    with pytest.raises(ValueError, match="judge_base_url"):
        factory(
            model=BUILDABLE_MODEL, judge_base_url="http://judge.example.com:11434/v1"
        )


@pytest.mark.parametrize("factory", FACTORIES)
def test_omitted_endpoint_is_rejected(factory):
    """An unnamed endpoint is refused, not warned about.

    Leaving resolution to the provider SDK would send prompts and trajectories to
    an address this library never sees, so there would be nothing to check. The
    message has to say which factory and which parameter, since the caller's only
    fix is to supply one.
    """
    with pytest.raises(ValueError, match="judge_base_url") as exc_info:
        factory(model=BUILDABLE_MODEL)

    assert factory.__name__ in str(exc_info.value)


@pytest.mark.parametrize("factory", FACTORIES)
def test_no_environment_variable_permits_an_unvalidated_judge(factory, monkeypatch):
    """There is no opt-out; a plausible bypass name changes nothing."""
    monkeypatch.setenv("FOUNDRY_AGENT_EVALS_ALLOW_HTTP", "1")

    with pytest.raises(ValueError, match="judge_base_url"):
        factory(model=BUILDABLE_MODEL)
    with pytest.raises(ValueError, match="judge_base_url"):
        factory(
            model=BUILDABLE_MODEL, judge_base_url="http://judge.example.com:11434/v1"
        )


@pytest.mark.parametrize("factory", FACTORIES)
def test_rejection_precedes_the_sdk_import_guard(factory, monkeypatch):
    """A bad endpoint raises even when the SDK is unavailable.

    Validation sits ahead of the ``ImportError`` guard, so an environment without
    ``agentevals`` gets the configuration error rather than a silently degraded
    scorer that hides it.
    """
    import builtins

    real_import = builtins.__import__

    def _no_agentevals(name, *args, **kwargs):
        if name.startswith("agentevals"):
            raise ImportError("simulated: agentevals not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_agentevals)

    with pytest.raises(ValueError, match="judge_base_url"):
        factory(
            model=BUILDABLE_MODEL, judge_base_url="http://judge.example.com:11434/v1"
        )
    with pytest.raises(ValueError, match="judge_base_url"):
        factory(model=BUILDABLE_MODEL)


@pytest.mark.parametrize("factory", FACTORIES)
def test_https_judge_endpoint_accepted(factory):
    scorer = factory(
        model=BUILDABLE_MODEL, judge_base_url="https://judge.example.com/v1"
    )
    assert callable(scorer)


@pytest.mark.parametrize("factory", FACTORIES)
def test_loopback_judge_endpoint_accepted(factory):
    """A judge served from the developer's own machine must keep working.

    This is the shape the rejected nickname default resolved to, so it stays
    reachable — but only when the caller names it, so the address the library
    connects to is one it has actually seen and checked.
    """
    scorer = factory(model=BUILDABLE_MODEL, judge_base_url="http://localhost:11434/v1")
    assert callable(scorer)


def test_validated_endpoint_reaches_the_provider():
    """The validated value must be the one the provider connects to.

    A validator whose result never reaches the transport is decoration. This
    asserts the endpoint arrives on the built judge, which the SDK receives via
    its ``judge=`` parameter.
    """
    endpoint = "https://judge.example.com/v1"
    from agent_evals.adapters.scorers.agentevals import _build_judge

    judge = _build_judge(BUILDABLE_MODEL, endpoint)

    assert judge is not None
    assert judge.openai_api_base == endpoint


@pytest.mark.parametrize("factory", FACTORIES)
def test_prompt_and_model_arguments_are_unaffected(factory):
    """The new parameter is keyword-only, so existing argument shapes still apply.

    Covers a positional prompt and a keyword prompt — the two shapes that carry
    caller-supplied configuration — each now naming an endpoint.
    """
    endpoint = "https://judge.example.com/v1"
    assert callable(
        factory(
            "Judge this: {outputs} vs {reference_outputs}",
            model=BUILDABLE_MODEL,
            judge_base_url=endpoint,
        )
    )
    assert callable(
        factory(
            prompt="Judge this: {outputs} vs {reference_outputs}",
            model=BUILDABLE_MODEL,
            judge_base_url=endpoint,
        )
    )


@pytest.mark.parametrize("factory", FACTORIES)
def test_local_path_is_not_an_encrypted_endpoint_for_a_judge(factory):
    """A judge endpoint is always a network address, so a bare path is a typo.

    The shared allowlist admits local filesystem paths for MLflow's sake. That is
    still the correct shared behaviour — but it is worth pinning what the judge
    path does with a malformed value, which is reject it.
    """
    with pytest.raises(ValueError, match="judge_base_url"):
        factory(model=BUILDABLE_MODEL, judge_base_url="htp://judge.example.com/v1")
