# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Transport validation of MLflow's caller-supplied tracking URI.

Both resolution paths are covered: ``aevaluate`` (MLFLOW_TRACKING_URI) and
``pull_traces`` (``config['tracking_uri']``). Validation sits in
``MLflowPlatform._configure_session``, not in the injectable session port, so
these tests inject a fake session and assert the fake is never configured on a
rejection — proving the value never reaches whatever would open a connection.
"""

from __future__ import annotations

import asyncio

import pytest
from helpers.fake_mlflow import FakeMlflowSession
from helpers.scorers import passing_scorer

from agent_evals.adapters.platforms.mlflow import MlflowConfig, MLflowPlatform
from agent_evals.core.types import ExampleData, ExpectedResult


def _dataset() -> list[ExampleData]:
    return [ExampleData(input="hi", expected=ExpectedResult(expected="echo: hi"))]


def _run_evaluate(adapter: MLflowPlatform) -> None:
    asyncio.run(
        adapter.aevaluate(
            task=lambda x: f"echo: {x}",
            dataset=_dataset(),
            evaluators=[passing_scorer],
            platform=MlflowConfig(experiment="test"),
        )
    )


class TestEvaluatePath:
    def test_cleartext_env_var_rejected(self, monkeypatch):
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example.com:5000")

        with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
            _run_evaluate(adapter)

        assert fake.configurations == [], (
            "the session must never be configured with a rejected URI"
        )

    def test_https_env_var_accepted(self, monkeypatch):
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "https://mlflow.example.com")

        _run_evaluate(adapter)

        assert len(fake.configurations) == 1
        assert fake.configurations[0]["tracking_uri"] == "https://mlflow.example.com"

    def test_local_store_accepted(self, monkeypatch):
        """A local path transmits nothing, so it must not be rejected.

        This is the case that would otherwise leave a local-development user
        with no accepted value at all.
        """
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "./mlruns")

        _run_evaluate(adapter)

        assert len(fake.configurations) == 1

    def test_loopback_tracking_server_accepted(self, monkeypatch):
        """A tracking server on the developer's own machine must keep working.

        MLflow's server speaks plaintext by default, so rejecting this would
        leave the documented local workflow unusable without a certificate or a
        TLS proxy. Loopback traffic reaches no network, so nothing is exposed.
        """
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

        _run_evaluate(adapter)

        assert len(fake.configurations) == 1
        assert fake.configurations[0]["tracking_uri"] == "http://localhost:5000"

    def test_no_environment_variable_accepts_cleartext(self, monkeypatch):
        """There is no opt-out; a plausible bypass name changes nothing."""
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example.com:5000")
        monkeypatch.setenv("FOUNDRY_AGENT_EVALS_ALLOW_HTTP", "1")

        with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
            _run_evaluate(adapter)

        assert fake.configurations == []

    def test_missing_env_var_still_reports_absence_not_transport(self, monkeypatch):
        """The pre-existing emptiness error is not taken over by validation."""
        adapter = MLflowPlatform(session=FakeMlflowSession())
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)

        with pytest.raises(ValueError, match="requires the MLFLOW_TRACKING_URI"):
            _run_evaluate(adapter)


class TestPullTracesPath:
    def test_cleartext_config_value_rejected(self):
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)

        with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
            adapter.pull_traces(
                config={
                    "tracking_uri": "http://mlflow.example.com:5000",
                    "experiment": "e",
                }
            )

        assert fake.configurations == []

    def test_https_config_value_accepted(self):
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)

        adapter.pull_traces(
            config={"tracking_uri": "https://mlflow.example.com", "experiment": "e"}
        )

        assert len(fake.configurations) == 1

    def test_no_environment_variable_accepts_cleartext(self, monkeypatch):
        """There is no opt-out; a plausible bypass name changes nothing."""
        fake = FakeMlflowSession()
        adapter = MLflowPlatform(session=fake)
        monkeypatch.setenv("FOUNDRY_AGENT_EVALS_ALLOW_HTTP", "1")

        with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
            adapter.pull_traces(
                config={
                    "tracking_uri": "http://mlflow.example.com:5000",
                    "experiment": "e",
                }
            )

        assert fake.configurations == []


class TestValidationSiteIsNotBypassableByInjection:
    def test_injected_session_does_not_remove_enforcement(self, monkeypatch):
        """The point of validating in the adapter rather than the port.

        A caller supplying their own session replaces ``configure``. If
        validation lived there, this would pass silently.
        """

        class NeverValidatingSession(FakeMlflowSession):
            pass

        adapter = MLflowPlatform(session=NeverValidatingSession())
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://mlflow.example.com:5000")

        with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
            _run_evaluate(adapter)
