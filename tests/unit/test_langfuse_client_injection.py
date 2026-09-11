# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Composition tests for the LangFusePlatform client-injection seam.

Pins that ``LangFusePlatform(client=fake)`` flows through to every
adapter call site that touches the SDK. Future refactors that
accidentally re-introduce a direct ``get_client()`` call inside an
adapter method must fail this test.
"""

import pytest
from helpers.fake_langfuse import FakeLangfuseClient

from agent_evals.adapters.platforms.langfuse import LangfuseConfig, LangFusePlatform
from agent_evals.adapters.platforms.langfuse_client import DatasetItemRecord
from agent_evals.core.types import (
    ExampleData,
    ExpectedResult,
    TaskResult,
)


class TestClientInjectionFlowsThrough:
    """The injected client must be the only client the adapter consults."""

    @pytest.mark.asyncio
    async def test_aevaluate_uses_injected_client(self):
        """``aevaluate`` must call the injected client, not module-level ``get_client``."""
        fake = FakeLangfuseClient()
        adapter = LangFusePlatform(client=fake)

        await adapter.aevaluate(
            task=lambda x: TaskResult(output="ok"),
            dataset=[ExampleData(input="hi", expected=ExpectedResult(expected="x"))],
            evaluators=[],
            platform=LangfuseConfig(experiment="injection-aevaluate"),
        )

        # If the adapter bypassed the injected client and called get_client()
        # directly, fake.experiments would be empty.
        assert len(fake.experiments) == 1
        assert fake.experiments[0]["name"] == "injection-aevaluate"

    def test_pull_traces_uses_injected_client(self):
        """``pull_traces`` must call the injected client, not module-level ``get_client``."""
        fake = FakeLangfuseClient(
            dataset_items={
                "d": [
                    DatasetItemRecord(
                        source_trace_id="t-1",
                        input="i",
                        expected_output="o",
                        metadata={},
                    )
                ]
            },
            trace_outputs={"t-1": "x"},
        )
        adapter = LangFusePlatform(client=fake)

        examples = adapter.pull_traces(config={"dataset": "d"})

        assert len(examples) == 1
        assert examples[0].input == "i"

    @pytest.mark.asyncio
    async def test_auth_failure_is_surfaced(self):
        """When the injected client's ``auth_check`` returns False, ``aevaluate`` raises."""
        fake = FakeLangfuseClient(auth_ok=False)
        adapter = LangFusePlatform(client=fake)

        with pytest.raises(RuntimeError, match="authentication failed"):
            await adapter.aevaluate(
                task=lambda x: TaskResult(output="ok"),
                dataset=[
                    ExampleData(input="hi", expected=ExpectedResult(expected="x"))
                ],
                evaluators=[],
                platform=LangfuseConfig(experiment="auth-fail"),
            )

    def test_default_construction_is_unchanged(self):
        """``LangFusePlatform()`` with no args still works — backward compatibility."""
        # We can't drive aevaluate without a real langfuse server, but we CAN
        # confirm construction succeeds and the lazy-cache field starts None.
        adapter = LangFusePlatform()
        assert adapter._client is None  # lazy: no get_client() yet

    def test_get_client_init_failure_wraps_in_runtime_error(self, monkeypatch):
        """SDK initialization failures surface as a uniform ``RuntimeError``.

        Without this wrap, the same root cause (bad creds, network, etc.)
        raises an uncaught SDK exception in ``aevaluate`` but gets
        re-wrapped as ``RuntimeError("Failed to export traces…")`` in
        ``pull_traces`` — two different error surfaces for the same
        problem, with the latter message misleadingly blaming trace
        export. This test pins the consistent surface.
        """
        from agent_evals.adapters.platforms import langfuse as langfuse_module

        def boom():
            raise ValueError("fake SDK init failure")

        monkeypatch.setattr(langfuse_module, "get_client", boom)

        adapter = LangFusePlatform()
        with pytest.raises(RuntimeError, match="could not obtain an SDK client"):
            adapter._get_client()

    def test_get_client_lazy_creates_real_wrapper_and_caches(self, monkeypatch):
        """The lazy-default path wraps ``get_client()`` in ``_RealLangfuseClient`` and caches.

        Without this test, the production fallback path is exercised only by
        live smoke tests (which need real credentials). A future commit that
        accidentally rewrote ``_get_client()`` to construct a fresh wrapper
        on every call — breaking SDK-level thread affinity assumptions —
        would silently pass unit tests but might break production.
        """
        from agent_evals.adapters.platforms import langfuse as langfuse_module
        from agent_evals.adapters.platforms.langfuse_client import (
            _RealLangfuseClient,
        )

        sentinel_sdk = object()
        call_count = {"n": 0}

        def fake_get_client():
            call_count["n"] += 1
            return sentinel_sdk

        monkeypatch.setattr(langfuse_module, "get_client", fake_get_client)

        adapter = LangFusePlatform()
        first = adapter._get_client()
        second = adapter._get_client()

        assert isinstance(first, _RealLangfuseClient)
        assert first is second, "the wrapper must be cached, not rebuilt per call"
        assert first._sdk is sentinel_sdk
        assert call_count["n"] == 1, (
            "langfuse.get_client() must be called once — the wrapper caches"
        )
