# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``_RealLangfuseClient`` (the production wrapper).

The wrapper is the *only* place in the adapter family that knows the
SDK's wire format. These tests exercise it directly against a stub SDK
object (a ``SimpleNamespace`` — not a ``Mock``) so that:

- The wrapper's defensive ``getattr`` paths in ``get_dataset_items`` are
  pinned (None metadata becomes ``{}``, missing attributes fall back).
- The empty-observations and missing-observations paths in
  ``get_trace_output`` are pinned (both return ``None``, not raise).
- The straightforward delegations (``auth_check``, ``run_experiment``)
  are pinned to actually delegate.

Without these tests, the wrapper is exercised only indirectly via the
1,490-LOC ``tests/integration/test_langfuse_adapter.py`` file, which
uses ``unittest.mock.Mock()`` heavily and inadvertently obscures
several of the wrapper's contracts (Mocks auto-stub attributes; a
real wrapper bug like reading ``dataset.itemz`` instead of ``items``
would still pass against a Mock).

These tests intentionally avoid ``unittest.mock`` for the SDK shape —
a ``SimpleNamespace`` is enough and makes the contract explicit.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from agent_evals.adapters.platforms.langfuse_client import (
    DatasetItemRecord,
    _RealLangfuseClient,
)


def _stub_sdk(**attrs: Any) -> SimpleNamespace:
    """Minimal stub SDK object the wrapper can call against."""
    return SimpleNamespace(**attrs)


class TestRealLangfuseClientAuthCheck:
    def test_delegates_true(self):
        sdk = _stub_sdk(auth_check=lambda: True)
        assert _RealLangfuseClient(sdk).auth_check() is True

    def test_delegates_false(self):
        sdk = _stub_sdk(auth_check=lambda: False)
        assert _RealLangfuseClient(sdk).auth_check() is False


class TestRealLangfuseClientRunExperiment:
    def test_delegates_all_kwargs(self):
        captured: dict[str, Any] = {}

        def fake_run_experiment(**kwargs: Any) -> str:
            captured.update(kwargs)
            return "result-sentinel"

        sdk = _stub_sdk(run_experiment=fake_run_experiment)
        client = _RealLangfuseClient(sdk)

        result = client.run_experiment(
            name="exp-1",
            description="d",
            data=[{"input": "x"}],
            task=lambda **_: None,
            evaluators=[],
            metadata={"k": "v"},
        )

        assert result == "result-sentinel"
        assert captured["name"] == "exp-1"
        assert captured["description"] == "d"
        assert captured["data"] == [{"input": "x"}]
        assert captured["evaluators"] == []
        assert captured["metadata"] == {"k": "v"}
        assert callable(captured["task"])


class TestRealLangfuseClientGetDatasetItems:
    """The wrapper extracts dataset-item attributes defensively.

    The langfuse SDK's dataset-item objects expose ``source_trace_id``,
    ``input``, ``expected_output``, and ``metadata`` as attributes.
    Each can legitimately be missing (None or absent). The wrapper
    must produce a clean ``DatasetItemRecord`` regardless.
    """

    def test_extracts_all_fields_when_present(self):
        item = SimpleNamespace(
            source_trace_id="t-1",
            input="hello",
            expected_output="world",
            metadata={"k": "v"},
        )
        sdk = _stub_sdk(get_dataset=lambda name: SimpleNamespace(items=[item]))

        records = _RealLangfuseClient(sdk).get_dataset_items("ds")

        assert records == [
            DatasetItemRecord(
                source_trace_id="t-1",
                input="hello",
                expected_output="world",
                metadata={"k": "v"},
            )
        ]

    def test_metadata_none_becomes_empty_dict(self):
        """Pins the ``or {}`` defensive branch in the wrapper."""
        item = SimpleNamespace(
            source_trace_id="t-1",
            input="x",
            expected_output=None,
            metadata=None,
        )
        sdk = _stub_sdk(get_dataset=lambda name: SimpleNamespace(items=[item]))

        records = _RealLangfuseClient(sdk).get_dataset_items("ds")

        assert records[0].metadata == {}

    def test_missing_attributes_fall_back_to_none(self):
        """Items missing ``source_trace_id`` / ``input`` / ``expected_output``
        attributes entirely produce records with ``None`` values, not
        ``AttributeError``.
        """
        # SimpleNamespace with no attrs at all — most defensive case
        item = SimpleNamespace()
        sdk = _stub_sdk(get_dataset=lambda name: SimpleNamespace(items=[item]))

        records = _RealLangfuseClient(sdk).get_dataset_items("ds")

        assert records == [
            DatasetItemRecord(
                source_trace_id=None,
                input=None,
                expected_output=None,
                metadata={},
            )
        ]

    def test_returns_empty_list_for_empty_dataset(self):
        sdk = _stub_sdk(get_dataset=lambda name: SimpleNamespace(items=[]))
        assert _RealLangfuseClient(sdk).get_dataset_items("ds") == []


class TestRealLangfuseClientGetTraceOutput:
    """The wrapper flattens ``client.api.trace.get(id).observations[-1].output``.

    Pins the three observation-shape paths: present, empty list,
    missing attribute.
    """

    def test_returns_last_observation_output(self):
        trace = SimpleNamespace(
            observations=[
                SimpleNamespace(output="first"),
                SimpleNamespace(output="last"),
            ]
        )
        sdk = _stub_sdk(
            api=SimpleNamespace(trace=SimpleNamespace(get=lambda trace_id: trace))
        )

        assert _RealLangfuseClient(sdk).get_trace_output("any") == "last"

    def test_empty_observations_returns_none(self):
        trace = SimpleNamespace(observations=[])
        sdk = _stub_sdk(
            api=SimpleNamespace(trace=SimpleNamespace(get=lambda trace_id: trace))
        )

        assert _RealLangfuseClient(sdk).get_trace_output("any") is None

    def test_missing_observations_attribute_returns_none(self):
        """If the SDK changes shape and removes ``.observations`` entirely,
        the wrapper returns ``None`` rather than raising ``AttributeError``.
        """
        trace = SimpleNamespace()  # no .observations
        sdk = _stub_sdk(
            api=SimpleNamespace(trace=SimpleNamespace(get=lambda trace_id: trace))
        )

        assert _RealLangfuseClient(sdk).get_trace_output("any") is None

    def test_propagates_sdk_lookup_failure(self):
        """If the SDK's ``api.trace.get`` raises, the wrapper does NOT swallow.

        The adapter's ``_convert_traces`` is the one place that catches
        per-item lookup failures and logs+skips. The wrapper must let
        the raise propagate so that pattern works.
        """

        def boom(trace_id: str) -> Any:
            raise RuntimeError("404 from the SDK")

        sdk = _stub_sdk(api=SimpleNamespace(trace=SimpleNamespace(get=boom)))

        with pytest.raises(RuntimeError, match="404 from the SDK"):
            _RealLangfuseClient(sdk).get_trace_output("missing")
