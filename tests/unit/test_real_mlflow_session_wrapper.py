# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Direct tests for _RealMlflowSession against SimpleNamespace stubs.

These tests pin the wrapper's defensive paths without touching the real
mlflow SDK. The wrapper is the only place in the adapter family that
calls set_tracking_uri / set_experiment / genai.evaluate / search_traces
and applies @scorer decoration; these tests verify each.
"""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from agent_evals.adapters.platforms.mlflow_client import (
    ScorerSpec,
    SessionInfo,
    _RealMlflowSession,
)


def _make_stub_mlflow(*, exp_id: str = "exp-1", search_return=None):
    stub = SimpleNamespace()
    stub.set_tracking_uri = lambda uri: setattr(stub, "_uri", uri)
    stub.set_experiment = lambda name: SimpleNamespace(experiment_id=exp_id)

    def search_traces(**kwargs):
        stub._last_search = kwargs
        return search_return if search_return is not None else []

    stub.search_traces = search_traces

    genai_ns = SimpleNamespace()

    def evaluate(*, data, predict_fn, scorers):
        stub._last_evaluate = {
            "data": data,
            "predict_fn": predict_fn,
            "scorers": scorers,
        }
        return SimpleNamespace(run_id="run-1")

    genai_ns.evaluate = evaluate
    stub.genai = genai_ns
    return stub


def test_configure_calls_setters_in_order():
    """Both setters must fire; experiment_id comes from set_experiment."""
    call_order: list[str] = []
    stub = _make_stub_mlflow(exp_id="exp-99")
    stub.set_tracking_uri = lambda uri: call_order.append(f"uri:{uri}")
    stub.set_experiment = lambda name: (
        call_order.append(f"exp:{name}"),
        SimpleNamespace(experiment_id="exp-99"),
    )[1]

    wrapper = _RealMlflowSession(stub)
    info = wrapper.configure(tracking_uri="https://x", experiment="my-exp")

    assert call_order == ["uri:https://x", "exp:my-exp"]
    assert info == SessionInfo(experiment_id="exp-99", tracking_uri="https://x")


def test_configure_rejects_empty_tracking_uri():
    stub = _make_stub_mlflow()
    wrapper = _RealMlflowSession(stub)
    with pytest.raises(ValueError, match="MLFLOW_TRACKING_URI"):
        wrapper.configure(tracking_uri="", experiment="my-exp")


def test_configure_rejects_empty_experiment():
    stub = _make_stub_mlflow()
    wrapper = _RealMlflowSession(stub)
    with pytest.raises(ValueError, match="experiment"):
        wrapper.configure(tracking_uri="https://x", experiment="")


def test_evaluate_applies_scorer_decoration():
    """Each ScorerSpec.fn becomes @scorer(name=spec.name)(spec.fn) before evaluate."""
    stub = _make_stub_mlflow()
    captured = {}

    def fake_evaluate(*, data, predict_fn, scorers):
        captured["scorers"] = scorers
        return SimpleNamespace(run_id="run-1")

    stub.genai.evaluate = fake_evaluate

    def my_scorer_fn(outputs=None, inputs=None, expectations=None):
        return None

    wrapper = _RealMlflowSession(stub)
    wrapper.evaluate(
        data=[{"inputs": {"x": 1}}],
        predict_fn=lambda x: x,
        scorers=[ScorerSpec(name="MyScorer", fn=my_scorer_fn)],
    )

    assert len(captured["scorers"]) == 1
    decorated = captured["scorers"][0]
    assert getattr(decorated, "name", None) == "MyScorer"


def test_search_traces_run_scoped_uses_return_type_list():
    """run_id non-None branch picks return_type='list' and converts Trace objects."""
    trace_obj = SimpleNamespace(
        info=SimpleNamespace(
            assessments=["FB", "EX"],
            trace_metadata={"k": "v"},
            execution_duration=250,
        ),
        data=SimpleNamespace(request="req-string", response="resp-string"),
    )
    stub = _make_stub_mlflow(search_return=[trace_obj])
    wrapper = _RealMlflowSession(stub)

    records = wrapper.search_traces(experiment_id="exp-1", run_id="run-1")

    assert stub._last_search["return_type"] == "list"
    assert stub._last_search["run_id"] == "run-1"

    assert len(records) == 1
    rec = records[0]
    assert rec.request == "req-string"
    assert rec.response == "resp-string"
    assert rec.assessments == ["FB", "EX"]
    assert rec.execution_duration_ms == 250
    assert rec.trace_metadata == {"k": "v"}


def test_search_traces_experiment_scoped_uses_return_type_pandas():
    """run_id None branch picks return_type='pandas' and converts DataFrame rows."""
    df = pd.DataFrame(
        [
            {
                "request": {"x": "hi"},
                "response": "hello",
                "assessments": [{"feedback": {"value": 1.0}}],
            }
        ]
    )
    stub = _make_stub_mlflow(search_return=df)
    wrapper = _RealMlflowSession(stub)

    records = wrapper.search_traces(experiment_id="exp-1")

    assert stub._last_search["return_type"] == "pandas"
    assert "run_id" not in stub._last_search

    assert len(records) == 1
    rec = records[0]
    assert rec.request == {"x": "hi"}
    assert rec.response == "hello"
    assert rec.assessments == [{"feedback": {"value": 1.0}}]
    # Pandas branch hardcodes these (DataFrame rows don't carry them);
    # pin the contract symmetrically with the run-scoped test.
    assert rec.execution_duration_ms is None
    assert rec.trace_metadata == {}


def test_search_traces_tolerates_none_in_optional_fields():
    """Trace.data.request/response None and trace_metadata absent must not crash."""
    trace_obj = SimpleNamespace(
        info=SimpleNamespace(assessments=None, execution_duration=None),
        data=SimpleNamespace(request=None, response=None),
    )
    stub = _make_stub_mlflow(search_return=[trace_obj])
    wrapper = _RealMlflowSession(stub)

    records = wrapper.search_traces(experiment_id="exp-1", run_id="run-1")
    rec = records[0]
    assert rec.request == ""
    assert rec.response == ""
    assert rec.assessments == []
    assert rec.execution_duration_ms is None
    assert rec.trace_metadata == {}


def test_search_traces_filter_string_forwarded():
    """The filter kwarg becomes filter_string on the SDK call."""
    stub = _make_stub_mlflow(search_return=pd.DataFrame())
    wrapper = _RealMlflowSession(stub)
    wrapper.search_traces(experiment_id="exp-1", filter="trace.status = 'OK'")
    assert stub._last_search["filter_string"] == "trace.status = 'OK'"
