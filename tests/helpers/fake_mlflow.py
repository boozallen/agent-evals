# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""FakeMlflowSession — flat test fake implementing MlflowSessionPort.

Drop-in substitute for _RealMlflowSession. The adapter under test sees
identical port semantics whether it has the real wrapper or this fake.
The fake speaks the adapter's domain — three flat methods that take
and return the dataclasses defined in mlflow_client.py.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from agent_evals.adapters.platforms.mlflow_client import (
    MlflowSessionPort,
    ScorerSpec,
    SessionInfo,
    TraceRecord,
)


class FakeMlflowSession(MlflowSessionPort):
    """Flat test fake for MlflowSessionPort.

    Records every port call on self.configurations / self.evaluations /
    self.searches so tests can inspect what the adapter did.

    Behavior:
      - configure: records the call, returns SessionInfo with a
        deterministic experiment_id (or seeded value).
      - evaluate: invokes wrapped task on each row, runs each
        ScorerSpec.fn, stores assessments on internal trace store keyed
        by run_id, returns SimpleNamespace(run_id=...).
      - search_traces: returns seeded traces (via seed_traces) or
        traces synthesized from the most recent evaluate call.
    """

    def __init__(self) -> None:
        self.configurations: list[dict[str, str]] = []
        self.evaluations: list[dict[str, Any]] = []
        self.searches: list[dict[str, Any]] = []
        self._experiment_id_overrides: dict[str, str] = {}
        self._seeded_traces: dict[str, list[TraceRecord]] = {}
        self._run_traces: dict[str, list[TraceRecord]] = {}
        self._next_run_id = 0

    def seed_experiment_id(self, *, experiment: str, experiment_id: str) -> None:
        """Make configure(experiment=<name>) return a specific id."""
        self._experiment_id_overrides[experiment] = experiment_id

    def seed_traces(self, *, experiment_id: str, traces: list[TraceRecord]) -> None:
        """Make search_traces(experiment_id=<id>, run_id=None) return these."""
        self._seeded_traces[experiment_id] = list(traces)

    def configure(self, *, tracking_uri: str, experiment: str) -> SessionInfo:
        if not tracking_uri:
            raise ValueError(
                "MLflow requires a tracking URI. "
                "For evaluations, set the MLFLOW_TRACKING_URI environment variable. "
                "For pull_traces, pass config={'tracking_uri': '...'}."
            )
        if not experiment:
            raise ValueError(
                "MLflow requires an experiment name. "
                "For evaluations, pass platform=MlflowConfig(experiment='...'). "
                "For pull_traces, pass config={'experiment': 'my-evaluation'}."
            )
        self.configurations.append(
            {"tracking_uri": tracking_uri, "experiment": experiment}
        )
        experiment_id = self._experiment_id_overrides.get(
            experiment, f"fake-exp-{experiment}"
        )
        return SessionInfo(
            experiment_id=experiment_id,
            tracking_uri=tracking_uri,
        )

    def evaluate(
        self,
        *,
        data: list[dict[str, Any]],
        predict_fn: Callable[..., Any] | None,
        scorers: list[ScorerSpec],
    ) -> Any:
        """Mimic genai.evaluate: drive task + scorers, build a result."""
        from mlflow.genai.evaluation.entities import Feedback

        self.evaluations.append(
            {"data": data, "predict_fn": predict_fn, "scorers": scorers}
        )

        run_id = f"fake-run-{self._next_run_id}"
        self._next_run_id += 1
        synthesized: list[TraceRecord] = []

        for row in data:
            inputs = row.get("inputs", {})
            expectations = row.get("expectations") or {}

            if predict_fn is not None:
                if isinstance(inputs, dict):
                    output = predict_fn(**inputs)
                else:
                    output = predict_fn(inputs)
                if asyncio.iscoroutine(output):
                    # MLflow's real genai.evaluate is sync and drives an async
                    # predict_fn with its own asyncio.run on a worker thread.
                    # The adapter now offloads session.evaluate via
                    # asyncio.to_thread (issue #239), so this fake also runs on
                    # a worker thread with no running loop. A fresh event loop
                    # mirrors MLflow's per-prediction asyncio.run and works
                    # whether the caller reached us via the sync or async path.
                    loop = asyncio.new_event_loop()
                    try:
                        output = loop.run_until_complete(output)
                    finally:
                        loop.close()
            else:
                output = row.get("outputs")

            assessments: list[Any] = []
            for spec in scorers:
                # The adapter's scorer wrapper may call asyncio.run() for
                # async user-scorers. That crashes if we're inside a running
                # event loop (tests typically drive aevaluate via
                # asyncio.run). Run spec.fn in a worker thread so its
                # asyncio.run gets a fresh loop. Production MLflow runs
                # scorers from worker threads anyway, so this matches the
                # real evaluate harness's call shape.
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                    feedback = executor.submit(
                        spec.fn,
                        outputs=output,
                        inputs=inputs,
                        expectations=expectations,
                    ).result()
                if isinstance(feedback, Feedback):
                    assessments.append(feedback)

            request_str = json.dumps({"messages": [{"content": str(inputs)}]})
            response_str = json.dumps(
                {"messages": [{"content": str(output) if output else ""}]}
            )

            synthesized.append(
                TraceRecord(
                    request=request_str,
                    response=response_str,
                    assessments=assessments,
                    execution_duration_ms=100,
                    trace_metadata={},
                )
            )

        self._run_traces[run_id] = synthesized
        return SimpleNamespace(run_id=run_id)

    def search_traces(
        self,
        *,
        experiment_id: str,
        run_id: str | None = None,
        filter: str | None = None,  # noqa: ARG002
    ) -> list[TraceRecord]:
        _ = filter  # vulture: intentionally unused
        self.searches.append(
            {"experiment_id": experiment_id, "run_id": run_id, "filter": filter}
        )

        if run_id is not None:
            return list(self._run_traces.get(run_id, []))

        return list(self._seeded_traces.get(experiment_id, []))
