# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""MLflow SDK seam: port + production wrapper + dataclasses.

Module-local artifacts the MLflow adapter (`mlflow.py`) uses to talk
to the mlflow SDK. Three-method port — MLflow's adapter ritual is
session-scoped (configure once, evaluate / search against the
configured session). The wrapper absorbs:

- The two-step ``set_tracking_uri`` + ``set_experiment`` configuration.
- The ``@mlflow.genai.scorer`` decoration ceremony.
- The dual-shape ``mlflow.search_traces`` returns (pandas DataFrame
  for filter-friendly experiment-scoped lookups, ``list[Trace]`` for
  run-scoped attribute traversal).

This is an **adapter-internal seam**, not a hexagonal port. Each
platform adapter has its own module-local seam — ports are NOT
unified across Langfuse, MLflow, and Braintrust. See
`docs/architecture.md` "Adapter Testability Pattern" for the structural
recipe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

try:
    import mlflow  # noqa: F401 — presence check
    from mlflow.genai import scorer
    from mlflow.genai.evaluation.entities import EvaluationResult, Feedback
except ImportError as e:
    raise ImportError(
        "MLflow adapter requires the mlflow package. "
        "Install it with: uv add 'agent-evals[mlflow]' "
        "or: uv add mlflow"
    ) from e


@dataclass
class SessionInfo:
    """Flat record returned from ``MlflowSessionPort.configure``.

    Carries the experiment_id (from ``set_experiment``'s return) and
    the (echoed-back) tracking_uri the wrapper just configured. The
    adapter caches this so ``_get_experiment_url`` doesn't need a
    second port method around ``mlflow.get_tracking_uri()``.
    """

    experiment_id: str
    tracking_uri: str


@dataclass
class TraceRecord:
    """Flat record the adapter consumes from ``search_traces``.

    Union of fields the adapter reads from MLflow's two return shapes:

    - ``return_type="pandas"`` rows (dict-style): ``request``,
      ``response``, ``assessments`` (list of mixed dicts).
    - ``return_type="list"`` rows (Trace objects):
      ``trace.info.assessments``, ``trace.data.request``,
      ``trace.data.response``, ``trace.info.execution_duration``,
      ``trace.info.trace_metadata``.

    The wrapper produces this record from either branch; the adapter
    speaks one shape.
    """

    # request/response are the wrapper's union of MLflow's two return
    # shapes: dict (pandas branch row) | str (list branch JSON payload) |
    # None (SDK Optional that the wrapper coerces to "" only on the list
    # branch — the pandas branch passes None through). Consumers must
    # tolerate all three.
    request: dict[str, Any] | str | None
    response: dict[str, Any] | str | None
    # assessments is genuinely heterogeneous: Feedback / Expectation
    # objects on the list branch, raw dicts on the pandas branch. The
    # consumer (`_convert_traces` / `_convert_from_platform_result`)
    # filters by isinstance/type-key.
    assessments: list[Any]
    execution_duration_ms: int | None = None
    trace_metadata: dict[str, str] = field(default_factory=dict)


@dataclass
class ScorerSpec:
    """Flat record carrying a user scorer + its registered name.

    The adapter builds these for each user evaluator; the wrapper
    applies ``@scorer(name=spec.name)(spec.fn)`` before calling
    ``mlflow.genai.evaluate``. The decoration step lives in the
    wrapper so the adapter doesn't import any ``mlflow.genai`` symbol.

    ``fn`` returns ``Feedback`` because that's MLflow's expected shape
    for `@scorer`-decorated callables; the adapter's wrapper closure
    converts user `Score` returns into `Feedback` before producing the
    spec.
    """

    name: str
    fn: Callable[..., Feedback]


class MlflowSessionPort(Protocol):
    """Module-local port describing what ``MLflowPlatform`` needs from mlflow.

    Three-method port: MLflow's adapter ritual is session-scoped. The
    wrapper absorbs ``set_tracking_uri`` + ``set_experiment``,
    ``@mlflow.genai.scorer`` decoration, and the dual-shape
    ``search_traces`` returns.
    """

    def configure(self, *, tracking_uri: str, experiment: str) -> SessionInfo:
        """Configure mlflow for this session; return info the adapter caches."""
        ...

    def evaluate(
        self,
        *,
        data: list[dict[str, Any]],
        predict_fn: Callable[..., Any] | None,
        scorers: list[ScorerSpec],
    ) -> EvaluationResult:
        """Run an evaluation; return the SDK's real EvaluationResult."""
        ...

    def search_traces(
        self,
        *,
        experiment_id: str,
        run_id: str | None = None,
        filter: str | None = None,
    ) -> list[TraceRecord]:
        """Return traces as flat TraceRecord instances."""
        ...


class _RealMlflowSession(MlflowSessionPort):
    """Production wrapper around the ``mlflow`` module."""

    def __init__(self, sdk_module: Any) -> None:
        # Typed as Any so unit tests can pass duck-typed stubs (e.g.,
        # SimpleNamespace) without ceremonial casts. Production callers
        # always pass the real `mlflow` module via the adapter's
        # _get_session() lazy accessor.
        self._sdk = sdk_module

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
        self._sdk.set_tracking_uri(tracking_uri)
        exp = self._sdk.set_experiment(experiment)
        return SessionInfo(experiment_id=exp.experiment_id, tracking_uri=tracking_uri)

    def evaluate(
        self,
        *,
        data: list[dict[str, Any]],
        predict_fn: Callable[..., Any] | None,
        scorers: list[ScorerSpec],
    ) -> EvaluationResult:
        decorated = [scorer(name=spec.name)(spec.fn) for spec in scorers]
        return self._sdk.genai.evaluate(  # type: ignore[possibly-missing-attribute]
            data=data,
            predict_fn=predict_fn,
            scorers=decorated,
        )

    def search_traces(
        self,
        *,
        experiment_id: str,
        run_id: str | None = None,
        filter: str | None = None,
    ) -> list[TraceRecord]:
        if run_id is not None:
            traces = self._sdk.search_traces(
                locations=[experiment_id],
                run_id=run_id,
                return_type="list",
            )
            return [self._trace_object_to_record(t) for t in traces]

        df = self._sdk.search_traces(
            locations=[experiment_id],
            return_type="pandas",
            filter_string=filter,
        )
        return [self._trace_dict_to_record(row) for row in df.to_dict("records")]

    @staticmethod
    def _trace_object_to_record(trace: Any) -> TraceRecord:
        info = getattr(trace, "info", None)
        data = getattr(trace, "data", None)
        return TraceRecord(
            request=getattr(data, "request", None) or "",
            response=getattr(data, "response", None) or "",
            assessments=list(getattr(info, "assessments", None) or []),
            execution_duration_ms=getattr(info, "execution_duration", None),
            trace_metadata=dict(getattr(info, "trace_metadata", None) or {}),
        )

    @staticmethod
    def _trace_dict_to_record(row: dict[str, Any]) -> TraceRecord:
        return TraceRecord(
            request=row.get("request"),
            response=row.get("response"),
            assessments=list(row.get("assessments") or []),
            execution_duration_ms=None,
            trace_metadata={},
        )
