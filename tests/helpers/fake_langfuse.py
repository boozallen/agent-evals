# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""FakeLangfuseClient — flat test fake implementing ``LangfuseClientPort``.

A drop-in substitute for ``_RealLangfuseClient`` (the production wrapper
around the langfuse SDK). The adapter under test sees identical port
semantics whether it has the real wrapper or this fake.

Shape rationale: the wrapper-port pattern (see ``docs/architecture.md``
"Adapter Testability Pattern") puts SDK-shape concerns inside the real
wrapper, so the fake speaks the adapter's *domain* — flat methods that
return scalars and ``DatasetItemRecord`` instances. Tests no longer
mimic the SDK's nested ``client.api.trace.get(...)`` shape; they seed
plain dicts and the fake returns flat values.

The fake builds REAL ``langfuse.experiment.ExperimentResult`` and
``ExperimentItemResult`` instances when its ``run_experiment`` runs;
this matches the existing pattern in
``tests/unit/test_langfuse_reconcile.py`` and ensures the adapter's
``_build_examples_from_platform_result`` path is exercised against
authentic SDK types, so SDK-shape drift in *value* types is visible.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any, cast

from langfuse import Evaluation
from langfuse.experiment import (
    ExperimentItemResult,
    ExperimentResult,
    LocalExperimentItem,
)

from agent_evals.adapters.platforms.langfuse_client import (
    DEFAULT_TRACE_PAGE_SIZE,
    DatasetItemRecord,
    LangfuseClientPort,
    TraceRecord,
)


def _drive_coroutine(coro: Any) -> Any:
    """Drive a coroutine to completion from a sync context.

    The real langfuse SDK's ``run_experiment`` is a sync entrypoint that
    internally uses ``run_async_safely`` to support being called from
    inside a running event loop (e.g., from an ``async def`` caller).
    Mirror that here: spin up a fresh loop in a worker thread and run
    the coroutine to completion. This keeps the fake usable from both
    sync tests and tests that drive ``aevaluate`` (which is itself
    ``async def`` and therefore has a running loop).
    """
    result: list[Any] = []
    error: list[BaseException] = []

    def _runner() -> None:
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result.append(loop.run_until_complete(coro))
        except BaseException as e:  # noqa: BLE001 - propagate to caller
            error.append(e)
        finally:
            loop.close()

    t = threading.Thread(target=_runner)
    t.start()
    t.join()
    if error:
        # Preserve the original traceback so test failures point at the
        # user's coroutine frame, not at this _drive_coroutine helper.
        # Python 3.11+ retains __traceback__ on the exception object even
        # after it's caught and stored, so reattaching it here surfaces
        # the original failure site at the worker-thread boundary.
        raise error[0].with_traceback(error[0].__traceback__)
    return result[0]


class FakeTraceNotFoundError(Exception):
    """Raised when ``FakeLangfuseClient.get_trace_output`` is called with an unseeded trace id.

    The real langfuse SDK surfaces a network/HTTP error from the underlying
    REST API in this case; the adapter's ``pull_traces`` catches the bare
    ``Exception`` and logs/skips. This fake type stands in for that —
    narrowly typed so test code can assert on the specific failure mode
    rather than matching any exception.
    """


class FakeLangfuseClient(LangfuseClientPort):
    """Flat test fake for ``LangfuseClientPort``.

    Explicitly inherits from ``LangfuseClientPort`` so a port drift
    (e.g., adding a method to the port) fails type-check at this class
    instead of at the adapter call site that consumes the fake.

    Constructor parameters seed the fake's responses; tests construct one
    instance per scenario. The fake records ``run_experiment`` calls on
    ``self.experiments`` so tests can inspect what the adapter did.

    Seeding shape (deliberately flat):

        FakeLangfuseClient(
            auth_ok=True,                                # default
            dataset_items={"my-ds": [DatasetItemRecord(...)]},
            trace_outputs={"trace-1": "the output"},     # already-extracted
            traces=[TraceRecord(id="t1", ...)],          # trace-listing path
        )

    Unseeded trace IDs unconditionally raise ``FakeTraceNotFoundError``,
    standing in for the real SDK's network/HTTP error. The adapter's
    ``pull_traces`` catches that with a bare ``except Exception``.
    """

    def __init__(
        self,
        *,
        auth_ok: bool = True,
        dataset_items: dict[str, list[DatasetItemRecord]] | None = None,
        trace_outputs: dict[str, Any] | None = None,
        trace_messages: dict[str, list[dict[str, Any]]] | None = None,
        traces: list[TraceRecord] | None = None,
    ) -> None:
        self._auth_ok = auth_ok
        self._dataset_items: dict[str, list[DatasetItemRecord]] = {
            name: list(items) for name, items in (dataset_items or {}).items()
        }
        self._trace_outputs: dict[str, Any] = dict(trace_outputs or {})
        self._trace_messages: dict[str, list[dict[str, Any]]] = {
            tid: list(msgs) for tid, msgs in (trace_messages or {}).items()
        }
        self._traces: list[TraceRecord] = list(traces or [])
        self.experiments: list[dict[str, Any]] = []
        self.list_traces_calls: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # LangfuseClientPort implementation
    # ------------------------------------------------------------------

    def auth_check(self) -> bool:
        return self._auth_ok

    def get_dataset_items(self, name: str) -> list[DatasetItemRecord]:
        if name not in self._dataset_items:
            raise KeyError(f"unknown dataset {name!r}")
        return self._dataset_items[name]

    def list_traces(
        self,
        *,
        session_id: str | None = None,
        user_id: str | None = None,
        limit: int | None = None,
        page_size: int = DEFAULT_TRACE_PAGE_SIZE,
    ) -> list[TraceRecord]:
        """Apply the seeded selection and return matching records, unpaginated.

        The fake does NOT simulate paging: the port's contract is that
        pagination is the implementation's concern and the caller sees
        the complete matching set, so a fake that returned pages would
        be asserting the adapter does work the port says it must not.
        Pagination itself is covered where it lives, against a stub SDK
        — see ``tests/unit/test_langfuse_client_pagination.py``.

        The fake *does* apply selection, so an adapter test can assert
        that a session or user constraint narrowed the result. Every call
        is recorded on ``self.list_traces_calls`` so a test can also
        assert the constraint was expressed here rather than applied by
        the adapter after the fact.
        """
        self.list_traces_calls.append(
            {
                "session_id": session_id,
                "user_id": user_id,
                "limit": limit,
                "page_size": page_size,
            }
        )
        matching = [
            record
            for record in self._traces
            if (session_id is None or record.session_id == session_id)
            and (user_id is None or record.user_id == user_id)
        ]
        return matching if limit is None else matching[:limit]

    def get_trace_output(self, trace_id: str) -> Any:
        if trace_id not in self._trace_outputs:
            raise FakeTraceNotFoundError(f"404: trace {trace_id!r} not found")
        return self._trace_outputs[trace_id]

    def get_trace_messages(self, trace_id: str) -> list[dict[str, Any]] | None:
        return self._trace_messages.get(trace_id)

    def run_experiment(
        self,
        *,
        name: str,
        description: str,
        data: list[dict[str, Any]],
        task: Any,
        evaluators: list[Any],
        metadata: dict[str, Any],
    ) -> ExperimentResult:
        """Mimic the SDK: invoke the wrapped task per item, run evaluators,
        return a real ExperimentResult populated with ExperimentItemResults.
        """
        self.experiments.append(
            {
                "name": name,
                "description": description,
                "data": data,
                "task": task,
                "evaluators": evaluators,
                "metadata": metadata,
            }
        )

        item_results: list[ExperimentItemResult] = []
        for item in data:
            # The langfuse SDK calls task(item={...}); the adapter's
            # _make_safe_task wrapper unwraps to task(input_value) before
            # invoking the user code. Tasks may be sync or async — drive
            # any returned coroutine to mirror the real SDK's behavior.
            output = task(item=item)
            if asyncio.iscoroutine(output):
                output = _drive_coroutine(output)

            # Run each evaluator on the output and collect Evaluations.
            # Evaluators may also be sync or async; the real SDK awaits
            # coroutine results inside its own async machinery, so the
            # fake must drive them too.
            evaluations: list[Evaluation] = []
            for evaluator in evaluators:
                ev = evaluator(
                    input=item.get("input"),
                    output=output,
                    expected_output=item.get("expected_output"),
                    metadata=item.get("metadata", {}),
                )
                if asyncio.iscoroutine(ev):
                    ev = _drive_coroutine(ev)
                if ev is not None:
                    evaluations.append(ev)

            item_results.append(
                ExperimentItemResult(
                    # Cast: at runtime the SDK accepts a plain dict; the static
                    # signature wants the TypedDict variant. The dict carries
                    # the same shape, so this is a safe widening for tests.
                    item=cast(LocalExperimentItem, item),
                    output=output,
                    evaluations=evaluations,
                    trace_id=None,
                    dataset_run_id=None,
                )
            )

        return ExperimentResult(
            name=name,
            run_name=f"{name}-run",
            description=description,
            item_results=item_results,
            run_evaluations=[],
            experiment_id=f"exp-{name}",
        )
