# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test fake implementing ``BraintrustClientPort`` for adapter tests."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

from agent_evals.adapters.platforms.braintrust_client import BraintrustClientPort


class FakeBraintrustClient(BraintrustClientPort):
    """Drives task + scorers; records ``run_eval`` calls on ``self.evals``.

    ``run_eval`` invokes the wrapped task on each item in ``data``, runs
    each scorer on the output, and returns a ``SimpleNamespace(results=[...])``
    matching the SDK's `EvalResult` shape consumed by the adapter.
    """

    def __init__(self) -> None:
        self.evals: list[dict[str, Any]] = []

    async def run_eval(
        self,
        *,
        project: str,
        experiment: str,
        data: list[dict[str, Any]],
        task: Callable[..., Any],
        scorers: list[Callable[..., Any]],
        metadata: dict[str, Any],
    ) -> Any:
        """Mimic braintrust.EvalAsync: drive task + scorers, build a result."""
        self.evals.append(
            {
                "project": project,
                "experiment": experiment,
                "data": data,
                "task": task,
                "scorers": scorers,
                "metadata": metadata,
            }
        )

        results: list[SimpleNamespace] = []
        for case in data:
            input_value = case["input"]
            output = task(input_value)
            if asyncio.iscoroutine(output):
                output = await output

            scores: dict[str, float] = {}
            for scorer in scorers:
                ev = scorer(output, case.get("expected"))
                if asyncio.iscoroutine(ev):
                    ev = await ev
                if ev is not None:
                    score_name = getattr(scorer, "__name__", "scorer")
                    scores[score_name] = float(ev)

            results.append(
                SimpleNamespace(
                    input=input_value,
                    output=output,
                    expected=case.get("expected"),
                    metadata=case.get("metadata") or {},
                    scores=scores,
                    error=None,
                    duration=0.0,
                )
            )

        return SimpleNamespace(results=results)
