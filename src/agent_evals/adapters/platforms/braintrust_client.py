# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Braintrust SDK seam: port + production wrapper.

Module-local Protocol (``BraintrustClientPort``) plus a thin wrapper
(``_RealBraintrustClient``) that absorbs ``EvalCase`` construction and
``EvalAsync`` invocation. The adapter speaks the port; the wrapper
speaks the SDK. See ``docs/architecture.md`` "Adapter Testability
Pattern".
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

try:
    import braintrust  # noqa: F401 — presence check
except ImportError as e:
    raise ImportError(
        "Braintrust adapter requires the braintrust package. "
        "Install it with: uv add 'agent-evals[braintrust]' "
        "or: uv add braintrust"
    ) from e


class BraintrustClientPort(Protocol):
    """What ``BraintrustPlatform`` needs from the braintrust SDK."""

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
        """Run a Braintrust evaluation, returning the SDK's ``EvalResult``.

        Implementations build ``braintrust.EvalCase`` from each dict in
        ``data`` and call ``braintrust.EvalAsync``.
        """
        ...


class _RealBraintrustClient(BraintrustClientPort):
    """Production wrapper around the ``braintrust`` module."""

    def __init__(self, sdk_module: Any) -> None:
        self._sdk = sdk_module

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
        eval_cases = [
            self._sdk.EvalCase(
                input=case["input"],
                expected=case.get("expected"),
                metadata=case.get("metadata"),
            )
            for case in data
        ]
        return await self._sdk.EvalAsync(
            name=project,
            data=eval_cases,
            task=task,
            scores=scorers,
            experiment_name=experiment,
            metadata=metadata,
        )
