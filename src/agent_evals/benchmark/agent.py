# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""User-facing base class for agents under evaluation.

`BaseAgent` is the single sanctioned pattern for wrapping an agent-under-test.
Subclass it, implement `run_case`, and optionally override `setup` / `teardown`.
Declare `name` and `version` as class attributes so results can identify
which agent produced them.

Lives in `benchmark/` because only the benchmark feature uses it — no layer
below `benchmark/` imports `BaseAgent`. Feature-private types stay with the
feature; see the architectural rule in CLAUDE.md.

Observability boundary: the framework scores only what `run_case` returns.
If a scorer needs tool calls or post-execution state, `run_case` must
include them in the returned `TaskResult`. How that data gets surfaced
(in-process snapshot, HTTP response field, follow-up API call) is an
application concern.
"""

from abc import ABC, abstractmethod
from typing import Any

from agent_evals.core.types import TaskResult


class BaseAgent(ABC):
    """Base class for an agent under evaluation.

    Subclass and implement `run_case`. Optionally override `setup` and
    `teardown` for expensive once-per-benchmark initialization and cleanup.
    Declare `name` and `version` as class attributes so reports can
    identify which agent produced them.

    The benchmark runner guarantees:
      - `setup` is awaited once before the case loop begins.
      - `teardown` is awaited in a `finally` block, even if `setup` raised
        or `run_case` raised mid-loop. Subclasses are responsible for
        tolerating partial initialization.
      - `run_case` is called once per case (sequentially or concurrently
        per the benchmark's `n_parallel_runs` setting).

    All observation flows through `TaskResult`. If a scorer needs tool calls
    or post-execution state, include them in the returned result. The
    framework does not introspect, intercept, or infer agent internals.

    Example:
        class HomeAgent(BaseAgent):
            name = "home-langgraph"
            version = "0.1.0"

            async def setup(self) -> None:
                self._agent = build_agent(...)

            async def run_case(self, prompt: str, **kwargs) -> TaskResult:
                result = await self._agent.ainvoke({"input": prompt})
                return TaskResult.from_messages(
                    langchain_to_openai(result["messages"]),
                    final_state=...,
                )
    """

    name: str = ""
    version: str | None = None

    async def setup(self) -> None:
        """Expensive once-per-benchmark initialization. Default: no-op.

        Override to open connections, start subprocesses, load fixtures, or
        construct framework agent objects. Runs exactly once per benchmark
        invocation before any `run_case` call.
        """
        return None

    @abstractmethod
    async def run_case(self, prompt: str, **kwargs: Any) -> TaskResult:
        """Handle one case and return a `TaskResult`.

        Subclasses must implement this. A "case" is one prompt in the
        benchmark's dataset.

        Args:
            prompt: The input prompt. Always provided.
            **kwargs: Forward-compat slot for per-case agent-config kwargs.
                Subclasses should accept ``**kwargs`` (or ``**_``) to stay
                forward-compatible with future per-case extensions. The
                benchmark layer is the current consumer of this slot —
                see ``docs/benchmarks.md`` for the keys it forwards.

        If your scorers need tool calls or post-execution state, populate
        `TaskResult.context` accordingly -- most framework users pair a
        framework normalizer (`langchain_to_openai`, `strands_to_openai`,
        `mink_to_openai`) with `TaskResult.from_messages(...)`.
        """

    async def teardown(self) -> None:
        """Release resources. Guaranteed to run even on error. Default: no-op.

        Override to close HTTP clients, terminate subprocesses, or exit
        framework context managers. The runner invokes this in a `finally`
        block, so it runs even if `setup` or `run_case` raised. Subclasses
        are responsible for tolerating partial initialization (e.g.,
        checking that `self._client` was assigned before closing it).
        """
        return None
