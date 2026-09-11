# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Live integration tests for the Braintrust adapter against the real API.

These tests are env-gated on `BRAINTRUST_API_KEY` and skipped without
credentials. They catch real-SDK behavioral drift that fake-driven unit
tests structurally cannot — the wrapper-port pattern in
`braintrust_client.py` protects against SDK *naming* drift (kwarg
renames, EvalCase shape), but only the real SDK can reveal *behavioral*
drift in async invocation, replay-task wiring, and BTQL response shape.

Test coverage:

- A. Public-API happy path through `run_eval_async(platform="braintrust", ...)`.
     This is the production entry point most users hit; routing through
     it covers registry/composition wiring on top of the SDK contract.
- B. Async task + async scorer end-to-end. The fake covers async-path
     wrapping in unit tests; this proves the real `EvalAsync` invokes
     async callables the way the adapter expects.
- C. Historical-replay round-trip: pull traces via BTQL, then re-score
     them with `task=None`. Catches BTQL response-shape drift and
     replay-task wiring in one shot. Requires `BRAINTRUST_PROJECT_ID`
     to point at a project with at least one logged trace.
"""

from __future__ import annotations

import asyncio
import os
import time

import pytest

from agent_evals import (
    EvalResult,
    ExampleData,
    ExpectedResult,
    Score,
    TaskResult,
    run_eval_async,
)
from agent_evals.adapters.platforms.braintrust import BraintrustPlatform

pytestmark = [
    pytest.mark.requires_braintrust,
    pytest.mark.skipif(
        not os.getenv("BRAINTRUST_API_KEY"),
        reason="Requires BRAINTRUST_API_KEY environment variable",
    ),
]

CI_PROJECT = "agent-evals-ci"


def _echo_task(x: str) -> str:
    return f"echo: {x}"


def _length_scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
    target = expected.expected if expected is not None else ""
    same_len = len(result.output) == len(target)
    return Score(
        name="_length_scorer",
        value=1.0 if same_len else 0.0,
        passed=same_len,
        metadata={"_threshold": 1.0},
    )


async def _async_echo_task(x: str) -> TaskResult:
    await asyncio.sleep(0)
    return TaskResult(output=f"echo: {x}")


async def _async_length_scorer(
    result: TaskResult, expected: ExpectedResult | None = None
) -> Score:
    await asyncio.sleep(0)
    target = expected.expected if expected is not None else ""
    same_len = len(result.output) == len(target)
    return Score(
        name="_async_length_scorer",
        value=1.0 if same_len else 0.0,
        passed=same_len,
        metadata={"_threshold": 1.0},
    )


def _replay_wiring_scorer(
    result: TaskResult, expected: ExpectedResult | None = None
) -> Score:
    # Wiring check, not a quality check: the replay path must deliver a
    # TaskResult to the scorer; we don't care what's in it.
    passed = isinstance(result, TaskResult) and bool(result.output)
    return Score(
        name="_replay_wiring_scorer",
        value=1.0 if passed else 0.0,
        passed=passed,
        metadata={"_threshold": 1.0},
    )


@pytest.mark.asyncio
async def test_run_eval_async_against_real_braintrust():
    """A. Public-API end-to-end: real EvalAsync via the registry path.

    Routes through `run_eval_async(platform="braintrust", ...)` rather
    than calling `adapter.aevaluate` directly. Covers the same SDK
    contract as before, plus the registry/composition wiring that
    production callers traverse.
    """
    dataset = [
        ExampleData(input="alpha", expected=ExpectedResult(expected="echo: alpha")),
        ExampleData(input="bravo", expected=ExpectedResult(expected="echo: bravo")),
        ExampleData(input="charlie", expected=ExpectedResult(expected="echo: ch")),
    ]

    from agent_evals.adapters.platforms.braintrust import BraintrustConfig

    result = await run_eval_async(
        task=_echo_task,
        dataset=dataset,
        scorers=[_length_scorer],
        platform=BraintrustConfig(
            project=CI_PROJECT,
            experiment=f"live-public-api-{int(time.time())}",
        ),
    )

    assert isinstance(result, EvalResult)
    assert result.platform == "braintrust"
    assert len(result.examples) == 3

    # Braintrust uses the scorer callable's __name__ as the score key
    # (BraintrustPlatform._infer_scorer_name), not Score.name.
    score_key = "_length_scorer"
    assert score_key in result.examples[0].scores, (
        f"Expected score key {score_key!r}, got {list(result.examples[0].scores.keys())!r}"
    )
    # First two pass length-match; third fails ("echo: charlie" vs "echo: ch").
    assert result.examples[0].scores[score_key].passed is True
    assert result.examples[1].scores[score_key].passed is True
    assert result.examples[2].scores[score_key].passed is False


@pytest.mark.asyncio
async def test_async_task_and_async_scorer_against_real_braintrust():
    """B. Async task + async scorer end-to-end against the real SDK.

    The fake-driven unit tests prove the *adapter* handles async paths;
    only the real SDK can prove `EvalAsync` invokes async callables in
    a way the adapter's wrapping survives.

    Task and scorer must be defined at module scope: ``_infer_scorer_name``
    truncates ``foo.<locals>.bar`` qualnames to the part before
    ``.<locals>.``, so a scorer defined inside the test function would
    register under the *test function's* name, not the scorer's.
    """
    dataset = [
        ExampleData(input="alpha", expected=ExpectedResult(expected="echo: alpha")),
        ExampleData(input="zz", expected=ExpectedResult(expected="echo: ZZZZZZZZZ")),
    ]

    from agent_evals.adapters.platforms.braintrust import BraintrustConfig

    result = await run_eval_async(
        task=_async_echo_task,
        dataset=dataset,
        scorers=[_async_length_scorer],
        platform=BraintrustConfig(
            project=CI_PROJECT,
            experiment=f"live-async-{int(time.time())}",
        ),
    )

    assert isinstance(result, EvalResult)
    assert len(result.examples) == 2
    score_key = "_async_length_scorer"
    # First passes (matched length), second fails (lengths differ).
    assert result.examples[0].scores[score_key].passed is True
    assert result.examples[1].scores[score_key].passed is False


@pytest.mark.skipif(
    not os.getenv("BRAINTRUST_PROJECT_ID"),
    reason=(
        "Requires BRAINTRUST_PROJECT_ID pointing at a project with at "
        "least one logged trace — needed for the BTQL replay round-trip."
    ),
)
@pytest.mark.asyncio
async def test_pull_traces_and_replay_round_trip():
    """C. BTQL pull_traces → run_eval_async(task=None) replay.

    This is the only test in the suite that exercises:
      - the BTQL HTTP path and response-shape dispatch
      - `_convert_traces` (Braintrust trace row → ExampleData)
      - the `task=None` historical-replay wiring through `aevaluate`

    Skipped without `BRAINTRUST_PROJECT_ID` because pull_traces needs a
    Braintrust project ID (not name) to query, and the test cannot
    bootstrap one — `aevaluate` returns an experiment URL but no
    project ID.
    """
    project_id = os.environ["BRAINTRUST_PROJECT_ID"]

    adapter = BraintrustPlatform()
    historical = adapter.pull_traces({"project_id": project_id})

    if not historical:
        pytest.skip(
            f"BRAINTRUST_PROJECT_ID={project_id!r} returned no root traces; "
            "the test needs at least one logged root span to replay."
        )

    # Limit to a small slice so the rescore experiment stays cheap.
    sample = historical[:3]
    assert all(isinstance(ex, ExampleData) for ex in sample)
    assert all(ex.output is None or isinstance(ex.output, TaskResult) for ex in sample)

    from agent_evals.adapters.platforms.braintrust import BraintrustConfig

    result = await run_eval_async(
        task=None,  # historical-replay mode
        dataset=sample,
        scorers=[_replay_wiring_scorer],
        platform=BraintrustConfig(
            project=CI_PROJECT,
            experiment=f"live-replay-{int(time.time())}",
        ),
    )

    assert isinstance(result, EvalResult)
    assert len(result.examples) == len(sample)
    # The replay path must deliver each example through the scorer.
    assert all("_replay_wiring_scorer" in ex.scores for ex in result.examples)
