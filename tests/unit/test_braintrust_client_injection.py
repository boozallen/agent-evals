# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin the BraintrustPlatform injection-seam contract.

These tests pin behavior observable through the public API:
  - constructing with `client=fake` flows the fake through to aevaluate
  - the env-var guard runs before reaching the client seam
  - the fake and the real wrapper share the port's `run_eval` signature
"""

from __future__ import annotations

import inspect
import os
from unittest.mock import patch

import pytest
from helpers.scorers import passing_scorer

from agent_evals.adapters.platforms.braintrust import BraintrustConfig
from agent_evals.core.types import ExampleData, ExpectedResult

pytestmark = pytest.mark.requires_braintrust


@pytest.mark.asyncio
async def test_aevaluate_uses_injected_client_not_real_wrapper():
    """When client= is passed, the adapter never constructs _RealBraintrustClient."""
    from helpers.fake_braintrust import FakeBraintrustClient

    from agent_evals.adapters.platforms.braintrust import BraintrustPlatform

    fake = FakeBraintrustClient()
    adapter = BraintrustPlatform(client=fake)

    with patch.dict(os.environ, {"BRAINTRUST_API_KEY": "test-key"}):
        await adapter.aevaluate(
            task=lambda x: "out",
            dataset=[ExampleData(input="a", expected=ExpectedResult(expected="x"))],
            evaluators=[passing_scorer],
            platform=BraintrustConfig(project="p", experiment="e"),
        )

    # The fake recorded the call; the seam routed through it.
    assert len(fake.evals) == 1
    assert fake.evals[0]["project"] == "p"


@pytest.mark.asyncio
async def test_aevaluate_failure_path_does_not_call_run_eval_when_env_missing():
    """Env-var guard runs BEFORE client.run_eval; fake.evals stays empty."""
    from helpers.fake_braintrust import FakeBraintrustClient

    from agent_evals.adapters.platforms.braintrust import BraintrustPlatform

    fake = FakeBraintrustClient()
    adapter = BraintrustPlatform(client=fake)

    with (
        patch.dict(os.environ, {}, clear=True),
        pytest.raises(ValueError, match="BRAINTRUST_API_KEY"),
    ):
        await adapter.aevaluate(
            task=lambda x: "out",
            dataset=[ExampleData(input="a")],
            evaluators=[passing_scorer],
            platform=BraintrustConfig(project="p", experiment="e"),
        )

    assert fake.evals == []  # env-var guard fired before reaching the seam


def test_fake_and_real_client_share_run_eval_signature():
    """FakeBraintrustClient.run_eval matches _RealBraintrustClient.run_eval.

    Python's Protocol conformance is structural and unchecked at runtime,
    so a kwarg added to the real wrapper without updating the fake (or
    vice versa) would let every fake-driven adapter test keep passing
    while production breaks. This signature-equality assertion catches
    such drift in one line.

    The Protocol itself is included in the comparison so a future change
    to BraintrustClientPort that the real wrapper hasn't caught up to is
    also surfaced here.
    """
    from helpers.fake_braintrust import FakeBraintrustClient

    from agent_evals.adapters.platforms.braintrust_client import (
        BraintrustClientPort,
        _RealBraintrustClient,
    )

    port_sig = inspect.signature(BraintrustClientPort.run_eval)
    real_sig = inspect.signature(_RealBraintrustClient.run_eval)
    fake_sig = inspect.signature(FakeBraintrustClient.run_eval)

    assert real_sig == port_sig, (
        f"_RealBraintrustClient.run_eval drifted from BraintrustClientPort: "
        f"{real_sig} vs {port_sig}"
    )
    assert fake_sig == port_sig, (
        f"FakeBraintrustClient.run_eval drifted from BraintrustClientPort: "
        f"{fake_sig} vs {port_sig}"
    )
