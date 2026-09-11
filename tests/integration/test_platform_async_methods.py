# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for async adapter protocol compliance."""

import inspect

import pytest

from agent_evals.adapters.platforms.braintrust import BraintrustPlatform
from agent_evals.adapters.platforms.local import LocalPlatform

pytestmark = pytest.mark.requires_braintrust


def test_braintrust_adapter_has_async_methods():
    """Test BraintrustPlatform implements all async protocol methods."""
    adapter = BraintrustPlatform()

    # Check methods exist
    assert hasattr(adapter, "aevaluate")

    # Check they are coroutine functions
    assert inspect.iscoroutinefunction(adapter.aevaluate)


def test_local_adapter_has_async_methods():
    """Test LocalPlatform implements all async protocol methods."""
    adapter = LocalPlatform()

    # Check methods exist
    assert hasattr(adapter, "aevaluate")

    # Check they are coroutine functions
    assert inspect.iscoroutinefunction(adapter.aevaluate)


def test_all_adapters_implement_async_protocol():
    """Test all adapters have async methods."""
    adapters = [BraintrustPlatform(), LocalPlatform()]

    for adapter in adapters:
        assert hasattr(adapter, "aevaluate")

        # Check they are coroutine functions
        assert inspect.iscoroutinefunction(adapter.aevaluate)
