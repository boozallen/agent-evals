# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EvalConfig schema validation.

Behavioral tests (serialization, concurrency cap, event-loop non-blocking)
live in test_parallel_scorers_contract.py.
"""

import pytest
from pydantic import ValidationError

from agent_evals.core.types import EvalConfig


class TestEvalConfigDefaults:
    def test_default_max_concurrent_tests(self):
        """Default is 1 (serial) — safe for stateful agents."""
        assert EvalConfig().max_concurrent_tests == 1


class TestEvalConfigValidation:
    def test_max_concurrent_tests_must_be_positive(self):
        with pytest.raises(ValidationError, match="max_concurrent_tests"):
            EvalConfig(max_concurrent_tests=0)

    def test_max_concurrent_tests_cannot_be_negative(self):
        with pytest.raises(ValidationError, match="max_concurrent_tests"):
            EvalConfig(max_concurrent_tests=-1)

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            EvalConfig(**{"not_a_real_field": 5})


class TestEvalConfigCustomValues:
    def test_custom_max_concurrent_tests(self):
        assert EvalConfig(max_concurrent_tests=5).max_concurrent_tests == 5

    def test_valid_small_value(self):
        assert EvalConfig(max_concurrent_tests=1).max_concurrent_tests == 1

    def test_valid_large_value(self):
        assert EvalConfig(max_concurrent_tests=100).max_concurrent_tests == 100
