# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Test ExampleData Pydantic validation."""

import pytest
from pydantic import ValidationError

from agent_evals.core.types import ExampleData, ExpectedResult, TaskResult


class TestExampleDataBasics:
    def test_minimal_valid_example(self):
        """Minimal example with just input."""
        example = ExampleData(input="test")
        assert example.input == "test"
        assert example.expected is None
        assert example.output is None
        assert example.metadata == {}

    def test_full_valid_example(self):
        """Example with all fields."""
        example = ExampleData(
            input="query",
            expected=ExpectedResult(expected="answer"),
            metadata={"test": "meta"},
            output=TaskResult(output="", context={"outputs": [{"role": "user"}]}),
        )
        assert example.input == "query"
        assert example.expected is not None
        assert example.expected.expected == "answer"
        assert example.metadata == {"test": "meta"}
        assert example.output is not None
        assert example.output.context is not None
        assert example.output.context["outputs"] == [{"role": "user"}]


class TestBackwardCompatibility:
    def test_from_dict_minimal(self):
        """Create from dict (backward compatible)."""
        data = {"input": "test"}
        example = ExampleData.model_validate(data)
        assert example.input == "test"

    def test_from_dict_full(self):
        """Create from dict with all fields."""
        data = {
            "input": "query",
            "expected": {"expected": "answer"},  # Use ExpectedResult format
            "metadata": {"key": "value"},
            "output": {
                "output": "",
                "context": {"outputs": []},
            },
        }
        example = ExampleData.model_validate(data)
        assert example.input == "query"
        assert example.expected is not None
        assert example.expected.expected == "answer"


class TestValidationErrors:
    def test_rejects_unknown_top_level_keys(self):
        """Test extra='forbid' rejects unknown keys."""
        with pytest.raises(ValidationError) as exc:
            ExampleData(
                input="x", outputs=[]
            )  # intentional - verifies runtime rejection

        errors = exc.value.errors()
        assert any("extra" in e["type"] for e in errors)

    def test_metadata_must_be_dict(self):
        """Validate metadata type."""
        with pytest.raises(ValidationError) as exc:
            ExampleData(input="x", metadata="not a dict")  # intentional

        assert "dict" in str(exc.value).lower()

    def test_context_must_be_dict(self):
        """Validate context type."""
        with pytest.raises(ValidationError) as exc:
            ExampleData(
                input="x", output=TaskResult(output="", context="not a dict")
            )  # intentional

        assert "dict" in str(exc.value).lower()

    def test_missing_input_rejected(self):
        """Input is required."""
        with pytest.raises(ValidationError) as exc:
            ExampleData.model_validate({"expected": "answer"})

        errors = exc.value.errors()
        assert any(e["loc"] == ("input",) for e in errors)
