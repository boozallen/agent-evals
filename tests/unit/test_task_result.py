# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for TaskResult validation and behavior."""

import pytest
from pydantic import ValidationError

from agent_evals import TaskResult


def test_task_result_basic():
    """Test basic TaskResult creation."""
    result = TaskResult(output="test output")
    assert result.output == "test output"
    assert result.context is None


def test_task_result_with_context():
    """Test TaskResult with context dictionary."""
    context = {"key": "value", "number": 42}
    result = TaskResult(output="test", context=context)
    assert result.output == "test"
    assert result.context == context


def test_task_result_empty_output():
    """Test TaskResult with empty string output."""
    result = TaskResult(output="")
    assert result.output == ""
    assert result.context is None


def test_task_result_empty_context():
    """Test TaskResult with empty context dict."""
    result = TaskResult(output="test", context={})
    assert result.output == "test"
    assert result.context == {}


def test_task_result_none_context():
    """Test TaskResult with explicit None context."""
    result = TaskResult(output="test", context=None)
    assert result.output == "test"
    assert result.context is None


def test_task_result_nested_context():
    """Test TaskResult with nested context structure."""
    context = {
        "metadata": {"user_id": "123", "session": "abc"},
        "outputs": [{"role": "user", "content": "test"}],
    }
    result = TaskResult(output="response", context=context)
    assert result.context == context
    if result.context:  # Type narrowing for type checker
        assert result.context["metadata"]["user_id"] == "123"


def test_task_result_missing_output():
    """Test that output is required."""
    with pytest.raises(ValidationError) as exc_info:
        TaskResult()  # type: ignore

    # Type narrowing: exc_info.value is ValidationError
    assert isinstance(exc_info.value, ValidationError)
    errors = exc_info.value.errors()
    assert any(
        error["loc"] == ("output",) and error["type"] == "missing" for error in errors
    )


def test_task_result_invalid_output_type():
    """Test that output must be a string."""
    with pytest.raises(ValidationError) as exc_info:
        TaskResult(output=123)

    # Type narrowing: exc_info.value is ValidationError
    assert isinstance(exc_info.value, ValidationError)
    errors = exc_info.value.errors()
    assert any("output" in error["loc"] for error in errors)


def test_task_result_invalid_context_type():
    """Test that context must be a dict or None."""
    with pytest.raises(ValidationError) as exc_info:
        TaskResult(output="test", context="invalid")

    # Type narrowing: exc_info.value is ValidationError
    assert isinstance(exc_info.value, ValidationError)
    errors = exc_info.value.errors()
    assert any("context" in error["loc"] for error in errors)


def test_task_result_extra_fields_forbidden():
    """Test that extra fields are forbidden."""
    with pytest.raises(ValidationError) as exc_info:
        TaskResult(output="test", extra_field="not allowed")

    # Type narrowing: exc_info.value is ValidationError
    assert isinstance(exc_info.value, ValidationError)
    errors = exc_info.value.errors()
    assert any(error["type"] == "extra_forbidden" for error in errors)


def test_task_result_immutable():
    """Test that TaskResult fields can be updated (Pydantic allows this by default)."""
    result = TaskResult(output="original")
    # Pydantic v2 allows mutation by default
    result.output = "modified"
    assert result.output == "modified"


def test_task_result_dict_conversion():
    """Test TaskResult to dict conversion."""
    result = TaskResult(output="test", context={"key": "value"})
    result_dict = result.model_dump()

    assert result_dict == {"output": "test", "context": {"key": "value"}}


def test_task_result_dict_conversion_none_context():
    """Test TaskResult to dict with None context."""
    result = TaskResult(output="test")
    result_dict = result.model_dump()

    assert result_dict == {"output": "test", "context": None}


def test_task_result_from_dict():
    """Test creating TaskResult from dict."""
    from typing import Any

    data: dict[str, Any] = {"output": "test", "context": {"key": "value"}}
    result = TaskResult(**data)

    assert result.output == "test"
    assert result.context == {"key": "value"}


def test_task_result_equality():
    """Test TaskResult equality comparison."""
    result1 = TaskResult(output="test", context={"key": "value"})
    result2 = TaskResult(output="test", context={"key": "value"})
    result3 = TaskResult(output="different")

    assert result1 == result2
    assert result1 != result3


def test_task_result_repr():
    """Test TaskResult string representation."""
    result = TaskResult(output="test output")
    repr_str = repr(result)

    assert "TaskResult" in repr_str
    assert "test output" in repr_str
