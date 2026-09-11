# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for core types (Score dataclass).

Tests follow TDD approach:
1. Write tests first (these should FAIL initially)
2. Implement Score to make tests pass
3. Refactor as needed

Test Coverage:
- T004: Score validation (value range, required fields)
- T005: Score serialization (to_dict, to_json, from_dict)
- T006: Score immutability (frozen dataclass)
- T007: Score error handling (error convention)
"""

import json

import pytest

from agent_evals.core.types import Score


class TestScoreValidation:
    """T004: Test Score validation with valid/invalid values."""

    def test_valid_score_creation_with_all_required_fields(self) -> None:
        """Create Score with all required fields (name, value, passed)."""
        score = Score(name="test_scorer", value=0.5, passed=True)

        assert score.name == "test_scorer"
        assert score.value == 0.5
        assert score.passed is True

    def test_score_validation_rejects_value_below_zero(self) -> None:
        """Score validation rejects value < 0.0."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            Score(name="test", value=-0.1, passed=False)

    def test_score_validation_rejects_value_above_one(self) -> None:
        """Score validation rejects value > 1.0."""
        with pytest.raises(ValueError, match="must be between 0.0 and 1.0"):
            Score(name="test", value=1.5, passed=True)

    def test_score_validation_accepts_value_zero(self) -> None:
        """Score validation accepts value = 0.0 (boundary case)."""
        score = Score(name="test", value=0.0, passed=False)
        assert score.value == 0.0

    def test_score_validation_accepts_value_one(self) -> None:
        """Score validation accepts value = 1.0 (boundary case)."""
        score = Score(name="test", value=1.0, passed=True)
        assert score.value == 1.0

    def test_score_with_optional_fields(self) -> None:
        """Test Score with optional fields (metadata, reasoning, execution_time_ms)."""
        score = Score(
            name="comprehensive",
            value=0.8,
            passed=True,
            metadata={"key": "value", "count": 42},
            reasoning="This is a detailed explanation",
            execution_time_ms=12.5,
        )

        assert score.metadata == {"key": "value", "count": 42}
        assert score.reasoning == "This is a detailed explanation"
        assert score.execution_time_ms == 12.5

    def test_score_with_none_optional_fields(self) -> None:
        """Test Score with None/omitted optional fields."""
        score = Score(name="minimal", value=0.5, passed=True)

        assert score.metadata == {}
        assert score.reasoning is None
        assert score.execution_time_ms is None


class TestScoreSerialization:
    """T005: Test Score serialization (model_dump, model_dump_json, model_validate)."""

    def test_model_dump_includes_all_fields(self) -> None:
        """Test model_dump() includes all fields when present."""
        score = Score(
            name="test",
            value=0.75,
            passed=True,
            metadata={"info": "data"},
            reasoning="explanation",
            execution_time_ms=5.0,
        )

        result = score.model_dump(exclude_none=True)

        assert result["name"] == "test"
        assert result["value"] == 0.75
        assert result["passed"] is True
        assert result["metadata"] == {"info": "data"}
        assert result["reasoning"] == "explanation"
        assert result["execution_time_ms"] == 5.0

    def test_model_dump_omits_none_values(self) -> None:
        """Test model_dump(exclude_none=True) omits None values for optional fields."""
        score = Score(name="test", value=0.5, passed=True)

        result = score.model_dump(exclude_none=True)

        assert "name" in result
        assert "value" in result
        assert "passed" in result
        assert "reasoning" not in result  # Should be omitted
        assert "execution_time_ms" not in result  # Should be omitted

    def test_model_dump_json_produces_valid_json_string(self) -> None:
        """Test model_dump_json() produces valid JSON string."""
        score = Score(
            name="json_test",
            value=0.9,
            passed=True,
            reasoning="test reasoning",
        )

        json_str = score.model_dump_json(exclude_none=True)

        # Should be valid JSON
        parsed = json.loads(json_str)
        assert parsed["name"] == "json_test"
        assert parsed["value"] == 0.9
        assert parsed["passed"] is True
        assert parsed["reasoning"] == "test reasoning"

    def test_model_validate_reconstructs_score_correctly(self) -> None:
        """Test model_validate() reconstructs Score correctly."""
        data = {
            "name": "reconstructed",
            "value": 0.6,
            "passed": False,
            "metadata": {"test": True},
            "reasoning": "failed test",
            "execution_time_ms": 10.0,
        }

        score = Score.model_validate(data)

        assert score.name == "reconstructed"
        assert score.value == 0.6
        assert score.passed is False
        assert score.metadata == {"test": True}
        assert score.reasoning == "failed test"
        assert score.execution_time_ms == 10.0

    def test_model_validate_handles_missing_optional_fields(self) -> None:
        """Test model_validate() handles missing optional fields."""
        data = {"name": "minimal", "value": 0.5, "passed": True}

        score = Score.model_validate(data)

        assert score.name == "minimal"
        assert score.value == 0.5
        assert score.passed is True
        assert score.metadata == {}
        assert score.reasoning is None
        assert score.execution_time_ms is None

    def test_round_trip_score_to_dict_to_score(self) -> None:
        """Test round-trip: Score → dict → Score preserves data."""
        original = Score(
            name="roundtrip",
            value=0.85,
            passed=True,
            metadata={"key": "value"},
            reasoning="test",
            execution_time_ms=3.14,
        )

        # Convert to dict and back
        data = original.model_dump(exclude_none=True)
        reconstructed = Score.model_validate(data)

        assert reconstructed.name == original.name
        assert reconstructed.value == original.value
        assert reconstructed.passed == original.passed
        assert reconstructed.metadata == original.metadata
        assert reconstructed.reasoning == original.reasoning
        assert reconstructed.execution_time_ms == original.execution_time_ms


class TestScoreImmutability:
    """T006: Test Score immutability (frozen Pydantic model)."""

    def test_score_fields_cannot_be_modified_after_creation(self) -> None:
        """Test Score fields cannot be modified after creation."""
        from pydantic import ValidationError

        score = Score(name="immutable", value=0.7, passed=True)

        # Attempting to modify should raise an error (Pydantic ValidationError)
        with pytest.raises((ValidationError, AttributeError, TypeError)):
            score.value = 0.8  # ty: ignore[invalid-assignment]

    def test_attempting_to_modify_raises_appropriate_error(self) -> None:
        """Test attempting to modify raises ValidationError for frozen Pydantic model."""
        from pydantic import ValidationError

        score = Score(name="frozen", value=0.5, passed=False)

        # Try to modify different fields
        with pytest.raises((ValidationError, AttributeError, TypeError)):
            score.name = "modified"  # ty: ignore[invalid-assignment]

        with pytest.raises((ValidationError, AttributeError, TypeError)):
            score.passed = True  # ty: ignore[invalid-assignment]


class TestScoreErrorHandling:
    """T007: Test Score error handling (error convention)."""

    def test_error_score_convention(self) -> None:
        """Test error Score convention (value=0.0, passed=False, error in metadata)."""
        error_score = Score(
            name="api_scorer",
            value=0.0,
            passed=False,
            metadata={"error": "timeout: API took 30s to respond"},
            reasoning="Scorer failed due to API timeout after 30 seconds",
        )

        assert error_score.value == 0.0
        assert error_score.passed is False
        assert "error" in error_score.metadata
        assert "timeout" in error_score.metadata["error"]
        assert error_score.reasoning is not None
        assert "failed" in error_score.reasoning.lower()

    def test_error_score_with_reasoning_field_populated(self) -> None:
        """Test error Score with reasoning field populated with error details."""
        error_score = Score(
            name="llm_judge",
            value=0.0,
            passed=False,
            metadata={
                "error": "API returned 500 Internal Server Error",
                "retry_attempted": True,
            },
            reasoning="LLM judge scorer encountered API error after 3 retry attempts",
        )

        assert error_score.value == 0.0
        assert error_score.passed is False
        assert error_score.reasoning is not None
        assert "error" in error_score.reasoning.lower()
        assert error_score.metadata["retry_attempted"] is True


class TestScoreIntegration:
    """T020: Integration tests for Score with multiple scorer types."""

    def test_score_aggregation_compatibility(self) -> None:
        """Test Score aggregation compatibility (create multiple Scores)."""
        scores = [
            Score(name="scorer1", value=0.8, passed=True),
            Score(name="scorer2", value=0.9, passed=True),
            Score(name="scorer3", value=0.7, passed=True),
        ]

        # Calculate aggregate metrics
        avg_value = sum(s.value for s in scores) / len(scores)
        all_passed = all(s.passed for s in scores)
        min_value = min(s.value for s in scores)
        max_value = max(s.value for s in scores)

        assert avg_value == pytest.approx(0.8, rel=0.01)
        assert all_passed is True
        assert min_value == 0.7
        assert max_value == 0.9


class TestJSONSchemaValidation:
    """T021: JSON schema validation tests."""

    def test_score_model_dump_matches_json_schema_structure(self) -> None:
        """Test Score.model_dump() output matches JSON schema structure."""
        score = Score(
            name="test_scorer",
            value=0.75,
            passed=True,
            metadata={"key": "value"},
            reasoning="Test reasoning",
            execution_time_ms=5.0,
        )

        score_dict = score.model_dump(exclude_none=True)

        # Verify all required fields present
        assert "name" in score_dict
        assert "value" in score_dict
        assert "passed" in score_dict

        # Verify types match schema
        assert isinstance(score_dict["name"], str)
        assert isinstance(score_dict["value"], float)
        assert isinstance(score_dict["passed"], bool)

        # Verify optional fields if present
        if "metadata" in score_dict:
            assert isinstance(score_dict["metadata"], dict)
        if "reasoning" in score_dict:
            assert isinstance(score_dict["reasoning"], str)
        if "execution_time_ms" in score_dict:
            assert isinstance(score_dict["execution_time_ms"], float)

    def test_all_required_fields_present_in_serialized_output(self) -> None:
        """Test all required fields present in serialized output."""
        score = Score(name="required_test", value=0.5, passed=False)

        score_dict = score.model_dump(exclude_none=True)

        # Required fields per JSON schema
        required_fields = ["name", "value", "passed"]
        for field in required_fields:
            assert field in score_dict, f"Required field '{field}' missing"

    def test_serialized_score_can_be_validated_against_json_schema(self) -> None:
        """Test serialized Score can be validated against JSON schema."""
        import json

        score = Score(
            name="schema_test",
            value=0.85,
            passed=True,
            metadata={"test": True},
            reasoning="Schema validation test",
        )

        # Convert to dict and then to JSON string
        score_json = score.model_dump_json(exclude_none=True)

        # Should be valid JSON
        parsed = json.loads(score_json)

        # Validate structure matches schema expectations
        assert parsed["name"] == "schema_test"
        assert 0.0 <= parsed["value"] <= 1.0
        assert isinstance(parsed["passed"], bool)
        assert isinstance(parsed["metadata"], dict)
        assert isinstance(parsed["reasoning"], str)

    def test_optional_fields_handled_correctly_per_schema(self) -> None:
        """Test optional fields handled correctly per schema."""
        # Score with minimal fields
        minimal_score = Score(name="minimal", value=0.5, passed=True)
        minimal_dict = minimal_score.model_dump(exclude_none=True)

        # Required fields should always be present
        assert "name" in minimal_dict
        assert "value" in minimal_dict
        assert "passed" in minimal_dict

        # Optional fields with None should be omitted (per schema design)
        # (Note: metadata defaults to {} so it's present)
        assert "reasoning" not in minimal_dict
        assert "execution_time_ms" not in minimal_dict

        # Score with all optional fields
        full_score = Score(
            name="full",
            value=0.9,
            passed=True,
            metadata={"data": "value"},
            reasoning="Full score",
            execution_time_ms=10.5,
        )
        full_dict = full_score.model_dump(exclude_none=True)

        # All fields should be present
        assert "name" in full_dict
        assert "value" in full_dict
        assert "passed" in full_dict
        assert "metadata" in full_dict
        assert "reasoning" in full_dict
        assert "execution_time_ms" in full_dict
