# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Contract validation tests for agentevals integration.

These tests validate that agentevals evaluation results are correctly
normalized to our Score dataclass format according to the contract
specifications in contracts/agentevals-result.schema.json and
contracts/score-trajectory.schema.json.
"""

import json

from agent_evals import TaskResult
from agent_evals.adapters.scorers.agentevals import (
    GraphTrajectoryStrictMatch,
    TrajectoryStrictMatch,
    TrajectoryUnorderedMatch,
)
from agent_evals.core.types import ExpectedResult, Score


class TestAgentEvalsResultNormalization:
    """Test that agentevals results are correctly normalized to Score format."""

    def test_match_evaluator_true_result_normalization(self):
        """Test normalization of agentevals match evaluator result (score=True)."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        trajectory = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]

        # Act
        score = scorer(
            result=TaskResult(
                output="",
                context={"outputs": trajectory},
            ),
            expected=ExpectedResult(
                expected="",
                context={"reference_outputs": trajectory},
            ),
        )

        # Assert - Validate Score contract
        assert isinstance(score, Score)
        assert score.name == "TrajectoryStrictMatch"
        assert score.value == 1.0  # Boolean True → 1.0
        assert score.passed is True  # Boolean mapped to passed
        assert isinstance(score.metadata, dict)
        assert "agentevals_key" in score.metadata
        assert score.metadata["agentevals_key"] == "trajectory_strict_match"

    def test_match_evaluator_false_result_normalization(self):
        """Test normalization of agentevals match evaluator result (score=False)."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        actual = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
        ]
        reference = [
            {"role": "user", "content": "Hello"},
            # Different structure - has extra step
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "greet",
                            "arguments": json.dumps({"name": "user"}),
                        }
                    }
                ],
            },
            {"role": "tool", "content": "Greeting sent"},
            {"role": "assistant", "content": "Hello!"},
        ]

        # Act
        score = scorer(
            result=TaskResult(
                output="", context={"outputs": actual, "reference_outputs": reference}
            ),
            expected="",
        )

        # Assert
        assert isinstance(score, Score)
        assert score.value == 0.0  # Boolean False → 0.0
        assert score.passed is False

    def test_trajectory_metadata_fields_present(self):
        """Test that trajectory-specific metadata fields are included."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        trajectory = [
            {"role": "user", "content": "What is 2+2?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calculate",
                            "arguments": json.dumps({"expr": "2+2"}),
                        }
                    }
                ],
            },
            {"role": "tool", "name": "calculate", "content": "4"},
            {"role": "assistant", "content": "The answer is 4"},
        ]

        # Act
        score = scorer(
            result=TaskResult(
                output="",
                context={"outputs": trajectory},
            ),
            expected=ExpectedResult(
                expected="",
                context={"reference_outputs": trajectory},
            ),
        )

        # Assert trajectory-specific metadata
        assert "trajectory_length" in score.metadata
        assert score.metadata["trajectory_length"] == 4
        assert "tool_args_match_mode" in score.metadata

    def test_graph_trajectory_metadata_fields(self):
        """Test that graph trajectory metadata includes graph-specific fields."""
        # Arrange
        scorer = GraphTrajectoryStrictMatch()
        graph_trajectory = {"steps": [["__start__", "agent", "tools", "__end__"]]}

        # Act
        score = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": graph_trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": graph_trajectory,
                },
            ),
        )

        # Assert graph-specific metadata
        assert isinstance(score, Score)
        assert "agentevals_key" in score.metadata
        assert score.metadata["agentevals_key"] == "graph_trajectory_strict_match"

    def test_error_score_normalization(self):
        """Test that errors are normalized to Score with error in metadata."""
        # Arrange
        scorer = TrajectoryStrictMatch()

        # Act - Missing trajectory data
        score = scorer(
            result=TaskResult(output="", context={}), expected=""
        )  # No outputs in context

        # Assert error Score format
        assert isinstance(score, Score)
        assert score.value == 0.0
        assert score.passed is False
        assert "error" in score.metadata
        assert "'outputs' not found in context" in score.metadata["error"]
        assert score.reasoning is not None

    def test_score_contract_all_required_fields_present(self):
        """Test that Score object has all required fields per contract."""
        # Arrange
        scorer = TrajectoryUnorderedMatch()
        trajectory = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]

        # Act
        score = scorer(
            result=TaskResult(
                output="",
                context={"outputs": trajectory},
            ),
            expected=ExpectedResult(
                expected="",
                context={"reference_outputs": trajectory},
            ),
        )

        # Assert all required Score fields exist
        assert hasattr(score, "name")
        assert hasattr(score, "value")
        assert hasattr(score, "passed")
        assert hasattr(score, "metadata")
        assert hasattr(score, "reasoning")

        # Assert types
        assert isinstance(score.name, str)
        assert isinstance(score.value, float)
        assert isinstance(score.passed, bool)
        assert isinstance(score.metadata, dict)
        assert score.reasoning is None or isinstance(score.reasoning, str)

    def test_score_value_in_valid_range(self):
        """Test that score value is always in [0.0, 1.0] range."""
        # Arrange
        scorers = [
            TrajectoryStrictMatch(),
            TrajectoryUnorderedMatch(),
        ]
        trajectory = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]

        # Act & Assert
        for scorer in scorers:
            score = scorer(
                result=TaskResult(
                    output="",
                    context={"outputs": trajectory},
                ),
                expected=ExpectedResult(
                    expected="",
                    context={"reference_outputs": trajectory},
                ),
            )
            assert 0.0 <= score.value <= 1.0

    def test_agentevals_key_preserved_in_metadata(self):
        """Test that original agentevals key is preserved in metadata."""
        # Arrange
        test_cases = [
            (TrajectoryStrictMatch(), "trajectory_strict_match"),
            (TrajectoryUnorderedMatch(), "trajectory_unordered_match"),
            (GraphTrajectoryStrictMatch(), "graph_trajectory_strict_match"),
        ]
        trajectory = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]
        graph_trajectory = {"steps": [["__start__", "agent", "__end__"]]}

        # Act & Assert
        for scorer, expected_key in test_cases[:2]:  # Message trajectories
            score = scorer(
                result=TaskResult(
                    output="",
                    context={"outputs": trajectory},
                ),
                expected=ExpectedResult(
                    expected="",
                    context={"reference_outputs": trajectory},
                ),
            )
            assert score.metadata["agentevals_key"] == expected_key

        # Test graph trajectory separately
        score = test_cases[2][0](
            result=TaskResult(
                output="",
                context={
                    "outputs": graph_trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": graph_trajectory,
                },
            ),
        )
        assert score.metadata["agentevals_key"] == test_cases[2][1]

    def test_wrapper_name_differs_from_agentevals_key(self):
        """Test that Score.name uses our wrapper name, not agentevals key."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        trajectory = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]

        # Act
        score = scorer(
            result=TaskResult(
                output="",
                context={"outputs": trajectory, "reference_outputs": trajectory},
            ),
            expected="",
        )

        # Assert
        # Our wrapper name (PascalCase)
        assert score.name == "TrajectoryStrictMatch"
        # Agentevals key (snake_case) preserved in metadata
        assert score.metadata["agentevals_key"] == "trajectory_strict_match"
        # They should differ
        assert score.name != score.metadata["agentevals_key"]


class TestContextParameterContract:
    """Test that context parameter contract is correctly implemented."""

    def test_outputs_key_extraction(self):
        """Test that 'outputs' key is correctly extracted from context."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        trajectory = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]

        # Act - Pass trajectory via context
        score = scorer(
            result=TaskResult(
                output="",
                context={"outputs": trajectory},
            ),
            expected=ExpectedResult(
                expected="",
                context={"reference_outputs": trajectory},
            ),
        )

        # Assert
        assert score.passed is True

    def test_missing_outputs_key_handled(self):
        """Test that missing 'outputs' key returns error Score."""
        # Arrange
        scorer = TrajectoryStrictMatch()

        # Act - No outputs key
        score = scorer(result=TaskResult(output="", context={}), expected="")

        # Assert
        assert score.passed is False
        assert "error" in score.metadata
        assert "outputs" in score.metadata["error"]

    def test_optional_reference_outputs(self):
        """Test that reference_outputs is optional (some evaluators don't require it)."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        trajectory = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]

        # Act - Only outputs, no reference_outputs
        score = scorer(
            result=TaskResult(output="", context={"outputs": trajectory}), expected=""
        )

        # Assert - Should handle gracefully (may fail match but not crash)
        assert isinstance(score, Score)

    def test_context_extensibility_extra_keys_ignored(self):
        """Test that extra context keys don't break trajectory scorers."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        trajectory = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]

        # Act - Include extra context keys
        score = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": trajectory,
                    # Extra keys that trajectory scorer should ignore
                    "user_id": "user_123",
                    "session_id": "session_456",
                    "custom_metadata": {"key": "value"},
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": trajectory,
                },
            ),
        )

        # Assert - Extra keys don't cause failures
        assert isinstance(score, Score)
        assert score.passed is True


class TestBackwardCompatibility:
    """Test that trajectory scorers are backward compatible."""

    def test_string_based_scorers_ignore_trajectory_data(self):
        """Test that string-based scorers ignore trajectory context (T069)."""
        from agent_evals.adapters.scorers.autoevals import Levenshtein

        # Arrange
        string_scorer = Levenshtein(threshold=0.8)

        # Act - Pass trajectory data that string scorer should ignore
        score = string_scorer(
            result=TaskResult(
                output="Hello world",
                context={
                    # String scorer should ignore these trajectory keys
                    "outputs": [{"role": "user", "content": "Test"}],
                    "reference_outputs": [{"role": "user", "content": "Test"}],
                },
            ),
            expected=ExpectedResult(expected="Hello world"),
        )

        # Assert - String scorer works normally, ignores trajectory data
        assert isinstance(score, Score)
        assert score.passed is True

    def test_trajectory_scorers_ignore_string_parameters(self):
        """Test that trajectory scorers ignore string output/expected (T070)."""
        # Arrange
        trajectory_scorer = TrajectoryStrictMatch()
        trajectory = [
            {"role": "user", "content": "Test"},
            {"role": "assistant", "content": "Response"},
        ]

        # Act - Pass string data that trajectory scorer should ignore
        score = trajectory_scorer(
            result=TaskResult(
                output="Some string output",  # Ignored by trajectory scorer
                context={"outputs": trajectory},
            ),
            expected=ExpectedResult(
                expected="Some expected string",  # Ignored by trajectory scorer
                context={"reference_outputs": trajectory},
            ),
        )

        # Assert - Trajectory scorer works normally, ignores string parameters
        assert isinstance(score, Score)
        assert score.passed is True


class TestImportError:
    """Test handling of agentevals import errors."""

    def test_missing_agentevals_library_returns_error_score(self):
        """Test that missing agentevals library is handled gracefully."""
        # This test validates the import error handling pattern,
        # but we can't actually test it without uninstalling agentevals.
        # The import happens inside the scorer function, so if agentevals
        # is missing, we'd get an error Score with metadata["error"].

        # This is a documentation test - the actual error handling
        # is validated by the error score test above.
        pass
