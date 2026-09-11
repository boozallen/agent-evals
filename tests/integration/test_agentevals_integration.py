# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for agentevals trajectory evaluators.

These tests validate that trajectory evaluators work correctly with the
full evaluation pipeline including adapters and result logging.
"""

import json

import pytest

from agent_evals import ExpectedResult, TaskResult, run_eval
from agent_evals.adapters.scorers.agentevals import (
    GraphTrajectoryStrictMatch,
    TrajectoryStrictMatch,
    TrajectorySubsetMatch,
    TrajectorySupsetMatch,
    TrajectoryUnorderedMatch,
)
from agent_evals.core.types import ExampleData


@pytest.fixture
def simple_trajectory_task():
    """Simple task for trajectory evaluation - returns empty string."""

    def task(input_val):
        return TaskResult(output="")  # Trajectory evaluators use context, not output

    return task


class TestTrajectoryStrictMatchIntegration:
    """Integration tests for TrajectoryStrictMatch evaluator."""

    def test_strict_match_with_valid_trajectory(self, simple_trajectory_task):
        """Test TrajectoryStrictMatch with matching trajectories (historical data pattern)."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(
                    expected="",
                    context={
                        "reference_outputs": [
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ],
                    },
                ),
                output=TaskResult(
                    output="",
                    context={
                        "outputs": [
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ],
                    },
                ),
            )
        ]

        # Act - no task, using historical data
        result = run_eval(
            dataset=dataset,
            scorers=[scorer],
        )

        # Assert
        assert len(result.examples) == 1
        assert "TrajectoryStrictMatch" in result.scores
        assert result.scores["TrajectoryStrictMatch"] == 1.0

    def test_strict_match_with_tool_calls(self, simple_trajectory_task):
        """Test TrajectoryStrictMatch with tool call trajectories (historical data pattern)."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(
                    expected="",
                    context={
                        "reference_outputs": [
                            {"role": "user", "content": "What's the weather in SF?"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "SF"}),
                                        }
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "name": "get_weather",
                                "content": "72°F, sunny",
                            },
                            {
                                "role": "assistant",
                                "content": "It's 72°F and sunny in SF.",
                            },
                        ],
                    },
                ),
                output=TaskResult(
                    output="",
                    context={
                        "outputs": [
                            {"role": "user", "content": "What's the weather in SF?"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "SF"}),
                                        }
                                    }
                                ],
                            },
                            {
                                "role": "tool",
                                "name": "get_weather",
                                "content": "72°F, sunny",
                            },
                            {
                                "role": "assistant",
                                "content": "It's 72°F and sunny in SF.",
                            },
                        ],
                    },
                ),
            )
        ]

        # Act - no task, using historical data
        result = run_eval(
            dataset=dataset,
            scorers=[scorer],
        )

        # Assert
        assert "TrajectoryStrictMatch" in result.scores
        assert result.scores["TrajectoryStrictMatch"] == 1.0

    def test_strict_match_with_mismatch(self, simple_trajectory_task):
        """Test TrajectoryStrictMatch with non-matching trajectories."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(expected=""),
                output=TaskResult(
                    output="",
                    context={
                        "outputs": [
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ],
                        "reference_outputs": [
                            {"role": "user", "content": "What is 2+2?"},
                            # Different structure - has tool call in reference
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
                            {"role": "tool", "content": "4"},
                            {"role": "assistant", "content": "4"},
                        ],
                    },
                ),
            )
        ]

        # Act
        result = run_eval(
            task=simple_trajectory_task,
            dataset=dataset,
            scorers=[scorer],
        )

        # Assert
        assert "TrajectoryStrictMatch" in result.scores
        assert result.scores["TrajectoryStrictMatch"] == 0.0

    def test_strict_match_missing_trajectory_data(self, simple_trajectory_task):
        """Test TrajectoryStrictMatch with missing trajectory data."""
        # Arrange
        scorer = TrajectoryStrictMatch()
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(expected=""),
                output=TaskResult(
                    output="",
                    context={},  # Missing 'outputs' and 'reference_outputs'
                ),
            )
        ]

        # Act
        result = run_eval(
            task=simple_trajectory_task,
            dataset=dataset,
            scorers=[scorer],
        )

        # Assert - Should handle gracefully with error score
        assert "TrajectoryStrictMatch" in result.scores
        assert result.scores["TrajectoryStrictMatch"] == 0.0


class TestTrajectoryMatchModesIntegration:
    """Integration tests for all 4 trajectory match modes (T032)."""

    def test_unordered_match_with_different_order(self, simple_trajectory_task):
        """Test TrajectoryUnorderedMatch with tools called in different order (historical data pattern)."""
        from agent_evals import ExpectedResult

        # Arrange
        scorer = TrajectoryUnorderedMatch()
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(
                    expected="",
                    context={
                        "reference_outputs": [
                            {"role": "user", "content": "Get weather for NYC and LA"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "NYC"}),
                                        }
                                    },
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "LA"}),
                                        }
                                    },
                                ],
                            },
                            {"role": "tool", "content": "NYC: 55°F"},
                            {"role": "tool", "content": "LA: 75°F"},
                            {"role": "assistant", "content": "NYC is 55°F, LA is 75°F"},
                        ],
                    },
                ),
                output=TaskResult(
                    output="",
                    context={
                        "outputs": [
                            {"role": "user", "content": "Get weather for NYC and LA"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "LA"}),
                                        }
                                    },
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "NYC"}),
                                        }
                                    },
                                ],
                            },
                            {"role": "tool", "content": "LA: 75°F"},
                            {"role": "tool", "content": "NYC: 55°F"},
                            {"role": "assistant", "content": "LA is 75°F, NYC is 55°F"},
                        ],
                    },
                ),
            )
        ]

        # Act - no task, using historical data
        result = run_eval(
            dataset=dataset,
            scorers=[scorer],
        )

        # Assert
        assert "TrajectoryUnorderedMatch" in result.scores
        assert result.scores["TrajectoryUnorderedMatch"] == 1.0

    def test_subset_match_actual_is_subset(self, simple_trajectory_task):
        """Test TrajectorySubsetMatch where actual is subset of reference (historical data pattern)."""
        from agent_evals import ExpectedResult

        # Arrange
        scorer = TrajectorySubsetMatch()
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(
                    expected="",
                    context={
                        "reference_outputs": [
                            {"role": "user", "content": "What's the weather?"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "SF"}),
                                        }
                                    }
                                ],
                            },
                            {"role": "tool", "content": "72°F"},
                            {"role": "assistant", "content": "It's 72°F"},
                            {"role": "assistant", "content": "Have a nice day!"},
                        ],
                    },
                ),
                output=TaskResult(
                    output="",
                    context={
                        "outputs": [
                            {"role": "user", "content": "What's the weather?"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "SF"}),
                                        }
                                    }
                                ],
                            },
                            {"role": "tool", "content": "72°F"},
                            {"role": "assistant", "content": "It's 72°F"},
                        ],
                    },
                ),
            )
        ]

        # Act - no task, using historical data
        result = run_eval(
            dataset=dataset,
            scorers=[scorer],
        )

        # Assert
        assert "TrajectorySubsetMatch" in result.scores
        assert result.scores["TrajectorySubsetMatch"] == 1.0

    def test_superset_match_actual_is_superset(self, simple_trajectory_task):
        """Test TrajectorySupsetMatch where actual is superset of reference."""
        # Arrange
        scorer = TrajectorySupsetMatch()
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(expected=""),
                output=TaskResult(
                    output="",
                    context={
                        "outputs": [
                            {"role": "user", "content": "What's the weather?"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "SF"}),
                                        }
                                    }
                                ],
                            },
                            {"role": "tool", "content": "72°F"},
                            {"role": "assistant", "content": "It's 72°F"},
                            {"role": "assistant", "content": "Have a nice day!"},
                        ],
                        "reference_outputs": [
                            {"role": "user", "content": "What's the weather?"},
                            {
                                "role": "assistant",
                                "content": "",
                                "tool_calls": [
                                    {
                                        "function": {
                                            "name": "get_weather",
                                            "arguments": json.dumps({"city": "SF"}),
                                        }
                                    }
                                ],
                            },
                            {"role": "tool", "content": "72°F"},
                            {"role": "assistant", "content": "It's 72°F"},
                        ],
                    },
                ),
            )
        ]

        # Act
        result = run_eval(
            task=simple_trajectory_task,
            dataset=dataset,
            scorers=[scorer],
        )

        # Assert
        assert "TrajectorySupsetMatch" in result.scores
        assert result.scores["TrajectorySupsetMatch"] == 1.0

    def test_all_four_match_modes_together(self, simple_trajectory_task):
        """Test all 4 match modes in a single evaluation."""
        # Arrange
        scorers = [
            TrajectoryStrictMatch(),
            TrajectoryUnorderedMatch(),
            TrajectorySubsetMatch(),
            TrajectorySupsetMatch(),
        ]
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(expected=""),
                output=TaskResult(
                    output="",
                    context={
                        "outputs": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                        ],
                        "reference_outputs": [
                            {"role": "user", "content": "Hello"},
                            {"role": "assistant", "content": "Hi there!"},
                        ],
                    },
                ),
            )
        ]

        # Act
        result = run_eval(
            task=simple_trajectory_task,
            dataset=dataset,
            scorers=scorers,
        )

        # Assert
        assert "TrajectoryStrictMatch" in result.scores
        assert "TrajectoryUnorderedMatch" in result.scores
        assert "TrajectorySubsetMatch" in result.scores
        assert "TrajectorySupsetMatch" in result.scores


class TestGraphTrajectoryIntegration:
    """Integration tests for graph trajectory evaluators (T064)."""

    def test_graph_trajectory_strict_match_dict_format(self, simple_trajectory_task):
        """Test GraphTrajectoryStrictMatch with dict format (historical data pattern)."""
        from agent_evals import ExpectedResult

        # Arrange
        scorer = GraphTrajectoryStrictMatch()
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(
                    expected="",
                    context={
                        "reference_outputs": {
                            "steps": [["__start__", "agent", "tools", "__end__"]]
                        },
                    },
                ),
                output=TaskResult(
                    output="",
                    context={
                        "outputs": {
                            "steps": [["__start__", "agent", "tools", "__end__"]]
                        }
                    },
                ),
            )
        ]

        # Act - no task, using historical data
        result = run_eval(
            dataset=dataset,
            scorers=[scorer],
        )

        # Assert
        assert "GraphTrajectoryStrictMatch" in result.scores
        assert result.scores["GraphTrajectoryStrictMatch"] == 1.0

    def test_graph_trajectory_with_langgraph_format(self, simple_trajectory_task):
        """Test GraphTrajectoryStrictMatch with LangGraph multi-step format (historical data pattern)."""
        from agent_evals import ExpectedResult

        # Arrange
        scorer = GraphTrajectoryStrictMatch()
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(
                    expected="",
                    context={
                        "reference_outputs": {
                            "steps": [
                                ["__start__", "agent", "tools", "__interrupt__"],
                                ["agent", "__end__"],
                            ]
                        },
                    },
                ),
                output=TaskResult(
                    output="",
                    context={
                        "outputs": {
                            "results": [],
                            "steps": [
                                ["__start__", "agent", "tools", "__interrupt__"],
                                ["agent", "__end__"],
                            ],
                        }
                    },
                ),
            )
        ]

        # Act - no task, using historical data
        result = run_eval(
            dataset=dataset,
            scorers=[scorer],
        )

        # Assert
        assert "GraphTrajectoryStrictMatch" in result.scores
        assert result.scores["GraphTrajectoryStrictMatch"] == 1.0

    def test_graph_trajectory_mismatch(self, simple_trajectory_task):
        """Test GraphTrajectoryStrictMatch with non-matching graph trajectories."""
        # Arrange
        scorer = GraphTrajectoryStrictMatch()
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(expected=""),
                output=TaskResult(
                    output="",
                    context={
                        "outputs": {"steps": [["__start__", "agent", "__end__"]]},
                        "reference_outputs": {
                            "steps": [["__start__", "agent", "tools", "__end__"]]
                        },
                    },
                ),
            )
        ]

        # Act
        result = run_eval(
            task=simple_trajectory_task,
            dataset=dataset,
            scorers=[scorer],
        )

        # Assert
        assert "GraphTrajectoryStrictMatch" in result.scores
        assert result.scores["GraphTrajectoryStrictMatch"] == 0.0


class TestMixedScorersIntegration:
    """Integration tests for mixing trajectory and string-based scorers (US5)."""

    def test_trajectory_and_string_scorers_together(self):
        """Test TrajectoryStrictMatch + Levenshtein in same evaluation (T071)."""
        from agent_evals.adapters.scorers.autoevals import Levenshtein

        # Arrange - task returns actual output for string scorer
        def task(input_val):
            return TaskResult(output="The weather in San Francisco is 72°F and sunny.")

        scorers = [
            TrajectoryStrictMatch(),
            Levenshtein(threshold=0.8),
        ]
        dataset = [
            ExampleData(
                input="What's the weather in SF?",
                expected=ExpectedResult(
                    expected="It's 72 degrees and sunny in San Francisco."
                ),
                output=TaskResult(
                    output="The weather in San Francisco is 72°F and sunny.",
                    context={
                        "outputs": [
                            {"role": "user", "content": "What's the weather in SF?"},
                            {
                                "role": "assistant",
                                "content": "The weather in San Francisco is 72°F and sunny.",
                            },
                        ],
                        "reference_outputs": [
                            {"role": "user", "content": "What's the weather in SF?"},
                            {
                                "role": "assistant",
                                "content": "It's 72 degrees and sunny in San Francisco.",
                            },
                        ],
                    },
                ),
            )
        ]

        # Act
        result = run_eval(
            task=task,
            dataset=dataset,
            scorers=scorers,
        )

        # Assert - Both scorers should return valid scores
        assert "TrajectoryStrictMatch" in result.scores
        assert "Levenshtein" in result.scores

    def test_multiple_trajectory_and_string_scorers(self):
        """Test mixing multiple trajectory + string scorers (T073)."""
        from agent_evals.adapters.scorers.autoevals import Levenshtein

        # Arrange
        def task(input_val):
            return TaskResult(output="4")

        scorers = [
            TrajectoryStrictMatch(),
            TrajectoryUnorderedMatch(),
            Levenshtein(threshold=0.8),
        ]
        dataset = [
            ExampleData(
                input="What is 2+2?",
                expected=ExpectedResult(expected="4"),
                output=TaskResult(
                    output="4",
                    context={
                        "outputs": [
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ],
                        "reference_outputs": [
                            {"role": "user", "content": "What is 2+2?"},
                            {"role": "assistant", "content": "4"},
                        ],
                    },
                ),
            )
        ]

        # Act
        result = run_eval(
            task=task,
            dataset=dataset,
            scorers=scorers,
        )

        # Assert - All 3 scorers should execute
        assert "TrajectoryStrictMatch" in result.scores
        assert "TrajectoryUnorderedMatch" in result.scores
        assert "Levenshtein" in result.scores

    def test_trajectory_scorer_failure_does_not_block_other_scorers(self):
        """Test that trajectory evaluator failure doesn't prevent evaluation continuing (T074)."""
        from agent_evals.adapters.scorers.autoevals import Levenshtein

        # Arrange - trajectory scorer will fail due to missing outputs
        def task(input_val):
            return TaskResult(output="Hello world")

        scorers = [
            TrajectoryStrictMatch(),
            Levenshtein(threshold=0.8),
        ]
        dataset = [
            ExampleData(
                input="test",
                expected=ExpectedResult(expected="Hello world"),
                output=TaskResult(output="Hello world", context={}),
            )
        ]

        # Act
        result = run_eval(
            task=task,
            dataset=dataset,
            scorers=scorers,
        )

        # Assert - String scorer should still work
        assert "TrajectoryStrictMatch" in result.scores
        assert "Levenshtein" in result.scores
        # Trajectory scorer should fail (score 0.0), string scorer should pass
        assert result.scores["TrajectoryStrictMatch"] == 0.0
        assert result.scores["Levenshtein"] > 0.0
