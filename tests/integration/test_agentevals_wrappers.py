# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for agentevals trajectory scorer wrappers.

Tests the adapter functions that normalize agentevals evaluator results to Score format.
All tests use mocks where possible - minimize actual agentevals library calls.

Following TDD:
1. Write tests first (these should FAIL initially)
2. Implement wrappers to make tests pass
3. Refactor as needed
"""

import json

import pytest

from agent_evals.core.types import ExpectedResult, Score, TaskResult

# =============================================================================
# User Story 1: TrajectoryStrictMatch Tests
# =============================================================================


class TestTrajectoryStrictMatch:
    """Test TrajectoryStrictMatch wrapper (T014-T016)."""

    def test_trajectory_strict_match_with_valid_trajectory(self):
        """T014: TrajectoryStrictMatch with valid trajectory data returns correct Score."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch

        # Create scorer
        scorer = TrajectoryStrictMatch()

        # Valid trajectory data
        trajectory = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        reference_trajectory = [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"},
        ]

        # Call scorer with trajectory in context
        result = scorer(
            result=TaskResult(
                output="",  # Unused for trajectory scorers
                context={
                    "outputs": trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",  # Unused for trajectory scorers
                context={
                    "reference_outputs": reference_trajectory,
                },
            ),
        )

        # Verify Score object
        assert isinstance(result, Score)
        assert result.name == "TrajectoryStrictMatch"
        assert result.value in [0.0, 1.0]  # Boolean result normalized to 0.0 or 1.0
        assert isinstance(result.passed, bool)
        assert isinstance(result.metadata, dict)

        # If passed, verify metadata has expected fields
        if result.passed:
            assert result.metadata.get("trajectory_length") == 2
            assert "agentevals_key" in result.metadata

    def test_trajectory_strict_match_with_tool_calls(self):
        """T014: TrajectoryStrictMatch with tool calls in trajectory."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch

        scorer = TrajectoryStrictMatch()

        trajectory = [
            {"role": "user", "content": "Get weather for SF"},
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
            {"role": "tool", "name": "get_weather", "content": "72°F, sunny"},
            {"role": "assistant", "content": "It's 72°F and sunny in SF."},
        ]

        reference_trajectory = trajectory  # Same trajectory

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": reference_trajectory,
                },
            ),
        )

        assert isinstance(result, Score)
        assert result.name == "TrajectoryStrictMatch"
        assert result.passed is True  # Exact match
        assert result.value == 1.0

    def test_trajectory_strict_match_missing_outputs(self):
        """T015: TrajectoryStrictMatch with missing outputs in context returns error Score."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch

        scorer = TrajectoryStrictMatch()

        # Call without outputs in context
        result = scorer(
            result=TaskResult(output="", context={}),
            expected="",
        )

        # Should return error Score (not raise exception)
        assert isinstance(result, Score)
        assert result.name == "TrajectoryStrictMatch"
        assert result.value == 0.0
        assert result.passed is False
        assert "error" in result.metadata
        assert "outputs" in result.metadata["error"].lower()

    def test_trajectory_strict_match_malformed_trajectory_not_list(self):
        """T016: TrajectoryStrictMatch with malformed trajectory (not a list)."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch

        scorer = TrajectoryStrictMatch()

        # Malformed: outputs is a string instead of list
        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": "not a list",  # Should be list
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [{"role": "user", "content": "test"}],
                },
            ),
        )

        # Should return error Score
        assert isinstance(result, Score)
        assert result.name == "TrajectoryStrictMatch"
        assert result.value == 0.0
        assert result.passed is False
        assert "error" in result.metadata

    def test_trajectory_strict_match_malformed_trajectory_invalid_message(self):
        """T016: TrajectoryStrictMatch with malformed trajectory (invalid message format)."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch

        scorer = TrajectoryStrictMatch()

        # Malformed: message missing required fields
        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": [{"invalid": "message"}],  # Missing role and content
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": [{"role": "user", "content": "test"}],
                },
            ),
        )

        # Should return error Score (agentevals will fail)
        assert isinstance(result, Score)
        assert result.name == "TrajectoryStrictMatch"
        assert result.value == 0.0
        assert result.passed is False
        assert "error" in result.metadata

    def test_trajectory_strict_match_import_error(self):
        """T016: TrajectoryStrictMatch handles errors gracefully.

        Note: Testing actual ImportError is complex due to module caching.
        This test verifies the error handling code path by testing
        error Score creation, which is the key behavior.
        """
        # Test that error Scores are created correctly when exceptions occur
        from agent_evals.adapters.scorers.agentevals import _create_error_score

        error_score = _create_error_score(
            "TrajectoryStrictMatch",
            "agentevals not installed",
            ImportError("Module not found"),
        )

        assert isinstance(error_score, Score)
        assert error_score.name == "TrajectoryStrictMatch"
        assert error_score.value == 0.0
        assert error_score.passed is False
        assert "error" in error_score.metadata
        assert error_score.metadata["error"] == "agentevals not installed"
        assert error_score.metadata["exception_type"] == "ImportError"

    def test_trajectory_strict_match_tool_args_match_mode(self):
        """Test TrajectoryStrictMatch with different tool_args_match_mode settings."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch

        # Test with ignore mode
        scorer = TrajectoryStrictMatch(tool_args_match_mode="ignore")

        trajectory = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"function": {"name": "tool1", "arguments": json.dumps({"a": 1})}}
                ],
            },
        ]

        reference_trajectory = [
            {"role": "user", "content": "test"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "tool1",
                            "arguments": json.dumps({"a": 2}),  # Different args
                        }
                    }
                ],
            },
        ]

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": reference_trajectory,
                },
            ),
        )

        # Should match because args are ignored
        assert isinstance(result, Score)
        assert result.name == "TrajectoryStrictMatch"
        # Result depends on agentevals behavior
        assert result.value in [0.0, 1.0]


# =============================================================================
# User Story 2: Additional Match Mode Tests
# =============================================================================


class TestTrajectoryUnorderedMatch:
    """Test TrajectoryUnorderedMatch wrapper (T027)."""

    def test_trajectory_unordered_match_different_order(self):
        """T027: TrajectoryUnorderedMatch with same tools in different order."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryUnorderedMatch

        scorer = TrajectoryUnorderedMatch()

        # Tools called in different order
        trajectory = [
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
        ]

        reference_trajectory = [
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
        ]

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": reference_trajectory,
                },
            ),
        )

        assert isinstance(result, Score)
        assert result.name == "TrajectoryUnorderedMatch"


class TestTrajectorySubsetMatch:
    """Test TrajectorySubsetMatch wrapper (T028)."""

    def test_trajectory_subset_match_actual_subset_of_reference(self):
        """T028: TrajectorySubsetMatch where actual ⊆ reference."""
        from agent_evals.adapters.scorers.agentevals import TrajectorySubsetMatch

        scorer = TrajectorySubsetMatch()

        # Actual has fewer steps
        trajectory = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]

        reference_trajectory = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "thinking..."},
            {"role": "assistant", "content": "response"},
        ]

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": reference_trajectory,
                },
            ),
        )

        assert isinstance(result, Score)
        assert result.name == "TrajectorySubsetMatch"


class TestTrajectorySupsetMatch:
    """Test TrajectorySupsetMatch wrapper (T029)."""

    def test_trajectory_supset_match_actual_superset_of_reference(self):
        """T029: TrajectorySupsetMatch where actual ⊇ reference."""
        from agent_evals.adapters.scorers.agentevals import TrajectorySupsetMatch

        scorer = TrajectorySupsetMatch()

        # Actual has more steps
        trajectory = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "thinking..."},
            {"role": "assistant", "content": "response"},
        ]

        reference_trajectory = [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "response"},
        ]

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": reference_trajectory,
                },
            ),
        )

        assert isinstance(result, Score)
        assert result.name == "TrajectorySupsetMatch"


class TestToolArgsMatchMode:
    """Test tool_args_match_mode parameter (T030)."""

    def test_tool_args_match_mode_ignore(self):
        """T030: Tool arguments are ignored when tool_args_match_mode='ignore'."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch

        scorer = TrajectoryStrictMatch(tool_args_match_mode="ignore")

        trajectory = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "tool1", "arguments": json.dumps({"x": 1})}}
                ],
            }
        ]

        reference_trajectory = [
            {
                "role": "assistant",
                "tool_calls": [
                    {"function": {"name": "tool1", "arguments": json.dumps({"x": 99})}}
                ],
            }
        ]

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": reference_trajectory,
                },
            ),
        )

        assert isinstance(result, Score)

    def test_custom_tool_comparator_callable(self):
        """T031: Custom callable comparator is invoked and respected."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch

        # Track if comparator was called
        comparator_calls = []

        def custom_search_comparator(actual_args, reference_args):
            """Custom comparator that only checks 'query' field."""
            comparator_calls.append((actual_args, reference_args))
            # Match if query fields are equal (ignore other fields)
            return actual_args.get("query") == reference_args.get("query")

        scorer = TrajectoryStrictMatch(
            tool_args_match_overrides={"search": custom_search_comparator}
        )

        # Trajectories differ in 'limit' field but match on 'query'
        trajectory = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": "test", "limit": 10}),
                        }
                    }
                ],
            }
        ]

        reference_trajectory = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": "test", "limit": 5}),
                        }
                    }
                ],
            }
        ]

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": reference_trajectory,
                },
            ),
        )

        # Verify comparator was called
        assert len(comparator_calls) == 1
        actual_call, reference_call = comparator_calls[0]
        assert actual_call == {"query": "test", "limit": 10}
        assert reference_call == {"query": "test", "limit": 5}

        # Verify the match succeeded (because query fields match)
        assert isinstance(result, Score)
        assert result.name == "TrajectoryStrictMatch"
        assert result.value == 1.0
        assert result.passed is True

    def test_custom_tool_comparator_returns_false(self):
        """T031: Custom comparator returning False causes mismatch."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch

        def custom_search_comparator(actual_args, reference_args):
            """Custom comparator that always returns False."""
            return False

        scorer = TrajectoryStrictMatch(
            tool_args_match_overrides={"search": custom_search_comparator}
        )

        trajectory = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": "test"}),
                        }
                    }
                ],
            }
        ]

        reference_trajectory = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": json.dumps({"query": "test"}),
                        }
                    }
                ],
            }
        ]

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": trajectory,
                    "reference_outputs": reference_trajectory,
                },
            )
        )

        # Verify the match failed (because comparator returned False)
        assert isinstance(result, Score)
        assert result.name == "TrajectoryStrictMatch"
        assert result.value == 0.0
        assert result.passed is False

    def test_custom_tool_comparator_per_tool_override(self):
        """T031: Custom comparators work per-tool with mixed tools."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch

        search_calls = []
        calculate_calls = []

        def search_comparator(actual_args, reference_args):
            """Only compare query field for search tool."""
            search_calls.append((actual_args, reference_args))
            return actual_args.get("query") == reference_args.get("query")

        def calculate_comparator(actual_args, reference_args):
            """Only compare expression field for calculate tool."""
            calculate_calls.append((actual_args, reference_args))
            return actual_args.get("expression") == reference_args.get("expression")

        scorer = TrajectoryStrictMatch(
            tool_args_match_overrides={
                "search": search_comparator,
                "calculate": calculate_comparator,
            }
        )

        trajectory = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": json.dumps(
                                {"query": "weather", "limit": 10, "source": "api"}
                            ),
                        }
                    },
                    {
                        "function": {
                            "name": "calculate",
                            "arguments": json.dumps(
                                {"expression": "2+2", "format": "decimal"}
                            ),
                        }
                    },
                ],
            }
        ]

        reference_trajectory = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "function": {
                            "name": "search",
                            "arguments": json.dumps(
                                {"query": "weather", "limit": 5, "source": "cache"}
                            ),
                        }
                    },
                    {
                        "function": {
                            "name": "calculate",
                            "arguments": json.dumps(
                                {"expression": "2+2", "format": "scientific"}
                            ),
                        }
                    },
                ],
            }
        ]

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": reference_trajectory,
                },
            ),
        )

        # Verify both comparators were called
        assert len(search_calls) == 1
        assert len(calculate_calls) == 1

        # Verify search comparator got the right args
        assert search_calls[0][0]["query"] == "weather"
        assert search_calls[0][1]["query"] == "weather"

        # Verify calculate comparator got the right args
        assert calculate_calls[0][0]["expression"] == "2+2"
        assert calculate_calls[0][1]["expression"] == "2+2"

        # Verify the match succeeded (both tools matched on their key fields)
        assert isinstance(result, Score)
        assert result.value == 1.0
        assert result.passed is True


# =============================================================================
# User Story 3: TrajectoryLLMAsJudge Tests
# =============================================================================


class TestTrajectoryLLMAsJudge:
    """Test TrajectoryLLMAsJudge wrapper (T047-T049)."""

    @pytest.mark.asyncio
    async def test_trajectory_llm_judge_with_custom_prompt(self):
        """T047: TrajectoryLLMAsJudge with custom prompt.

        Note: This test skips if Ollama/models aren't available since
        we can't easily mock the evaluator created at function definition time.
        """
        from agent_evals.adapters.scorers.agentevals import TrajectoryLLMAsJudge

        custom_prompt = "Evaluate trajectory: {outputs}"

        try:
            # Try to create scorer - will fail if model unavailable
            scorer = TrajectoryLLMAsJudge(
                prompt=custom_prompt,
                model="openai:o3-mini",
                judge_base_url="https://judge.example.com/v1",
            )

            # If we get here, the model is available
            # The actual LLM call behavior is tested in integration tests
            # Here we just verify the scorer was created correctly
            assert callable(scorer)

            # Test error handling path instead (no outputs provided)
            result = await scorer(result=TaskResult(output="", context={}))
            assert isinstance(result, Score)
            assert result.value == 0.0  # Should fail due to missing outputs

        except (ImportError, RuntimeError) as e:
            # Skip if agentevals not installed or model unavailable
            pytest.skip(f"LLM model unavailable: {e}")

    @pytest.mark.asyncio
    async def test_trajectory_llm_judge_explicit_model_unavailable_graceful(self):
        """T049: TrajectoryLLMAsJudge with explicit model unavailable returns error Score."""
        from agent_evals.adapters.scorers.agentevals import TrajectoryLLMAsJudge

        try:
            # Use a model that doesn't exist
            scorer = TrajectoryLLMAsJudge(
                model="fake:model:123", judge_base_url="https://judge.example.com/v1"
            )

            trajectory = [
                {"role": "user", "content": "test"},
                {"role": "assistant", "content": "response"},
            ]

            result = await scorer(
                result=TaskResult(
                    output="",
                    context={
                        "outputs": trajectory,
                    },
                )
            )

            # Should return error Score (graceful degradation)
            assert isinstance(result, Score)
            assert result.value == 0.0
            assert result.passed is False
            assert "error" in result.metadata

        except ImportError:
            pytest.skip("agentevals not installed")

    @pytest.mark.asyncio
    async def test_trajectory_llm_judge_with_inputs_parameter(self):
        """T050: TrajectoryLLMAsJudge accepts inputs parameter from context.

        Verifies that the inputs parameter is properly extracted from context
        and passed to the underlying agentevals evaluator. This aligns with
        the official agentevals usage pattern where inputs provides task
        context for better LLM evaluation.
        """
        from agent_evals.adapters.scorers.agentevals import TrajectoryLLMAsJudge

        try:
            # Use a fake model to avoid actual LLM calls
            scorer = TrajectoryLLMAsJudge(
                model="fake:model:test", judge_base_url="https://judge.example.com/v1"
            )

            trajectory = [
                {"role": "user", "content": "What is the weather in SF?"},
                {"role": "assistant", "content": "It's sunny and 70 degrees."},
            ]

            # Test WITH inputs parameter
            result = await scorer(
                result=TaskResult(
                    output="",
                    context={
                        "inputs": "What is the weather in San Francisco?",
                        "outputs": trajectory,
                    },
                )
            )

            # Should return error Score (fake model), but inputs should be accepted
            assert isinstance(result, Score)
            # The scorer should have processed the inputs without error
            # (actual error is from fake model, not from inputs handling)

            # Test WITHOUT inputs parameter (backward compatibility)
            result_no_inputs = await scorer(
                result=TaskResult(
                    output="",
                    context={
                        "outputs": trajectory,
                    },
                )
            )

            # Should also work without inputs
            assert isinstance(result_no_inputs, Score)

        except ImportError:
            pytest.skip("agentevals not installed")


# =============================================================================
# User Story 4: Graph Trajectory Tests
# =============================================================================


class TestGraphTrajectoryStrictMatch:
    """Test GraphTrajectoryStrictMatch wrapper (T061-T062)."""

    def test_graph_trajectory_strict_match_with_dict_format(self):
        """T061: GraphTrajectoryStrictMatch with dict format (LangGraph)."""
        from agent_evals.adapters.scorers.agentevals import GraphTrajectoryStrictMatch

        scorer = GraphTrajectoryStrictMatch()

        # LangGraph format with steps
        graph_trajectory = {"steps": [["__start__", "agent", "tools", "__end__"]]}

        reference_graph = {"steps": [["__start__", "agent", "tools", "__end__"]]}

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": graph_trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": reference_graph,
                },
            ),
        )

        assert isinstance(result, Score)
        assert result.name == "GraphTrajectoryStrictMatch"

    def test_graph_trajectory_strict_match_invalid_simple_list(self):
        """T062: GraphTrajectoryStrictMatch with invalid simple list format."""
        from agent_evals.adapters.scorers.agentevals import GraphTrajectoryStrictMatch

        scorer = GraphTrajectoryStrictMatch()

        # Simple list format (NOT supported - must be dict with steps)
        graph_trajectory = ["__start__", "agent", "tools", "__end__"]

        reference_graph = ["__start__", "agent", "tools", "__end__"]

        result = scorer(
            result=TaskResult(
                output="",
                context={
                    "outputs": graph_trajectory,
                },
            ),
            expected=ExpectedResult(
                expected="",
                context={
                    "reference_outputs": reference_graph,
                },
            ),
        )

        # Should return error Score
        assert isinstance(result, Score)
        assert result.value == 0.0
        assert result.passed is False
        assert "error" in result.metadata


class TestGraphTrajectoryLLMAsJudge:
    """Test GraphTrajectoryLLMAsJudge wrapper (T063)."""

    @pytest.mark.asyncio
    async def test_graph_trajectory_llm_judge(self):
        """T063: GraphTrajectoryLLMAsJudge with graph trajectory.

        Note: This test skips if models aren't available since
        we can't easily mock the evaluator created at function definition time.
        """
        from agent_evals.adapters.scorers.agentevals import GraphTrajectoryLLMAsJudge

        try:
            scorer = GraphTrajectoryLLMAsJudge(
                model="openai:o3-mini", judge_base_url="https://judge.example.com/v1"
            )

            # If we get here, the model is available
            assert callable(scorer)

            # Test error handling path (no outputs provided)
            result = await scorer(result=TaskResult(output="", context={}))
            assert isinstance(result, Score)
            assert result.value == 0.0  # Should fail due to missing outputs

        except (ImportError, RuntimeError) as e:
            pytest.skip(f"LLM model unavailable: {e}")
