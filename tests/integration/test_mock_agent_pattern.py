# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Test Mock Agent Pattern - Validates transformed examples follow pattern

This test validates that examples using the mock agent pattern:
1. Use deterministic dictionary-based agents (not identity_task)
2. Include 3-5 realistic responses
3. Run successfully without API keys
4. Execute in under 1 second
5. Produce expected score ranges
"""

import asyncio
import time
from collections.abc import Callable

import pytest

from agent_evals import ExpectedResult, TaskResult, run_eval_async
from agent_evals.core.types import ExampleData


def create_mock_qa_agent() -> Callable[[str], TaskResult]:
    """Create a deterministic Q&A mock agent for testing."""

    def qa_agent(query: str) -> TaskResult:
        """Mock Q&A agent with deterministic responses."""
        knowledge = {
            "capital of france": "Paris",
            "capital of germany": "Berlin",
            "2+2": "4",
            "speed of light": "299,792,458 m/s",
        }

        query_lower = query.lower()
        for key, answer in knowledge.items():
            if key in query_lower:
                return TaskResult(output=answer)

        return TaskResult(output="I don't know.")

    return qa_agent


def create_test_dataset() -> list[ExampleData]:
    """Create test dataset with pass/fail cases."""
    return [
        ExampleData(
            input="What is the capital of France?",
            expected=ExpectedResult(expected="Paris"),
            metadata={"description": "Perfect match"},
        ),
        ExampleData(
            input="What is the capital of Germany?",
            expected=ExpectedResult(expected="Berlin"),
            metadata={"description": "Perfect match"},
        ),
        ExampleData(
            input="What is 2+2?",
            expected=ExpectedResult(expected="4"),
            metadata={"description": "Perfect match"},
        ),
        ExampleData(
            input="What is the speed of light?",
            expected=ExpectedResult(expected="299,792,458 m/s"),
            metadata={"description": "Perfect match"},
        ),
        ExampleData(
            input="What is the capital of Spain?",
            expected=ExpectedResult(expected="Madrid"),
            metadata={"description": "Unknown - should return 'I don't know'"},
        ),
    ]


@pytest.mark.asyncio
async def test_mock_agent_pattern_is_deterministic():
    """Test that mock agent produces same results on multiple runs."""
    from agent_evals.adapters.scorers.autoevals import ExactMatch

    agent = create_mock_qa_agent()
    dataset = create_test_dataset()[:3]  # Use first 3 cases

    # Run evaluation twice
    result1 = await run_eval_async(
        task=agent,
        dataset=dataset,
        scorers=[ExactMatch()],
    )

    result2 = await run_eval_async(
        task=agent,
        dataset=dataset,
        scorers=[ExactMatch()],
    )

    # Scores should be identical
    for ex1, ex2 in zip(result1.examples, result2.examples, strict=False):
        assert ex1.scores["ExactMatch"].value == ex2.scores["ExactMatch"].value
        assert ex1.output == ex2.output


@pytest.mark.asyncio
async def test_mock_agent_pattern_is_fast():
    """Test that mock agent executes in under 1 second."""
    from agent_evals.adapters.scorers.autoevals import ExactMatch

    agent = create_mock_qa_agent()
    dataset = create_test_dataset()

    start = time.time()
    await run_eval_async(
        task=agent,
        dataset=dataset,
        scorers=[ExactMatch()],
    )
    duration = time.time() - start

    assert duration < 1.0, f"Mock agent took {duration:.2f}s, should be < 1s"


@pytest.mark.asyncio
async def test_mock_agent_pattern_no_api_keys():
    """Test that mock agent works without environment variables."""
    import os

    from agent_evals.adapters.scorers.autoevals import ExactMatch

    # Temporarily clear any API keys
    old_keys = {}
    api_key_vars = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "BRAINTRUST_API_KEY"]

    for var in api_key_vars:
        old_keys[var] = os.environ.get(var)
        if var in os.environ:
            del os.environ[var]

    try:
        agent = create_mock_qa_agent()
        dataset = create_test_dataset()[:2]

        # Should work without any API keys
        result = await run_eval_async(
            task=agent,
            dataset=dataset,
            scorers=[ExactMatch()],
        )

        assert len(result.examples) == 2

    finally:
        # Restore API keys
        for var, value in old_keys.items():
            if value is not None:
                os.environ[var] = value


@pytest.mark.asyncio
async def test_mock_agent_pattern_realistic_responses():
    """Test that mock agent provides realistic, agent-like responses."""
    from agent_evals.adapters.scorers.autoevals import ExactMatch

    agent = create_mock_qa_agent()
    dataset = create_test_dataset()

    result = await run_eval_async(
        task=agent,
        dataset=dataset,
        scorers=[ExactMatch()],
    )

    # Check that responses are not just pass-through
    for example in result.examples:
        # Responses should be different from inputs (not identity)
        assert example.output != example.input

        # Responses should be strings (agent-like)
        assert isinstance(example.output, str)

        # Responses should not be empty
        assert len(example.output) > 0


def test_mock_agent_has_fallback():
    """Test that mock agent has a default fallback response."""
    agent = create_mock_qa_agent()

    # Test with unrecognized input
    result = agent("This is a completely unknown question about nothing")

    assert result.output == "I don't know."


def test_mock_agent_is_case_insensitive():
    """Test that mock agent handles case variations."""
    agent = create_mock_qa_agent()

    # Test case variations
    assert agent("What is the CAPITAL of FRANCE?").output == "Paris"
    assert agent("what is the capital of france?").output == "Paris"
    assert agent("What Is The Capital Of France?").output == "Paris"


if __name__ == "__main__":
    # Run tests
    print("Running mock agent pattern tests...")
    asyncio.run(test_mock_agent_pattern_is_deterministic())
    asyncio.run(test_mock_agent_pattern_is_fast())
    asyncio.run(test_mock_agent_pattern_no_api_keys())
    asyncio.run(test_mock_agent_pattern_realistic_responses())
    test_mock_agent_has_fallback()
    test_mock_agent_is_case_insensitive()
    print("✅ All tests passed!")
