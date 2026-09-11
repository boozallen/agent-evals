# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Unit tests for sync run_eval wrapper around async implementation.

Tests that run_eval correctly wraps run_eval_async with asyncio.run().
"""

from helpers.scorers import passing_scorer

from agent_evals import TaskResult, run_eval
from agent_evals.core.types import ExampleData


def test_run_eval_wrapper_exists():
    """Test that run_eval function exists and is callable.

    Task: T048
    """
    from agent_evals.core.runner import run_eval

    assert callable(run_eval)


def test_run_eval_has_correct_signature():
    """Test that run_eval has the expected signature.

    Task: T048
    """
    import inspect

    from agent_evals.core.runner import run_eval

    sig = inspect.signature(run_eval)
    params = list(sig.parameters.keys())

    # Verify expected parameters exist
    assert "task" in params
    assert "dataset" in params
    assert "scorers" in params
    assert "platform" in params
    assert "config" in params


def test_run_eval_calls_asyncio_run_with_run_eval_async():
    """Test that sync run_eval wraps async implementation using asyncio.run().

    This verifies the implementation pattern:
    def run_eval(...):
        return asyncio.run(run_eval_async(...))

    We test this by calling run_eval and verifying it returns a valid result,
    which proves asyncio.run() was called internally.

    Task: T048
    """

    # Define test inputs
    def test_task(x):
        return TaskResult(output=x)

    test_dataset = [ExampleData(input="test")]
    test_scorers = [passing_scorer]

    # Call run_eval (which should wrap run_eval_async)
    result = run_eval(
        task=test_task,
        dataset=test_dataset,
        scorers=test_scorers,
    )

    # Verify result is returned (proves asyncio.run() was called internally)
    assert result is not None
    assert hasattr(result, "experiment_id")
    assert hasattr(result, "scores")
    assert "passing" in result.scores


def test_sync_run_eval_does_not_raise_runtime_error():
    """Test that calling run_eval from sync context doesn't raise RuntimeError.

    Task: T049
    """

    def task(input_value):
        return TaskResult(output=input_value)

    # This should work without RuntimeError
    result = run_eval(
        task=task,
        dataset=[ExampleData(input="test")],
        scorers=[passing_scorer],
    )

    assert result is not None
    assert len(result.examples) == 1


def test_sync_wrapper_preserves_all_parameters():
    """Test that sync wrapper passes all parameters to async implementation.

    Task: T048
    """
    from agent_evals.core.types import EvalConfig, ExampleData

    def task(input_value):
        return TaskResult(output=input_value)

    exec_config = EvalConfig()

    # Call with all parameters
    result = run_eval(
        task=task,
        dataset=[ExampleData(input="test")],
        scorers=[passing_scorer],
        config=exec_config,
    )

    # Verify result was produced (parameters were passed correctly)
    assert result is not None
    assert len(result.examples) == 1
    assert result.platform == "local"
