# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for LangFusePlatform with mocked SDK.

Tests the LangFuse adapter that uses langfuse.get_client().run_experiment() API
in isolation by mocking the langfuse SDK, ensuring no actual API calls are made
during unit tests.
"""

import asyncio
import time
from unittest.mock import Mock, patch

import pytest
from langfuse.experiment import ExperimentItemResult, ExperimentResult

from agent_evals.adapters.platforms.langfuse import LangfuseConfig
from agent_evals.core.types import EvalResult, ExampleData, ExpectedResult, Score


class TestLangFusePlatformBasic:
    """Test basic adapter functionality."""

    @patch("agent_evals.adapters.platforms.langfuse.get_client")
    def test_adapter_initialization(self, mock_get_client):
        """Test that adapter can be instantiated."""
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        adapter = LangFusePlatform()
        assert adapter is not None

    @patch("agent_evals.adapters.platforms.langfuse.get_client")
    def test_evaluate_wraps_aevaluate(self, mock_get_client):
        """Test that sync evaluate() wraps async aevaluate()."""
        # Mock LangFuse components
        mock_client = Mock()
        mock_client.auth_check.return_value = True
        mock_get_client.return_value = mock_client

        mock_result = Mock()
        mock_result.experiment_id = "test-exp-id"
        mock_result.dataset_run_url = None
        mock_result.item_results = []
        mock_client.run_experiment.return_value = mock_result

        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        adapter = LangFusePlatform()

        def simple_task(x):
            return f"output for {x}"

        def simple_scorer(result, expected=None, **kwargs):
            return Score(name="test", value=0.8, passed=True)

        dataset = [
            ExampleData(
                input="test input",
                expected=ExpectedResult(expected="expected output"),
            )
        ]

        # This should call aevaluate under the hood
        result = adapter.evaluate(
            simple_task,
            dataset,
            [simple_scorer],
            platform=LangfuseConfig(experiment="test-experiment"),
        )

        assert isinstance(result, EvalResult)
        assert result.platform == "langfuse"
        # dataset_run_url is None for list[dict] runs (no Langfuse Dataset);
        # the adapter must coerce to "" rather than propagate None.
        assert result.experiment_url == ""
        mock_client.run_experiment.assert_called_once()

    @patch("agent_evals.adapters.platforms.langfuse.get_client")
    def test_aevaluate_basic_flow(self, mock_get_client):
        """Test basic async evaluation flow."""
        # Mock LangFuse components
        mock_client = Mock()
        mock_client.auth_check.return_value = True
        mock_get_client.return_value = mock_client

        mock_result = Mock()
        mock_result.experiment_id = "test-exp-id"
        mock_result.dataset_run_url = (
            "http://localhost:3000/project/p1/datasets/d1/runs/r1"
        )
        mock_result.item_results = []
        mock_client.run_experiment.return_value = mock_result

        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        adapter = LangFusePlatform()

        def simple_task(input_value):
            return f"output for {input_value}"

        def simple_scorer(result, expected=None, **kwargs):
            return Score(name="test", value=0.8, passed=True)

        dataset = [
            ExampleData(
                input={"input": "test input"},
                expected=ExpectedResult(expected="expected output"),
            )
        ]

        async def run_test():
            result = await adapter.aevaluate(
                simple_task,
                dataset,
                [simple_scorer],
                platform=LangfuseConfig(
                    experiment="test-experiment", description="test description"
                ),
            )
            return result

        result = asyncio.run(run_test())

        assert isinstance(result, EvalResult)
        assert result.platform == "langfuse"
        assert result.experiment_id == "test-exp-id"
        assert (
            result.experiment_url
            == "http://localhost:3000/project/p1/datasets/d1/runs/r1"
        )
        mock_client.run_experiment.assert_called_once()

    @pytest.mark.asyncio
    async def test_config_validation(self):
        """Test that aevaluate raises when no LangfuseConfig is provided."""
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform
        from agent_evals.core.types import TaskResult

        adapter = LangFusePlatform()

        # No platform config should raise ValueError
        with pytest.raises(ValueError, match="requires a LangfuseConfig"):
            await adapter.aevaluate(
                task=lambda x: TaskResult(output="ok"),
                dataset=[
                    ExampleData(input="hi", expected=ExpectedResult(expected="x"))
                ],
                evaluators=[],
                platform=None,
            )

    def test_convert_to_platform_dataset(self):
        """Test dataset conversion to LangFuse format."""
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        adapter = LangFusePlatform()

        dataset = [
            ExampleData(
                input={"question": "What is 2+2?"},
                expected=ExpectedResult(expected="4"),
                metadata={"source": "math"},
            ),
            ExampleData(
                input={"question": "What is the capital of France?"},
                expected=ExpectedResult(expected="Paris"),
            ),
        ]

        result = adapter._convert_to_platform_dataset(dataset)

        assert len(result) == 2
        # Full ExpectedResult objects are preserved so scorers receive context
        assert result[0] == {
            "input": {"question": "What is 2+2?"},
            "expected_output": ExpectedResult(expected="4"),
            "metadata": {"source": "math"},
        }
        assert result[1] == {
            "input": {"question": "What is the capital of France?"},
            "expected_output": ExpectedResult(expected="Paris"),
        }

    def test_convert_to_platform_scorer(self):
        """Test scorer conversion to LangFuse evaluator format."""
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        adapter = LangFusePlatform()

        def simple_scorer(result, expected=None, **kwargs):
            return Score(name="test", value=0.8, passed=True)

        langfuse_evaluators = adapter._convert_to_platform_scorer([simple_scorer])

        assert len(langfuse_evaluators) == 1
        evaluator = langfuse_evaluators[0]

        # Test calling the converted evaluator
        with patch("agent_evals.adapters.platforms.langfuse.Evaluation") as mock_eval:
            mock_eval.return_value = Mock()

            evaluator(
                input="test input",
                output="test output",
                expected_output="expected",
                metadata={"test": "data"},
            )

            mock_eval.assert_called_once()

    def test_infer_scorer_name(self):
        """Test scorer name inference from different types of callables."""
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        adapter = LangFusePlatform()

        # Test closure-style name extraction like autoevals (most important case)
        def create_closure_scorer():
            scorer_name = "NumericDiff"

            def scorer():
                return scorer_name

            return scorer

        closure_scorer = create_closure_scorer()
        assert adapter._infer_scorer_name(closure_scorer) == "NumericDiff"

        # Test factory pattern with <locals> normalization
        class MockFactory:
            pass

        def mock_factory_scorer():
            pass

        mock_factory_scorer.__qualname__ = "MockFactory.<locals>.scorer"
        assert adapter._infer_scorer_name(mock_factory_scorer) == "MockFactory"

        # Test basic callable fallback
        class SimpleScorer:
            def __call__(self):
                pass

        simple_scorer = SimpleScorer()
        assert adapter._infer_scorer_name(simple_scorer) == "SimpleScorer"

    @patch("agent_evals.adapters.platforms.langfuse.get_client")
    def test_pull_traces_basic(self, mock_get_client):
        """Test basic pull traces functionality."""
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        # Mock LangFuse API
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock trace list
        mock_trace = Mock()
        mock_trace.id = "trace-1"
        mock_response = Mock()
        mock_response.data = [mock_trace]
        mock_client.api.trace.list.return_value = mock_response

        # Mock dataset
        mock_dataset_item = Mock()
        mock_dataset_item.source_trace_id = "trace-1"
        mock_dataset_item.input = "test input"
        mock_dataset_item.expected_output = "expected output"
        mock_dataset_item.metadata = {"source": "test"}

        mock_dataset = Mock()
        mock_dataset.items = [mock_dataset_item]
        mock_client.get_dataset.return_value = mock_dataset

        # Mock trace get
        mock_observation = Mock()
        mock_observation.output = "test output"
        mock_original_trace = Mock()
        mock_original_trace.observations = [mock_observation]
        mock_client.api.trace.get.return_value = mock_original_trace

        adapter = LangFusePlatform()
        examples = adapter.pull_traces(config={"dataset": "test-dataset"})

        assert len(examples) == 1
        assert examples[0].input == "test input"
        assert examples[0].output is not None
        assert examples[0].output.output == "test output"
        assert examples[0].expected is not None
        assert examples[0].expected.expected == "expected output"
        assert examples[0].metadata["source"] == "test"
        mock_client.get_dataset.assert_called_once_with("test-dataset")

    @patch("agent_evals.adapters.platforms.langfuse.get_client")
    def test_aevaluate_auth_failure(self, mock_get_client):
        """Test handling of LangFuse authentication failure."""
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        # Mock client with failed auth
        mock_client = Mock()
        mock_client.auth_check.return_value = False
        mock_get_client.return_value = mock_client

        adapter = LangFusePlatform()

        def simple_task(item):
            return "output"

        def simple_scorer(result, expected=None, **kwargs):
            return Score(name="test", value=0.8, passed=True)

        dataset = [
            ExampleData(input="test", expected=ExpectedResult(expected="expected"))
        ]

        async def run_test():
            with pytest.raises(
                RuntimeError, match="LangFuse client authentication failed"
            ):
                await adapter.aevaluate(
                    simple_task,
                    dataset,
                    [simple_scorer],
                    platform=LangfuseConfig(experiment="test-experiment"),
                )

        asyncio.run(run_test())


class TestLangFusePlatformErrorHandling:
    """Test error handling and edge cases."""

    @patch("agent_evals.adapters.platforms.langfuse.get_client")
    def test_pull_traces_error_handling(self, mock_get_client):
        """Test pull traces error handling."""
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        mock_client = Mock()
        mock_get_client.return_value = mock_client
        mock_client.api.trace.list.side_effect = Exception("API Error")

        adapter = LangFusePlatform()
        with pytest.raises(RuntimeError, match="Failed to export traces from LangFuse"):
            adapter.pull_traces(config={"dataset": "test-dataset"})

    @patch("agent_evals.adapters.platforms.langfuse.get_client")
    def test_pull_traces_empty_data(self, mock_get_client):
        """Test pull traces with empty or malformed data."""
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        # Mock LangFuse API with empty dataset
        mock_client = Mock()
        mock_get_client.return_value = mock_client

        # Mock trace list
        mock_response = Mock()
        mock_response.data = []
        mock_client.api.trace.list.return_value = mock_response

        # Mock empty dataset
        mock_dataset = Mock()
        mock_dataset.items = []
        mock_client.get_dataset.return_value = mock_dataset

        adapter = LangFusePlatform()
        examples = adapter.pull_traces(config={"dataset": "empty-dataset"})
        assert examples == []

        # Test with missing config
        with pytest.raises(ValueError, match="requires config with 'dataset' key"):
            adapter.pull_traces(config=None)

        with pytest.raises(ValueError, match="requires config with 'dataset' key"):
            adapter.pull_traces(config={"other_key": "value"})


def _stub_run_experiment(experiment_id: str = "exp"):
    """Build a fake ``run_experiment`` that invokes the wrapped task per item
    and returns a real ``ExperimentResult``."""

    def fake(*, name, description, data, task, evaluators, metadata):
        item_results = []
        for item in data:
            output = task(item=item)
            if asyncio.iscoroutine(output):
                output = asyncio.new_event_loop().run_until_complete(output)
            item_results.append(
                ExperimentItemResult(
                    item=item,
                    output=output,
                    evaluations=[],
                    trace_id=None,
                    dataset_run_id=None,
                )
            )
        return ExperimentResult(
            name=name,
            run_name=f"{name}-run",
            description=description,
            item_results=item_results,
            run_evaluations=[],
            experiment_id=experiment_id,
        )

    return fake


class TestErrorAndDurationCaptureThroughAevaluate:
    """Failed tasks produce EvalExamples with captured errors; successful
    tasks have non-zero durations. Live counterparts in
    ``test_langfuse_adapter_live.py`` cover the same contract against a
    real langfuse server.
    """

    @pytest.mark.asyncio
    @patch("agent_evals.adapters.platforms.langfuse.get_client")
    async def test_user_exception_surfaces_in_examples(self, mock_get_client):
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        client = Mock()
        client.auth_check.return_value = True
        client.run_experiment = _stub_run_experiment()
        mock_get_client.return_value = client

        def task(input_value):
            raise RuntimeError(f"boom on {input_value}")

        adapter = LangFusePlatform()
        result = await adapter.aevaluate(
            task=task,
            dataset=[
                ExampleData(input="alpha", expected=ExpectedResult(expected="alpha")),
                ExampleData(input="beta", expected=ExpectedResult(expected="beta")),
            ],
            evaluators=[],
            platform=LangfuseConfig(experiment="issue-171-aevaluate"),
        )

        assert isinstance(result, EvalResult)
        assert result.summary == {
            "total_examples": 2,
            "successful_examples": 0,
            "failed_examples": 2,
        }
        errors_by_input = {ex.input: ex.error for ex in result.examples}
        assert errors_by_input == {
            "alpha": "boom on alpha",
            "beta": "boom on beta",
        }

    @pytest.mark.asyncio
    @patch("agent_evals.adapters.platforms.langfuse.get_client")
    async def test_successful_task_duration_recorded(self, mock_get_client):
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        client = Mock()
        client.auth_check.return_value = True
        client.run_experiment = _stub_run_experiment()
        mock_get_client.return_value = client

        def task(input_value):
            time.sleep(0.02)
            return f"answer-{input_value}"

        adapter = LangFusePlatform()
        result = await adapter.aevaluate(
            task=task,
            dataset=[
                ExampleData(input="alpha", expected=ExpectedResult(expected="alpha")),
            ],
            evaluators=[],
            platform=LangfuseConfig(experiment="duration-aevaluate"),
        )

        assert result.summary["successful_examples"] == 1
        assert result.examples[0].duration > 0

    @pytest.mark.asyncio
    @patch("agent_evals.adapters.platforms.langfuse.get_client")
    async def test_empty_dataset_returns_empty_eval_result(self, mock_get_client):
        from agent_evals.adapters.platforms.langfuse import LangFusePlatform

        client = Mock()
        client.auth_check.return_value = True
        client.run_experiment = _stub_run_experiment()
        mock_get_client.return_value = client

        def task(input_value):
            return "unused"

        adapter = LangFusePlatform()
        result = await adapter.aevaluate(
            task=task,
            dataset=[],
            evaluators=[],
            platform=LangfuseConfig(experiment="empty-dataset"),
        )

        assert result.examples == []
        assert result.summary == {
            "total_examples": 0,
            "successful_examples": 0,
            "failed_examples": 0,
        }
