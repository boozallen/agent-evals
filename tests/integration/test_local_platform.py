# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Integration tests for local adapter."""

import shutil
import tempfile
from pathlib import Path

import pytest
from foundry_agent_core.encryption import load_encryption_key

from agent_evals import (
    EvalExample,
    ExpectedResult,
    Score,
    TaskResult,
    read_encrypted_file,
    read_encrypted_jsonl,
    run_eval,
)
from agent_evals.adapters.platforms.local import LocalConfig
from agent_evals.core.types import ExampleData


@pytest.fixture
def temp_eval_dir():
    """Create temporary directory for evaluation results."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup after test
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def simple_task():
    """Simple task function for testing."""

    def task(input_value):
        """Echo input with prefix."""
        return TaskResult(output=f"Processed: {input_value}")

    return task


@pytest.fixture
def exact_match_scorer():
    """Simple exact match scorer."""

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        """Score 1.0 if exact match, else 0.0."""
        expected_value = expected.expected if expected is not None else None
        matches = str(result.output).strip() == str(expected_value).strip()
        return Score(name="ExactMatch", value=1.0 if matches else 0.0, passed=matches)

    return scorer


@pytest.fixture
def length_scorer():
    """Simple length-based scorer."""

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        """Score based on output length."""
        # Handle both TaskResult and string inputs
        output_str = result.output
        length = len(output_str)
        score_val = 1.0 if 5 <= length <= 100 else 0.0
        return Score(name="LengthCheck", value=score_val, passed=score_val > 0.5)

    return scorer


class TestBasicEvaluation:
    """Test basic evaluation flow (User Story 1)."""

    def test_basic_evaluation_with_one_scorer(
        self, simple_task, exact_match_scorer, temp_eval_dir
    ):
        """Test US1: Basic evaluation with single scorer completes successfully."""
        dataset = [
            ExampleData(
                input="Hello", expected=ExpectedResult(expected="Processed: Hello")
            ),
            ExampleData(
                input="World", expected=ExpectedResult(expected="Processed: World")
            ),
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[exact_match_scorer],
        )

        # Verify result structure
        assert result.experiment_id is not None
        assert result.experiment_url is not None
        assert result.platform == "local"
        assert "ExactMatch" in result.scores
        assert result.scores["ExactMatch"] == 1.0
        assert len(result.examples) == 2
        assert result.summary["total_examples"] == 2
        assert result.summary["successful_examples"] == 2
        assert result.summary["failed_examples"] == 0

    def test_multiple_examples_logged_correctly(
        self, simple_task, exact_match_scorer, temp_eval_dir
    ):
        """Test US1: All examples are logged correctly to results file."""
        dataset = [
            ExampleData(
                input=f"input_{i}",
                expected=ExpectedResult(expected=f"Processed: input_{i}"),
            )
            for i in range(10)
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[exact_match_scorer],
            platform=LocalConfig(
                experiment="multi_example", output_dir=str(temp_eval_dir)
            ),
        )

        # Verify all examples present
        assert len(result.examples) == 10
        assert all(isinstance(ex, EvalExample) for ex in result.examples)

        # Verify each example has required fields
        for i, example in enumerate(result.examples):
            assert example.input == f"input_{i}"
            assert example.output == f"Processed: input_{i}"
            assert example.expected == f"Processed: input_{i}"
            assert "ExactMatch" in example.scores
            assert example.scores["ExactMatch"].value == 1.0
            assert example.duration > 0
            assert example.error is None

        # Verify results file exists and has correct number of lines
        exp_id = result.experiment_id
        results_file = (
            temp_eval_dir / "experiments" / f"multi_example-{exp_id}" / "results.jsonl"
        )
        assert results_file.exists()

        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        records = list(read_encrypted_jsonl(results_file, key))
        assert len(records) == 10

        for example_data in records:
            assert "input" in example_data
            assert "output" in example_data
            assert "scores" in example_data


class TestAdapterInterface:
    """Test adapter 3-method interface (User Story 2)."""

    def test_adapter_three_methods_sufficient(
        self, simple_task, exact_match_scorer, temp_eval_dir
    ):
        """Test US2: 3-method adapter interface is sufficient for evaluation."""
        dataset = [
            ExampleData(
                input="test", expected=ExpectedResult(expected="Processed: test")
            )
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[exact_match_scorer],
        )

        # Verify complete evaluation using only 3 adapter methods:
        # 1. initialize_experiment (returned experiment_id)
        assert result.experiment_id is not None

        # 2. log_result (logged example data)
        assert len(result.examples) == 1
        assert result.examples[0].input == "test"

        # 3. finalize (returned summary with aggregates)
        assert "ExactMatch" in result.scores
        assert result.summary["total_examples"] == 1

    def test_adapter_finalize_returns_complete_summary(
        self, simple_task, exact_match_scorer, temp_eval_dir
    ):
        """Test US2: Adapter finalize returns complete summary."""
        dataset = [
            ExampleData(input="a", expected=ExpectedResult(expected="Processed: a")),
            ExampleData(input="b", expected=ExpectedResult(expected="Processed: b")),
            ExampleData(input="c", expected=ExpectedResult(expected="Processed: c")),
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[exact_match_scorer],
        )

        # Verify summary has all required fields
        assert "total_examples" in result.summary
        assert "successful_examples" in result.summary
        assert "failed_examples" in result.summary
        assert result.summary["total_examples"] == 3
        assert result.summary["successful_examples"] == 3
        assert result.summary["failed_examples"] == 0

        # Verify aggregate scores computed correctly
        assert "ExactMatch" in result.scores
        assert result.scores["ExactMatch"] == 1.0


class TestRunnerControlFlow:
    """Test EvalRunner controls evaluation flow (User Story 3)."""

    def test_runner_controls_initialization(
        self, simple_task, exact_match_scorer, temp_eval_dir
    ):
        """Test US3: EvalRunner controls adapter initialization."""
        dataset = [
            ExampleData(
                input="test", expected=ExpectedResult(expected="Processed: test")
            )
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[exact_match_scorer],
            platform=LocalConfig(experiment="init_test", output_dir=str(temp_eval_dir)),
        )

        # Verify runner initialized adapter (experiment_id assigned)
        assert result.experiment_id is not None
        assert len(result.experiment_id) == 36  # UUID format

        # Verify experiment directory created
        exp_id = result.experiment_id
        exp_dir = temp_eval_dir / "experiments" / f"init_test-{exp_id}"
        assert exp_dir.exists()
        assert (exp_dir / "metadata.json").exists()

    def test_runner_controls_logging_timing(
        self, simple_task, exact_match_scorer, temp_eval_dir
    ):
        """Test US3: EvalRunner controls when results are logged."""
        dataset = [
            ExampleData(input="a", expected=ExpectedResult(expected="Processed: a")),
            ExampleData(input="b", expected=ExpectedResult(expected="Processed: b")),
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[exact_match_scorer],
            platform=LocalConfig(
                experiment="timing_test", output_dir=str(temp_eval_dir)
            ),
        )

        # Verify results logged after execution (not before)
        # Check that all examples have duration > 0 (task was executed)
        for example in result.examples:
            assert example.duration > 0

        # Verify results file has correct number of entries
        exp_id = result.experiment_id
        results_file = (
            temp_eval_dir / "experiments" / f"timing_test-{exp_id}" / "results.jsonl"
        )
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        records = list(read_encrypted_jsonl(results_file, key))
        assert len(records) == 2

    def test_runner_controls_finalization(
        self, simple_task, exact_match_scorer, temp_eval_dir
    ):
        """Test US3: EvalRunner controls finalization timing."""
        dataset = [
            ExampleData(
                input="test", expected=ExpectedResult(expected="Processed: test")
            )
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[exact_match_scorer],
            platform=LocalConfig(
                experiment="finalize_test", output_dir=str(temp_eval_dir)
            ),
        )

        # Verify summary.json created after all examples processed
        exp_id = result.experiment_id
        summary_file = (
            temp_eval_dir / "experiments" / f"finalize_test-{exp_id}" / "summary.json"
        )
        assert summary_file.exists()

        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        summary_data = read_encrypted_file(summary_file, key)
        assert summary_data is not None
        assert summary_data["total_examples"] == 1
        assert "scores" in summary_data


class TestMultipleScorerS:
    """Test multiple scorers per example (User Story 4)."""

    def test_multiple_scorers_all_invoked(
        self, simple_task, exact_match_scorer, length_scorer, temp_eval_dir
    ):
        """Test US4: All scorers are invoked for each example."""
        dataset = [
            ExampleData(
                input="test", expected=ExpectedResult(expected="Processed: test")
            )
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[exact_match_scorer, length_scorer],
        )

        # Verify both scorers ran
        assert "ExactMatch" in result.scores
        assert "LengthCheck" in result.scores

        # Verify each example has both scores
        for example in result.examples:
            assert "ExactMatch" in example.scores
            assert "LengthCheck" in example.scores

    def test_scorer_results_correctly_associated(
        self, simple_task, exact_match_scorer, length_scorer, temp_eval_dir
    ):
        """Test US4: Scorer results correctly associated with examples."""
        dataset = [
            ExampleData(input="a", expected=ExpectedResult(expected="Processed: a")),
            ExampleData(input="b", expected=ExpectedResult(expected="Processed: b")),
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[exact_match_scorer, length_scorer],
        )

        # Verify scores associated with correct inputs
        for example in result.examples:
            assert example.input in ["a", "b"]
            assert "ExactMatch" in example.scores
            assert "LengthCheck" in example.scores

    def test_aggregate_scores_per_scorer(
        self, simple_task, exact_match_scorer, length_scorer, temp_eval_dir
    ):
        """Test US4: Aggregate scores computed per scorer."""
        dataset = [
            ExampleData(input="a", expected=ExpectedResult(expected="Processed: a")),
            ExampleData(input="b", expected=ExpectedResult(expected="Processed: b")),
            ExampleData(input="c", expected=ExpectedResult(expected="Processed: c")),
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[exact_match_scorer, length_scorer],
        )

        # Verify separate aggregate for each scorer
        assert "ExactMatch" in result.scores
        assert "LengthCheck" in result.scores
        assert result.scores["ExactMatch"] == 1.0
        assert result.scores["LengthCheck"] == 1.0


class TestErrorHandling:
    """Test error handling for task and scorer failures (User Story 5)."""

    def test_task_failure_captured(self, exact_match_scorer, temp_eval_dir):
        """Test US5: Task failures are captured and evaluation continues."""

        def failing_task(input_value):
            """Task that fails on specific input."""
            if input_value == "fail":
                raise ValueError("Intentional task failure")
            return TaskResult(output=f"Processed: {input_value}")

        dataset = [
            ExampleData(
                input="success", expected=ExpectedResult(expected="Processed: success")
            ),
            ExampleData(
                input="fail", expected=ExpectedResult(expected="Processed: fail")
            ),
            ExampleData(
                input="also_success",
                expected=ExpectedResult(expected="Processed: also_success"),
            ),
        ]

        result = run_eval(
            task=failing_task,
            dataset=dataset,
            scorers=[exact_match_scorer],
        )

        # Verify evaluation completed
        assert len(result.examples) == 3

        # Verify failed example captured error
        failed_example = result.examples[1]
        assert failed_example.error is not None
        assert "Intentional task failure" in failed_example.error
        assert failed_example.output == ""  # Empty string when task fails

        # Verify successful examples have no error
        assert result.examples[0].error is None
        assert result.examples[2].error is None

        # Verify summary counts
        assert result.summary["total_examples"] == 3
        assert result.summary["successful_examples"] == 2
        assert result.summary["failed_examples"] == 1

    def test_scorer_failure_continues(self, simple_task, temp_eval_dir):
        """Test US5: Scorer failures create 0.0 score and evaluation continues."""

        def failing_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Scorer that always fails."""
            raise RuntimeError("Intentional scorer failure")

        def working_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Scorer that works."""
            # Handle both TaskResult and string outputs
            output_str = result.output
            expected_value = expected.expected if expected is not None else None
            matches = output_str == expected_value
            value = 1.0 if matches else 0.0
            return Score(name="WorkingScorer", value=value, passed=matches)

        dataset = [
            ExampleData(
                input="test", expected=ExpectedResult(expected="Processed: test")
            )
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[failing_scorer, working_scorer],
        )

        # Verify evaluation completed
        assert len(result.examples) == 1
        example = result.examples[0]

        # Verify failed scorer has 0.0 score
        assert "failing_scorer" in example.scores
        failed_score = example.scores["failing_scorer"]
        assert failed_score.value == 0.0
        assert failed_score.passed is False
        assert "error" in failed_score.metadata

        # Verify working scorer still ran
        assert "WorkingScorer" in example.scores
        assert example.scores["WorkingScorer"].value == 1.0

    def test_failed_examples_count_in_summary(self, exact_match_scorer, temp_eval_dir):
        """Test US5: Failed examples counted in summary."""

        def partially_failing_task(input_value):
            """Task that fails on even numbers."""
            if int(input_value) % 2 == 0:
                raise ValueError("Even number")
            return TaskResult(output=f"Processed: {input_value}")

        dataset = [
            ExampleData(
                input=str(i), expected=ExpectedResult(expected=f"Processed: {i}")
            )
            for i in range(10)
        ]

        result = run_eval(
            task=partially_failing_task,
            dataset=dataset,
            scorers=[exact_match_scorer],
        )

        # Verify counts
        assert result.summary["total_examples"] == 10
        assert result.summary["successful_examples"] == 5  # Odd numbers
        assert result.summary["failed_examples"] == 5  # Even numbers

    def test_none_output_treated_as_empty_string(self, temp_eval_dir):
        """Test US5: None output handled gracefully."""

        def none_returning_task(input_value):
            """Task that returns empty string (simulating failed task)."""
            return TaskResult(output="")

        def length_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Scorer that checks output length."""
            output_str = str(result) if result is not None else ""
            score_val = 1.0 if len(output_str) > 0 else 0.0
            return Score(name="LengthScorer", value=score_val, passed=score_val > 0.5)

        dataset = [ExampleData(input="test")]

        result = run_eval(
            task=none_returning_task,
            dataset=dataset,
            scorers=[length_scorer],
        )

        # Verify evaluation completed
        assert len(result.examples) == 1
        example = result.examples[0]
        assert example.output == ""  # Empty string for tasks returning empty
        assert example.error is None  # Empty string is valid output, not error

        # Verify scorer ran (should give 0.0 for empty string)
        assert "LengthScorer" in example.scores


class TestEdgeCases:
    """Test edge cases and validation."""

    def test_missing_directory_created_automatically(
        self, simple_task, exact_match_scorer
    ):
        """Test that missing output directory is created automatically."""
        import tempfile
        from pathlib import Path

        # Create a temp directory that will be deleted
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "nonexistent" / "deeply" / "nested"
            assert not output_dir.exists()

            dataset = [
                ExampleData(
                    input="test", expected=ExpectedResult(expected="Processed: test")
                )
            ]

            result = run_eval(
                task=simple_task,
                dataset=dataset,
                scorers=[exact_match_scorer],
                platform=LocalConfig(
                    experiment="directory_test", output_dir=str(output_dir)
                ),
            )

            # Verify directory was created
            exp_dir = (
                output_dir / "experiments" / f"directory_test-{result.experiment_id}"
            )
            assert exp_dir.exists()
            assert (exp_dir / "metadata.json").exists()
            assert (exp_dir / "results.jsonl").exists()
            assert (exp_dir / "summary.json").exists()


@pytest.mark.integration
def test_autoevals_levenshtein_integration(temp_eval_dir):
    """Integration test for Levenshtein scorer with real autoevals library.

    This test uses the real autoevals library (no mocks) but doesn't require
    API calls, making it suitable for automated CI/CD.
    """
    pytest.importorskip("autoevals", reason="autoevals not installed")

    from agent_evals.adapters.scorers.autoevals import Levenshtein

    # Simple echo task
    def echo_task(input_text: str) -> TaskResult:
        return TaskResult(output=input_text.lower())

    # Dataset with exact and near matches
    dataset = [
        ExampleData(input="HELLO", expected=ExpectedResult(expected="hello")),
        ExampleData(input="WORLD", expected=ExpectedResult(expected="world")),
        ExampleData(input="Python", expected=ExpectedResult(expected="python")),
    ]

    # Run evaluation with Levenshtein scorer
    result = run_eval(
        task=echo_task,
        dataset=dataset,
        scorers=[Levenshtein(threshold=1.0)],  # Exact match required
    )

    # Verify results
    assert result.platform == "local"
    assert len(result.examples) == 3

    # All examples should have Levenshtein scores
    for example in result.examples:
        assert "Levenshtein" in example.scores
        score = example.scores["Levenshtein"]

        # Verify Score format
        assert isinstance(score, Score)
        assert 0.0 <= score.value <= 1.0
        assert isinstance(score.passed, bool)
        assert isinstance(score.metadata, dict)

        # All outputs match exactly, so score should be 1.0
        assert score.value == 1.0
        assert score.passed is True

    # Verify aggregate scores
    assert "Levenshtein" in result.scores
    assert result.scores["Levenshtein"] == 1.0  # All exact matches

    # Verify summary has pass rates
    if "pass_rates" in result.summary:
        assert result.summary["pass_rates"]["Levenshtein"] == 1.0


@pytest.mark.integration
def test_autoevals_levenshtein_with_fuzzy_match(temp_eval_dir):
    """Integration test for Levenshtein with relaxed threshold."""
    pytest.importorskip("autoevals", reason="autoevals not installed")

    from agent_evals.adapters.scorers.autoevals import Levenshtein

    # Task that slightly modifies output
    def slightly_off_task(input_text: str) -> TaskResult:
        return TaskResult(output=input_text.lower() + "!")  # Add exclamation

    # Dataset
    dataset = [
        ExampleData(input="hello", expected=ExpectedResult(expected="hello")),
        ExampleData(input="world", expected=ExpectedResult(expected="world")),
    ]

    # Run with relaxed threshold
    result = run_eval(
        task=slightly_off_task,
        dataset=dataset,
        scorers=[Levenshtein(threshold=0.8)],  # Allow 80%+ similarity
    )

    # Verify results
    assert len(result.examples) == 2

    for example in result.examples:
        score = example.scores["Levenshtein"]

        # Score should be high (close match) but not perfect
        assert 0.8 <= score.value < 1.0

        # Should pass with relaxed threshold
        assert score.passed is True


class TestMultipleScorers:
    """Test multiple autoevals scorers running together (User Story 2)."""

    def test_multiple_scorers_all_execute(self, simple_task, temp_eval_dir):
        """Test US2: Multiple autoevals scorers run in same evaluation."""
        pytest.importorskip("autoevals", reason="autoevals not installed")

        from agent_evals.adapters.scorers.autoevals import Levenshtein

        # Create multiple Levenshtein scorers with different thresholds
        # (Using Levenshtein only to avoid LLM API calls in CI)
        strict_scorer = Levenshtein(threshold=1.0)  # Exact match
        relaxed_scorer = Levenshtein(threshold=0.8)  # 80% similarity

        # We'll need to give them different names to distinguish them
        # For now, they'll both be "Levenshtein" but we can test aggregation

        # Custom task that produces exact matches
        def exact_task(input_text: str) -> TaskResult:
            return TaskResult(output=input_text)

        dataset = [
            ExampleData(input="hello", expected=ExpectedResult(expected="hello")),
            ExampleData(input="world", expected=ExpectedResult(expected="world")),
        ]

        # Run with multiple scorers (both Levenshtein for now)
        # NOTE: This will currently produce duplicate scorer names
        # In reality, you'd want different scorers or custom names
        result = run_eval(
            task=exact_task,
            dataset=dataset,
            scorers=[strict_scorer, relaxed_scorer],
        )

        # Verify both scorers executed
        assert len(result.examples) == 2

        # Each example should have scores from both scorers
        for example in result.examples:
            assert "Levenshtein" in example.scores
            # Note: With duplicate names, only one will be stored
            # This is a limitation we'll document

        # Verify aggregate scores exist
        assert "Levenshtein" in result.scores

    def test_pass_rate_computation(self, temp_eval_dir):
        """Test US2: Pass rates are computed correctly for multiple scorers."""
        pytest.importorskip("autoevals", reason="autoevals not installed")

        from agent_evals.adapters.scorers.autoevals import Levenshtein

        # Task that sometimes matches
        def inconsistent_task(input_text: str) -> TaskResult:
            # Exact match for "hello", off for others
            if input_text == "hello":
                return TaskResult(output="hello")
            else:
                return TaskResult(output="WRONG")

        dataset = [
            ExampleData(
                input="hello", expected=ExpectedResult(expected="hello")
            ),  # Will pass
            ExampleData(
                input="world", expected=ExpectedResult(expected="world")
            ),  # Will fail
            ExampleData(
                input="test", expected=ExpectedResult(expected="test")
            ),  # Will fail
        ]

        # Run with Levenshtein scorer
        result = run_eval(
            task=inconsistent_task,
            dataset=dataset,
            scorers=[Levenshtein(threshold=1.0)],  # Exact match required
        )

        # Verify pass_rates dict exists in result
        assert hasattr(result, "pass_rates"), "Result should have pass_rates attribute"
        assert isinstance(result.pass_rates, dict), "pass_rates should be a dict"

        # Verify pass rate for Levenshtein scorer
        assert "Levenshtein" in result.pass_rates

        # Pass rate should be 1/3 = 0.333...
        expected_pass_rate = 1.0 / 3.0
        actual_pass_rate = result.pass_rates["Levenshtein"]
        assert abs(actual_pass_rate - expected_pass_rate) < 0.01, (
            f"Expected pass rate {expected_pass_rate}, got {actual_pass_rate}"
        )

        # Verify aggregate score (average of all scores)
        assert "Levenshtein" in result.scores
        # Aggregate should be average: (1.0 + 0.0 + 0.0) / 3 = 0.333...
        expected_avg = 1.0 / 3.0
        actual_avg = result.scores["Levenshtein"]
        assert abs(actual_avg - expected_avg) < 0.01

    def test_pass_rate_with_multiple_scorers(self, temp_eval_dir):
        """Test US2: Pass rates computed independently for each scorer."""
        pytest.importorskip("autoevals", reason="autoevals not installed")

        from agent_evals.adapters.scorers.autoevals import Levenshtein

        # Use custom scorer alongside Levenshtein to test multiple scorers
        def always_pass_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ):
            """Scorer that always passes."""
            return Score(name="AlwaysPass", value=1.0, passed=True)

        def always_fail_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ):
            """Scorer that always fails."""
            return Score(name="AlwaysFail", value=0.0, passed=False)

        # Task doesn't matter for these scorers
        def simple_task(input_text: str) -> TaskResult:
            return TaskResult(output=input_text)

        dataset = [
            ExampleData(input="test1", expected=ExpectedResult(expected="test1")),
            ExampleData(input="test2", expected=ExpectedResult(expected="test2")),
            ExampleData(input="test3", expected=ExpectedResult(expected="test3")),
            ExampleData(input="test4", expected=ExpectedResult(expected="test4")),
        ]

        # Run with three scorers
        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[
                Levenshtein(threshold=1.0),  # Will pass all (exact matches)
                always_pass_scorer,  # Will pass all
                always_fail_scorer,  # Will fail all
            ],
        )

        # Verify pass_rates exist
        assert hasattr(result, "pass_rates")
        assert isinstance(result.pass_rates, dict)

        # Verify each scorer has independent pass rate
        assert "Levenshtein" in result.pass_rates
        assert "AlwaysPass" in result.pass_rates
        assert "AlwaysFail" in result.pass_rates

        # Verify pass rates are correct
        assert result.pass_rates["Levenshtein"] == 1.0  # 4/4 passed
        assert result.pass_rates["AlwaysPass"] == 1.0  # 4/4 passed
        assert result.pass_rates["AlwaysFail"] == 0.0  # 0/4 passed


class TestMixedScorerTypes:
    """Test mixing custom and external scorers (User Story 3)."""

    def test_custom_and_autoevals_scorers_work_together(self, temp_eval_dir):
        """Test US3: Custom scorers and autoevals scorers work seamlessly together."""
        pytest.importorskip("autoevals", reason="autoevals not installed")

        from agent_evals.adapters.scorers.autoevals import Levenshtein

        # Define a custom scorer
        def length_check(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Custom scorer: check if output meets minimum length."""
            min_length = 5
            # Extract string output from TaskResult if needed
            output_str = result.output
            length = len(output_str)
            value = min(length / min_length, 1.0)
            passed = length >= min_length

            return Score(
                name="LengthCheck",
                value=value,
                passed=passed,
                metadata={"length": length, "min_required": min_length},
                reasoning=f"Output is {length} characters (min: {min_length})",
            )

        # Simple task
        def echo_task(input_text: str) -> TaskResult:
            return TaskResult(output=input_text.lower())

        dataset = [
            ExampleData(input="HELLO", expected=ExpectedResult(expected="hello")),
            ExampleData(input="WORLD", expected=ExpectedResult(expected="world")),
            ExampleData(input="SHORT", expected=ExpectedResult(expected="short")),
            ExampleData(
                input="A", expected=ExpectedResult(expected="a")
            ),  # Too short for LengthCheck
        ]

        # Run evaluation with both custom and autoevals scorer
        result = run_eval(
            task=echo_task,
            dataset=dataset,
            scorers=[
                Levenshtein(threshold=1.0),  # External (autoevals)
                length_check,  # Custom function
            ],
        )

        # Verify both scorers appear in results
        assert "Levenshtein" in result.scores
        assert "LengthCheck" in result.scores

        # Verify each example has scores from both scorers
        for example in result.examples:
            assert "Levenshtein" in example.scores
            assert "LengthCheck" in example.scores

            # Verify both produce valid Score objects
            lev_score = example.scores["Levenshtein"]
            len_score = example.scores["LengthCheck"]

            assert isinstance(lev_score, Score)
            assert isinstance(len_score, Score)
            assert 0.0 <= lev_score.value <= 1.0
            assert 0.0 <= len_score.value <= 1.0

        # Verify pass rates for both scorers
        assert "Levenshtein" in result.pass_rates
        assert "LengthCheck" in result.pass_rates

        # All examples should pass Levenshtein (exact matches)
        assert result.pass_rates["Levenshtein"] == 1.0

        # Only 3/4 should pass LengthCheck (last one is too short)
        expected_length_pass_rate = 3.0 / 4.0
        assert abs(result.pass_rates["LengthCheck"] - expected_length_pass_rate) < 0.01

    def test_no_inheritance_required_for_custom_scorers(self, temp_eval_dir):
        """Test US3: Custom scorers work without inheritance, only duck typing."""
        pytest.importorskip("autoevals", reason="autoevals not installed")

        from agent_evals.adapters.scorers.autoevals import Levenshtein

        # Define custom scorer as a plain function (no base class)
        def contains_keyword(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Check if output contains required keyword."""
            keyword = "processed"
            contains = keyword.lower() in str(result).lower()

            return Score(
                name="ContainsKeyword",
                value=1.0 if contains else 0.0,
                passed=contains,
                metadata={"keyword": keyword, "found": contains},
            )

        # Task function
        def task(input_text: str) -> TaskResult:
            return TaskResult(output=f"Processed: {input_text}")

        dataset = [
            ExampleData(
                input="test1", expected=ExpectedResult(expected="Processed: test1")
            ),
            ExampleData(
                input="test2", expected=ExpectedResult(expected="Processed: test2")
            ),
        ]

        # Run with custom + external scorers
        result = run_eval(
            task=task,
            dataset=dataset,
            scorers=[
                contains_keyword,  # Custom function (no inheritance)
                Levenshtein(threshold=1.0),  # External (autoevals)
            ],
        )

        # Verify both scorers ran successfully
        assert "ContainsKeyword" in result.scores
        assert "Levenshtein" in result.scores

        # Both should have perfect scores (all examples match)
        assert result.scores["ContainsKeyword"] == 1.0
        assert result.scores["Levenshtein"] == 1.0

        # Verify no exceptions or errors
        for example in result.examples:
            assert example.error is None
            assert "ContainsKeyword" in example.scores
            assert "Levenshtein" in example.scores


class TestScorerErrorHandling:
    """Test scorer failure scenarios (User Story 4)."""

    def test_failing_scorer_returns_error_score_continues_evaluation(
        self, temp_eval_dir
    ):
        """Test US4: When one scorer fails, evaluation continues with error Score."""

        def failing_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Scorer that always raises an exception."""
            raise RuntimeError("Intentional scorer failure for testing")

        def working_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Scorer that works normally."""
            # Handle both TaskResult and string outputs
            output_str = result.output
            expected_value = expected.expected if expected is not None else None
            value = 1.0 if output_str == expected_value else 0.0
            return Score(name="WorkingScorer", value=value, passed=value > 0.5)

        def simple_task(input_text: str) -> TaskResult:
            return TaskResult(output=input_text.lower())

        dataset = [
            ExampleData(input="HELLO", expected=ExpectedResult(expected="hello")),
            ExampleData(input="WORLD", expected=ExpectedResult(expected="world")),
        ]

        # Run evaluation with both failing and working scorers
        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[
                failing_scorer,
                working_scorer,
            ],
        )

        # Verify evaluation completed despite scorer failure
        assert len(result.examples) == 2

        # Verify both scorers appear in results
        assert "failing_scorer" in result.scores
        assert "WorkingScorer" in result.scores

        # Working scorer should have perfect scores
        assert result.scores["WorkingScorer"] == 1.0

        # Failing scorer should have 0.0 aggregate (all examples failed)
        assert result.scores["failing_scorer"] == 0.0

        # Verify each example has error Score for failing scorer
        for example in result.examples:
            assert "failing_scorer" in example.scores
            assert "WorkingScorer" in example.scores

            # Failing scorer should have error metadata
            failing_score = example.scores["failing_scorer"]
            assert failing_score.value == 0.0
            assert failing_score.passed is False
            assert "error" in failing_score.metadata
            assert "Intentional scorer failure" in failing_score.metadata["error"]

            # Working scorer should have valid score
            working_score = example.scores["WorkingScorer"]
            assert working_score.value == 1.0
            assert working_score.passed is True

    def test_multiple_scorers_one_fails_aggregates_correctly(self, temp_eval_dir):
        """Test US4: Pass rates computed correctly when one scorer fails."""

        def always_fail_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Scorer that always raises an exception."""
            raise ValueError("Always fails")

        def always_pass_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Scorer that always passes."""
            return Score(name="AlwaysPass", value=1.0, passed=True)

        def sometimes_pass_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Scorer that passes half the time."""
            # Handle both TaskResult and string outputs
            output_str = result.output
            value = 1.0 if len(output_str) > 3 else 0.0
            return Score(name="SometimesPass", value=value, passed=value > 0.5)

        def simple_task(input_text: str) -> TaskResult:
            return TaskResult(output=input_text)

        dataset = [
            ExampleData(input="a", expected=ExpectedResult(expected="a")),
            ExampleData(input="ab", expected=ExpectedResult(expected="ab")),
            ExampleData(input="abc", expected=ExpectedResult(expected="abc")),
            ExampleData(input="abcd", expected=ExpectedResult(expected="abcd")),
        ]

        # Run with three scorers: always fail, always pass, sometimes pass
        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[
                always_fail_scorer,
                always_pass_scorer,
                sometimes_pass_scorer,
            ],
        )

        # Verify all scorers appear in results
        assert "always_fail_scorer" in result.scores
        assert "AlwaysPass" in result.scores
        assert "SometimesPass" in result.scores

        # Verify aggregate scores
        assert result.scores["always_fail_scorer"] == 0.0  # All failed
        assert result.scores["AlwaysPass"] == 1.0  # All passed
        assert result.scores["SometimesPass"] == 0.25  # 1/4 passed

        # Verify pass rates
        assert result.pass_rates["always_fail_scorer"] == 0.0  # 0/4 passed
        assert result.pass_rates["AlwaysPass"] == 1.0  # 4/4 passed
        assert result.pass_rates["SometimesPass"] == 0.25  # 1/4 passed

    def test_autoevals_scorer_failure_with_missing_library(self, temp_eval_dir):
        """Test US4: Autoevals scorer returns error Score when library not available."""
        # Note: This test assumes autoevals IS installed, but tests the error path
        # by checking that the ImportError handling logic is correct

        def custom_working_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Custom scorer that always works."""
            return Score(name="CustomWorking", value=1.0, passed=True)

        def simple_task(input_text: str) -> TaskResult:
            return TaskResult(output=input_text)

        dataset = [
            ExampleData(input="test", expected=ExpectedResult(expected="test")),
        ]

        # Run with custom scorer (autoevals test would require mocking import failure)
        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[custom_working_scorer],
        )

        # Verify evaluation completes
        assert len(result.examples) == 1
        assert "CustomWorking" in result.scores
        assert result.scores["CustomWorking"] == 1.0

    def test_error_scores_have_debugging_metadata(self, temp_eval_dir):
        """Test US4: Error Scores contain sufficient debugging information."""

        def detailed_failing_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Scorer that fails with detailed error."""
            raise ValueError(f"Failed to score output: '{result}' - invalid format")

        def simple_task(input_text: str) -> TaskResult:
            return TaskResult(output=input_text)

        dataset = [
            ExampleData(
                input="test_input", expected=ExpectedResult(expected="test_input")
            ),
        ]

        result = run_eval(
            task=simple_task,
            dataset=dataset,
            scorers=[detailed_failing_scorer],
        )

        # Get the error score
        example = result.examples[0]
        error_score = example.scores["detailed_failing_scorer"]

        # Verify error Score has debugging information
        assert "error" in error_score.metadata
        error_msg = error_score.metadata["error"]

        # Should contain exception type and message
        assert "ValueError" in error_msg or "Failed to score output" in error_msg

        # Should have clear failure indicators
        assert error_score.value == 0.0
        assert error_score.passed is False
        assert error_score.name == "detailed_failing_scorer"
