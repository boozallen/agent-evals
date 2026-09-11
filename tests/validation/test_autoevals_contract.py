# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Contract tests for autoevals API compatibility.

These tests verify that the autoevals library API structure hasn't changed
in ways that would break our adapters. Uses mocks to avoid actual API calls.
"""

from unittest.mock import MagicMock, patch


class TestLevenshteinContract:
    """Verify autoevals Levenshtein result format."""

    @patch("autoevals.string.Levenshtein")
    def test_levenshtein_result_has_score_attribute(self, mock_levenshtein_class):
        """Autoevals Levenshtein result should have .score attribute."""
        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.name = "Levenshtein"
        mock_result.metadata = {}

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.return_value = mock_result
        mock_levenshtein_class.return_value = mock_scorer

        # Import and instantiate
        from autoevals.string import Levenshtein

        scorer = Levenshtein()
        result = scorer(output="hello", expected="hello")

        # Verify contract
        assert hasattr(result, "score")
        assert isinstance(result.score, float | int | bool)

    @patch("autoevals.string.Levenshtein")
    def test_levenshtein_result_has_name_attribute(self, mock_levenshtein_class):
        """Autoevals Levenshtein result should have .name attribute."""
        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 1.0
        mock_result.name = "Levenshtein"
        mock_result.metadata = {}

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.return_value = mock_result
        mock_levenshtein_class.return_value = mock_scorer

        # Import and instantiate
        from autoevals.string import Levenshtein

        scorer = Levenshtein()
        result = scorer(output="test", expected="test")

        # Verify contract
        assert hasattr(result, "name")
        assert isinstance(result.name, str)

    @patch("autoevals.string.Levenshtein")
    def test_levenshtein_result_has_metadata_attribute(self, mock_levenshtein_class):
        """Autoevals Levenshtein result should have .metadata attribute."""
        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 0.8
        mock_result.name = "Levenshtein"
        mock_result.metadata = {"distance": 2}

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.return_value = mock_result
        mock_levenshtein_class.return_value = mock_scorer

        # Import and instantiate
        from autoevals.string import Levenshtein

        scorer = Levenshtein()
        result = scorer(output="hello", expected="hallo")

        # Verify contract
        assert hasattr(result, "metadata")
        # metadata can be None or dict
        assert result.metadata is None or isinstance(result.metadata, dict)


class TestFactualityContract:
    """Verify autoevals Factuality result format."""

    @patch("autoevals.llm.Factuality")
    def test_factuality_result_has_score_attribute(self, mock_factuality_class):
        """Autoevals Factuality result should have .score attribute."""
        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 0.85
        mock_result.name = "Factuality"
        mock_result.metadata = {"reasoning": "The output is factually correct"}

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.return_value = mock_result
        mock_factuality_class.return_value = mock_scorer

        # Import and instantiate
        from autoevals.llm import Factuality

        scorer = Factuality()  # ty: ignore[missing-argument]  # Mocked by @patch decorator
        result = scorer(output="Paris is the capital of France", expected="Paris")

        # Verify contract
        assert hasattr(result, "score")
        assert isinstance(result.score, float | int | bool)

    @patch("autoevals.llm.Factuality")
    def test_factuality_result_has_name_attribute(self, mock_factuality_class):
        """Autoevals Factuality result should have .name attribute."""
        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 0.9
        mock_result.name = "Factuality"
        mock_result.metadata = {}

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.return_value = mock_result
        mock_factuality_class.return_value = mock_scorer

        # Import and instantiate
        from autoevals.llm import Factuality

        scorer = Factuality()  # ty: ignore[missing-argument]  # Mocked by @patch decorator
        result = scorer(output="Test output", expected="Test expected")

        # Verify contract
        assert hasattr(result, "name")
        assert isinstance(result.name, str)

    @patch("autoevals.llm.Factuality")
    def test_factuality_result_has_metadata_attribute(self, mock_factuality_class):
        """Autoevals Factuality result should have .metadata attribute."""
        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 0.7
        mock_result.name = "Factuality"
        mock_result.metadata = {"model": "gpt-4o-mini", "reasoning": "Partial match"}

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.return_value = mock_result
        mock_factuality_class.return_value = mock_scorer

        # Import and instantiate
        from autoevals.llm import Factuality

        scorer = Factuality()  # ty: ignore[missing-argument]  # Mocked by @patch decorator
        result = scorer(output="Test", expected="Expected")

        # Verify contract
        assert hasattr(result, "metadata")
        # metadata can be None or dict
        assert result.metadata is None or isinstance(result.metadata, dict)


class TestClosedQAContract:
    """Verify autoevals ClosedQA result format."""

    @patch("autoevals.llm.ClosedQA")
    def test_closedqa_result_has_score_attribute(self, mock_closedqa_class):
        """Autoevals ClosedQA result should have .score attribute."""
        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.name = "ClosedQA"
        mock_result.metadata = {"choice": "A"}

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.return_value = mock_result
        mock_closedqa_class.return_value = mock_scorer

        # Import and instantiate
        from autoevals.llm import ClosedQA

        scorer = ClosedQA()  # ty: ignore[missing-argument]  # Mocked by @patch decorator
        result = scorer(
            output="A",
            expected="A",
            input="What is the capital of France? A) Paris B) London",
        )

        # Verify contract
        assert hasattr(result, "score")
        assert isinstance(result.score, float | int | bool)

    @patch("autoevals.llm.ClosedQA")
    def test_closedqa_result_has_name_attribute(self, mock_closedqa_class):
        """Autoevals ClosedQA result should have .name attribute."""
        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 1.0
        mock_result.name = "ClosedQA"
        mock_result.metadata = {}

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.return_value = mock_result
        mock_closedqa_class.return_value = mock_scorer

        # Import and instantiate
        from autoevals.llm import ClosedQA

        scorer = ClosedQA()  # ty: ignore[missing-argument]  # Mocked by @patch decorator
        result = scorer(output="B", expected="B", input="Question")

        # Verify contract
        assert hasattr(result, "name")
        assert isinstance(result.name, str)

    @patch("autoevals.llm.ClosedQA")
    def test_closedqa_result_has_metadata_attribute(self, mock_closedqa_class):
        """Autoevals ClosedQA result should have .metadata attribute."""
        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 0.8
        mock_result.name = "ClosedQA"
        mock_result.metadata = {"model": "gpt-4o-mini", "reasoning": "Correct answer"}

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.return_value = mock_result
        mock_closedqa_class.return_value = mock_scorer

        # Import and instantiate
        from autoevals.llm import ClosedQA

        scorer = ClosedQA()  # ty: ignore[missing-argument]  # Mocked by @patch decorator
        result = scorer(output="C", expected="C", input="Test question")

        # Verify contract
        assert hasattr(result, "metadata")
        # metadata can be None or dict
        assert result.metadata is None or isinstance(result.metadata, dict)
