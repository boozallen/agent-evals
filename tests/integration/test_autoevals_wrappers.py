# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for autoevals scorer wrappers.

Tests the adapter functions that normalize autoevals scorer results to Score format.
All tests use mocks - no actual autoevals library calls or LLM API calls.
"""

from unittest.mock import patch

import pytest

from agent_evals.adapters.scorers.autoevals import _normalize_value
from agent_evals.core.types import ExampleData, ExpectedResult, TaskResult


class TestNormalization:
    """Test score normalization logic (_normalize_value helper)."""

    def test_valid_range_passthrough(self):
        """Values in [0.0, 1.0] should pass through unchanged."""
        # Test boundary values
        assert _normalize_value(0.0) == 0.0
        assert _normalize_value(1.0) == 1.0

        # Test mid-range values
        assert _normalize_value(0.5) == 0.5
        assert _normalize_value(0.75) == 0.75
        assert _normalize_value(0.123456) == 0.123456

    def test_out_of_range_clamping_negative(self):
        """Negative values should be clamped to 0.0."""
        assert _normalize_value(-0.1) == 0.0
        assert _normalize_value(-1.0) == 0.0
        assert _normalize_value(-100.0) == 0.0

    def test_out_of_range_clamping_above_one(self):
        """Values > 1.0 should be clamped to 1.0."""
        assert _normalize_value(1.1) == 1.0
        assert _normalize_value(2.0) == 1.0
        assert _normalize_value(100.0) == 1.0

    def test_boolean_conversion_true(self):
        """True should convert to 1.0."""
        assert _normalize_value(True) == 1.0

    def test_boolean_conversion_false(self):
        """False should convert to 0.0."""
        assert _normalize_value(False) == 0.0

    def test_none_handling(self):
        """None values should raise TypeError (caught by wrapper)."""
        with pytest.raises(TypeError):
            _normalize_value(None)

    def test_int_conversion(self):
        """Integer values should be converted to float."""
        assert _normalize_value(0) == 0.0
        assert _normalize_value(1) == 1.0
        # Out of range integers should also be clamped
        assert _normalize_value(-5) == 0.0
        assert _normalize_value(5) == 1.0

    def test_warning_logged_for_negative_clamping(self):
        """Should log warning when clamping negative values."""
        with patch(
            "agent_evals.adapters.scorers.autoevals.logger.warning"
        ) as mock_warning:
            _normalize_value(-0.5)
            mock_warning.assert_called_once()
            # Verify warning message contains the value
            call_args = mock_warning.call_args[0]
            assert "-0.5" in str(call_args) or -0.5 in call_args

    def test_warning_logged_for_above_one_clamping(self):
        """Should log warning when clamping values > 1.0."""
        with patch(
            "agent_evals.adapters.scorers.autoevals.logger.warning"
        ) as mock_warning:
            _normalize_value(1.5)
            mock_warning.assert_called_once()
            # Verify warning message contains the value
            call_args = mock_warning.call_args[0]
            assert "1.5" in str(call_args) or 1.5 in call_args

    def test_no_warning_for_valid_range(self):
        """Should not log warning for values in valid range."""
        with patch(
            "agent_evals.adapters.scorers.autoevals.logger.warning"
        ) as mock_warning:
            _normalize_value(0.5)
            mock_warning.assert_not_called()

    def test_no_warning_for_boolean_conversion(self):
        """Should not log warning when converting booleans."""
        with patch(
            "agent_evals.adapters.scorers.autoevals.logger.warning"
        ) as mock_warning:
            _normalize_value(True)
            _normalize_value(False)
            mock_warning.assert_not_called()


class TestExactMatch:
    """Test ExactMatch() convenience factory."""

    def test_exact_match_returns_callable(self):
        """ExactMatch() should return a callable."""
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        scorer = ExactMatch()
        assert callable(scorer)

    def test_string_comparison(self):
        """Test exact string matching (case-sensitive)."""
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        scorer = ExactMatch()

        # Exact match
        result = scorer(
            result=TaskResult(output="hello"), expected=ExpectedResult(expected="hello")
        )
        assert result.value == 1.0
        assert result.passed
        assert result.name == "ExactMatch"

        # Case difference
        result = scorer(
            result=TaskResult(output="hello"), expected=ExpectedResult(expected="Hello")
        )
        assert result.value == 0.0
        assert not result.passed

        # Whitespace difference
        result = scorer(
            result=TaskResult(output="hello"),
            expected=ExpectedResult(expected="hello "),
        )
        assert result.value == 0.0
        assert not result.passed

    def test_number_comparison(self):
        """Test numeric value comparison with string conversion."""
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        scorer = ExactMatch()

        # Same numbers
        result = scorer(
            result=TaskResult(output="123"), expected=ExpectedResult(expected="123")
        )
        assert result.value == 1.0
        assert result.passed

        # Number vs string (should match after conversion)
        result = scorer(
            result=TaskResult(output="123"), expected=ExpectedResult(expected="123")
        )
        assert result.value == 1.0
        assert result.passed

        # Different numbers
        result = scorer(
            result=TaskResult(output="123"), expected=ExpectedResult(expected="456")
        )
        assert result.value == 0.0
        assert not result.passed

    def test_json_object_comparison(self):
        """Test JSON object comparison - order matters!"""
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        scorer = ExactMatch()

        # Same dict, same order
        result = scorer(
            result=TaskResult(output='{"a": 1, "b": 2}'),
            expected=ExpectedResult(expected='{"a": 1, "b": 2}'),
        )
        assert result.value == 1.0
        assert result.passed

        # Same content, DIFFERENT ORDER (should NOT match!)
        result = scorer(
            result=TaskResult(output='{"a": 1, "b": 2}'),
            expected=ExpectedResult(expected='{"b": 2, "a": 1}'),
        )
        assert result.value == 0.0
        assert not result.passed

        # Different values
        result = scorer(
            result=TaskResult(output='{"a": 1, "b": 2}'),
            expected=ExpectedResult(expected='{"a": 1, "b": 3}'),
        )
        assert result.value == 0.0
        assert not result.passed

        # Extra key
        result = scorer(
            result=TaskResult(output='{"a": 1, "b": 2}'),
            expected=ExpectedResult(expected='{"a": 1, "b": 2, "c": 3}'),
        )
        assert result.value == 0.0
        assert not result.passed

    def test_json_string_normalization(self):
        """Test JSON string formatting normalization."""
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        scorer = ExactMatch()

        # Dict vs JSON string (same order)
        result = scorer(
            result=TaskResult(output='{"a": 1, "b": 2}'),
            expected=ExpectedResult(expected='{"a": 1, "b": 2}'),
        )
        assert result.value == 1.0
        assert result.passed

        # Different whitespace formatting (should NOT match for exact match)
        result = scorer(
            result=TaskResult(output='{"a": 1, "b": 2}'),
            expected=ExpectedResult(expected='{"a":1, "b":2}'),
        )
        assert result.value == 0.0  # ExactMatch requires exact string match
        assert not result.passed

        # Different order (should NOT match)
        result = scorer(
            result=TaskResult(output='{"a": 1, "b": 2}'),
            expected=ExpectedResult(expected='{"b": 2, "a": 1}'),
        )
        assert result.value == 0.0
        assert not result.passed

    def test_array_comparison(self):
        """Test array comparison - order matters."""
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        scorer = ExactMatch()

        # Same array
        result = scorer(
            result=TaskResult(output="[1, 2, 3]"),
            expected=ExpectedResult(expected="[1, 2, 3]"),
        )
        assert result.value == 1.0
        assert result.passed

        # Different order (should NOT match)
        result = scorer(
            result=TaskResult(output="[1, 2, 3]"),
            expected=ExpectedResult(expected="[3, 2, 1]"),
        )
        assert result.value == 0.0
        assert not result.passed

        # Array vs JSON string
        result = scorer(
            result=TaskResult(output="[1, 2, 3]"),
            expected=ExpectedResult(expected="[1, 2, 3]"),
        )
        assert result.value == 1.0
        assert result.passed

    def test_none_handling(self):
        """Test None value comparison."""
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        scorer = ExactMatch()

        # None vs None
        result = scorer(
            result=TaskResult(output="None"), expected=ExpectedResult(expected="None")
        )
        assert result.value == 1.0
        assert result.passed

        # None vs string "None" (converts to string)
        result = scorer(
            result=TaskResult(output="None"), expected=ExpectedResult(expected="None")
        )
        assert result.value == 1.0
        assert result.passed

        # None vs other
        result = scorer(
            result=TaskResult(output="None"), expected=ExpectedResult(expected="hello")
        )
        assert result.value == 0.0
        assert not result.passed

    def test_comprehensive_cases(self):
        """Test comprehensive set of edge cases."""
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        scorer = ExactMatch()

        cases = [
            # [output, expected, expected_score]
            ["hello", "hello", 1],
            ["hello", "world", 0],
            [123, 123, 1],
            [123, "123", 1],
            [{"a": 1, "b": 2}, '{"a":1,"b":2}', 1],  # Dict to JSON string
            [{"a": 1, "b": 2}, '{"a":1,"b":3}', 0],  # Different values
            [[1, 2, 3], "[1,2,3]", 1],  # Array to JSON string
            [[1, 2, 3], "[3,2,1]", 0],  # Different order
            [{"a": 1, "b": 2}, '{"b":2,"a":1}', 0],  # Order matters
            [
                {"a": 1, "b": 2},
                '{"a":1,"b":2}',
                1,
            ],  # String matches dict (compact JSON)
            [{"a": 1, "b": 2}, '{"a": 1, "b": 2}', 0],  # Different whitespace format
            [{"a": 1, "b": 2}, '{"b": 2, "a": 1}', 0],  # Different order (with spaces)
            [{"a": 1, "b": 2}, '{"a":1,"b":2,"c":3}', 0],  # Extra key
            [None, None, 1],  # None matches None
            [None, "None", 1],  # None string matches None
        ]

        for output, expected, expected_score in cases:
            # Convert output to string for TaskResult
            if isinstance(output, dict | list):
                import json

                output_str = json.dumps(output, separators=(",", ":"))
            elif output is None:
                output_str = "None"
            else:
                output_str = str(output)

            # Convert expected to string for ExpectedResult
            if isinstance(expected, dict | list):
                import json

                expected_str = json.dumps(expected, separators=(",", ":"))
            elif expected is None:
                expected_str = "None"
            else:
                expected_str = str(expected)

            result = scorer(
                result=TaskResult(output=output_str),
                expected=ExpectedResult(expected=expected_str),
            )
            assert result.value == pytest.approx(expected_score, abs=1e-4), (
                f"Failed for output={output}, expected={expected}, "
                f"got score={result.value}, expected score={expected_score}"
            )

    def test_threshold_configuration(self):
        """Test threshold configuration."""
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        # Default threshold (1.0)
        scorer_default = ExactMatch()
        result = scorer_default(
            result=TaskResult(output="hello"), expected=ExpectedResult(expected="world")
        )
        assert not result.passed

        # Custom threshold (should still be binary)
        scorer_custom = ExactMatch(threshold=0.5)
        result = scorer_custom(
            result=TaskResult(output="hello"), expected=ExpectedResult(expected="world")
        )
        assert result.value == 0.0
        assert not result.passed  # 0.0 < 0.5

        result = scorer_custom(
            result=TaskResult(output="hello"), expected=ExpectedResult(expected="hello")
        )
        assert result.value == 1.0
        assert result.passed  # 1.0 >= 0.5

    def test_metadata_and_reasoning(self):
        """Test that result includes proper metadata and reasoning."""
        from agent_evals.adapters.scorers.autoevals import ExactMatch

        scorer = ExactMatch()
        result = scorer(
            result=TaskResult(output="hello"), expected=ExpectedResult(expected="world")
        )

        assert result.name == "ExactMatch"
        assert isinstance(result.metadata, dict)
        # Reasoning may be None for simple scorers
        assert result.reasoning is None or isinstance(result.reasoning, str)


class TestLevenshtein:
    """Test Levenshtein() convenience factory."""

    def test_levenshtein_returns_callable(self):
        """Levenshtein() should return a callable."""
        from agent_evals.adapters.scorers.autoevals import Levenshtein

        scorer = Levenshtein()
        assert callable(scorer)

    def test_levenshtein_returns_score_object_with_mock(self):
        """Levenshtein scorer should return Score object."""
        from unittest.mock import MagicMock, patch

        from agent_evals.core.types import Score

        # Mock the autoevals module before importing Levenshtein
        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals Levenshtein result
        mock_result = MagicMock()
        mock_result.score = 1.0
        mock_result.name = "Levenshtein"
        mock_result.metadata = {}
        mock_result.rationale = None  # Wrapper extracts 'rationale' not 'reasoning'

        mock_lev_instance = MagicMock()
        mock_lev_instance.return_value = mock_result
        mock_autoevals_string.Levenshtein.return_value = mock_lev_instance
        mock_autoevals.string = mock_autoevals_string

        # Inject mock into sys.modules
        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import Levenshtein

            # Create and call scorer
            scorer = Levenshtein()
            result = scorer(
                result=TaskResult(output="hello"),
                expected=ExpectedResult(expected="hello"),
            )

            assert isinstance(result, Score)
            assert result.name == "Levenshtein"

    def test_levenshtein_default_threshold_with_mock(self):
        """Levenshtein should use threshold=1.0 by default (exact match)."""
        from unittest.mock import MagicMock, patch

        # Mock the autoevals module
        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals with exact match
        mock_result = MagicMock()
        mock_result.score = 1.0
        mock_result.name = "Levenshtein"
        mock_result.metadata = {}
        mock_result.rationale = None  # Wrapper extracts 'rationale' not 'reasoning'

        mock_lev_instance = MagicMock()
        mock_lev_instance.return_value = mock_result
        mock_autoevals_string.Levenshtein.return_value = mock_lev_instance
        mock_autoevals.string = mock_autoevals_string

        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import Levenshtein

            scorer = Levenshtein()  # No threshold specified
            result = scorer(
                result=TaskResult(output="hello"),
                expected=ExpectedResult(expected="hello"),
            )

            # Should pass with exact match (score=1.0, threshold=1.0)
            assert result.passed is True

            # Mock with near match
            mock_result.score = 0.95
            result = scorer(
                result=TaskResult(output="hello"),
                expected=ExpectedResult(expected="hallo"),
            )

            # Should fail with near match (score=0.95, threshold=1.0)
            assert result.passed is False

    def test_levenshtein_custom_threshold_with_mock(self):
        """Should accept custom threshold parameter."""
        from unittest.mock import MagicMock, patch

        # Mock the autoevals module
        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals with near match
        mock_result = MagicMock()
        mock_result.score = 0.95
        mock_result.name = "Levenshtein"
        mock_result.metadata = {}
        mock_result.rationale = None  # Wrapper extracts 'rationale' not 'reasoning'

        mock_lev_instance = MagicMock()
        mock_lev_instance.return_value = mock_result
        mock_autoevals_string.Levenshtein.return_value = mock_lev_instance
        mock_autoevals.string = mock_autoevals_string

        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import Levenshtein

            # Custom threshold
            scorer = Levenshtein(threshold=0.9)
            result = scorer(
                result=TaskResult(output="hello"),
                expected=ExpectedResult(expected="hallo"),
            )

            # Should pass with relaxed threshold (score=0.95, threshold=0.9)
            assert result.passed is True

    def test_levenshtein_requires_expected_value(self):
        """Levenshtein should handle missing expected value gracefully."""
        from unittest.mock import MagicMock, patch

        from agent_evals.core.types import Score

        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals to raise ValueError when expected is empty string
        def mock_scorer_call(output, expected="", **kwargs):
            if expected == "":  # Our wrapper converts None to empty string
                raise ValueError("LevenshteinScorer requires an expected value")
            mock_result = MagicMock()
            mock_result.score = 1.0
            mock_result.metadata = {}
            mock_result.rationale = None  # Wrapper extracts 'rationale' not 'reasoning'
            return mock_result

        mock_lev_instance = MagicMock()
        mock_lev_instance.side_effect = mock_scorer_call
        mock_autoevals_string.Levenshtein.return_value = mock_lev_instance
        mock_autoevals.string = mock_autoevals_string

        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import Levenshtein

            scorer = Levenshtein()
            result = scorer(result=TaskResult(output="hello"), expected=None)

            # Should return error Score (our wrapper handles the exception)
            assert isinstance(result, Score)
            assert result.value == 0.0
            assert result.passed is False
            assert "error" in result.metadata


class TestEmbeddingSimilarity:
    """Test EmbeddingSimilarity() convenience factory."""

    def test_embedding_similarity_returns_callable(self):
        """EmbeddingSimilarity() should return a callable."""
        from agent_evals.adapters.scorers.autoevals import EmbeddingSimilarity

        scorer = EmbeddingSimilarity()
        assert callable(scorer)

    @pytest.mark.asyncio
    async def test_embedding_similarity_returns_score_object_with_mock(self):
        """EmbeddingSimilarity scorer should return Score object."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from agent_evals.core.types import Score

        # Mock the autoevals module before importing EmbeddingSimilarity
        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals EmbeddingSimilarity result
        mock_result = MagicMock()
        mock_result.score = 0.92
        mock_result.name = "EmbeddingSimilarity"
        mock_result.metadata = {"model": "text-embedding-ada-002"}
        mock_result.rationale = None

        mock_embed_instance = MagicMock()
        mock_embed_instance.eval_async = AsyncMock(return_value=mock_result)
        mock_autoevals_string.EmbeddingSimilarity.return_value = mock_embed_instance
        mock_autoevals.string = mock_autoevals_string

        # Inject mock into sys.modules
        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import EmbeddingSimilarity

            # Create and call scorer
            scorer = EmbeddingSimilarity()
            result = await scorer(
                result=TaskResult(output="The dog ran"),
                expected=ExpectedResult(expected="The canine sprinted"),
            )

            assert isinstance(result, Score)
            assert result.name == "EmbeddingSimilarity"

    @pytest.mark.asyncio
    async def test_embedding_similarity_default_threshold_with_mock(self):
        """EmbeddingSimilarity should use threshold=0.8 by default."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Mock the autoevals module
        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals with high similarity
        mock_result = MagicMock()
        mock_result.score = 0.85
        mock_result.name = "EmbeddingSimilarity"
        mock_result.metadata = {}
        mock_result.rationale = None

        mock_embed_instance = MagicMock()
        mock_embed_instance.eval_async = AsyncMock(return_value=mock_result)
        mock_autoevals_string.EmbeddingSimilarity.return_value = mock_embed_instance
        mock_autoevals.string = mock_autoevals_string

        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import EmbeddingSimilarity

            scorer = EmbeddingSimilarity()  # No threshold specified
            result = await scorer(
                result=TaskResult(output="car"),
                expected=ExpectedResult(expected="automobile"),
            )

            # Should pass with high similarity (score=0.85, threshold=0.8)
            assert result.passed is True

            # Mock with lower similarity
            mock_result.score = 0.75
            result = await scorer(
                result=TaskResult(output="car"), expected=ExpectedResult(expected="sky")
            )

            # Should fail with lower similarity (score=0.75, threshold=0.8)
            assert result.passed is False

    @pytest.mark.asyncio
    async def test_embedding_similarity_custom_threshold_with_mock(self):
        """Should accept custom threshold parameter."""
        from unittest.mock import AsyncMock, MagicMock, patch

        # Mock the autoevals module
        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals with moderate similarity
        mock_result = MagicMock()
        mock_result.score = 0.75
        mock_result.name = "EmbeddingSimilarity"
        mock_result.metadata = {}
        mock_result.rationale = None

        mock_embed_instance = MagicMock()
        mock_embed_instance.eval_async = AsyncMock(return_value=mock_result)
        mock_autoevals_string.EmbeddingSimilarity.return_value = mock_embed_instance
        mock_autoevals.string = mock_autoevals_string

        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import EmbeddingSimilarity

            # Custom threshold
            scorer = EmbeddingSimilarity(threshold=0.7)
            result = await scorer(
                result=TaskResult(output="car"),
                expected=ExpectedResult(expected="vehicle"),
            )

            # Should pass with relaxed threshold (score=0.75, threshold=0.7)
            assert result.passed is True

    @pytest.mark.asyncio
    async def test_embedding_similarity_parameters_passed_to_autoevals(self):
        """Should pass all parameters to autoevals scorer."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from agent_evals.core.types import Score

        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals result
        mock_result = MagicMock()
        mock_result.score = 0.9
        mock_result.metadata = {}
        mock_result.rationale = None

        mock_embed_instance = MagicMock()
        mock_embed_instance.eval_async = AsyncMock(return_value=mock_result)
        mock_embed_class = MagicMock(return_value=mock_embed_instance)
        mock_autoevals_string.EmbeddingSimilarity = mock_embed_class
        mock_autoevals.string = mock_autoevals_string

        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import EmbeddingSimilarity

            # Create scorer with custom parameters
            scorer = EmbeddingSimilarity(
                threshold=0.85,
                model="custom-embedding-model",
                prefix="Code: ",
                api_key="test-key",
                base_url="https://localhost:11434/v1",
                client=MagicMock(),
            )

            # Call the scorer
            result = await scorer(
                result=TaskResult(output="test"),
                expected=ExpectedResult(expected="test"),
            )

            # Verify result is valid Score object
            assert isinstance(result, Score)
            assert result.value == 0.9
            assert result.passed is True

            # Verify autoevals class was instantiated with correct parameters
            mock_embed_class.assert_called_once()
            call_kwargs = mock_embed_class.call_args[1]

            assert call_kwargs["model"] == "custom-embedding-model"
            assert call_kwargs["prefix"] == "Code: "
            assert call_kwargs["expected_min"] == 0.85  # threshold maps to expected_min
            assert call_kwargs["api_key"] == "test-key"
            assert call_kwargs["base_url"] == "https://localhost:11434/v1"
            assert "client" in call_kwargs

    @pytest.mark.asyncio
    async def test_embedding_similarity_requires_expected_value(self):
        """EmbeddingSimilarity should handle missing expected value gracefully."""
        from unittest.mock import AsyncMock, MagicMock, patch

        from agent_evals.core.types import Score

        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals to raise ValueError when expected is empty string
        async def mock_scorer_call(output, expected="", **kwargs):
            if expected == "":  # Our wrapper converts None to empty string
                raise ValueError("EmbeddingSimilarity requires an expected value")
            mock_result = MagicMock()
            mock_result.score = 0.85
            mock_result.metadata = {}
            mock_result.rationale = None
            return mock_result

        mock_embed_instance = MagicMock()
        mock_embed_instance.eval_async = AsyncMock(side_effect=mock_scorer_call)
        mock_autoevals_string.EmbeddingSimilarity.return_value = mock_embed_instance
        mock_autoevals.string = mock_autoevals_string

        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import EmbeddingSimilarity

            scorer = EmbeddingSimilarity()
            result = await scorer(result=TaskResult(output="test"), expected=None)

            # Should return error Score (our wrapper handles the exception)
            assert isinstance(result, Score)
            assert result.value == 0.0
            assert result.passed is False
            assert "error" in result.metadata

    @pytest.mark.asyncio
    async def test_embedding_similarity_threshold_maps_to_expected_min(self):
        """EmbeddingSimilarity threshold parameter should map to expected_min."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals result
        mock_result = MagicMock()
        mock_result.score = 0.8
        mock_result.metadata = {}
        mock_result.rationale = None

        mock_embed_instance = MagicMock()
        mock_embed_instance.eval_async = AsyncMock(return_value=mock_result)
        mock_embed_class = MagicMock(return_value=mock_embed_instance)
        mock_autoevals_string.EmbeddingSimilarity = mock_embed_class
        mock_autoevals.string = mock_autoevals_string

        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import EmbeddingSimilarity

            # Create scorer with custom threshold
            scorer = EmbeddingSimilarity(threshold=0.75)
            result = await scorer(
                result=TaskResult(output="test"),
                expected=ExpectedResult(expected="test"),
            )

            # Verify expected_min was set to threshold value
            mock_embed_class.assert_called_once()
            call_kwargs = mock_embed_class.call_args[1]
            assert call_kwargs["expected_min"] == 0.75

            # Also verify the result
            assert result.value == 0.8
            assert result.passed is True  # 0.8 >= 0.75

    @pytest.mark.asyncio
    async def test_embedding_similarity_prefix_parameter(self):
        """EmbeddingSimilarity should pass prefix parameter to autoevals."""
        from unittest.mock import AsyncMock, MagicMock, patch

        mock_autoevals = MagicMock()
        mock_autoevals_string = MagicMock()

        # Mock autoevals result
        mock_result = MagicMock()
        mock_result.score = 0.9
        mock_result.metadata = {}
        mock_result.rationale = None

        mock_embed_instance = MagicMock()
        mock_embed_instance.eval_async = AsyncMock(return_value=mock_result)
        mock_embed_class = MagicMock(return_value=mock_embed_instance)
        mock_autoevals_string.EmbeddingSimilarity = mock_embed_class
        mock_autoevals.string = mock_autoevals_string

        with patch.dict(
            "sys.modules",
            {"autoevals": mock_autoevals, "autoevals.string": mock_autoevals_string},
        ):
            from agent_evals.adapters.scorers.autoevals import EmbeddingSimilarity

            # Create scorer with prefix
            scorer = EmbeddingSimilarity(prefix="Code explanation: ")
            result = await scorer(
                result=TaskResult(output="test"),
                expected=ExpectedResult(expected="test"),
            )

            # Verify result is valid
            assert result.value == 0.9
            assert result.passed is True

            # Verify prefix was passed
            mock_embed_class.assert_called_once()
            call_kwargs = mock_embed_class.call_args[1]
            assert call_kwargs["prefix"] == "Code explanation: "


class TestFactuality:
    """Test Factuality() convenience factory."""

    @patch("autoevals.llm.Factuality")
    def test_factuality_returns_callable(self, mock_factuality_class):
        """Factuality() should return a callable."""
        from agent_evals.adapters.scorers.autoevals import Factuality

        scorer = Factuality()
        assert callable(scorer)

    @patch("autoevals.llm.Factuality")
    @pytest.mark.asyncio
    async def test_factuality_default_threshold(self, mock_factuality_class):
        """Factuality should use threshold=0.7 by default."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_evals.adapters.scorers.autoevals import Factuality
        from agent_evals.core.types import Score

        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 0.75  # Above default threshold (0.7)
        mock_result.name = "Factuality"
        mock_result.metadata = {}
        mock_result.rationale = None

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.eval_async = AsyncMock(return_value=mock_result)
        mock_factuality_class.return_value = mock_scorer

        # Use default threshold
        scorer = Factuality()
        result = await scorer(
            result=TaskResult(output="Paris"),
            expected=ExpectedResult(expected="Paris is the capital"),
        )

        # Should pass with score above threshold (0.75 >= 0.7)
        assert isinstance(result, Score)
        assert result.passed is True

    @patch("autoevals.llm.Factuality")
    @pytest.mark.asyncio
    async def test_factuality_default_model(self, mock_factuality_class):
        """Factuality should use model=gpt-4o-mini by default."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_evals.adapters.scorers.autoevals import Factuality

        # Mock the result
        mock_result = MagicMock()
        mock_result.score = 0.8
        mock_result.name = "Factuality"
        mock_result.metadata = {}
        mock_result.rationale = None

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.eval_async = AsyncMock(return_value=mock_result)
        mock_factuality_class.return_value = mock_scorer

        # Create Factuality scorer with defaults
        scorer = Factuality()
        await scorer(
            result=TaskResult(output="test"), expected=ExpectedResult(expected="test")
        )

        # Verify Factuality was instantiated with model="gpt-4o-mini"
        mock_factuality_class.assert_called_once_with(model="gpt-4o-mini")

    @patch("autoevals.llm.Factuality")
    @pytest.mark.asyncio
    async def test_factuality_custom_threshold_and_model(self, mock_factuality_class):
        """Factuality should respect custom threshold and model."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_evals.adapters.scorers.autoevals import Factuality
        from agent_evals.core.types import Score

        # Mock the result
        mock_result = MagicMock()
        mock_result.score = 0.85  # Between 0.7 and 0.9
        mock_result.name = "Factuality"
        mock_result.metadata = {}
        mock_result.rationale = None

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.eval_async = AsyncMock(return_value=mock_result)
        mock_factuality_class.return_value = mock_scorer

        # Custom threshold and model
        scorer = Factuality(threshold=0.9, model="gpt-4")
        result = await scorer(
            result=TaskResult(output="test"), expected=ExpectedResult(expected="test")
        )

        # Verify custom model was used
        mock_factuality_class.assert_called_once_with(model="gpt-4")

        # Should fail with high threshold (0.85 < 0.9)
        assert isinstance(result, Score)
        assert result.passed is False


class TestClosedQA:
    """Test ClosedQA() convenience factory."""

    @patch("autoevals.llm.ClosedQA")
    def test_closedqa_returns_callable(self, mock_closedqa_class):
        """ClosedQA() should return a callable."""
        from agent_evals.adapters.scorers.autoevals import ClosedQA

        scorer = ClosedQA()
        assert callable(scorer)

    @patch("autoevals.llm.ClosedQA")
    @pytest.mark.asyncio
    async def test_closedqa_default_threshold(self, mock_closedqa_class):
        """ClosedQA should use threshold=0.8 by default."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_evals.adapters.scorers.autoevals import ClosedQA
        from agent_evals.core.types import Score

        # Mock the result object
        mock_result = MagicMock()
        mock_result.score = 0.85  # Above default threshold (0.8)
        mock_result.name = "ClosedQA"
        mock_result.metadata = {}
        mock_result.rationale = None

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.eval_async = AsyncMock(return_value=mock_result)
        mock_closedqa_class.return_value = mock_scorer

        # Use default threshold
        scorer = ClosedQA()
        result = await scorer(
            result=TaskResult(output="A"), expected=ExpectedResult(expected="A")
        )

        # Should pass with score above threshold (0.85 >= 0.8)
        assert isinstance(result, Score)
        assert result.passed is True

    @patch("autoevals.llm.ClosedQA")
    @pytest.mark.asyncio
    async def test_closedqa_default_model(self, mock_closedqa_class):
        """ClosedQA should use model=gpt-4o-mini by default."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_evals.adapters.scorers.autoevals import ClosedQA

        # Mock the result
        mock_result = MagicMock()
        mock_result.score = 0.9
        mock_result.name = "ClosedQA"
        mock_result.metadata = {}
        mock_result.rationale = None

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.eval_async = AsyncMock(return_value=mock_result)
        mock_closedqa_class.return_value = mock_scorer

        # Create ClosedQA scorer with defaults
        scorer = ClosedQA()
        await scorer(
            result=TaskResult(output="B"), expected=ExpectedResult(expected="B")
        )

        # Verify ClosedQA was instantiated with model="gpt-4o-mini"
        mock_closedqa_class.assert_called_once_with(model="gpt-4o-mini")

    @patch("autoevals.llm.ClosedQA")
    @pytest.mark.asyncio
    async def test_closedqa_custom_threshold_and_model(self, mock_closedqa_class):
        """ClosedQA should respect custom threshold and model."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_evals.adapters.scorers.autoevals import ClosedQA
        from agent_evals.core.types import Score

        # Mock the result
        mock_result = MagicMock()
        mock_result.score = 0.9  # Between 0.8 and 0.95
        mock_result.name = "ClosedQA"
        mock_result.metadata = {}
        mock_result.rationale = None

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.eval_async = AsyncMock(return_value=mock_result)
        mock_closedqa_class.return_value = mock_scorer

        # Custom threshold and model
        scorer = ClosedQA(threshold=0.95, model="gpt-4")
        result = await scorer(
            result=TaskResult(output="C"), expected=ExpectedResult(expected="C")
        )

        # Verify custom model was used
        mock_closedqa_class.assert_called_once_with(model="gpt-4")

        # Should fail with high threshold (0.9 < 0.95)
        assert isinstance(result, Score)
        assert result.passed is False


class TestLLMParameterPassthrough:
    """Test that explicit LLM parameters are passed through correctly."""

    @patch("autoevals.llm.Factuality")
    @pytest.mark.asyncio
    async def test_factuality_with_explicit_params(self, mock_factuality_class):
        """Factuality should pass explicit parameters to autoevals."""
        from unittest.mock import AsyncMock, MagicMock, Mock

        from agent_evals.adapters.scorers.autoevals import Factuality

        # Mock the result
        mock_result = MagicMock()
        mock_result.score = 0.8
        mock_result.name = "Factuality"
        mock_result.metadata = {}
        mock_result.rationale = None

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.eval_async = AsyncMock(return_value=mock_result)
        mock_factuality_class.return_value = mock_scorer

        # Create mock client
        mock_client = Mock()

        # Create scorer with all explicit params
        scorer = Factuality(
            threshold=0.75,
            model="gpt-4",
            temperature=0.5,
            use_cot=False,
            max_tokens=256,
            client=mock_client,
            base_url="https://localhost:11434/v1",
        )
        await scorer(
            result=TaskResult(output="Test output"),
            expected=ExpectedResult(expected="Test expected"),
        )

        # Verify Factuality was instantiated with all params
        mock_factuality_class.assert_called_once_with(
            model="gpt-4",
            temperature=0.5,
            use_cot=False,
            max_tokens=256,
            client=mock_client,
            base_url="https://localhost:11434/v1",
        )

    @patch("autoevals.llm.Factuality")
    @pytest.mark.asyncio
    async def test_factuality_with_partial_params(self, mock_factuality_class):
        """Factuality should only pass non-None parameters."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_evals.adapters.scorers.autoevals import Factuality

        # Mock the result
        mock_result = MagicMock()
        mock_result.score = 0.8
        mock_result.name = "Factuality"
        mock_result.metadata = {}
        mock_result.rationale = None

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.eval_async = AsyncMock(return_value=mock_result)
        mock_factuality_class.return_value = mock_scorer

        # Create scorer with only some params (others default to None)
        scorer = Factuality(
            threshold=0.75,
            model="gpt-4",
            temperature=0.3,
            # use_cot, max_tokens, client, api_key, base_url all default to None
        )
        await scorer(
            result=TaskResult(output="Test"),
            expected=ExpectedResult(expected="Expected"),
        )

        # Verify only non-None params were passed
        mock_factuality_class.assert_called_once_with(
            model="gpt-4",
            temperature=0.3,
        )

    @patch("autoevals.llm.ClosedQA")
    @pytest.mark.asyncio
    async def test_closedqa_with_base_url(self, mock_closedqa_class):
        """ClosedQA should pass base_url for Ollama/custom endpoints."""
        from unittest.mock import AsyncMock, MagicMock

        from agent_evals.adapters.scorers.autoevals import ClosedQA

        # Mock the result
        mock_result = MagicMock()
        mock_result.score = 0.9
        mock_result.name = "ClosedQA"
        mock_result.metadata = {}
        mock_result.rationale = None

        # Mock the scorer instance
        mock_scorer = MagicMock()
        mock_scorer.eval_async = AsyncMock(return_value=mock_result)
        mock_closedqa_class.return_value = mock_scorer

        # Create scorer with base_url for Ollama
        scorer = ClosedQA(
            model="gpt-oss:20b",
            base_url="https://localhost:11434/v1",
        )
        await scorer(
            result=TaskResult(output="Answer"),
            expected=ExpectedResult(expected="Answer"),
        )

        # Verify base_url was passed
        mock_closedqa_class.assert_called_once_with(
            model="gpt-oss:20b",
            base_url="https://localhost:11434/v1",
        )


class TestErrorHandling:
    """Test error handling in autoevals wrappers."""

    def test_levenshtein_with_missing_autoevals(self):
        """Levenshtein factory should handle missing autoevals gracefully."""
        import builtins

        from agent_evals.core.types import Score

        # Mock __import__ to raise ImportError only for autoevals (not agent_evals)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            # Only block autoevals, not agent_evals
            if name == "autoevals.string" or name == "autoevals":
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import  # ty: ignore[invalid-assignment]  # Intentional temporary mock for testing ImportError
        try:
            # Need to reload the module to trigger ImportError in factory
            import sys

            if "agent_evals.adapters.scorers.autoevals" in sys.modules:
                del sys.modules["agent_evals.adapters.scorers.autoevals"]

            from agent_evals.adapters.scorers.autoevals import Levenshtein

            scorer = Levenshtein()
            result = scorer(
                result=TaskResult(output="test"),
                expected=ExpectedResult(expected="test"),
            )

            # Should return error Score
            assert isinstance(result, Score)
            assert result.value == 0.0
            assert result.passed is False
            assert "error" in result.metadata
        finally:
            builtins.__import__ = original_import
            # Clean up - reload the module normally
            if "agent_evals.adapters.scorers.autoevals" in sys.modules:
                del sys.modules["agent_evals.adapters.scorers.autoevals"]

    def test_embedding_similarity_with_missing_autoevals(self):
        """EmbeddingSimilarity factory should handle missing autoevals gracefully."""
        import builtins

        from agent_evals.core.types import Score

        # Mock __import__ to raise ImportError only for autoevals (not agent_evals)
        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            # Only block autoevals, not agent_evals
            if name == "autoevals.string" or name == "autoevals":
                raise ImportError(f"No module named '{name}'")
            return original_import(name, *args, **kwargs)

        builtins.__import__ = mock_import  # ty: ignore[invalid-assignment]  # Intentional temporary mock for testing ImportError
        try:
            # Need to reload the module to trigger ImportError in factory
            import sys

            if "agent_evals.adapters.scorers.autoevals" in sys.modules:
                del sys.modules["agent_evals.adapters.scorers.autoevals"]

            from agent_evals.adapters.scorers.autoevals import EmbeddingSimilarity

            scorer = EmbeddingSimilarity()
            result = scorer(
                result=TaskResult(output="test"),
                expected=ExpectedResult(expected="test"),
            )

            # Should return error Score
            assert isinstance(result, Score)
            assert result.value == 0.0
            assert result.passed is False
            assert "error" in result.metadata
        finally:
            builtins.__import__ = original_import
            # Clean up - reload the module normally
            if "agent_evals.adapters.scorers.autoevals" in sys.modules:
                del sys.modules["agent_evals.adapters.scorers.autoevals"]


class TestNumericDiff:
    """Test NumericDiff wrapper."""

    def test_exact_match(self):
        """Test exact numeric match returns score of 1.0."""
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        cases = [
            ("0", "0", 1.0),
            ("42", "42", 1.0),
            ("42.5", "42.5", 1.0),
            ("-10", "-10", 1.0),
            ("100.123", "100.123", 1.0),
        ]

        scorer = NumericDiff()
        for output, expected, expected_score in cases:
            result = scorer(
                result=TaskResult(output=output),
                expected=ExpectedResult(expected=expected),
            )
            assert result.value == pytest.approx(expected_score, abs=1e-4)
            assert result.passed  # Default threshold is 0.9
            assert result.name == "NumericDiff"

    def test_small_differences(self):
        """Test small differences produce high scores."""
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        # Formula: score = 1 - abs(output - expected) / (abs(output) + abs(expected))
        cases = [
            ("100", "99", 0.99497),  # 1 - 1/199 ≈ 0.99497
            ("42.5", "42.0", 0.99408),  # 1 - 0.5/84.5 ≈ 0.99408
            ("10", "11", 0.95238),  # 1 - 1/21 ≈ 0.95238
        ]

        scorer = NumericDiff()
        for output, expected, expected_score in cases:
            result = scorer(
                result=TaskResult(output=output),
                expected=ExpectedResult(expected=expected),
            )
            assert result.value == pytest.approx(expected_score, abs=1e-4)
            assert result.passed  # All should pass with threshold 0.9

    def test_large_differences(self):
        """Test large differences produce low scores."""
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        cases = [
            ("100", "50", 0.66667),  # 1 - 50/150 ≈ 0.66667
            ("100", "42", 0.59155),  # 1 - 58/142 ≈ 0.59155
            ("42", "10", 0.38462),  # 1 - 32/52 ≈ 0.38462
            ("100", "0", 0.0),  # 1 - 100/100 = 0.0
        ]

        scorer = NumericDiff()
        for output, expected, expected_score in cases:
            result = scorer(
                result=TaskResult(output=output),
                expected=ExpectedResult(expected=expected),
            )
            assert result.value == pytest.approx(expected_score, abs=1e-4)
            assert not result.passed  # All should fail with threshold 0.9

    def test_both_zeros_edge_case(self):
        """Test special case: both values are 0 returns score of 1.0."""
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        scorer = NumericDiff()
        result = scorer(
            result=TaskResult(output="0"), expected=ExpectedResult(expected="0")
        )
        assert result.value == 1.0
        assert result.passed

    def test_negative_numbers(self):
        """Test scorer works with negative numbers."""
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        cases = [
            ("-10", "-10", 1.0),  # Exact match
            ("-10", "-11", 0.95238),  # 1 - 1/21 ≈ 0.95238
            ("-100", "-50", 0.66667),  # 1 - 50/150 ≈ 0.66667
            ("10", "-10", 0.0),  # 1 - 20/20 = 0.0 (opposite signs)
        ]

        scorer = NumericDiff()
        for output, expected, expected_score in cases:
            result = scorer(
                result=TaskResult(output=output),
                expected=ExpectedResult(expected=expected),
            )
            assert result.value == pytest.approx(expected_score, abs=1e-4)

    def test_type_conversion(self):
        """Test automatic string to float conversion."""
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        scorer = NumericDiff()

        # Integer strings
        result = scorer(
            result=TaskResult(output="42"), expected=ExpectedResult(expected="42")
        )
        assert result.value == 1.0
        assert "converted_output" in result.metadata
        assert result.metadata["converted_output"] == 42.0

        # Float strings
        result = scorer(
            result=TaskResult(output="42.5"), expected=ExpectedResult(expected="42.0")
        )
        assert result.value == pytest.approx(0.99408, abs=1e-4)
        assert result.metadata["converted_expected"] == 42.0

        # Scientific notation
        result = scorer(
            result=TaskResult(output="1e2"), expected=ExpectedResult(expected="100")
        )
        assert result.value == 1.0

    def test_metadata_enrichment(self):
        """Test metadata includes conversion details."""
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        scorer = NumericDiff()
        result = scorer(
            result=TaskResult(output="42.5"), expected=ExpectedResult(expected="42.0")
        )

        # Check metadata contains conversion details
        assert "original_output" in result.metadata
        assert "original_expected" in result.metadata
        assert "converted_output" in result.metadata
        assert "converted_expected" in result.metadata

        assert result.metadata["original_output"] == "42.5"
        assert result.metadata["original_expected"] == "42.0"
        assert result.metadata["converted_output"] == 42.5
        assert result.metadata["converted_expected"] == 42.0

    def test_threshold_behavior(self):
        """Test threshold correctly determines pass/fail."""
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        scorer_strict = NumericDiff(threshold=0.99)
        scorer_lenient = NumericDiff(threshold=0.5)

        # Perfect match passes both
        result = scorer_strict(
            result=TaskResult(output="42"), expected=ExpectedResult(expected="42")
        )
        assert result.passed
        assert result.value == 1.0

        result = scorer_lenient(
            result=TaskResult(output="42"), expected=ExpectedResult(expected="42")
        )
        assert result.passed
        assert result.value == 1.0

        # Score ~0.95 fails strict, passes lenient
        result = scorer_strict(
            result=TaskResult(output="10"), expected=ExpectedResult(expected="11")
        )
        assert not result.passed
        assert result.value == pytest.approx(0.95238, abs=1e-4)

        result = scorer_lenient(
            result=TaskResult(output="10"), expected=ExpectedResult(expected="11")
        )
        assert result.passed
        assert result.value == pytest.approx(0.95238, abs=1e-4)

        # Score ~0.67 fails strict, passes lenient
        result = scorer_strict(
            result=TaskResult(output="100"), expected=ExpectedResult(expected="50")
        )
        assert not result.passed
        assert result.value == pytest.approx(0.66667, abs=1e-4)

        result = scorer_lenient(
            result=TaskResult(output="100"), expected=ExpectedResult(expected="50")
        )
        assert result.passed
        assert result.value == pytest.approx(0.66667, abs=1e-4)

    def test_error_handling_non_numeric_string(self):
        """Test error handling when strings cannot be converted to numbers."""
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        scorer = NumericDiff()

        # Non-numeric output
        result = scorer(
            result=TaskResult(output="not_a_number"),
            expected=ExpectedResult(expected="42"),
        )
        assert result.value == 0.0
        assert not result.passed
        assert "error" in result.metadata
        assert "Failed to convert to numeric" in result.metadata["error"]
        assert result.reasoning is not None

        # Non-numeric expected
        result = scorer(
            result=TaskResult(output="42"),
            expected=ExpectedResult(expected="not_a_number"),
        )
        assert result.value == 0.0
        assert not result.passed
        assert "error" in result.metadata

    def test_error_handling_none_expected(self):
        """Test error handling when expected is None."""
        from agent_evals.adapters.scorers.autoevals import NumericDiff

        scorer = NumericDiff()

        # NumericDiff requires expected value
        result = scorer(result=TaskResult(output="42"), expected=None)
        assert result.value == 0.0
        assert not result.passed
        assert "error" in result.metadata


class TestJSONDiff:
    """Test JSONDiff wrapper."""

    def test_string_as_json(self):
        """Test string similarity when strings are treated as JSON."""
        from agent_evals.adapters.scorers.autoevals import JSONDiff

        cases = [
            ("", "", 1),
            ("", "a", 0),
            ("a", "", 0),
            ("a", "a", 1),
            ("a", "b", 0),
            ("ab", "ac", 0.5),
            ("ac", "bc", 0.5),
            ("abc", "axc", 0.66667),
            ("xabxcdxxefxgx", "1ab2cd34ef5g6", 0.53846),
        ]

        scorer = JSONDiff()
        for output, expected, expected_score in cases:
            result = scorer(
                result=TaskResult(output=output),
                expected=ExpectedResult(expected=expected),
            )
            assert result.value == pytest.approx(expected_score, abs=1e-4)

    def test_json_objects(self):
        """Test JSON object comparison."""
        from agent_evals.adapters.scorers.autoevals import JSONDiff

        cases = [
            (None, "None", 1),  # None as string should match "None"
            (None, "", 0),
            ([], {}, 0),
            ([], [], 1),
            ({}, {}, 1),
            ({"a": 1}, {"a": 1}, 1),
            ({"a": 1}, {"a": 2}, 0.66667),
            ({"a": 1}, ["a", 1], 0.5714285714285714),
            ({"a": 1}, {"b": {"a": 1}}, 0),
            ({"a": 1}, {"a": None}, 0),
            (
                {"mapping": {"a": "foo", "b": "bar"}},
                {"mapping": {"a": "Foo", "b": "Bar"}, "Extra": 5},
                0.33333333333333337,
            ),
        ]

        scorer = JSONDiff()
        for output, expected, expected_score in cases:
            # Convert output to string for TaskResult
            if isinstance(output, dict | list):
                import json

                output_str = json.dumps(output)
            elif output is None:
                output_str = "None"
            else:
                output_str = str(output)

            # Convert expected to string for ExpectedResult
            if isinstance(expected, dict | list):
                import json

                expected_str = json.dumps(expected)
            elif expected is None:
                expected_str = "None"
            else:
                expected_str = str(expected)

            result = scorer(
                result=TaskResult(output=output_str),
                expected=ExpectedResult(expected=expected_str),
            )
            assert result.value == pytest.approx(expected_score, abs=1e-4)

    def test_semantic_json(self):
        """Test semantic JSON comparison with key reordering."""
        from agent_evals.adapters.scorers.autoevals import JSONDiff

        cases = [
            ('{"x": 1, "y": 2}', '{"y": 2, "x": 1}', 1),
            (
                '{"zs": ["a", "b"], "x": 1, "y": 2}',
                '{"y": 2, "zs": ["a", "b"], "x": 1}',
                1,
            ),
            (
                '{"o1": {"x": 1, "y": 2}}',
                '{"o1": {"y": 2, "x": 1}}',
                1,
            ),
            (
                '{"xs": [{"o1": {"x": 1, "y": [2]}}]}',
                '{"xs": [{"o1": {"y": [2], "x": 1}}]}',
                1,
            ),
            (
                '{"o1": {"x": 2, "y": 2}}',
                '{"o1": {"y": 2, "x": 1}}',
                0.83333,
            ),
            (
                {"o1": {"x": 2, "y": 2}},
                '{"o1": {"y": 2, "x": 1}}',
                0.83333,
            ),
            ('{"x": 1, "y": 2}', '{"x": 1, "z": 2}', 0.3333),
            ("[1, 2]", "[1, 2]", 1),
            ("[1, 2]", "[2, 1]", 0.66667),
        ]

        scorer = JSONDiff()
        for output, expected, expected_score in cases:
            # Convert output to string for TaskResult
            if isinstance(output, dict | list):
                import json

                output_str = json.dumps(output)
            elif output is None:
                output_str = "None"
            else:
                output_str = str(output)

            # Convert expected to string for ExpectedResult
            if isinstance(expected, dict | list):
                import json

                expected_str = json.dumps(expected)
            elif expected is None:
                expected_str = "None"
            else:
                expected_str = str(expected)

            result = scorer(
                result=TaskResult(output=output_str),
                expected=ExpectedResult(expected=expected_str),
            )
            assert result.value == pytest.approx(expected_score, abs=1e-4)

    def test_threshold_behavior(self):
        """Test that threshold correctly determines pass/fail."""
        from agent_evals.adapters.scorers.autoevals import JSONDiff

        scorer_strict = JSONDiff(threshold=1.0)
        scorer_lenient = JSONDiff(threshold=0.8)

        # Perfect match passes both
        result = scorer_strict(
            result=TaskResult(output='{"a": 1}'),
            expected=ExpectedResult(expected='{"a": 1}'),
        )
        assert result.passed
        assert result.value == 1.0

        result = scorer_lenient(
            result=TaskResult(output='{"a": 1}'),
            expected=ExpectedResult(expected='{"a": 1}'),
        )
        assert result.passed
        assert result.value == 1.0

        # Near match (0.83) fails strict, passes lenient
        result = scorer_strict(
            result=TaskResult(output='{"a": 2}'),
            expected=ExpectedResult(expected='{"a": 1}'),
        )
        assert not result.passed
        assert result.value == pytest.approx(0.66667, abs=1e-4)

        result = scorer_lenient(
            result=TaskResult(output='{"o1": {"x": 2, "y": 2}}'),
            expected=ExpectedResult(expected='{"o1": {"y": 2, "x": 1}}'),
        )
        assert result.passed
        assert result.value == pytest.approx(0.83333, abs=1e-4)

    def test_preserve_strings_true(self):
        """Test preserve_strings parameter prevents JSON string parsing."""
        from agent_evals.adapters.scorers.autoevals import JSONDiff

        scorer_default = JSONDiff(preserve_strings=False)
        result = scorer_default(
            result=TaskResult(output='{"config": "{\\"port\\": 8080}"}'),
            expected=ExpectedResult(expected='{"config": "{\\"port\\": 8080}"}'),
        )
        assert result.passed
        assert result.value == 1.0

        scorer_preserve = JSONDiff(preserve_strings=True)
        result = scorer_preserve(
            result=TaskResult(output='{"config": "{\\"port\\": 8080}"}'),
            expected=ExpectedResult(expected='{"config": "{\\"port\\": 8080}"}'),
        )
        assert result.passed
        assert result.value == 1.0

        result = scorer_preserve(
            result=TaskResult(output='{"value": "plain_string"}'),
            expected=ExpectedResult(expected='{"value": "plain_string"}'),
        )
        assert result.passed
        assert result.value == 1.0

    def test_custom_string_scorer(self):
        """Test JSONDiff with custom string scorer (ExactMatch for strict comparison)."""
        from agent_evals.adapters.scorers.autoevals import JSONDiff

        try:
            from autoevals import ExactMatch

            scorer = JSONDiff(string_scorer=ExactMatch())

            result = scorer(
                result=TaskResult(output='{"name": "John"}'),
                expected=ExpectedResult(expected='{"name": "John"}'),
            )
            assert result.passed
            assert result.value == 1.0

            result = scorer(
                result=TaskResult(output='{"name": "Jon"}'),
                expected=ExpectedResult(expected='{"name": "John"}'),
            )
            assert result.value == 0.0

        except ImportError:
            pytest.skip("autoevals.ExactMatch not available")

    def test_custom_number_scorer(self):
        """Test JSONDiff with custom number scorer."""
        from agent_evals.adapters.scorers.autoevals import JSONDiff

        try:
            from autoevals.number import NumericDiff

            scorer = JSONDiff(number_scorer=NumericDiff())

            result = scorer(
                result=TaskResult(output='{"age": 30, "score": 85}'),
                expected=ExpectedResult(expected='{"age": 31, "score": 85}'),
            )
            assert result.value > 0.8
            assert result.value < 1.0

        except ImportError:
            pytest.skip("autoevals.number.NumericDiff not available")


class TestValidJSON:
    """Test ValidJSON wrapper."""

    def test_basic_json_validation(self):
        """Test basic JSON syntax validation without schema."""
        from agent_evals.adapters.scorers.autoevals import ValidJSON

        cases = [
            ("1", 0),  # Bare number is not considered valid JSON object/array
            ('{ "a": 1, "b": "hello" }', 1),
            ('[{ "a": 1 }]', 1),
            ('[{ "a": 1 }', 0),  # Missing closing bracket
            ('{ "mapping": { "a": "foo", "b": "bar" }, "extra": 4 }', 1),
            ('{ mapping: { "a": "foo", "b": "bar" }, "extra": 4 }', 0),  # Unquoted key
            ({"a": "1", "b": "1"}, 1),  # Pre-parsed object
            ([{"a": "1"}, {"a": "1", "b": 22}], 1),  # Pre-parsed array
            (100, 0),  # Bare number
            ("100", 0),  # String number (ambiguous, treated as unparsed invalid)
        ]

        scorer = ValidJSON()
        for output, expected_score in cases:
            # Convert output to string for TaskResult
            if isinstance(output, dict | list):
                import json

                output_str = json.dumps(output, separators=(",", ":"))
            elif output is None:
                output_str = "None"
            else:
                output_str = str(output)
            result = scorer(result=TaskResult(output=output_str))
            assert result.value == expected_score, f"Failed for {output}"

    def test_schema_validation(self):
        """Test JSON Schema validation."""
        from agent_evals.adapters.scorers.autoevals import ValidJSON

        # Valid against schema
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}},
            "required": ["a"],
        }
        scorer = ValidJSON(schema=schema)
        result = scorer(result=TaskResult(output='{ "a": "1" }'))
        assert result.value == 1
        assert result.passed

        # Invalid - wrong type for property
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "number"},
            },
            "required": ["a"],
        }
        scorer = ValidJSON(schema=schema)
        result = scorer(
            result=TaskResult(output='{"a": "1", "b": "1"}')
        )  # b should be number, not string
        assert result.value == 0
        assert not result.passed

        # Valid array with schema
        schema = {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "a": {"type": "string"},
                    "b": {"type": "number"},
                },
                "required": ["a"],
            },
            "uniqueItems": True,
        }
        scorer = ValidJSON(schema=schema)
        result = scorer(result=TaskResult(output='[{"a": "1"}, {"a": "1", "b": 22}]'))
        assert result.value == 1
        assert result.passed

    def test_threshold_behavior(self):
        """Test that threshold correctly determines pass/fail."""
        from agent_evals.adapters.scorers.autoevals import ValidJSON

        # ValidJSON is binary, so threshold doesn't change much
        # But we should test it still works
        scorer = ValidJSON(threshold=1.0)

        result = scorer(result=TaskResult(output='{"valid": "json"}'))
        assert result.passed
        assert result.value == 1.0

        result = scorer(result=TaskResult(output="{invalid json}"))
        assert not result.passed
        assert result.value == 0.0

    def test_schema_in_metadata(self):
        """Test that schema is included in metadata when provided."""
        from agent_evals.adapters.scorers.autoevals import ValidJSON

        schema = {"type": "object", "properties": {"a": {"type": "string"}}}
        scorer = ValidJSON(schema=schema)

        result = scorer(result=TaskResult(output='{"a": "test"}'))
        assert "schema" in result.metadata
        assert result.metadata["schema"] == schema


class TestListContains:
    """Test ListContains wrapper."""

    def test_listcontains_returns_callable(self):
        """ListContains() should return a callable."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        assert callable(scorer)

    def test_exact_match_all_items(self):
        """Test perfect match with all expected items."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        result = scorer(
            result=TaskResult(output='["python", "java", "javascript"]'),
            expected=ExpectedResult(expected='["python", "java", "javascript"]'),
        )
        assert result.value == 1.0
        assert result.passed
        assert result.name == "ListContains"

    def test_missing_expected_items(self):
        """Test when some expected items are missing from output."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        result = scorer(
            result=TaskResult(output='["python", "java"]'),
            expected=ExpectedResult(expected='["python", "java", "javascript"]'),
        )
        # Should get 2/3 = 0.67 (approximately)
        assert result.value == pytest.approx(0.6667, abs=0.01)
        assert not result.passed  # Default threshold is 1.0

    def test_with_extra_items_allowed(self):
        """Test with extra items in output (default allow_extra_entities=True)."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        result = scorer(
            result=TaskResult(output='["python", "java", "javascript", "rust", "go"]'),
            expected=ExpectedResult(expected='["python", "java", "javascript"]'),
        )
        # All expected items found, extras don't penalize
        assert result.value == 1.0
        assert result.passed

    def test_with_extra_items_not_allowed(self):
        """Test with extra items penalized (allow_extra_entities=False)."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains(allow_extra_entities=False)
        result = scorer(
            result=TaskResult(output='["python", "java", "javascript", "rust", "go"]'),
            expected=ExpectedResult(expected='["python", "java", "javascript"]'),
        )
        # 3 matched out of max(5,3)=5, so score = 3/5 = 0.6
        assert result.value == pytest.approx(0.6, abs=0.01)
        assert not result.passed  # Below threshold of 1.0

    def test_custom_threshold(self):
        """Test custom threshold parameter."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        # Lenient threshold
        scorer = ListContains(threshold=0.6)
        result = scorer(
            result=TaskResult(output='["python", "java"]'),
            expected=ExpectedResult(expected='["python", "java", "javascript"]'),
        )
        # Score ~0.67 should pass with threshold 0.6
        assert result.value == pytest.approx(0.6667, abs=0.01)
        assert result.passed

        # Strict threshold
        scorer_strict = ListContains(threshold=0.8)
        result_strict = scorer_strict(
            result=TaskResult(output='["python", "java"]'),
            expected=ExpectedResult(expected='["python", "java", "javascript"]'),
        )
        # Score ~0.67 should fail with threshold 0.8
        assert result_strict.value == pytest.approx(0.6667, abs=0.01)
        assert not result_strict.passed

    def test_string_to_list_conversion_json(self):
        """Test automatic JSON string to list conversion."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        # JSON format strings
        result = scorer(
            result=TaskResult(output='["python", "java", "javascript"]'),
            expected=ExpectedResult(expected='["python", "java", "javascript"]'),
        )
        assert result.value == 1.0
        assert result.passed

    def test_string_to_list_conversion_literal(self):
        """Test automatic Python literal string to list conversion."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        # Python literal format (single quotes, ast.literal_eval fallback)
        result = scorer(
            result=TaskResult(output="['python', 'java', 'javascript']"),
            expected=ExpectedResult(expected="['python', 'java', 'javascript']"),
        )
        assert result.value == 1.0
        assert result.passed

    def test_empty_lists(self):
        """Test behavior with empty lists."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        # Both empty - should be perfect match
        result = scorer(
            result=TaskResult(output="[]"), expected=ExpectedResult(expected="[]")
        )
        assert result.value == 1.0
        assert result.passed

    def test_output_empty_expected_not(self):
        """Test when output is empty but expected has items."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        result = scorer(
            result=TaskResult(output="[]"),
            expected=ExpectedResult(expected='["python", "java"]'),
        )
        # No items found
        assert result.value == 0.0
        assert not result.passed

    def test_metadata_populated(self):
        """Test that metadata contains expected fields."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains(allow_extra_entities=True)
        result = scorer(
            result=TaskResult(output='["python", "java"]'),
            expected=ExpectedResult(expected='["python", "java"]'),
        )
        # Check metadata keys
        assert "allow_extra_entities" in result.metadata
        assert result.metadata["allow_extra_entities"] is True
        assert "pairwise_scorer" in result.metadata
        assert result.metadata["pairwise_scorer"] == "Levenshtein"  # Default

    def test_with_custom_pairwise_scorer(self):
        """Test with custom pairwise scorer parameter."""
        # Use ExactMatch as pairwise scorer (simpler test, avoids type issues)
        from autoevals.value import ExactMatch

        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains(
            threshold=0.8, pairwise_scorer=ExactMatch(), allow_extra_entities=True
        )
        result = scorer(
            result=TaskResult(output='["a", "b", "c"]'),
            expected=ExpectedResult(expected='["a", "b", "c"]'),
        )

        # Should work with exact match comparison
        assert result.value >= 0.8
        assert result.passed

        # Check metadata records custom scorer
        assert "pairwise_scorer" in result.metadata
        assert result.metadata["pairwise_scorer"] == "ExactMatch"

    def test_threshold_behavior_boundary(self):
        """Test threshold behavior at boundary conditions."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        # Test with threshold slightly below actual score
        scorer = ListContains(threshold=0.66)
        result = scorer(
            result=TaskResult(output='["python", "java"]'),
            expected=ExpectedResult(expected='["python", "java", "javascript"]'),
        )
        # Score should be ~0.6667, which is >= 0.66
        assert result.value == pytest.approx(0.6667, abs=0.01)
        assert result.passed  # Should pass with threshold 0.66

        # Test with threshold slightly above actual score
        scorer_high = ListContains(threshold=0.67)
        result_high = scorer_high(
            result=TaskResult(output='["python", "java"]'),
            expected=ExpectedResult(expected='["python", "java", "javascript"]'),
        )
        # Score ~0.6667 is slightly less than 0.67
        assert result_high.value == pytest.approx(0.6667, abs=0.01)
        assert not result_high.passed  # Should fail with threshold 0.67

    def test_error_handling_invalid_string(self):
        """Test error handling with invalid string input."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        # Invalid JSON/literal string
        result = scorer(
            result=TaskResult(output="not a valid list"),
            expected=ExpectedResult(expected='["python"]'),
        )

        # Should return error Score, not crash
        assert result.value == 0.0
        assert not result.passed
        assert "error" in result.metadata

    def test_oversized_literal_input_returns_error_score(self):
        """Oversized non-JSON output is rejected by the length cap, not parsed.

        The literal-eval fallback runs on untrusted LLM output, so it is bounded
        by ``_MAX_LITERAL_EVAL_LEN`` to keep pathological/oversized strings out of
        the CPython parser. This test guards that cap: the payload below is a
        valid Python literal that ``ast.literal_eval`` would happily parse, so it
        fails if the length guard is removed.
        """
        from agent_evals.adapters.scorers.autoevals import (
            _MAX_LITERAL_EVAL_LEN,
            ListContains,
        )

        scorer = ListContains()
        # Single-quoted -> json.loads raises JSONDecodeError -> literal_eval path.
        # Sized past the cap so the guard rejects it before parsing.
        items = (_MAX_LITERAL_EVAL_LEN // len("'x',")) + 100
        payload = "[" + "'x'," * items + "]"
        assert len(payload) > _MAX_LITERAL_EVAL_LEN
        result = scorer(
            result=TaskResult(output=payload),
            expected=ExpectedResult(expected='["x"]'),
        )
        assert result.value == 0.0
        assert not result.passed
        assert "error" in result.metadata
        assert "too large" in result.metadata["error"]

    def test_undersized_python_literal_still_parses(self):
        """Inputs within the cap keep using the Python-literal fallback."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        result = scorer(
            result=TaskResult(output="['python', 'java']"),
            expected=ExpectedResult(expected="['python', 'java']"),
        )
        assert result.value == 1.0
        assert result.passed
        assert "error" not in result.metadata

    def test_reasoning_field(self):
        """Test that reasoning field is populated."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        result = scorer(
            result=TaskResult(output='["python", "java"]'),
            expected=ExpectedResult(expected='["python", "java", "javascript"]'),
        )
        assert result.reasoning is not None
        assert isinstance(result.reasoning, str)
        # Should mention percentage or items found
        assert "67%" in result.reasoning or "items" in result.reasoning.lower()

    def test_output_has_items_expected_empty(self):
        """Test when output has items but expected is empty."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains()
        result = scorer(
            result=TaskResult(output='["python", "java"]'),
            expected=ExpectedResult(expected="[]"),
        )
        # No expected items to find
        assert result.value == 0.0
        assert not result.passed

    def test_exact_score_calculations(self):
        """Test exact score calculations match autoevals behavior.

        Note: autoevals defaults to allow_extra_entities=False, but our wrapper
        defaults to True (more lenient). This test uses False to match autoevals.
        """
        from agent_evals.adapters.scorers.autoevals import ListContains

        # Test cases from autoevals test suite
        cases = [
            ([], [], 1.0),
            ([0], [], 0.0),
            ([], [0], 0.0),
            (["a"], ["a"], 1.0),
            (["a"], ["a", "b"], 0.5),
            (["a", "b"], ["a"], 0.5),  # Extra items penalize when allow_extra=False
        ]

        # Use allow_extra_entities=False to match autoevals default behavior
        scorer = ListContains(threshold=0.0, allow_extra_entities=False)
        for output, expected, expected_score in cases:
            # Convert output to string for TaskResult
            if isinstance(output, list):
                import json

                output_str = json.dumps(output)
            else:
                output_str = str(output)

            # Convert expected to string for ExpectedResult
            if isinstance(expected, list):
                import json

                expected_str = json.dumps(expected)
            else:
                expected_str = str(expected)

            result = scorer(
                result=TaskResult(output=output_str),
                expected=ExpectedResult(expected=expected_str),
            )
            assert result.value == pytest.approx(expected_score, abs=1e-4), (
                f"Failed for output={output}, expected={expected}"
            )

    def test_with_extra_items_exact_score(self):
        """Test exact score with allow_extra_entities=True."""
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains(threshold=0.0, allow_extra_entities=True)
        result = scorer(
            result=TaskResult(output='["a", "b"]'),
            expected=ExpectedResult(expected='["a"]'),
        )
        # All expected items found, extras don't penalize
        assert result.value == 1.0

    @pytest.mark.asyncio
    async def test_works_with_async_runner(self):
        """Test that sync scorer works seamlessly with run_eval_async.

        This demonstrates that the runner automatically wraps sync scorers
        with asyncio.to_thread() for non-blocking execution.
        """
        import json

        from agent_evals import TaskResult, run_eval_async
        from agent_evals.adapters.scorers.autoevals import ListContains

        scorer = ListContains(threshold=1.0, allow_extra_entities=False)

        # Run evaluation with async runner
        result = await run_eval_async(
            task=lambda x: TaskResult(
                output=json.dumps(x)
            ),  # Convert list to JSON string
            dataset=[
                ExampleData(
                    input=["a", "b"],
                    expected=ExpectedResult(expected=json.dumps(["a", "b"])),
                ),  # Expected as JSON string to match output format
                ExampleData(
                    input=["a", "b", "c"],
                    expected=ExpectedResult(expected=json.dumps(["a", "b"])),
                ),  # Expected as JSON string to match output format
            ],
            scorers=[scorer],
        )

        # Check aggregate results
        assert "ListContains" in result.scores
        # Aggregate is average of scores: (1.0 + score_with_extras) / 2
        # With allow_extra_entities=False, extra items penalize
        assert result.scores["ListContains"] > 0.5  # Should be between 0.5 and 1.0

        # Check individual examples
        assert result.examples[0].scores["ListContains"].passed  # Exact match
        assert result.examples[0].scores["ListContains"].value == 1.0
        # Second example has extra items, which penalize with allow_extra_entities=False
        assert (
            not result.examples[1].scores["ListContains"].passed
        )  # Extra items fail threshold
