# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Autoevals scorer adapters.

Convenience wrappers for 25+ autoevals scorers that conform to the agent-evals
Scorer Protocol. LLM-based scorers use native async I/O, while heuristic scorers
remain sync for optimal performance.

Usage:
    from agent_evals.adapters.scorers.autoevals import (
        Levenshtein, Factuality, NumericDiff, JSONDiff, ListContains
    )

    scorers = [
        Levenshtein(threshold=0.95),           # Sync (CPU-bound)
        Factuality(model="gpt-4o-mini", threshold=0.7),  # Async (LLM I/O)
        NumericDiff(threshold=0.9),            # Sync (CPU-bound)
    ]

Async Execution:
    LLM-based scorers (12 total) use native async I/O via eval_async() for
    true concurrent execution in the event loop. Heuristic scorers remain sync
    and execute in a thread pool. Framework auto-detects scorer type and routes
    appropriately.

    Async scorers (12): Factuality, ClosedQA, Humor, Battle, Security, Summary,
                        Translation, Sql, Possible, LLMClassifier, Moderation,
                        EmbeddingSimilarity
    Sync scorers (6): Levenshtein, ExactMatch, NumericDiff, JSONDiff,
                      ValidJSON, ListContains

    Sync scorers run via ``asyncio.to_thread`` so they don't block the event
    loop. Rate-limiting LLM calls inside your scorer is the scorer's
    responsibility; the framework does not throttle scorer calls.

Available Scorers:
    • String: Levenshtein (sync), ExactMatch (sync), EmbeddingSimilarity (async)
    • Numeric/Structure: NumericDiff (sync), JSONDiff (sync), ValidJSON (sync),
                         ListContains (sync)
    • LLM Judges (all async): Factuality, ClosedQA, Humor, Battle, Security,
                              Summary, Translation, Sql, Possible, LLMClassifier,
                              Moderation

Note: RAG scorers (Faithfulness, AnswerCorrectness, etc.) are NOT included because
they require OpenAI's function calling with tool_choice parameter, which is not
supported by Ollama. See CONTRIBUTING.md for details.

For detailed usage examples, see docs/scorers.md.
See CONTRIBUTING.md for implementation patterns (if adding new wrappers).
"""

import logging
from collections.abc import Callable
from typing import Any

from agent_evals.core._registries import scorer_registry
from agent_evals.core.transport import validate_secure_transport
from agent_evals.core.types import ExpectedResult, Score, TaskResult

logger = logging.getLogger(__name__)


def _normalize_value(raw_value: float | bool | int) -> float:
    """Normalize score value to [0.0, 1.0] range.

    Handles various input types from external scorers and normalizes them
    to the standard [0.0, 1.0] range required by the Score dataclass.

    Args:
        raw_value: Raw score value from external scorer. Can be:
            - float in any range (will be clamped)
            - bool (True → 1.0, False → 0.0)
            - int (converted to float, then clamped)

    Returns:
        Normalized value in [0.0, 1.0] range

    Raises:
        TypeError: If raw_value is None or unsupported type

    Side Effects:
        Logs warning if value is clamped (out of range)

    Examples:
        >>> _normalize_value(0.5)
        0.5
        >>> _normalize_value(True)
        1.0
        >>> _normalize_value(1.5)  # Logs warning
        1.0
        >>> _normalize_value(-0.2)  # Logs warning
        0.0
    """
    # Handle boolean values first (before conversion to float)
    if isinstance(raw_value, bool):
        return 1.0 if raw_value else 0.0

    # Convert to float (handles int and float)
    value = float(raw_value)

    # Check if value is in valid range
    if not (0.0 <= value <= 1.0):
        logger.warning(
            "Score value %s outside valid range [0.0, 1.0], clamping to nearest bound",
            value,
        )
        return max(0.0, min(1.0, value))

    return value


# The ast.literal_eval fallback runs on untrusted LLM output. CPython's parser
# documents that deeply-nested input can exhaust the C stack; current CPython
# guards the cited payloads (catchable SyntaxError/RecursionError), but we don't
# want to depend on that guard holding across versions for untrusted input.
# Bounding the input length keeps pathological or oversized strings out of the
# parser entirely. 32 KiB comfortably covers any flat list an LLM would emit for
# containment scoring.
_MAX_LITERAL_EVAL_LEN = 1 << 15


def _safe_literal_eval(value: str) -> Any:
    """Length-bounded ast.literal_eval for untrusted string input.

    Oversized input raises ValueError before reaching the parser; callers
    let it propagate to their ``except Exception`` so it becomes an error Score.
    """
    import ast

    if len(value) > _MAX_LITERAL_EVAL_LEN:
        raise ValueError(
            f"input too large for literal_eval fallback "
            f"({len(value)} > {_MAX_LITERAL_EVAL_LEN} chars)"
        )
    return ast.literal_eval(value)


# Internal helper to build kwargs dict from optional parameters
def _build_scorer_kwargs(
    model: str,
    temperature: float | None = None,
    use_cot: bool | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Build kwargs dict, only including non-None values."""
    kwargs: dict[str, Any] = {"model": model}
    if temperature is not None:
        kwargs["temperature"] = temperature
    if use_cot is not None:
        kwargs["use_cot"] = use_cot
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if client is not None:
        kwargs["client"] = client
    if api_key is not None:
        kwargs["api_key"] = api_key
    if base_url is not None:
        # This helper is not the only entry point for base_url. Three public
        # factories build their kwargs inline and never call it —
        # EmbeddingSimilarity, LLMClassifier, and Moderation — so each carries
        # its own validate_secure_transport call. A new factory accepting
        # base_url must either route through here or validate in its own body;
        # tests/validation/test_transport_enforcement.py fails it if neither.
        validate_secure_transport("base_url", base_url)
        kwargs["base_url"] = base_url
    return kwargs


def _missing_expected_score(scorer_name: str, output: Any) -> Score:
    """Build the failing Score for a comparison with no reference value.

    Matches the shape the three factories that already guard explicitly
    (``NumericDiff``, ``ListContains``, ``JSONDiff``) return, rather than
    introducing a second convention for the same condition. Returning a
    Score rather than raising is deliberate too: one unscoreable example
    must not abort a run of two hundred, and the platform adapters record
    a Score where an exception would be lost.
    """
    return Score(
        name=scorer_name,
        value=0.0,
        passed=False,
        metadata={
            "error": f"Expected value is required for {scorer_name}",
            "output": str(output),
            "expected": None,
        },
        reasoning=f"{scorer_name} requires an expected value for comparison",
    )


def _resolve_expected(
    expected: ExpectedResult | None,
    requires_expected: bool,
) -> str | None:
    """Return the reference string to score against, or None if it is missing.

    ``None`` means the caller supplied no usable reference and the scorer
    declared that it needs one — the wrapper must then report the absence
    instead of calling the underlying scorer.

    This replaces the ``expected.expected if expected else ""`` coercion
    rather than sitting on top of it. That coercion is what made the
    defect invisible: an absent reference became an empty string, the
    underlying scorer performed a real comparison against nothing, and the
    caller read the result as a low score from their agent. Whitespace-only
    counts as empty for the same reason.

    A scorer that does not require a reference still receives ``""``,
    because the underlying call shape passes ``expected=`` unconditionally
    and those scorers demonstrably ignore it — their prompt templates
    never interpolate it.
    """
    value = expected.expected if expected is not None else ""
    if requires_expected and not value.strip():
        return None
    return value


# Internal helper function for creating autoevals adapters
# This is NOT exported - users should use convenience wrappers instead
def _create_autoevals_scorer(
    scorer_class: type,
    scorer_name: str,
    threshold: float,
    requires_expected: bool = False,
    **scorer_kwargs: Any,
) -> Callable:
    """Internal helper to create autoevals scorer adapters (sync).

    This is an internal implementation detail. Users should use the convenience
    wrapper functions like Levenshtein(), Factuality(), etc.

    Args:
        scorer_class: The autoevals class to wrap.
        scorer_name: Name reported on the returned Score.
        threshold: Value at or above which the Score passes.
        requires_expected: Whether the wrapped scorer performs a comparison
            against a reference value. When True and no usable reference is
            supplied, the wrapper reports the absence and never calls the
            underlying scorer. Declared per factory from what the wrapped
            class actually does with ``expected`` — not from its name.
        **scorer_kwargs: Forwarded to ``scorer_class``.
    """

    def scorer(
        result: TaskResult,
        expected: ExpectedResult | None = None,
    ) -> Score:
        """Adapted scorer function that returns Score."""
        try:
            # Import autoevals inside the function to handle ImportError
            import autoevals  # noqa: F401

            # Extract items from TaskResult and ExpectedResult
            # simple scorers implementing this wrapper do not take context
            # (input, instructions, etc.)
            output_str = result.output
            resolved = _resolve_expected(expected, requires_expected)
            if resolved is None:
                # Reported, not scored. Comparing against a substituted
                # empty string would return a real number the caller reads
                # as an agent regression, with nothing to distinguish it
                # from a genuine low score.
                return _missing_expected_score(scorer_name, result.output)
            expected_str = resolved

            # Instantiate the autoevals scorer
            autoevals_scorer = scorer_class(**scorer_kwargs)

            # Call the autoevals scorer with output, expected, and context
            # Pass context kwargs for scorers needing additional data
            # (input, instructions, etc.)
            # Fallback to no context if scorer doesn't accept extra kwargs
            if result.context:
                try:
                    ae_result = autoevals_scorer(
                        output=output_str, expected=expected_str, **result.context
                    )
                except TypeError:
                    # Heuristic scorer doesn't accept extra kwargs
                    # Call without context
                    ae_result = autoevals_scorer(
                        output=output_str, expected=expected_str
                    )
            else:
                # No context to pass - call directly
                ae_result = autoevals_scorer(output=output_str, expected=expected_str)

            # Normalize the score value
            normalized_value = _normalize_value(ae_result.score)

            # Determine passed based on threshold
            passed = normalized_value >= threshold

            # Handle metadata (may be None)
            metadata = ae_result.metadata if ae_result.metadata is not None else {}
            metadata["_threshold"] = threshold

            # Extract reasoning if available
            reasoning = getattr(ae_result, "rationale", None)

            return Score(
                name=scorer_name,
                value=normalized_value,
                passed=passed,
                metadata=metadata,
                reasoning=reasoning,
            )

        except ImportError as e:
            # Handle missing autoevals library
            return Score(
                name=scorer_name,
                value=0.0,
                passed=False,
                metadata={"error": f"autoevals not installed: {e}"},
                reasoning="Scorer failed due to missing autoevals library",
            )

        except Exception as e:
            # Handle any other errors
            return Score(
                name=scorer_name,
                value=0.0,
                passed=False,
                metadata={"error": str(e)},
                reasoning=f"Scorer failed with exception: {type(e).__name__}",
            )

    return scorer


def _create_autoevals_async_scorer(
    scorer_class: type,
    scorer_name: str,
    threshold: float,
    requires_expected: bool = False,
    **scorer_kwargs: Any,
) -> Callable:
    """Internal helper to create async autoevals scorer adapters (for LLM scorers).

    This is an internal implementation detail. Users should use the convenience
    wrapper functions like Factuality(), ClosedQA(), etc.

    Uses autoevals' eval_async() method for true async I/O instead of thread pool.

    Args:
        scorer_class: The autoevals class to wrap.
        scorer_name: Name reported on the returned Score.
        threshold: Value at or above which the Score passes.
        requires_expected: Whether the wrapped scorer performs a comparison
            against a reference value. For the LLM-judged scorers this is
            decided by whether the class's prompt template interpolates
            ``{{expected}}``; a template that does will otherwise be sent an
            empty reference and still return a confident verdict.
        **scorer_kwargs: Forwarded to ``scorer_class``.
    """

    async def scorer(
        result: TaskResult, expected: ExpectedResult | None = None
    ) -> Score:
        """Adapted async scorer function that returns Score."""
        try:
            # Import autoevals inside the function to handle ImportError
            import autoevals  # noqa: F401

            # Extract items from TaskResult
            output_str = result.output
            resolved = _resolve_expected(expected, requires_expected)
            if resolved is None:
                # Reported before the model call, not after. An LLM judge
                # handed an empty reference does not error — it returns a
                # confident verdict about nothing, and for `Battle` and
                # `Summary` that verdict is a *passing* one.
                return _missing_expected_score(scorer_name, result.output)
            expected_str = resolved

            # Instantiate the autoevals scorer
            autoevals_scorer = scorer_class(**scorer_kwargs)

            # Call the autoevals scorer's async method
            # Pass context kwargs for scorers needing additional data
            # (input, instructions, etc.)
            # Fallback to no context if scorer doesn't accept extra kwargs
            if result.context:
                try:
                    ae_result = await autoevals_scorer.eval_async(
                        output=output_str,
                        expected=expected_str,
                        **result.context,
                    )
                except TypeError:
                    # Heuristic scorer doesn't accept extra kwargs
                    # Call without context
                    ae_result = await autoevals_scorer.eval_async(
                        output=output_str,
                        expected=expected_str,
                    )
            else:
                # No context to pass - call directly
                ae_result = await autoevals_scorer.eval_async(
                    output=output_str,
                    expected=expected_str,
                )

            # Normalize the score value
            normalized_value = _normalize_value(ae_result.score)

            # Determine passed based on threshold
            passed = normalized_value >= threshold

            # Handle metadata (may be None)
            metadata = ae_result.metadata if ae_result.metadata is not None else {}
            metadata["_threshold"] = threshold

            # Extract reasoning if available
            reasoning = getattr(ae_result, "rationale", None)

            return Score(
                name=scorer_name,
                value=normalized_value,
                passed=passed,
                metadata=metadata,
                reasoning=reasoning,
            )

        except ImportError as e:
            # Handle missing autoevals library
            return Score(
                name=scorer_name,
                value=0.0,
                passed=False,
                metadata={"error": f"autoevals not installed: {e}"},
                reasoning="Scorer failed due to missing autoevals library",
            )

        except Exception as e:
            # Handle any other errors
            return Score(
                name=scorer_name,
                value=0.0,
                passed=False,
                metadata={"error": str(e)},
                reasoning=f"Scorer failed with exception: {type(e).__name__}",
            )

    return scorer


@scorer_registry.register("Levenshtein")
def Levenshtein(threshold: float = 1.0) -> Callable:  # noqa: N802
    """Convenience factory for autoevals Levenshtein distance scorer.

    String similarity scorer using Levenshtein (edit) distance. Returns
    normalized similarity score where 1.0 = exact match, 0.0 = no similarity.

    Default threshold is 1.0 (exact match required) because Levenshtein is
    typically used for strict output validation where minor differences matter.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 1.0 = exact match)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import Levenshtein
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Exact match required (default)
        >>> exact_match = Levenshtein()
        >>> result = exact_match(
        ...     TaskResult(output="hello"),
        ...     expected=ExpectedResult(expected="hello")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Allow 95%+ similarity
        >>> fuzzy_match = Levenshtein(threshold=0.95)
        >>> result = fuzzy_match(
        ...     TaskResult(output="hello"),
        ...     expected=ExpectedResult(expected="hallo")
        ... )
        >>> print(result.passed)
        True
    """
    try:
        import autoevals.string

        return _create_autoevals_scorer(
            autoevals.string.Levenshtein,
            scorer_name="Levenshtein",
            threshold=threshold,
            # Edit distance to a reference string. With no reference the
            # distance is the output's own length, which normalises to a
            # plausible similarity number about nothing.
            requires_expected=True,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="Levenshtein",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("Factuality")
def Factuality(  # noqa: N802
    threshold: float = 0.7,
    model: str = "gpt-4o-mini",
    temperature: float | None = None,
    use_cot: bool | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals Factuality LLM-based scorer.

    LLM-based scorer that evaluates whether the output is factually consistent
    with the expected answer. Uses an LLM to assess factual accuracy.

    Default threshold is 0.7 (70% confidence) because Factuality scoring is
    probabilistic and some uncertainty is acceptable for nuanced factual claims.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.7)
        model: LLM model to use (default: "gpt-4o-mini")
        temperature: Randomness control (0=deterministic, 1=creative, default: 0)
        use_cot: Enable chain-of-thought reasoning (default: True)
        max_tokens: Maximum tokens in response (default: 512)
        client: OpenAI client (default: uses global from autoevals.init())
        api_key: OpenAI API key (deprecated, use client instead)
        base_url: Custom API base URL (for Ollama/custom endpoints)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import Factuality
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration
        >>> factuality = Factuality()
        >>> result = factuality(
        ...     TaskResult(output="Paris is the capital of France"),
        ...     expected=ExpectedResult(expected="Paris")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Stricter threshold with GPT-4
        >>> strict_factuality = Factuality(threshold=0.9, model="gpt-4")
    """
    try:
        import autoevals.llm

        scorer_kwargs = _build_scorer_kwargs(
            model, temperature, use_cot, max_tokens, client, api_key, base_url
        )
        return _create_autoevals_async_scorer(
            autoevals.llm.Factuality,
            scorer_name="Factuality",
            threshold=threshold,
            # factuality.yaml interpolates {{expected}} as the reference facts.
            requires_expected=True,
            **scorer_kwargs,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="Factuality",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("ClosedQA")
def ClosedQA(  # noqa: N802
    threshold: float = 0.8,
    model: str = "gpt-4o-mini",
    temperature: float | None = None,
    use_cot: bool | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals ClosedQA LLM-based scorer.

    LLM-based scorer for closed-ended question answering. Evaluates whether the
    output correctly answers a multiple-choice or short-answer question against
    the expected answer.

    Default threshold is 0.8 (80% confidence) because ClosedQA typically has
    clearer right/wrong answers compared to open-ended tasks.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.8)
        model: LLM model to use (default: "gpt-4o-mini")
        temperature: Randomness control (0=deterministic, 1=creative, default: 0)
        use_cot: Enable chain-of-thought reasoning (default: True)
        max_tokens: Maximum tokens in response (default: 512)
        client: OpenAI client (default: uses global from autoevals.init())
        api_key: OpenAI API key (deprecated, use client instead)
        base_url: Custom API base URL (for Ollama/custom endpoints)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import ClosedQA
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration
        >>> closedqa = ClosedQA()
        >>> result = closedqa(
        ...     TaskResult(
        ...         output="B",
        ...         context={"input": "What is 2+2? A) 3 B) 4 C) 5"}
        ...     ),
        ...     expected=ExpectedResult(expected="B")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Custom threshold
        >>> strict_qa = ClosedQA(threshold=0.95, model="gpt-4")
    """
    try:
        import autoevals.llm

        scorer_kwargs = _build_scorer_kwargs(
            model, temperature, use_cot, max_tokens, client, api_key, base_url
        )
        return _create_autoevals_async_scorer(
            autoevals.llm.ClosedQA,
            scorer_name="ClosedQA",
            threshold=threshold,
            # closed_q_a.yaml interpolates {{criteria}}/{{input}}/{{output}} only —
            # the model answers from its own knowledge, so a reference is not
            # part of the judgement.
            requires_expected=False,
            **scorer_kwargs,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="ClosedQA",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("Humor")
def Humor(  # noqa: N802
    threshold: float = 0.6,
    model: str = "gpt-4o-mini",
    temperature: float | None = None,
    use_cot: bool | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals Humor LLM-based scorer.

    LLM-based scorer that evaluates whether the output is humorous or funny.
    Uses an LLM to assess humor quality and comedic timing.

    Default threshold is 0.6 (60% confidence) because humor is subjective and
    what's funny varies by context and audience. Lower threshold allows for
    different styles of humor.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.6)
        model: LLM model to use (default: "gpt-4o-mini")
        temperature: Randomness control (0=deterministic, 1=creative, default: 0)
        use_cot: Enable chain-of-thought reasoning (default: True)
        max_tokens: Maximum tokens in response (default: 512)
        client: OpenAI client (default: uses global from autoevals.init())
        api_key: OpenAI API key (deprecated, use client instead)
        base_url: Custom API base URL (for Ollama/custom endpoints)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import Humor
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration
        >>> humor = Humor()
        >>> result = humor(
        ...     TaskResult(
        ...         output="Why did the chicken cross the road? "
        ...                "To get to the other side!"
        ...     ),
        ...     expected=ExpectedResult(expected="")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Stricter threshold
        >>> strict_humor = Humor(threshold=0.8, model="gpt-4")
    """
    try:
        import autoevals.llm

        scorer_kwargs = _build_scorer_kwargs(
            model, temperature, use_cot, max_tokens, client, api_key, base_url
        )
        return _create_autoevals_async_scorer(
            autoevals.llm.Humor,
            scorer_name="Humor",
            threshold=threshold,
            # humor.yaml interpolates {{output}} only.
            requires_expected=False,
            **scorer_kwargs,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="Humor",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("Battle")
def Battle(  # noqa: N802
    threshold: float = 0.7,
    model: str = "gpt-4o-mini",
    temperature: float | None = None,
    use_cot: bool | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals Battle LLM-based scorer.

    LLM-based scorer that compares two outputs head-to-head and determines
    which one is better. The output is compared against the expected output
    to assess relative quality.

    Default threshold is 0.7 (70% confidence) because comparative judgments
    typically have more confidence than absolute quality assessments.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.7)
        model: LLM model to use (default: "gpt-4o-mini")
        temperature: Randomness control (0=deterministic, 1=creative, default: 0)
        use_cot: Enable chain-of-thought reasoning (default: True)
        max_tokens: Maximum tokens in response (default: 512)
        client: OpenAI client (default: uses global from autoevals.init())
        api_key: OpenAI API key (deprecated, use client instead)
        base_url: Custom API base URL (for Ollama/custom endpoints)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import Battle
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration
        >>> battle = Battle()
        >>> result = battle(
        ...     TaskResult(output="The quick brown fox jumps over the lazy dog."),
        ...     expected=ExpectedResult(expected="A fast fox leaps over a sleepy dog.")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Custom threshold with GPT-4
        >>> strict_battle = Battle(threshold=0.85, model="gpt-4")
    """
    try:
        import autoevals.llm

        scorer_kwargs = _build_scorer_kwargs(
            model, temperature, use_cot, max_tokens, client, api_key, base_url
        )
        return _create_autoevals_async_scorer(
            autoevals.llm.Battle,
            scorer_name="Battle",
            threshold=threshold,
            # battle.yaml interpolates {{expected}} as Response 2. Despite the
            # name reading reference-free, the judgement is a comparison: with
            # an empty Response 2, "is the first response better?" resolves to
            # "Yes" and the scorer reports 1.0 — a false pass, the worst
            # failure mode of the three.
            requires_expected=True,
            **scorer_kwargs,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="Battle",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("Security")
def Security(  # noqa: N802
    threshold: float = 0.8,
    model: str = "gpt-4o-mini",
    temperature: float | None = None,
    use_cot: bool | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals Security LLM-based scorer.

    LLM-based scorer that evaluates whether the output contains security
    vulnerabilities, sensitive information leaks, or unsafe recommendations.
    Higher scores indicate more secure outputs.

    Default threshold is 0.8 (80% confidence) because security is critical
    and we want high confidence before marking content as secure.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.8)
        model: LLM model to use (default: "gpt-4o-mini")
        temperature: Randomness control (0=deterministic, 1=creative, default: 0)
        use_cot: Enable chain-of-thought reasoning (default: True)
        max_tokens: Maximum tokens in response (default: 512)
        client: OpenAI client (default: uses global from autoevals.init())
        api_key: OpenAI API key (deprecated, use client instead)
        base_url: Custom API base URL (for Ollama/custom endpoints)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import Security
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration
        >>> security = Security()
        >>> result = security(
        ...     TaskResult(output="Use environment variables for API keys."),
        ...     expected=ExpectedResult(expected="")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Very strict security check
        >>> strict_security = Security(threshold=0.95, model="gpt-4")
    """
    try:
        import autoevals.llm

        scorer_kwargs = _build_scorer_kwargs(
            model, temperature, use_cot, max_tokens, client, api_key, base_url
        )
        return _create_autoevals_async_scorer(
            autoevals.llm.Security,
            scorer_name="Security",
            threshold=threshold,
            # security.yaml interpolates {{output}} only.
            requires_expected=False,
            **scorer_kwargs,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="Security",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("Summary")
def Summary(  # noqa: N802
    threshold: float = 0.7,
    model: str = "gpt-4o-mini",
    temperature: float | None = None,
    use_cot: bool | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals Summary LLM-based scorer.

    LLM-based scorer that evaluates the quality of a summary against the
    original text or expected summary. Assesses whether key information is
    preserved and the summary is concise.

    Default threshold is 0.7 (70% confidence) because good summaries can
    vary in style and emphasis while still being valid.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.7)
        model: LLM model to use (default: "gpt-4o-mini")
        temperature: Randomness control (0=deterministic, 1=creative, default: 0)
        use_cot: Enable chain-of-thought reasoning (default: True)
        max_tokens: Maximum tokens in response (default: 512)
        client: OpenAI client (default: uses global from autoevals.init())
        api_key: OpenAI API key (deprecated, use client instead)
        base_url: Custom API base URL (for Ollama/custom endpoints)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import Summary
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration
        >>> summary = Summary()
        >>> result = summary(
        ...     TaskResult(
        ...         output="Climate change is accelerating globally.",
        ...         context={"input": "Long article about climate change..."}
        ...     ),
        ...     expected=ExpectedResult(expected="Global warming is increasing.")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Stricter summary evaluation
        >>> strict_summary = Summary(threshold=0.85, model="gpt-4")
    """
    try:
        import autoevals.llm

        scorer_kwargs = _build_scorer_kwargs(
            model, temperature, use_cot, max_tokens, client, api_key, base_url
        )
        return _create_autoevals_async_scorer(
            autoevals.llm.Summary,
            scorer_name="Summary",
            threshold=threshold,
            # summary.yaml interpolates {{expected}} as the *expert summary* and
            # asks which of A/B better describes the text. With A empty the
            # judge picks B, scoring the candidate 1.0 against nothing.
            requires_expected=True,
            **scorer_kwargs,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="Summary",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("Translation")
def Translation(  # noqa: N802
    threshold: float = 0.8,
    model: str = "gpt-4o-mini",
    temperature: float | None = None,
    use_cot: bool | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals Translation LLM-based scorer.

    LLM-based scorer that evaluates the quality of translations by comparing
    the output translation against an expected translation or assessing
    translation accuracy.

    Default threshold is 0.8 (80% confidence) because translation quality
    should be high to preserve meaning across languages.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.8)
        model: LLM model to use (default: "gpt-4o-mini")
        temperature: Randomness control (0=deterministic, 1=creative, default: 0)
        use_cot: Enable chain-of-thought reasoning (default: True)
        max_tokens: Maximum tokens in response (default: 512)
        client: OpenAI client (default: uses global from autoevals.init())
        api_key: OpenAI API key (deprecated, use client instead)
        base_url: Custom API base URL (for Ollama/custom endpoints)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import Translation
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration
        >>> translation = Translation()
        >>> result = translation(
        ...     TaskResult(output="Bonjour le monde"),
        ...     expected=ExpectedResult(expected="Hello world")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Very strict translation check
        >>> strict_translation = Translation(threshold=0.95, model="gpt-4")
    """
    try:
        import autoevals.llm

        scorer_kwargs = _build_scorer_kwargs(
            model, temperature, use_cot, max_tokens, client, api_key, base_url
        )
        return _create_autoevals_async_scorer(
            autoevals.llm.Translation,
            scorer_name="Translation",
            threshold=threshold,
            # translation.yaml interpolates {{expected}} as the reference translation.
            requires_expected=True,
            **scorer_kwargs,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="Translation",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("ExactMatch")
def ExactMatch(threshold: float = 1.0) -> Callable:  # noqa: N802
    """Convenience factory for autoevals ExactMatch scorer.

    Exact value equality scorer with JSON normalization. Returns 1.0 for
    exact matches, 0.0 for any difference.

    For primitive values (strings, numbers):
    - Case-sensitive and whitespace-sensitive comparison
    - "hello" != "Hello", "hello " != "hello"
    - Numbers are converted to strings: 123 == "123"

    For JSON objects and arrays:
    - Automatically serializes dicts/lists to JSON strings for comparison
    - Normalizes JSON string formatting: '{"a":1}' == '{"a": 1}'
    - ⚠️ ORDER MATTERS: {"a": 1, "b": 2} != {"b": 2, "a": 1}
    - For order-independent JSON comparison, use JSONDiff instead

    Default threshold is 1.0 (exact match required) since this scorer is
    binary - it either matches exactly or not at all.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 1.0 = exact match)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import ExactMatch
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # String comparison (case-sensitive)
        >>> exact = ExactMatch()
        >>> result = exact(
        ...     TaskResult(output="hello"),
        ...     expected=ExpectedResult(expected="hello")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> result = exact(
        ...     TaskResult(output="hello"),
        ...     expected=ExpectedResult(expected="Hello")
        ... )
        >>> print(result.passed)
        False
        >>>
        >>> # Number to string conversion
        >>> result = exact(
        ...     TaskResult(output="123"),
        ...     expected=ExpectedResult(expected="123")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # JSON: whitespace normalized but order matters
        >>> result = exact(
        ...     TaskResult(output='{"a":1}'),
        ...     expected=ExpectedResult(expected='{"a": 1}')
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # JSON: order matters!
        >>> result = exact(
        ...     TaskResult(output='{"a": 1, "b": 2}'),
        ...     expected=ExpectedResult(expected='{"b": 2, "a": 1}')
        ... )
        >>> print(result.passed)
        False
    """
    try:
        import autoevals.value

        return _create_autoevals_scorer(
            autoevals.value.ExactMatch,
            scorer_name="ExactMatch",
            threshold=threshold,
            # Equality against a reference. The coercion this replaces made
            # an empty output *match* an absent reference and report 1.0 —
            # a passing verdict from no comparison at all.
            requires_expected=True,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="ExactMatch",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("NumericDiff")
def NumericDiff(threshold: float = 0.9) -> Callable:  # noqa: N802
    """Convenience factory for autoevals NumericDiff scorer with type conversion.

    Numeric difference scorer that compares numeric values. Automatically converts
    string outputs to floats before comparison.

    ⚠️ IMPORTANT: This convenience function is REQUIRED for NumericDiff because:
    - Agent-evals tasks return strings (our interface requirement)
    - NumericDiff expects numeric types (int/float)
    - This function handles automatic string→float conversion
    - Must use this wrapper (not the internal helper) for proper type conversion

    The score is calculated as:
        1.0 - abs(output - expected) / (abs(output) + abs(expected))
    Special case: When both values are 0, score = 1.0
    Score of 1.0 = exact match, lower scores = larger differences.

    Default threshold is 0.9 (90% similarity) since numeric comparisons often
    need high precision.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.9)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import NumericDiff
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration
        >>> numeric = NumericDiff()
        >>> result = numeric(
        ...     TaskResult(output="42.0"),
        ...     expected=ExpectedResult(expected="42.0")  # Strings!
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Small difference still passes with threshold 0.9
        >>> result = numeric(
        ...     TaskResult(output="100"),
        ...     expected=ExpectedResult(expected="99")
        ... )
        >>> print(result.value, result.passed)
        0.99 True
        >>>
        >>> # Large difference fails
        >>> result = numeric(
        ...     TaskResult(output="100"),
        ...     expected=ExpectedResult(expected="50")
        ... )
        >>> print(result.passed)
        False

    Raises:
        Returns error Score if output/expected cannot be converted to numbers.
    """
    try:
        import autoevals.number

        def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
            """NumericDiff scorer with automatic type conversion."""
            try:
                # Check if expected is None
                if expected is None:
                    return Score(
                        name="NumericDiff",
                        value=0.0,
                        passed=False,
                        metadata={
                            "error": "Expected value is required for NumericDiff",
                            "output": str(result.output),
                            "expected": None,
                        },
                        reasoning=(
                            "NumericDiff requires an expected value for comparison"
                        ),
                    )

                # Convert string outputs to floats
                numeric_output = float(result.output)
                numeric_expected = float(expected.expected)
                # Create and call the autoevals NumericDiff scorer
                autoevals_scorer = autoevals.number.NumericDiff()
                ae_result = autoevals_scorer(
                    output=numeric_output, expected=numeric_expected
                )
                score_value = ae_result.score if ae_result.score is not None else 0.0

                passed = score_value >= threshold

                # Handle metadata
                metadata = ae_result.metadata if ae_result.metadata is not None else {}
                metadata.update(
                    {
                        "original_output": result.output,
                        "original_expected": expected.expected,
                        "converted_output": numeric_output,
                        "converted_expected": numeric_expected,
                        "_threshold": threshold,
                    }
                )

                return Score(
                    name="NumericDiff",
                    value=score_value,
                    passed=passed,
                    metadata=metadata,
                    reasoning=getattr(ae_result, "rationale", None),
                )

            except (ValueError, TypeError) as e:
                # Handle conversion failures
                return Score(
                    name="NumericDiff",
                    value=0.0,
                    passed=False,
                    metadata={
                        "error": f"Failed to convert to numeric: {e}",
                        "output": str(result.output),
                        "expected": str(expected),
                    },
                    reasoning=(
                        "NumericDiff requires output and expected to be "
                        "numeric or convertible to numeric"
                    ),
                )

        return scorer

    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="NumericDiff",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("EmbeddingSimilarity")
def EmbeddingSimilarity(  # noqa: N802
    threshold: float = 0.8,
    model: str = "text-embedding-ada-002",
    prefix: str = "",
    api_key: str | None = None,
    base_url: str | None = None,
    client: Any | None = None,
) -> Callable:
    """Convenience factory for autoevals EmbeddingSimilarity scorer.

    Semantic similarity scorer using text embeddings. Compares the semantic
    meaning of output vs expected using embedding vectors, independent of
    exact word matching.

    Default threshold is 0.8 (80% similarity) because semantic similarity
    should be relatively high for outputs to be considered equivalent in meaning.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.8)
        model: Embedding model to use (default: "text-embedding-ada-002")
        prefix: Optional text to prepend to inputs for domain context
        api_key: Optional API key for embedding service
        base_url: Optional base URL for embedding service (e.g., Ollama)
        client: Optional OpenAI/AsyncOpenAI client instance

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import EmbeddingSimilarity
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration (OpenAI)
        >>> embedding = EmbeddingSimilarity()
        >>> result = embedding(
        ...     TaskResult(output="The dog ran quickly"),
        ...     expected=ExpectedResult(expected="The canine sprinted fast")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # With Ollama behind TLS
        >>> ollama_embedding = EmbeddingSimilarity(
        ...     model="embeddinggemma:latest",
        ...     base_url="https://ollama.example.com/v1",
        ...     api_key="ollama"
        ... )
        >>>
        >>> # With custom OpenAI client
        >>> from openai import AsyncOpenAI
        >>> custom_client = AsyncOpenAI(api_key="sk-...")
        >>> embedding = EmbeddingSimilarity(client=custom_client)
    """
    # Validated here rather than in _build_scorer_kwargs: this factory passes
    # base_url straight to the creator and never calls that helper.
    if base_url is not None:
        validate_secure_transport("base_url", base_url)

    try:
        import autoevals.string

        return _create_autoevals_async_scorer(
            autoevals.string.EmbeddingSimilarity,
            scorer_name="EmbeddingSimilarity",
            threshold=threshold,
            # Cosine similarity against a reference embedding. An empty
            # reference still embeds, so the comparison silently returns a
            # similarity to the empty string.
            requires_expected=True,
            model=model,
            prefix=prefix,
            expected_min=threshold,
            api_key=api_key,
            base_url=base_url,
            client=client,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="EmbeddingSimilarity",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("ListContains")
def ListContains(  # noqa: N802
    threshold: float = 1.0,
    pairwise_scorer: Any | None = None,
    allow_extra_entities: bool = True,
) -> Callable:
    """Convenience factory for list containment checker using autoevals.

    Checks if all expected items are present in the output list. By default,
    allows extra items in the output (allow_extra_entities=True).

    Uses autoevals.list.ListContains scorer which compares two lists using
    pairwise similarity scoring (via Linear Sum Assignment) to find the best
    matching pairs between output and expected items.

    **Async Compatibility:**
    This scorer is sync but works seamlessly in async contexts. The eval runner
    automatically wraps sync scorers with asyncio.to_thread() for non-blocking
    execution. You don't need to worry about async/sync - it just works!

    **Scoring:**
    - Returns 1.0 if all expected items are present (extra items OK by default)
    - Returns fractional score (0.0-1.0) based on % of expected items found
    - Returns 0.0 if no expected items are found

    **Interface:**
    - Output: List of items (can be list object or JSON string)
    - Expected: List of items that should be present
    - Returns: Score based on how many expected items are found

    **Pairwise Scoring:**
    The pairwise_scorer parameter allows you to customize how individual items
    are compared. By default, uses Levenshtein distance for string similarity.
    You can provide any autoevals scorer instance for custom comparison logic.

    Default threshold is 1.0 (100% match) because list containment is typically
    used for strict validation where all items must be present.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 1.0)
        pairwise_scorer: Optional autoevals scorer for comparing list items.
                        If None, uses Levenshtein distance (default: None).
                        Must be an autoevals scorer instance, not our wrapped scorer.
        allow_extra_entities: If True, extra items in output don't penalize score.
                             If False, penalizes based on output/expected ratio.
                             (default: True)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import ListContains
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # All expected items present (with extras) - passes
        >>> list_check = ListContains(threshold=1.0)
        >>> result = list_check(
        ...     TaskResult(output='["python", "java", "javascript", "rust"]'),
        ...     expected=ExpectedResult(expected='["python", "java", "javascript"]')
        ... )
        >>> print(result.value, result.passed)
        1.0 True
        >>>
        >>> # Missing expected items - fails
        >>> result = list_check(
        ...     TaskResult(output='["python", "java"]'),
        ...     expected=ExpectedResult(expected='["python", "java", "javascript"]')
        ... )
        >>> print(result.value, result.passed)
        0.67 False
        >>>
        >>> # All expected items + extras - passes (allow_extra_entities=True)
        >>> result = list_check(
        ...     TaskResult(output='["python", "java", "javascript", "go", "ruby"]'),
        ...     expected=ExpectedResult(expected='["python", "java", "javascript"]')
        ... )
        >>> print(result.value, result.passed)
        1.0 True
        >>>
        >>> # Custom pairwise scorer for semantic similarity
        >>> from autoevals.string import EmbeddingSimilarity
        >>> semantic_list = ListContains(
        ...     threshold=0.9,
        ...     pairwise_scorer=EmbeddingSimilarity()
        ... )
        >>> result = semantic_list(
        ...     TaskResult(output='["dog", "cat", "bird"]'),
        ...     expected=ExpectedResult(expected='["canine", "feline"]')
        ... )
        >>> # Uses semantic similarity to match dog~canine, cat~feline
        >>>
        >>> # Works in async contexts automatically via run_eval_async()
        >>> async def evaluate():
        ...     from agent_evals import run_eval_async
        ...     result = await run_eval_async(
        ...         task=lambda x: x,
        ...         dataset=[{"input": ["a", "b"],
        ...                   "expected": ExpectedResult(expected='["a", "b"]')}],
        ...         scorers=[list_check],
        ...         platform="local",
        ...         config={"project": "test", "experiment": "test"}
        ...     )

    Raises:
        Returns error Score if scorer fails.
    """
    try:
        from autoevals.list import ListContains as AutoevalsListContains

        def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
            """List containment checker using autoevals."""
            try:
                # Check if expected is None
                if expected is None:
                    return Score(
                        name="ListContains",
                        value=0.0,
                        passed=False,
                        metadata={
                            "error": "Expected value is required for ListContains"
                        },
                        reasoning=(
                            "ListContains requires an expected value for comparison"
                        ),
                    )

                # Parse string inputs to lists if needed
                output_list = result.output
                if isinstance(result.output, str):
                    import json

                    try:
                        output_list = json.loads(result.output)
                    except json.JSONDecodeError:
                        output_list = _safe_literal_eval(result.output)

                expected_list = expected.expected
                if isinstance(expected_list, str):
                    import json

                    try:
                        expected_list = json.loads(expected_list)
                    except json.JSONDecodeError:
                        expected_list = _safe_literal_eval(expected_list)

                # Instantiate autoevals scorer per-call
                autoevals_scorer = AutoevalsListContains(
                    pairwise_scorer=pairwise_scorer,
                    allow_extra_entities=allow_extra_entities,
                )

                # Call autoevals scorer
                ae_score = autoevals_scorer(output=output_list, expected=expected_list)

                # Normalize the score value
                normalized_value = _normalize_value(
                    ae_score.score if ae_score.score is not None else 0.0
                )

                # Determine passed based on threshold
                passed = normalized_value >= threshold

                # Handle metadata
                metadata = ae_score.metadata if ae_score.metadata is not None else {}
                scorer_name = (
                    type(pairwise_scorer).__name__ if pairwise_scorer else "Levenshtein"
                )
                metadata.update(
                    {
                        "allow_extra_entities": allow_extra_entities,
                        "pairwise_scorer": scorer_name,
                        "_threshold": threshold,
                    }
                )

                # Extract reasoning
                reasoning = getattr(ae_score, "rationale", None)
                if reasoning is None:
                    items_msg = f"Found {normalized_value:.0%} of expected items. "
                    extras_msg = (
                        "Extra items allowed."
                        if allow_extra_entities
                        else "Extra items penalize."
                    )
                    reasoning = items_msg + extras_msg

                return Score(
                    name="ListContains",
                    value=normalized_value,
                    passed=passed,
                    metadata=metadata,
                    reasoning=reasoning,
                )

            except Exception as e:
                # Handle any errors
                exc_name = type(e).__name__
                return Score(
                    name="ListContains",
                    value=0.0,
                    passed=False,
                    metadata={"error": str(e)},
                    reasoning=f"Scorer failed with exception: {exc_name}",
                )

        return scorer

    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="ListContains",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("Sql")
def Sql(  # noqa: N802
    threshold: float = 0.8,
    model: str = "gpt-4o-mini",
    temperature: float | None = None,
    use_cot: bool | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals Sql LLM-based scorer.

    LLM-based scorer that evaluates SQL query quality, correctness, and
    adherence to best practices. Useful for validating generated SQL queries.

    Default threshold is 0.8 (80% confidence) because SQL correctness is
    important but there can be multiple valid ways to write the same query.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.8)
        model: LLM model to use (default: "gpt-4o-mini")
        temperature: Randomness control (0=deterministic, 1=creative, default: 0)
        use_cot: Enable chain-of-thought reasoning (default: True)
        max_tokens: Maximum tokens in response (default: 512)
        client: OpenAI client (default: uses global from autoevals.init())
        api_key: OpenAI API key (deprecated, use client instead)
        base_url: Custom API base URL (for Ollama/custom endpoints)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import Sql
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration
        >>> sql_scorer = Sql()
        >>> result = sql_scorer(
        ...     TaskResult(output="SELECT * FROM users WHERE id = 1"),
        ...     expected=ExpectedResult(
        ...         expected="SELECT id, name FROM users WHERE id = 1"
        ...     )
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Stricter evaluation
        >>> strict_sql = Sql(threshold=0.95, model="gpt-4")
    """
    try:
        import autoevals.llm

        scorer_kwargs = _build_scorer_kwargs(
            model, temperature, use_cot, max_tokens, client, api_key, base_url
        )
        return _create_autoevals_async_scorer(
            autoevals.llm.Sql,
            scorer_name="Sql",
            threshold=threshold,
            # sql.yaml interpolates {{expected}} as the reference query.
            requires_expected=True,
            **scorer_kwargs,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="Sql",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("Possible")
def Possible(  # noqa: N802
    threshold: float = 0.7,
    model: str = "gpt-4o-mini",
    temperature: float | None = None,
    use_cot: bool | None = None,
    max_tokens: int | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals Possible LLM-based scorer.

    LLM-based scorer that evaluates whether an output is possible, plausible,
    or reasonable given the input context. Useful for checking if generated
    content makes logical sense.

    Default threshold is 0.7 (70% confidence) because plausibility judgments
    can be subjective and context-dependent.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.7)
        model: LLM model to use (default: "gpt-4o-mini")
        temperature: Randomness control (0=deterministic, 1=creative, default: 0)
        use_cot: Enable chain-of-thought reasoning (default: True)
        max_tokens: Maximum tokens in response (default: 512)
        client: OpenAI client (default: uses global from autoevals.init())
        api_key: OpenAI API key (deprecated, use client instead)
        base_url: Custom API base URL (for Ollama/custom endpoints)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import Possible
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration
        >>> possible = Possible()
        >>> result = possible(
        ...     TaskResult(
        ...         output="Paris",
        ...         context={"input": "What's the capital of France?"}
        ...     ),
        ...     expected=ExpectedResult(expected="")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Stricter plausibility check
        >>> strict_possible = Possible(threshold=0.9, model="gpt-4")
    """
    try:
        import autoevals.llm

        scorer_kwargs = _build_scorer_kwargs(
            model, temperature, use_cot, max_tokens, client, api_key, base_url
        )
        return _create_autoevals_async_scorer(
            autoevals.llm.Possible,
            scorer_name="Possible",
            threshold=threshold,
            # possible.yaml interpolates {{input}}/{{output}} only — it judges
            # whether a solution was achievable, not whether it matched one.
            requires_expected=False,
            **scorer_kwargs,
        )
    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="Possible",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("LLMClassifier")
def LLMClassifier(  # noqa: N802
    threshold: float = 0.8,
    model: str = "gpt-4o-mini",
    categories: dict[str, str] | None = None,
    use_cot: bool = True,
    temperature: float = 0.0,
    max_tokens: int = 512,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals LLMClassifier scorer.

    General-purpose LLM-based classification scorer. Classifies outputs into
    custom categories based on the categories dict.

    Default threshold is 0.8 (80% confidence) because classification decisions
    should have high confidence before marking as passed.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 0.8)
        model: LLM model to use (default: "gpt-4o-mini")
        categories: Dict mapping category names to descriptions
            (e.g., {"positive": "Satisfaction", "negative": "Dissatisfaction"})
        use_cot: Enable chain of thought reasoning (default: True)
        temperature: Controls randomness 0-1 (default: 0.0)
        max_tokens: Maximum tokens to generate (default: 512)
        client: Optional OpenAI client instance
        api_key: Optional OpenAI API key
        base_url: Optional OpenAI API base URL

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import LLMClassifier
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Custom classification
        >>> categories = {
        ...     "positive": "Expresses satisfaction or approval",
        ...     "negative": "Expresses dissatisfaction or disapproval",
        ...     "neutral": "Factual statement without strong sentiment"
        ... }
        >>> sentiment = LLMClassifier(threshold=0.8, categories=categories)
        >>> result = sentiment(
        ...     TaskResult(output="This product is amazing!"),
        ...     expected=ExpectedResult(expected="positive")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Stricter classification
        >>> strict_classifier = LLMClassifier(
        ...     threshold=0.95,
        ...     model="gpt-4",
        ...     categories=categories
        ... )
    """
    # Validated in the factory body, not in the inner scorer closure below.
    # The closure consumes base_url once per scored example, so validating
    # there would turn a configuration error into a mid-run failure repeated
    # on every example. This factory builds its client kwargs inline and never
    # calls _build_scorer_kwargs.
    if base_url is not None:
        validate_secure_transport("base_url", base_url)

    try:
        import autoevals.llm

        if categories is None:
            categories = {
                "positive": "Positive sentiment",
                "negative": "Negative sentiment",
                "neutral": "Neutral sentiment",
            }

        # Build prompt template
        category_descriptions = "\n".join(
            [f"- {name}: {desc}" for name, desc in categories.items()]
        )
        prompt_template = f"""Classify the following text into one of these categories:

{category_descriptions}

Text: {{{{output}}}}

Select the most appropriate category."""

        async def scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            # Build choice_scores dynamically based on expected category
            if expected and expected.expected in categories:
                # Assign 1.0 to expected category, 0.0 to others
                dynamic_choice_scores = {
                    name: 1.0 if name == expected.expected else 0.0
                    for name in categories
                }
            else:
                # If no expected, treat all categories equally
                dynamic_choice_scores = dict.fromkeys(categories, 1.0)

            # Build client kwargs
            client_kwargs: dict[str, Any] = {}
            if client is not None:
                client_kwargs["client"] = client
            if api_key is not None:
                client_kwargs["api_key"] = api_key
            if base_url is not None:
                client_kwargs["base_url"] = base_url

            # Create classifier instance
            classifier = autoevals.llm.LLMClassifier(
                name="LLMClassifier",
                prompt_template=prompt_template,
                choice_scores=dynamic_choice_scores,
                model=model,
                use_cot=use_cot,
                max_tokens=max_tokens,
                temperature=temperature,
                **client_kwargs,
            )

            # Evaluate (using async method)
            try:
                ae_result = await classifier.eval_async(
                    output=result.output,
                    expected=expected.expected if expected else "",
                )
                # Handle case where ae_result.score might be None
                if ae_result.score is None:
                    normalized_value = 0.0
                else:
                    normalized_value = _normalize_value(ae_result.score)

                return Score(
                    name="LLMClassifier",
                    value=normalized_value,
                    passed=normalized_value >= threshold,
                    metadata={**(ae_result.metadata or {}), "_threshold": threshold},
                    reasoning=getattr(ae_result, "rationale", None),
                )
            except Exception as e:
                return Score(
                    name="LLMClassifier",
                    value=0.0,
                    passed=False,
                    metadata={"error": str(e)},
                    reasoning=f"Classification failed: {e}",
                )

        return scorer

    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="LLMClassifier",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("JSONDiff")
def JSONDiff(  # noqa: N802
    threshold: float = 1.0,
    string_scorer: Any | None = None,
    number_scorer: Any | None = None,
    preserve_strings: bool = False,
) -> Callable:
    """Convenience factory for autoevals JSONDiff scorer.

    Compare JSON objects for structural and content similarity. Recursively
    compares nested structures using configurable scorers for strings and numbers.

    Default threshold is 1.0 (exact match) because JSON comparison is typically
    used for structured output validation where precision matters.

    ⚠️ IMPORTANT: The string_scorer and number_scorer parameters accept autoevals
    scorer instances (not our wrapped scorers). If not provided, uses autoevals
    defaults (Levenshtein for strings, NumericDiff for numbers).

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 1.0)
        string_scorer: Optional autoevals scorer for string comparisons
                      (default: autoevals.string.Levenshtein())
        number_scorer: Optional autoevals scorer for number comparisons
                      (default: autoevals.number.NumericDiff())
        preserve_strings: Don't attempt to parse strings as JSON (default: False)

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import JSONDiff
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration (exact structural match)
        >>> json_diff = JSONDiff()
        >>> result = json_diff(
        ...     TaskResult(output='{"name": "John", "age": 30}'),
        ...     expected=ExpectedResult(expected='{"name": "John", "age": 30}')
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Allow some fuzzy matching (90% similarity)
        >>> fuzzy_json = JSONDiff(threshold=0.90)
        >>> result = fuzzy_json(
        ...     TaskResult(output='{"name": "Jon", "age": 30}'),  # Minor typo
        ...     expected=ExpectedResult(expected='{"name": "John", "age": 30}')
        ... )
        >>> print(result.value >= 0.90)  # High similarity despite typo
        True
        >>>
        >>> # Semantic comparison (conceptual similarity for strings)
        >>> # Use EmbeddingSimilarity to match synonyms/paraphrases
        >>> from autoevals.string import EmbeddingSimilarity
        >>> import os
        >>> os.environ["OPENAI_BASE_URL"] = "https://ollama.example.com/v1"
        >>> os.environ["OPENAI_API_KEY"] = "ollama"
        >>>
        >>> semantic_json = JSONDiff(
        ...     string_scorer=EmbeddingSimilarity(
        ...         model="embeddinggemma:latest",
        ...         api_key=os.environ["OPENAI_API_KEY"],
        ...         base_url=os.environ["OPENAI_BASE_URL"],
        ...     ),
        ...     threshold=0.25  # Lower threshold for semantic matching
        ... )
        >>> result = semantic_json(
        ...     TaskResult(output='{"status": "completed"}'),
        ...     expected=ExpectedResult(expected='{"status": "done"}')
        ... )  # Semantic match!
        >>> # Note: Semantic scores typically 0.2-0.5, not 0.8+
        >>> # Use lower thresholds (0.2-0.4) for semantic matching
        >>>
        >>> # Preserve string literals (don't parse nested JSON strings)
        >>> literal_json = JSONDiff(preserve_strings=True)
        >>> result = literal_json(
        ...     TaskResult(output='{"config": "{\\"port\\": 8080}"}'),
        ...     expected=ExpectedResult(expected='{"config": "{\\"port\\": 8080}"}')
        ... )
        >>> # String value compared literally, not parsed as JSON

    See Also:
        docs/scorers.md for comprehensive examples showing default, semantic,
        and preserve_strings approaches.

    Raises:
        Returns error Score if autoevals is not installed.
    """
    try:
        # JSONDiff requires parameters at evaluation time, not construction time
        # Create a custom wrapper that captures the parameters
        from typing import cast

        from autoevals.json import JSONDiff as AutoevalsJSONDiff

        autoevals_scorer = AutoevalsJSONDiff(
            string_scorer=cast(Any, string_scorer),
            number_scorer=cast(Any, number_scorer),
            preserve_strings=preserve_strings,
        )

        def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
            """JSONDiff scorer with captured parameters."""
            try:
                # Check if expected is None
                if expected is None:
                    return Score(
                        name="JSONDiff",
                        value=0.0,
                        passed=False,
                        metadata={"error": "Expected value is required for JSONDiff"},
                        reasoning=(
                            "JSONDiff requires an expected value for comparison"
                        ),
                    )

                # Call the autoevals scorer
                ae_result = autoevals_scorer(
                    output=result.output, expected=expected.expected
                )

                # Normalize the score value
                normalized_value = _normalize_value(
                    ae_result.score if ae_result.score is not None else 0.0
                )

                # Determine passed based on threshold
                passed = normalized_value >= threshold

                # Handle metadata
                metadata = ae_result.metadata if ae_result.metadata is not None else {}
                metadata["_threshold"] = threshold

                # Extract reasoning if available
                reasoning = getattr(ae_result, "rationale", None)

                return Score(
                    name="JSONDiff",
                    value=normalized_value,
                    passed=passed,
                    metadata=metadata,
                    reasoning=reasoning,
                )

            except Exception as e:
                # Handle any errors
                return Score(
                    name="JSONDiff",
                    value=0.0,
                    passed=False,
                    metadata={"error": str(e)},
                    reasoning=f"Scorer failed with exception: {type(e).__name__}",
                )

        return scorer

    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="JSONDiff",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("ValidJSON")
def ValidJSON(  # noqa: N802
    threshold: float = 1.0,
    schema: dict | None = None,
) -> Callable:
    """Convenience factory for autoevals ValidJSON scorer.

    Validate if a string is valid JSON and optionally matches a JSON schema.
    Binary scorer that checks JSON syntax and schema compliance.

    Default threshold is 1.0 (must be valid) because JSON validation is binary -
    either the JSON is valid or it isn't.

    Args:
        threshold: Pass threshold [0.0, 1.0] (default: 1.0)
        schema: Optional JSON Schema (dict) to validate against.
                Follows JSON Schema specification (draft-07).

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import ValidJSON
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Simple JSON validation
        >>> validator = ValidJSON()
        >>> result = validator(
        ...     TaskResult(output='{"name": "John", "age": 30}'),
        ...     expected=ExpectedResult(expected="")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Invalid JSON
        >>> result = validator(
        ...     TaskResult(output='{"name": "John", age: 30}'),  # Missing quotes
        ...     expected=ExpectedResult(expected="")
        ... )
        >>> print(result.passed)
        False
        >>>
        >>> # With JSON Schema validation
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "number"}
        ...     },
        ...     "required": ["name", "age"]
        ... }
        >>> strict_validator = ValidJSON(schema=schema)
        >>> result = strict_validator(
        ...     TaskResult(output='{"name": "John", "age": 30}'),
        ...     expected=ExpectedResult(expected="")
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Invalid against schema (missing required field)
        >>> result = strict_validator(
        ...     TaskResult(output='{"name": "John"}'),
        ...     expected=ExpectedResult(expected="")
        ... )
        >>> print(result.passed)
        False

    Raises:
        Returns error Score if autoevals is not installed.
    """
    try:
        from autoevals.json import ValidJSON as AutoevalsValidJSON

        # ValidJSON requires schema at evaluation time, not construction time
        # Create a custom wrapper that captures the schema
        autoevals_scorer = AutoevalsValidJSON()

        def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
            """ValidJSON scorer with captured schema."""
            try:
                # Call the autoevals scorer with schema as positional argument
                # autoevals ValidJSON signature is: (output, schema=None)
                ae_result = autoevals_scorer(result.output, schema)

                # Normalize the score value
                normalized_value = _normalize_value(
                    ae_result.score if ae_result.score is not None else 0.0
                )

                # Determine passed based on threshold
                passed = normalized_value >= threshold

                # Handle metadata
                metadata = ae_result.metadata if ae_result.metadata is not None else {}
                metadata["_threshold"] = threshold
                if schema is not None:
                    metadata["schema"] = schema

                # Extract reasoning if available
                reasoning = getattr(ae_result, "rationale", None)

                return Score(
                    name="ValidJSON",
                    value=normalized_value,
                    passed=passed,
                    metadata=metadata,
                    reasoning=reasoning,
                )

            except Exception as e:
                # Handle any errors
                return Score(
                    name="ValidJSON",
                    value=0.0,
                    passed=False,
                    metadata={"error": str(e)},
                    reasoning=f"Scorer failed with exception: {type(e).__name__}",
                )

        return scorer

    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="ValidJSON",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


@scorer_registry.register("Moderation")
def Moderation(  # noqa: N802
    threshold: float | None = None,
    client: Any | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
) -> Callable:
    """Convenience factory for autoevals Moderation scorer.

    Content safety scorer that evaluates whether output contains harmful,
    inappropriate, or policy-violating content using OpenAI's Moderation API.
    Returns 1.0 for safe content, 0.0 for flagged content (binary scorer).

    ⚠️ IMPORTANT: This wrapper uses Pattern 2 (inline) because the threshold
    parameter controls HOW the score is computed, not just pass/fail logic.
    The autoevals Moderation class always returns binary 0 or 1.

    **Threshold Semantics:**
    - `threshold=None` (default): Uses OpenAI's default content flagging logic
    - `threshold=<value>`: Content is flagged if ANY category score exceeds this value

    Note: Moderation uses OpenAI's moderation endpoint which requires an OpenAI API key.
    It does not support Ollama or other local models.

    Args:
        threshold: Category score threshold for flagging content (default: None)
                  If None, uses OpenAI's default flagging.
                  If set, flags content when any category score > threshold.
        client: Optional OpenAI client instance
        api_key: Optional OpenAI API key (deprecated, use client instead)
        base_url: Optional OpenAI API base URL

    Returns:
        Callable scorer conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.autoevals import Moderation
        >>> from agent_evals.core.types import TaskResult, ExpectedResult
        >>>
        >>> # Default configuration (uses OpenAI's flagging logic)
        >>> moderation = Moderation()
        >>> result = moderation(
        ...     TaskResult(
        ...         output="Python is a great programming language for beginners."
        ...     ),
        ...     expected=ExpectedResult(expected="")
        ... )
        >>> print(result.value)  # 1.0 (safe)
        >>> print(result.passed)  # True
        >>>
        >>> # Custom threshold - stricter than OpenAI default
        >>> # Flags if any category score > 0.25
        >>> strict_moderation = Moderation(threshold=0.25)
        >>> result = strict_moderation(
        ...     TaskResult(output="here are my suicidal thoughts"),
        ...     expected=ExpectedResult(expected="")
        ... )
        >>> print(result.value)  # 0.0 (flagged)
        >>> print(result.passed)  # False
        >>>
        >>> # Lenient threshold - only flag very extreme content
        >>> lenient_moderation = Moderation(threshold=0.99)
        >>> result = lenient_moderation(
        ...     TaskResult(output="here are my suicidal thoughts"),
        ...     expected=ExpectedResult(expected="")
        ... )
        >>> print(result.value)  # 1.0 (not flagged - below threshold)
        >>> print(result.passed)  # True
    """
    # Validated in the factory body, not in the inner scorer closure below.
    # The closure re-instantiates the autoevals scorer per example (threshold
    # affects score computation), so validating there would defer a
    # configuration error into a mid-run failure repeated on every example.
    # This factory builds its arguments inline and never calls
    # _build_scorer_kwargs.
    if base_url is not None:
        validate_secure_transport("base_url", base_url)

    try:
        from autoevals.moderation import Moderation as AutoevalsModeration

        async def scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            """Moderation scorer with threshold parameter passed to autoevals."""
            try:
                # Instantiate autoevals Moderation with threshold
                # This MUST be per-call because threshold affects score computation
                autoevals_scorer = AutoevalsModeration(
                    threshold=threshold,
                    client=client,
                    api_key=api_key,
                    base_url=base_url,
                )

                # Call the autoevals scorer (using async method)
                # Note: Moderation doesn't use 'expected', only 'output'
                ae_result = await autoevals_scorer.eval_async(
                    output=result.output,
                )

                # Normalize the score value (should be 0 or 1)
                normalized_value = _normalize_value(
                    ae_result.score if ae_result.score is not None else 0.0
                )

                # For binary scorers, passed = (value == 1.0)
                # Content is "safe" if score is 1.0, "flagged" if 0.0
                passed = normalized_value >= 0.5  # Binary: 1.0 passes, 0.0 fails

                # Handle metadata
                metadata = ae_result.metadata if ae_result.metadata is not None else {}
                metadata["_threshold"] = 0.5

                # Extract reasoning if available
                reasoning = getattr(ae_result, "rationale", None)
                if reasoning is None:
                    reasoning = (
                        "Content passed moderation"
                        if passed
                        else "Content flagged by moderation"
                    )

                return Score(
                    name="Moderation",
                    value=normalized_value,
                    passed=passed,
                    metadata=metadata,
                    reasoning=reasoning,
                )

            except Exception as e:
                # Handle any errors
                return Score(
                    name="Moderation",
                    value=0.0,
                    passed=False,
                    metadata={"error": str(e)},
                    reasoning=f"Scorer failed with exception: {type(e).__name__}",
                )

        return scorer

    except ImportError as e:
        # Capture error message for closure
        error_message = f"autoevals not installed: {e}"

        # Return a scorer that always returns error Score
        def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return Score(
                name="Moderation",
                value=0.0,
                passed=False,
                metadata={"error": error_message},
                reasoning="Scorer failed due to missing autoevals library",
            )

        return error_scorer


# Export all convenience wrapper functions
__all__ = [
    # String scorers
    "Levenshtein",
    "EmbeddingSimilarity",
    # Number scorers
    "NumericDiff",
    # Value scorers
    "ExactMatch",
    # List scorers
    "ListContains",
    # JSON scorers
    "JSONDiff",
    "ValidJSON",
    # LLM scorers
    "Factuality",
    "ClosedQA",
    "Humor",
    "Battle",
    "Security",
    "Summary",
    "Translation",
    "Sql",
    "Possible",
    "LLMClassifier",
    "Moderation",
]
