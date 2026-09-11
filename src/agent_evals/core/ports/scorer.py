# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scorer protocol.

Defines:
- Scorer: Protocol for duck-typed scorer interface
"""

from collections.abc import Awaitable
from typing import Protocol, runtime_checkable

from agent_evals.core.types import ExpectedResult, Score, TaskResult


@runtime_checkable
class Scorer(Protocol):
    """Protocol for scorer callables.

    The Scorer Protocol defines the interface that any scorer must satisfy to be
    compatible with the evaluation framework. Any callable (function, lambda, or
    callable class) matching this signature can be used as a scorer without
    inheritance.

    This uses Python's structural subtyping (PEP 544) - if it walks like a scorer
    and quacks like a scorer, it IS a scorer. No inheritance required!

    Signature:
        scorer(result: TaskResult, expected: ExpectedResult | None = None,
               **context) -> Score | Awaitable[Score]

    Parameters:
        result: The output from the system being evaluated (TaskResult type)
        expected: Optional reference/expected output for comparison
                 (ExpectedResult type)
        **context: Optional extra keyword arguments a scorer may accept;
                 ignored by scorers that don't need them.

    Returns:
        A Score, or an Awaitable[Score] for async (``async def``) scorers.
        The runtime awaits coroutine scorers and runs sync scorers on a
        thread, so both shapes are accepted wherever a Scorer is required.

    Type Flexibility:
        Output is always TaskResult with .output (str) and optional .context (dict).
        Expected is ExpectedResult type (or None) with .expected field containing the
        reference value and optional .context field for additional data.

        Examples of expected values:

        String comparison:
            result.output="hello", expected.expected="hello"

        Numeric comparison:
            result.output="42", expected.expected="42.5"

        Classification:
            result.output="positive", expected.expected="negative" (class labels)

        Context-based scoring:
            result.context={"trajectory": [...], "tool_calls": [...]}

        Scorers should extract data from result.output (string) and result.context
        (dict) and handle type conversion internally with appropriate error handling.

    Context Pattern:
        Scorers should access context through the TaskResult.context dict:

        ```python
        def my_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            context = result.context or {}
            trajectory = context.get('trajectory', [])
            tool_calls = context.get('tool_calls', [])
            output_text = result.output

            # Use trajectory, tool_calls, and output_text in scoring logic...

            return Score(name="my_scorer", value=0.8, passed=True)
        ```

    Common Context Fields (not enforced, use as needed):
        Note: Fields are provided in TaskResult.context dict from task execution
        or dataset examples. None are guaranteed to be present - always use .get()
        pattern.

        - input: Any - Original input to the task
        - trajectory: list[dict] - Execution steps with standardized format
        - tool_calls: list[dict] - Tool invocations with inputs/outputs
        - outputs: list[dict] - Actual trajectory for trajectory scorers
        - reference_outputs: list[dict] - Expected trajectory for trajectory scorers
        - retrieved_docs: list[str] - Documents retrieved during RAG
        - metadata: dict - General purpose metadata

    Examples:
        Simple function scorer:
        >>> def keyword_scorer(
        ...     result: TaskResult, expected: ExpectedResult | None = None
        ... ) -> Score:
        ...     keywords = ["python", "evaluation"]
        ...     output_str = result.output
        ...     found = [kw for kw in keywords if kw.lower() in output_str.lower()]
        ...     score_value = len(found) / len(keywords)
        ...     return Score(
        ...         name="keyword_presence",
        ...         value=score_value,
        ...         passed=score_value >= 0.5,
        ...         metadata={"keywords_found": found}
        ...     )

        Callable class scorer with configuration:
        >>> class LengthScorer:
        ...     def __init__(self, min_length: int):
        ...         self.min_length = min_length
        ...
        ...     def __call__(
        ...         self, result: TaskResult, expected: ExpectedResult | None = None
        ...     ) -> Score:
        ...         output_str = result.output
        ...         length = len(output_str)
        ...         passed = length >= self.min_length
        ...         score_value = min(length / self.min_length, 1.0)
        ...         return Score(
        ...             name="length_check",
        ...             value=score_value,
        ...             passed=passed,
        ...             reasoning=f"Output length {length} (min: {self.min_length})"
        ...         )

        Trajectory-aware scorer using context:
        >>> def trajectory_scorer(
        ...     result: TaskResult, expected: ExpectedResult | None = None
        ... ) -> Score:
        ...     context = result.context or {}
        ...     trajectory = context.get('trajectory', [])
        ...     if not trajectory:
        ...         return Score(name="trajectory_check", value=0.0, passed=False,
        ...                     reasoning="No trajectory data available")
        ...
        ...     step_count = len(trajectory)
        ...     passed = step_count <= 10  # Prefer efficient executions
        ...     score_value = max(0.0, 1.0 - (step_count - 5) * 0.1)
        ...     return Score(
        ...         name="trajectory_efficiency",
        ...         value=score_value,
        ...         passed=passed,
        ...         metadata={"step_count": step_count},
        ...         reasoning=f"Agent completed task in {step_count} steps"
        ...     )

    Type Checking:
        The @runtime_checkable decorator enables isinstance() checks:
        >>> isinstance(keyword_scorer, Scorer)  # True
        >>> isinstance(LengthScorer(50), Scorer)  # True

    Notes:
        - No inheritance required - any callable with matching signature works
        - Context fields are optional - scorers only extract what they need
        - New context fields can be added without breaking existing scorers
        - Type checkers (mypy, pyright, ty) verify conformance statically
    """

    def __call__(
        self,
        result: TaskResult,
        expected: ExpectedResult | None = None,
    ) -> Score | Awaitable[Score]:
        """Score the output.

        Args:
            result: The TaskResult to evaluate containing output string and context
            expected: Optional expected/reference output for comparison

        Returns:
            Score object with evaluation results

        Note:
            This is a Protocol method signature - implementing callables don't
            need to explicitly implement this. Any callable matching this signature
            automatically satisfies the Protocol.
        """
        ...
