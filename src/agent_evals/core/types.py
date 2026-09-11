# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Core data structures for evaluation.

Defines:
- Score: Unified score format returned by all scorers
- EvalExample: Single evaluation example with results
- EvalResult: Complete evaluation results
- EvalConfig: Configuration for evaluation execution
- PlatformConfig: Base model for platform adapter configuration
- ExampleData: Pydantic model for dataset example validation

All types use Pydantic for validation, IDE support, and consistency.
"""

import logging
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

logger = logging.getLogger(__name__)

# Maximum length for an identifier that becomes a filesystem path segment.
# Chosen well below the 255-byte limit common to ext4, APFS and NTFS, leaving
# room for the suffixes and separators callers compose around the identifier.
_MAX_IDENTIFIER_LENGTH = 96

PathSafeIdentifier = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=_MAX_IDENTIFIER_LENGTH,
        pattern=r"^[.]*[A-Za-z0-9_-][A-Za-z0-9._-]*$",
    ),
]
"""A bounded identifier that cannot alter the structure of a path built from it.

Values annotated with this type are names, not paths. The character set admits
only letters, digits, dot, underscore and hyphen, which excludes every
construct that changes where a joined path lands: both separator characters,
the colon that makes a drive-relative form anchor away from its intended
parent, and the null byte.

The pattern additionally requires at least one character that is not a dot, so
a value consisting only of dots — ``.``, ``..`` — is rejected. The character
class alone would admit those, and each is a directory reference rather than a
name. Expressing the rule as ``[.]*`` before a mandatory non-dot rather than as
a lookahead keeps it within the regex dialect the validation layer compiles.

This is an allow-list because a name never legitimately needs a separator. It
is deliberately the wrong tool for a value that *is* a path: paths require
separators and dot segments to function, and are constrained by resolving them
and testing containment instead.
"""


class Score(BaseModel):
    """Unified score format for evaluation results.

    Score objects represent the result of evaluating an output against some
    criteria. They provide a normalized format (0.0-1.0 value) with pass/fail
    indication and optional metadata/reasoning.

    All scorers must return Score objects, enabling consistent aggregation and
    reporting across different scorer types.

    Attributes:
        name: Identifier for the scorer that produced this score
        value: Normalized score between 0.0 and 1.0 (inclusive)
        passed: Binary pass/fail indicator
        error: Failure message if the scorer could not produce a verdict
            (default: None). A score with ``error`` set carries no opinion
            about the output: its ``value`` and ``passed`` are placeholders,
            not measurements, and aggregation excludes it rather than
            averaging it in as genuine performance. This is what separates
            "the scorer says 0.0" from "the scorer never ran".
        metadata: Additional scorer-specific information (default: {})
        reasoning: Human-readable explanation of the score (default: None)
        execution_time_ms: Time taken to compute score in ms (default: None)

    Examples:
        Basic score creation:
        >>> score = Score(name="keyword_check", value=1.0, passed=True)

        Score with metadata and reasoning:
        >>> score = Score(
        ...     name="length_check",
        ...     value=0.8,
        ...     passed=True,
        ...     metadata={"length": 80, "min_required": 50},
        ...     reasoning="Output length is 80 chars (min: 50)"
        ... )

        Error score convention:
        >>> error_score = Score(
        ...     name="api_scorer",
        ...     value=0.0,
        ...     passed=False,
        ...     error="timeout: API took 30s to respond",
        ...     reasoning="Scorer failed due to API timeout"
        ... )

        Serialization:
        >>> score = Score(name="test", value=0.5, passed=True)
        >>> score.model_dump(exclude_none=True)
        {'name': 'test', 'value': 0.5, 'passed': True, 'metadata': {}}

    Raises:
        ValueError: If value is not between 0.0 and 1.0 (inclusive)
    """

    name: str
    value: float
    passed: bool
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    reasoning: str | None = None
    execution_time_ms: float | None = None

    model_config = ConfigDict(frozen=True)

    @field_validator("value")
    @classmethod
    def validate_score_range(cls, v: float) -> float:
        """Validate score value is in valid range [0.0, 1.0]."""
        if not (0.0 <= v <= 1.0):
            raise ValueError(
                f"Score value must be between 0.0 and 1.0 (inclusive), "
                f"got {v}. "
                f"Ensure scorer returns normalized scores in the valid range. "
                f"For external scorers (autoevals, etc.), use wrapper adapters "
                f"that handle normalization automatically "
                f"(see agent_evals.scorers.autoevals module)."
            )
        return v


class EvalExample(BaseModel):
    """Complete result for a single evaluation example.

    Represents the full evaluation pipeline for one dataset example:
    input → task execution → scorer evaluation → final result.

    Attributes:
        input: Input value passed to task function
        output: Output returned by task (or None if task failed)
        expected: Expected/reference output for scoring (optional)
        scores: Dictionary of scorer_name → Score object
        metadata: Additional example-level metadata
        duration: Task execution time in seconds
        error: Error message if the *task* failed (None if it ran). Scorer
            failures do not appear here — they live on the individual
            ``Score.error``. Use ``failed`` to ask whether anything went
            wrong, rather than testing this field directly.
        trajectory: Step-by-step execution trace
        tool_calls: Tool invocations during execution

    Examples:
        >>> example = EvalExample(
        ...     input="What is 2+2?",
        ...     output="4",
        ...     expected="4",
        ...     scores={"ExactMatch": Score(name="ExactMatch", value=1.0, passed=True)},
        ...     duration=0.123,
        ...     error=None
        ... )
    """

    input: Any
    output: str
    expected: str | None = None
    scores: dict[str, Score] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration: float = 0.0
    error: str | None = None
    trajectory: list[dict[str, Any]] | None = None
    tool_calls: list[dict[str, Any]] | None = None

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def scorer_errors(self) -> dict[str, str]:
        """Scorer name → failure message, for scorers that could not score.

        Empty when every scorer returned a genuine verdict.
        """
        return {
            name: score.error
            for name, score in self.scores.items()
            if score.error is not None
        }

    @property
    def failed(self) -> bool:
        """Whether anything went wrong for this example.

        True when the task failed *or* any scorer failed. The summary counts
        and the aggregates both derive from this, so they cannot disagree
        about which examples failed — an example is successful only if it
        produced a real verdict from every scorer that ran.
        """
        return self.error is not None or bool(self.scorer_errors)


class EvalResult(BaseModel):
    """Complete summary of an evaluation run.

    Contains aggregate metrics, all individual examples, and summary statistics
    from a complete evaluation run.

    Attributes:
        experiment_id: Unique UUID identifier for this evaluation run
        experiment_url: Platform-specific URL or file path to results
        platform: Platform name identifying the adapter that produced this result.
        scores: Aggregate scores (scorer_name → average value)
        pass_rates: Pass rates per scorer (scorer_name → fraction passed)
        examples: All example results from evaluation
        summary: Additional summary statistics
        metadata: Evaluation-level metadata
        duration: Total evaluation time in seconds

    Examples:
        >>> result = EvalResult(
        ...     experiment_id="550e8400-e29b-41d4-a716-446655440000",
        ...     experiment_url="file:///Users/user/.agent-evals/experiments/550e8400.../",
        ...     platform="local",
        ...     scores={"ExactMatch": 0.95},
        ...     pass_rates={"ExactMatch": 0.95},
        ...     examples=[...],
        ...     summary={"total_examples": 100, "successful_examples": 95},
        ...     duration=45.67
        ... )
    """

    experiment_id: str
    experiment_url: str
    platform: str
    scores: dict[str, float]
    pass_rates: dict[str, float]
    examples: list[EvalExample]
    summary: dict[str, Any]
    metadata: dict[str, Any] = Field(default_factory=dict)
    duration: float = 0.0

    model_config = ConfigDict(arbitrary_types_allowed=True)


class EvalConfig(BaseModel):
    """Non-platform settings for one eval run.

    One "test" = one ``task(input)`` invocation plus its scorers. Sync scorers
    run via ``asyncio.to_thread`` so they don't block the event loop.

    Rate-limiting LLM calls is the agent's responsibility, not the framework's.

    Attributes:
        max_concurrent_tests: Maximum test cases processed at once.
            Defaults to 1 (serial) because the framework cannot detect whether
            your task/agent mutates shared state (in-memory caches, DB
            connections, conversation history). Raise when your agent is
            stateless and each invocation is independent.

    Examples:
        >>> EvalConfig()                            # serial, safe default
        >>> EvalConfig(max_concurrent_tests=5)      # 5 tests at a time
    """

    model_config = ConfigDict(extra="forbid")

    max_concurrent_tests: int = 1

    @field_validator("max_concurrent_tests")
    @classmethod
    def _must_be_positive(cls, v: int, info) -> int:
        if v < 1:
            raise ValueError(f"{info.field_name} must be >= 1, got {v}")
        return v


class PlatformConfig(BaseModel):
    """Platform-agnostic base for platform adapter configuration.

    Holds only the fields every adapter uses. Each adapter declares a
    subclass with its own platform-specific fields. The ``name`` field is
    the registered adapter name and selects the adapter; subclasses pin it
    via ``Literal``.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    experiment: str | None = Field(
        default=None,
        description=(
            "A name you choose to label this run on the platform. Not "
            "looked up — the platform records your run under it."
        ),
    )


def _last_assistant_text(openai_messages: list[dict]) -> str:
    """Return the last assistant message's text content, or ''.

    Walks the message list in reverse; the first assistant message whose
    `content` is a non-empty string wins. Used as the default for
    `TaskResult.output` when messages are the only input — keeps text
    scorers (Levenshtein, ExactMatch, Factuality) functional without
    requiring the caller to supply `output` explicitly.
    """
    for msg in reversed(openai_messages):
        if msg.get("role") == "assistant":
            content = msg.get("content", "")
            if isinstance(content, str) and content:
                return content
    return ""


class TaskResult(BaseModel):
    """Type-safe task return value.

    All tasks must return TaskResult. The output field is the string
    result shown to scorers. The context dict is accessible to scorers
    via the TaskResult.context attribute.

    Examples:
        Simple output:
        >>> return TaskResult(output="Paris")

        With trajectory context:
        >>> return TaskResult(
        ...     output="Paris",
        ...     context={"outputs": trajectory, "inputs": inputs}
        ... )

        Trajectory-only (no meaningful string output):
        >>> return TaskResult(output="", context={"outputs": trajectory})

        From normalized OpenAI-format messages (applies library defaults):
        >>> return TaskResult.from_messages(
        ...     openai_format_messages,
        ...     final_state=snapshot_world(),
        ... )
    """

    output: str  # Required - use "" if no meaningful output
    context: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_messages(
        cls,
        openai_messages: list[dict],
        **context: Any,
    ) -> TaskResult:
        """Build a TaskResult from OpenAI-format messages.

        Applies the library's two conventions:
          - Messages are placed under `context["outputs"]`.
          - `output` defaults to the last assistant message's text content.

        Any keyword arguments are merged into `context` alongside `outputs`.
        This is where scorer-specific data goes: `final_state=` for
        StateMatch, `reference_outputs=` for trajectory comparison, or
        anything a custom scorer reads.

        If you need a non-default `output`, use the `TaskResult(...)`
        constructor directly — there is deliberately no `output=`
        override on this helper.
        """
        return cls(
            output=_last_assistant_text(openai_messages),
            context={"outputs": openai_messages, **context},
        )


def _derive_tool_calls(messages: list[Any]) -> list[dict]:
    """Flatten ``tool_calls`` arrays from assistant turns, in order.

    Returns an empty list if the trajectory has assistant turns but
    none carry tool_calls. Returns an empty list (not None) so callers
    can distinguish "no trajectory at all" (None) from "trajectory
    present but no tools were invoked" ([]).

    Non-dict elements in ``messages`` are skipped — the function tolerates
    mixed-type lists rather than raising ``AttributeError`` on a malformed
    trajectory. The caller is expected to pass OpenAI-format message
    dicts; this guard is a defensive default for user code that strays.
    The signature accepts ``list[Any]`` to make the tolerance explicit
    at the type level rather than relying on isinstance at runtime alone.
    """
    out: list[dict] = []
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            out.append(tc)
    return out


def task_result_to_example(
    *,
    task_result: TaskResult,
    input: Any,
    expected: ExpectedResult | None,
    scores: dict[str, Score],
    metadata: dict[str, Any],
    duration: float,
    error: str | None,
) -> EvalExample:
    """Build an EvalExample from a TaskResult plus example-level metadata.

    Consumer-side counterpart of ``TaskResult.from_messages``: where
    ``from_messages`` defines how a trajectory ENTERS a ``TaskResult``
    (under ``context["outputs"]``), this function defines how it
    LEAVES and lands on ``EvalExample.trajectory``. The two together
    establish the trajectory convention end-to-end.

    ``trajectory`` reads ``task_result.context["outputs"]`` if present
    (None otherwise). ``tool_calls`` is derived by flattening
    assistant-turn ``tool_calls`` arrays in order.

    Forward-path platform adapters that hold a ``TaskResult`` at the
    moment of constructing an ``EvalExample`` should delegate here.
    Adapters whose forward path reads from platform traces populate the
    fields directly without using this helper.

    Returns:
        Fully-populated ``EvalExample`` with ``trajectory`` and
        ``tool_calls`` set when the ``TaskResult`` carries them,
        ``None`` otherwise.
    """
    context = task_result.context or {}
    raw_trajectory = context.get("outputs")
    if isinstance(raw_trajectory, list):
        trajectory: list[dict[str, Any]] | None = raw_trajectory
        tool_calls: list[dict[str, Any]] | None = _derive_tool_calls(raw_trajectory)
    else:
        if raw_trajectory is not None:
            # Non-list, non-None outputs is a contract violation by the
            # caller — TaskResult.from_messages always produces a list.
            # Drop trajectory but log so the malformed shape is visible
            # rather than silently swallowed.
            logger.warning(
                "TaskResult.context['outputs'] present but not a list "
                "(got %s); dropping trajectory. Use TaskResult.from_messages "
                "or pass a list of OpenAI-format message dicts.",
                type(raw_trajectory).__name__,
            )
        trajectory = None
        tool_calls = None

    return EvalExample(
        input=input,
        output=task_result.output,
        expected=expected.expected if expected is not None else None,
        scores=scores,
        metadata=metadata,
        duration=duration,
        error=error,
        trajectory=trajectory,
        tool_calls=tool_calls,
    )


class ExpectedResult(BaseModel):
    """Type-safe task return value.

    A class mirroring TaskResult for passing expected or reference values.
    The output field is the string result shown to scorers. The context dict is
    accessible to scorers via the ExpectedResult.context attribute.

    Examples:
        Simple output:
        >>> return ExpectedResult(output="Paris")

        With trajectory context:
        >>> return ExpectedResult(
        ...     output="Paris",
        ...     context={"outputs": trajectory, "inputs": inputs}
        ... )

        Trajectory-only (no meaningful string output):
        >>> return ExpectedResult(output="", context={"outputs": trajectory})
    """

    expected: str  # Required - use "" if scorer only needs context
    context: dict[str, Any] | None = None

    model_config = ConfigDict(extra="forbid")


class ExampleData(BaseModel):
    """Dataset example structure with runtime validation.

    Use explicit 'context' dict for scorer context data (outputs,
    reference_outputs, etc.). Top-level keys are reserved for framework use.

    Reserved keys:
        input: Data passed to task function
        expected: Reference output for comparison
        output: Pre-populated agent output (for historical data evaluation)
        metadata: Tracking metadata for results

    Examples:
        Basic example:
        >>> example = ExampleData(input="What is 2+2?", expected="4")

        With pre-populated output (historical data pattern):
        >>> example = ExampleData(
        ...     input="What is 2+2?",
        ...     output=TaskResult(output="4"),
        ...     expected="4"
        ... )

        With scorer context:
        >>> example = ExampleData(
        ...     input="query",
        ...     output = TaskResult(output = "", context={
        ...         "outputs": [...],
        ...         "reference_outputs": [...]
        ...     }
        ... )

        From dict (backward compatible):
        >>> data = {"input": "query", "expected": "answer"}
        >>> example = ExampleData.model_validate(data)

    Note:
        Like Scorer Protocol, this uses structural validation - you can pass
        dicts at runtime and Pydantic converts them. Type hints guide toward
        using ExampleData directly for better IDE support.
    """

    input: Any
    expected: ExpectedResult | None = None
    output: TaskResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        extra="forbid",  # Reject unknown top-level keys
        validate_assignment=True,  # Validate on mutation
        arbitrary_types_allowed=True,  # Allow Any types
    )

    @field_validator("metadata")
    @classmethod
    def must_be_dict(cls, v: Any, info) -> dict[str, Any]:
        """Ensure metadata is a dict."""
        if v is None:
            return {}
        if not isinstance(v, dict):
            field_name = info.field_name
            raise ValueError(f"'{field_name}' must be a dict, got {type(v).__name__}")
        return v


# Type alias for datasets
Dataset = list[ExampleData]
