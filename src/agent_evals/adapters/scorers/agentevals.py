# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""AgentEvals scorer wrappers for trajectory evaluation.

Trajectory evaluation assesses multi-step agent behavior by comparing the sequence
of actions (tool calls, LLM interactions, state transitions) against expected patterns.
Unlike traditional output-only scoring, trajectory scorers validate the HOW not just
the WHAT - ensuring agents follow correct reasoning paths and tool usage patterns.

Usage:
    from agent_evals.adapters.scorers.agentevals import (
        TrajectoryStrictMatch, TrajectoryLLMAsJudge, GraphTrajectoryStrictMatch
    )

    # Strict sequential validation
    scorers = [TrajectoryStrictMatch(tool_args_match_mode="exact")]

    # Flexible LLM-based assessment
    scorers = [
        TrajectoryLLMAsJudge(
            model="ollama:gpt-oss:20b",
            judge_base_url="https://ollama.example.com/v1",
        )
    ]

Data Format Requirements:
    All trajectory scorers require data passed via context parameters:

    TaskResult.context:
    • outputs: List[dict] - Actual agent trajectory

    ExpectedResult.context:
    • reference_outputs: List[dict] - Expected trajectory for comparison

    Optional TaskResult.context:
    • inputs: Any - Input context (query, task description, etc.)

    Message trajectories: List of dicts with role/content/tool_calls
    Graph trajectories: Dict with 'steps' key containing node sequences

Async Execution:
    LLM-based trajectory scorers (2 total) use native async I/O for true concurrent
    execution in the event loop. Match-based scorers remain sync (CPU-bound operations).
    Framework auto-detects scorer type and routes appropriately.

    Async scorers (2): TrajectoryLLMAsJudge, GraphTrajectoryLLMAsJudge
    Sync scorers (5): TrajectoryStrictMatch, TrajectoryUnorderedMatch,
                      TrajectorySubsetMatch, TrajectorySupsetMatch,
                      GraphTrajectoryStrictMatch

    Sync scorers run via ``asyncio.to_thread`` so they don't block the event
    loop.

Available Scorers:
    Message Trajectory Evaluators:
    • TrajectoryStrictMatch (sync): Exact order and content matching
    • TrajectoryUnorderedMatch (sync): Same tools, any order
    • TrajectorySubsetMatch (sync): Actual ⊆ reference
    • TrajectorySupsetMatch (sync): Actual ⊇ reference
    • TrajectoryLLMAsJudge (async): LLM-based trajectory assessment

    Graph Trajectory Evaluators (LangGraph):
    • GraphTrajectoryStrictMatch (sync): Graph node sequence matching
    • GraphTrajectoryLLMAsJudge (async): LLM-based graph trajectory assessment

For detailed usage examples, see docs/scorers.md.
See CONTRIBUTING.md for implementation patterns (if adding new wrappers).
"""

import logging
from collections.abc import Callable, Coroutine
from typing import TYPE_CHECKING, Any, Literal, cast

from agent_evals.core._registries import scorer_registry
from agent_evals.core.transport import validate_secure_transport
from agent_evals.core.types import ExpectedResult, Score, TaskResult

if TYPE_CHECKING:
    from agentevals.types import GraphTrajectory  # noqa: F401

logger = logging.getLogger(__name__)


# =============================================================================
# Helper Functions
# =============================================================================


def _normalize_agentevals_result(
    wrapper_name: str,
    result: Any,  # agentevals returns EvaluatorResult, but we treat as dict-like
    trajectory_length: int | None = None,
    reference_trajectory_length: int | None = None,
) -> Score:
    """Normalize agentevals result dict to Score dataclass.

    Args:
        wrapper_name: Name of our wrapper (e.g., "TrajectoryStrictMatch")
        result: AgentEvals result dict with keys: key, score, comment, metadata
        trajectory_length: Length of actual trajectory (optional)
        reference_trajectory_length: Length of reference trajectory (optional)

    Returns:
        Score object with normalized fields
    """
    # Extract fields from agentevals result
    agentevals_key = result.get("key", "unknown")
    score_bool = result.get("score", False)
    comment = result.get("comment")
    agentevals_metadata = result.get("metadata") or {}

    # Convert boolean score to 0.0/1.0
    value = 1.0 if score_bool else 0.0

    # Build metadata
    metadata = {
        "agentevals_key": agentevals_key,
        **agentevals_metadata,  # Merge agentevals metadata
    }

    # Add trajectory lengths if provided
    if trajectory_length is not None:
        metadata["trajectory_length"] = trajectory_length
    if reference_trajectory_length is not None:
        metadata["reference_trajectory_length"] = reference_trajectory_length

    # Return Score
    return Score(
        name=wrapper_name,
        value=value,
        passed=score_bool,
        metadata=metadata,
        reasoning=comment,  # Map comment to reasoning
    )


def _create_error_score(
    wrapper_name: str, error_msg: str, exception: Exception | None = None
) -> Score:
    """Create error Score for failed evaluations.

    Args:
        wrapper_name: Name of wrapper
        error_msg: Human-readable error message
        exception: Original exception (optional)

    Returns:
        Score with value=0.0, passed=False, error in metadata
    """
    metadata = {"error": error_msg}
    if exception:
        metadata["exception_type"] = type(exception).__name__

    return Score(
        name=wrapper_name,
        value=0.0,
        passed=False,
        metadata=metadata,
        reasoning=f"Scorer failed: {error_msg}",
    )


def _validate_judge_endpoint(wrapper_name: str, judge_base_url: str | None) -> str:
    """Require a judge endpoint and check that its transport is accepted.

    The LLM-judge factories take a provider-prefixed model *nickname*
    (``"ollama:gpt-oss:20b"``), not a URL. The address behind that nickname is
    resolved by the provider SDK, downstream of this library, and lazily: the
    third-party chain is ``agentevals.trajectory.llm`` -> ``openevals.llm``,
    which calls ``init_chat_model(model=model)`` **bare** from inside its own
    scorer closure, forwarding nothing. So a nickname alone leaves this library
    with no address to inspect and no keyword route to influence — the endpoint
    would be unknown at the moment prompts and trajectories are sent to it.

    An unknown endpoint is therefore refused rather than warned about: the
    caller must name one, and ``_build_judge`` binds it through the SDK's
    ``judge=`` seam so the checked value is the connected value.

    Kept separate from ``_build_judge`` so it can run before the SDK import
    guard in each factory: a missing or unacceptable endpoint is then a
    configuration error whether or not the SDK is installed, rather than being
    masked by a degraded scorer.

    Args:
        wrapper_name: Factory name, so the message says what to fix.
        judge_base_url: Caller-supplied judge endpoint, or None.

    Returns:
        The same endpoint, now known to be present, so a caller can hand the
        result straight to ``_build_judge`` without re-narrowing the type.

    Raises:
        ValueError: If judge_base_url is absent, or names a transport that is
            not accepted — it must encrypt in transit or address only the
            machine this process runs on.
    """
    if judge_base_url is None:
        raise ValueError(
            f"{wrapper_name} requires judge_base_url. The model identifier is a "
            f"provider nickname, not an address, so leaving the endpoint to the "
            f"provider SDK would send prompts and trajectories somewhere this "
            f"library cannot check. Pass judge_base_url= with an encrypted "
            f"endpoint, for example 'https://ollama.example.com/v1'."
        )

    validate_secure_transport("judge_base_url", judge_base_url)
    return judge_base_url


def _build_judge(model: str, judge_base_url: str) -> Any:
    """Build the judge that will use an already-validated endpoint.

    Binds the validated endpoint to the provider so the value this library
    checked is the value the provider connects to — verified against
    ``langchain/chat_models/base.py``, which forwards ``**kwargs`` to the
    provider class (``ChatOpenAI.openai_api_base`` receives ``base_url``).
    Passing the result as the SDK's ``judge=`` is the only route that reaches the
    address; the nickname path forwards no keywords at all.

    ``init_chat_model`` needs no new dependency: ``agentevals`` requires
    ``openevals``, which requires ``langchain``. Callers who have the SDK at all
    have this.

    Called only after ``_validate_judge_endpoint``, and from inside each
    factory's ``try`` block, so a provider that cannot be constructed (an
    uninstalled ``langchain-<provider>`` package, say) routes through that
    factory's existing fail-fast / graceful-degradation branches instead of
    escaping as a raw import error.

    Args:
        model: Provider-prefixed model identifier.
        judge_base_url: Validated judge endpoint.

    Returns:
        A chat model bound to the validated endpoint.
    """
    from langchain.chat_models import init_chat_model

    return init_chat_model(model=model, base_url=judge_base_url)


# =============================================================================
# User Story 1: TrajectoryStrictMatch
# =============================================================================


@scorer_registry.register("TrajectoryStrictMatch")
def TrajectoryStrictMatch(  # noqa: N802
    tool_args_match_mode: Literal["exact", "ignore", "subset", "superset"] = "exact",
    tool_args_match_overrides: dict[str, Any] | None = None,
) -> Callable[..., Score]:
    """Create trajectory strict match scorer.

    Strict trajectory evaluation requiring exact order AND content matching.
    Every step must appear in the same sequence with matching tool calls and
    arguments. Use this for deterministic workflows where execution order matters.

    When to use:
    - API call sequences that must happen in specific order (auth → query → close)
    - Multi-step workflows with dependencies (validate → process → save)
    - Testing that agents follow exact reference implementations
    - Regression testing against golden trajectories

    Args:
        tool_args_match_mode: How to compare tool arguments
            - "exact": All args must match exactly (default) - strictest validation
            - "ignore": Only tool names match, args ignored - test tool selection
            - "subset": Actual args ⊆ expected args - agent can omit optional params
            - "superset": Actual args ⊇ expected args - agent can add extra params
        tool_args_match_overrides: Per-tool custom comparison config
            Example: {"search_tool": "ignore", "api_call": "exact"}

    Returns:
        Scorer function conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch
        >>>
        >>> # Exact match validation (default)
        >>> scorer = TrajectoryStrictMatch()
        >>> from agent_evals.core.types import TaskResult
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",  # Not used for trajectory scoring
        ...         context={
        ...             "outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "search", "args": {"q": "AI"}}]},
        ...                 {"role": "tool", "content": "Results..."},
        ...                 {"role": "assistant",
        ...                  "content": "Based on search..."}
        ...             ]
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",  # Not used for trajectory scoring
        ...         context={
        ...             "reference_outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "search", "args": {"q": "AI"}}]},
        ...                 {"role": "tool", "content": "Results..."},
        ...                 {"role": "assistant",
        ...                  "content": "Based on search..."}
        ...             ]
        ...         }
        ...     )
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Ignore tool arguments - just validate tool selection
        >>> flexible_scorer = TrajectoryStrictMatch(
        ...     tool_args_match_mode="ignore")
        >>> result = flexible_scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "search", "args": {"q": "AI"}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "summarize", "args": {}}]}
        ...             ]
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "search",
        ...                                  "args": {"query": "different"}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "summarize",
        ...                                  "args": {"length": 100}}]}
        ...             ]
        ...         }
        ...     )
        ... )
        >>> print(result.passed)  # Passes - tool names match, args ignored
        True
        >>>
        >>> # Per-tool custom matching
        >>> custom_scorer = TrajectoryStrictMatch(
        ...     tool_args_match_mode="exact",
        ...     tool_args_match_overrides={"search": "ignore"}
        ... )

    Note:
        Order matters! ["search", "summarize"] != ["summarize", "search"]
        For order-independent matching, use TrajectoryUnorderedMatch instead.
    """

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        try:
            # Import agentevals
            from agentevals.trajectory.match import create_trajectory_match_evaluator
        except ImportError as e:
            return _create_error_score(
                "TrajectoryStrictMatch",
                "agentevals not installed. Install with: uv add agentevals",
                e,
            )

        # Extract trajectory data from context
        outputs = result.context.get("outputs") if result.context else None

        # Extract reference_outputs from ExpectedResult (semantically correct location)
        reference_outputs = None
        if isinstance(expected, ExpectedResult) and expected.context:
            reference_outputs = expected.context.get("reference_outputs")

        # Validate required data
        if outputs is None:
            return _create_error_score(
                "TrajectoryStrictMatch",
                "'outputs' not found in context. "
                "Pass trajectory data via context parameter.",
            )

        # Validate trajectory format (should be list)
        if not isinstance(outputs, list):
            return _create_error_score(
                "TrajectoryStrictMatch",
                f"'outputs' must be a list, got {type(outputs).__name__}",
            )

        try:
            # Create agentevals evaluator with strict mode
            evaluator = create_trajectory_match_evaluator(
                trajectory_match_mode="strict",
                tool_args_match_mode=tool_args_match_mode,
                tool_args_match_overrides=tool_args_match_overrides,
            )

            # Call evaluator
            eval_result = evaluator(
                outputs=outputs, reference_outputs=reference_outputs
            )

            # Normalize to Score and add configuration to metadata
            score = _normalize_agentevals_result(
                wrapper_name="TrajectoryStrictMatch",
                result=eval_result,
                trajectory_length=len(outputs) if isinstance(outputs, list) else None,
                reference_trajectory_length=(
                    len(reference_outputs)
                    if reference_outputs and isinstance(reference_outputs, list)
                    else None
                ),
            )
            # Add tool matching configuration to metadata for transparency
            score.metadata["tool_args_match_mode"] = tool_args_match_mode
            if tool_args_match_overrides:
                score.metadata["tool_args_match_overrides"] = tool_args_match_overrides
            return score

        except Exception as e:
            logger.exception("TrajectoryStrictMatch evaluation failed")
            return _create_error_score(
                "TrajectoryStrictMatch",
                f"Evaluation failed: {str(e)}",
                e,
            )

    return scorer


# =============================================================================
# User Story 2: Additional Match Modes
# =============================================================================


@scorer_registry.register("TrajectoryUnorderedMatch")
def TrajectoryUnorderedMatch(  # noqa: N802
    tool_args_match_mode: Literal["exact", "ignore", "subset", "superset"] = "exact",
    tool_args_match_overrides: dict[str, Any] | None = None,
) -> Callable[..., Score]:
    """Create trajectory unordered match scorer.

    Flexible trajectory evaluation where order doesn't matter. Same set of tools
    must be called with matching arguments, but execution order is ignored. Use
    this for parallel workflows or when tools have no interdependencies.

    When to use:
    - Parallel tool execution (multiple API calls that can run simultaneously)
    - Data collection workflows (gather from multiple sources, order irrelevant)
    - Testing tool coverage rather than execution sequence
    - Validating that all required steps happened, regardless of order

    Args:
        tool_args_match_mode: How to compare tool arguments
            - "exact": All args must match exactly (default)
            - "ignore": Only tool names match, args ignored
            - "subset": Actual args ⊆ expected args
            - "superset": Actual args ⊇ expected args
        tool_args_match_overrides: Per-tool custom comparison config

    Returns:
        Scorer function conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.agentevals import TrajectoryUnorderedMatch
        >>>
        >>> # Order-independent validation
        >>> scorer = TrajectoryUnorderedMatch()
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "weather",
        ...                                  "args": {"city": "NYC"}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "news",
        ...                                  "args": {"topic": "tech"}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "stocks",
        ...                                  "args": {"symbol": "AAPL"}}]}
        ...             ]
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "stocks",
        ...                                  "args": {"symbol": "AAPL"}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "weather",
        ...                                  "args": {"city": "NYC"}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "news",
        ...                                  "args": {"topic": "tech"}}]}
        ...             ]
        ...         }
        ...     )
        ... )
        >>> print(result.passed)  # Passes - same tools, different order
        True
        >>>
        >>> # Different order fails with strict match
        >>> strict = TrajectoryStrictMatch()
        >>> result = strict(
        ...     TaskResult(
        ...         output="",
        ...         context={"outputs": [...]}
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={"reference_outputs": [...]}
        ...     )
        ... )
        >>> print(result.passed)  # Fails - order matters for strict
        False

    Note:
        This scorer validates that the SAME SET of tools were called.
        For subset/superset relationships, use TrajectorySubsetMatch or
        TrajectorySupsetMatch instead.
    """

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        try:
            from agentevals.trajectory.match import create_trajectory_match_evaluator
        except ImportError as e:
            return _create_error_score(
                "TrajectoryUnorderedMatch",
                "agentevals not installed",
                e,
            )

        outputs = result.context.get("outputs") if result.context else None

        # Extract reference_outputs from ExpectedResult (semantically correct location)
        reference_outputs = None
        if isinstance(expected, ExpectedResult) and expected.context:
            reference_outputs = expected.context.get("reference_outputs")

        if outputs is None:
            return _create_error_score(
                "TrajectoryUnorderedMatch",
                "'outputs' not found in context",
            )

        if not isinstance(outputs, list):
            return _create_error_score(
                "TrajectoryUnorderedMatch",
                f"'outputs' must be a list, got {type(outputs).__name__}",
            )

        try:
            evaluator = create_trajectory_match_evaluator(
                trajectory_match_mode="unordered",
                tool_args_match_mode=tool_args_match_mode,
                tool_args_match_overrides=tool_args_match_overrides,
            )

            eval_result = evaluator(
                outputs=outputs, reference_outputs=reference_outputs
            )

            score = _normalize_agentevals_result(
                wrapper_name="TrajectoryUnorderedMatch",
                result=eval_result,
                trajectory_length=len(outputs),
                reference_trajectory_length=(
                    len(reference_outputs) if reference_outputs else None
                ),
            )
            score.metadata["tool_args_match_mode"] = tool_args_match_mode
            if tool_args_match_overrides:
                score.metadata["tool_args_match_overrides"] = tool_args_match_overrides
            return score

        except Exception as e:
            logger.exception("TrajectoryUnorderedMatch evaluation failed")
            return _create_error_score(
                "TrajectoryUnorderedMatch",
                f"Evaluation failed: {str(e)}",
                e,
            )

    return scorer


@scorer_registry.register("TrajectorySubsetMatch")
def TrajectorySubsetMatch(  # noqa: N802
    tool_args_match_mode: Literal["exact", "ignore", "subset", "superset"] = "exact",
    tool_args_match_overrides: dict[str, Any] | None = None,
) -> Callable[..., Score]:
    """Create trajectory subset match scorer.

    Subset trajectory evaluation where actual ⊆ reference. Every step in the
    actual trajectory must appear in the reference trajectory, but the agent
    can skip optional steps. Use this to validate minimum required behavior.

    When to use:
    - Testing that agent performs all critical steps (can skip optional ones)
    - Validating minimum viable workflows (auth + query, skip caching)
    - Ensuring required tools are called (agent can call additional tools)
    - Progressive enhancement scenarios (basic features must work)

    Relationship:
        actual ⊆ reference
        ├─ actual = ["auth", "query"]
        └─ reference = ["auth", "cache_check", "query", "log"]
           → PASSES (actual is subset of reference)

    Args:
        tool_args_match_mode: How to compare tool arguments
            - "exact": All args must match exactly (default)
            - "ignore": Only tool names match, args ignored
            - "subset": Actual args ⊆ expected args
            - "superset": Actual args ⊇ expected args
        tool_args_match_overrides: Per-tool custom comparison config

    Returns:
        Scorer function conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.agentevals import TrajectorySubsetMatch
        >>>
        >>> # Agent performs minimum required steps
        >>> scorer = TrajectorySubsetMatch()
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "authenticate", "args": {}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "query_db",
        ...                                  "args": {"q": "data"}}]}
        ...             ]
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "authenticate", "args": {}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "check_cache", "args": {}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "query_db",
        ...                                  "args": {"q": "data"}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "log_metrics", "args": {}}]}
        ...             ]
        ...         }
        ...     )
        ... )
        >>> print(result.passed)  # Passes - actual is subset of reference
        True
        >>>
        >>> # Agent calls extra tool not in reference - fails
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "authenticate", "args": {}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "unauthorized_tool",
        ...                                  "args": {}}]}
        ...             ]
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "authenticate", "args": {}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "query_db", "args": {}}]}
        ...             ]
        ...         }
        ...     )
        ... )
        >>> print(result.passed)  # Fails - unauthorized_tool not in reference
        False

    Note:
        For the inverse (reference ⊆ actual), use TrajectorySupsetMatch instead.
    """

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        try:
            from agentevals.trajectory.match import create_trajectory_match_evaluator
        except ImportError as e:
            return _create_error_score(
                "TrajectorySubsetMatch",
                "agentevals not installed",
                e,
            )

        outputs = result.context.get("outputs") if result.context else None

        # Extract reference_outputs from ExpectedResult (semantically correct location)
        reference_outputs = None
        if isinstance(expected, ExpectedResult) and expected.context:
            reference_outputs = expected.context.get("reference_outputs")

        if outputs is None:
            return _create_error_score(
                "TrajectorySubsetMatch",
                "'outputs' not found in context",
            )

        if not isinstance(outputs, list):
            return _create_error_score(
                "TrajectorySubsetMatch",
                f"'outputs' must be a list, got {type(outputs).__name__}",
            )

        try:
            evaluator = create_trajectory_match_evaluator(
                trajectory_match_mode="subset",
                tool_args_match_mode=tool_args_match_mode,
                tool_args_match_overrides=tool_args_match_overrides,
            )

            eval_result = evaluator(
                outputs=outputs, reference_outputs=reference_outputs
            )

            score = _normalize_agentevals_result(
                wrapper_name="TrajectorySubsetMatch",
                result=eval_result,
                trajectory_length=len(outputs),
                reference_trajectory_length=(
                    len(reference_outputs) if reference_outputs else None
                ),
            )
            score.metadata["tool_args_match_mode"] = tool_args_match_mode
            if tool_args_match_overrides:
                score.metadata["tool_args_match_overrides"] = tool_args_match_overrides
            return score

        except Exception as e:
            logger.exception("TrajectorySubsetMatch evaluation failed")
            return _create_error_score(
                "TrajectorySubsetMatch",
                f"Evaluation failed: {str(e)}",
                e,
            )

    return scorer


@scorer_registry.register("TrajectorySupsetMatch")
def TrajectorySupsetMatch(  # noqa: N802
    tool_args_match_mode: Literal["exact", "ignore", "subset", "superset"] = "exact",
    tool_args_match_overrides: dict[str, Any] | None = None,
) -> Callable[..., Score]:
    """Create trajectory superset match scorer.

    Superset trajectory evaluation where actual ⊇ reference. The agent must
    perform ALL reference steps but can add additional steps. Use this to
    ensure comprehensive workflows that may include enhancements.

    When to use:
    - Ensuring all required baseline steps are executed
    - Validating that agent doesn't skip critical steps (can add extra)
    - Testing enhanced workflows that build on reference implementation
    - Regression testing where new steps are acceptable

    Relationship:
        actual ⊇ reference (reference ⊆ actual)
        ├─ actual = ["auth", "cache_check", "query", "log", "alert"]
        └─ reference = ["auth", "query"]
           → PASSES (reference is subset of actual)

    Args:
        tool_args_match_mode: How to compare tool arguments
            - "exact": All args must match exactly (default)
            - "ignore": Only tool names match, args ignored
            - "subset": Actual args ⊆ expected args
            - "superset": Actual args ⊇ expected args
        tool_args_match_overrides: Per-tool custom comparison config

    Returns:
        Scorer function conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.agentevals import TrajectorySupsetMatch
        >>>
        >>> # Agent performs all required steps plus enhancements
        >>> scorer = TrajectorySupsetMatch()
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "authenticate", "args": {}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "check_cache", "args": {}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "query_db",
        ...                                  "args": {"q": "data"}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "log_metrics", "args": {}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "send_alert", "args": {}}]}
        ...             ]
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "authenticate", "args": {}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "query_db",
        ...                                  "args": {"q": "data"}}]}
        ...             ]
        ...         }
        ...     )
        ... )
        >>> print(result.passed)  # Passes - all reference steps present + extras
        True
        >>>
        >>> # Agent skips required step - fails
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "query_db",
        ...                                  "args": {"q": "data"}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "log_metrics", "args": {}}]}
        ...             ]
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "authenticate", "args": {}}]},
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "query_db",
        ...                                  "args": {"q": "data"}}]}
        ...             ]
        ...         }
        ...     )
        ... )
        >>> print(result.passed)  # Fails - missing authenticate step
        False

    Note:
        For the inverse (actual ⊆ reference), use TrajectorySubsetMatch instead.
    """

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        try:
            from agentevals.trajectory.match import create_trajectory_match_evaluator
        except ImportError as e:
            return _create_error_score(
                "TrajectorySupsetMatch",
                "agentevals not installed",
                e,
            )

        outputs = result.context.get("outputs") if result.context else None

        # Extract reference_outputs from ExpectedResult (semantically correct location)
        reference_outputs = None
        if isinstance(expected, ExpectedResult) and expected.context:
            reference_outputs = expected.context.get("reference_outputs")

        if outputs is None:
            return _create_error_score(
                "TrajectorySupsetMatch",
                "'outputs' not found in context",
            )

        if not isinstance(outputs, list):
            return _create_error_score(
                "TrajectorySupsetMatch",
                f"'outputs' must be a list, got {type(outputs).__name__}",
            )

        try:
            evaluator = create_trajectory_match_evaluator(
                trajectory_match_mode="superset",
                tool_args_match_mode=tool_args_match_mode,
                tool_args_match_overrides=tool_args_match_overrides,
            )

            eval_result = evaluator(
                outputs=outputs, reference_outputs=reference_outputs
            )

            score = _normalize_agentevals_result(
                wrapper_name="TrajectorySupsetMatch",
                result=eval_result,
                trajectory_length=len(outputs),
                reference_trajectory_length=(
                    len(reference_outputs) if reference_outputs else None
                ),
            )
            score.metadata["tool_args_match_mode"] = tool_args_match_mode
            if tool_args_match_overrides:
                score.metadata["tool_args_match_overrides"] = tool_args_match_overrides
            return score

        except Exception as e:
            logger.exception("TrajectorySupsetMatch evaluation failed")
            return _create_error_score(
                "TrajectorySupsetMatch",
                f"Evaluation failed: {str(e)}",
                e,
            )

    return scorer


# =============================================================================
# User Story 3: TrajectoryLLMAsJudge
# =============================================================================


@scorer_registry.register("TrajectoryLLMAsJudge")
def TrajectoryLLMAsJudge(  # noqa: N802
    prompt: str | None = None,
    model: str = "ollama:gpt-oss:20b",
    *,
    judge_base_url: str | None = None,
) -> Callable[..., Coroutine[Any, Any, Score]]:
    """Create trajectory LLM-as-judge scorer.

    Semantic trajectory evaluation using LLM reasoning instead of exact matching.
    The LLM evaluates whether the trajectory achieves the intended goal using
    appropriate reasoning, even if steps differ from reference. Use this for
    flexible evaluation where multiple valid paths exist.

    When to use:
    - Evaluating agent reasoning quality over exact step matching
    - Testing creative problem-solving where multiple approaches are valid
    - Comparing different agent implementations that achieve same goal
    - Situations where reference trajectory is guidance, not strict requirement
    - Assessing trajectory quality when heuristic rules are too rigid

    When NOT to use:
    - Security-critical workflows requiring exact steps (use TrajectoryStrictMatch)
    - Compliance scenarios with mandatory procedures (use TrajectoryStrictMatch)
    - High-volume evaluation where LLM cost/latency is prohibitive
    - Deterministic workflows where heuristic matching is sufficient

    Args:
        prompt: Evaluation prompt template. If None, uses TRAJECTORY_ACCURACY_PROMPT.
            Template should include {outputs} and {reference_outputs} placeholders.
            Customize to focus on specific aspects (efficiency, correctness, safety).
        model: Model identifier in format "provider:model_name"
            Examples:
            - "ollama:gpt-oss:20b" (default - local, free, requires Ollama)
            - "openai:gpt-4o-mini" (faster, paid API)
            - "openai:o3-mini" (most capable, higher cost)
            - "anthropic:claude-3-sonnet" (alternative provider)
        judge_base_url: Endpoint for the judge model. Required, keyword-only,
            and checked for an accepted transport before use — it must encrypt
            in transit or address only the machine this process runs on. The
            model identifier
            above is a provider nickname rather than an address, so without this
            the endpoint would be resolved by the provider SDK and this library
            would have nothing to check — which is why omitting it is an error
            rather than a warning.
            Example: "https://ollama.example.com/v1"

    Returns:
        Scorer function conforming to Scorer Protocol

    Raises:
        ValueError: If judge_base_url is omitted, or names a transport that is
            not accepted.
        RuntimeError: If default Ollama model is unavailable (fail-fast for clear UX)

    Examples:
        >>> from agent_evals.adapters.scorers.agentevals import TrajectoryLLMAsJudge
        >>>
        >>> # Local Ollama judge, reached over TLS
        >>> scorer = TrajectoryLLMAsJudge(
        ...     judge_base_url="https://ollama.example.com/v1",
        ... )
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "search",
        ...                                  "args": {"q": "Python"}}]},
        ...                 {"role": "assistant",
        ...                  "content": "Python is a programming language..."}
        ...             ]
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": [
        ...                 {"role": "assistant",
        ...                  "tool_calls": [{"name": "wikipedia",
        ...                                  "args": {"topic": "Python"}}]},
        ...                 {"role": "assistant",
        ...                  "content": "Python is used for..."}
        ...             ]
        ...         }
        ...     )
        ... )
        >>> # LLM evaluates: Different tools but semantically equivalent
        >>> print(result.passed)
        True
        >>>
        >>> # Custom prompt focusing on efficiency
        >>> efficiency_prompt = '''
        ... Evaluate if the trajectory is efficient:
        ... Actual: {outputs}
        ... Reference: {reference_outputs}
        ...
        ... Consider:
        ... - Minimal tool calls
        ... - No redundant steps
        ... - Optimal order
        ...
        ... Return 1 if efficient, 0 if wasteful.
        ... '''
        >>> efficiency_scorer = TrajectoryLLMAsJudge(
        ...     prompt=efficiency_prompt,
        ...     judge_base_url="https://ollama.example.com/v1",
        ... )
        >>>
        >>> # Use GPT-4 for higher quality evaluation
        >>> premium_scorer = TrajectoryLLMAsJudge(
        ...     model="openai:gpt-4o-mini",
        ...     judge_base_url="https://judge.example.com/v1",
        ... )
        >>>
        >>> # Provide input context for better evaluation
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "inputs": "What is the weather in San Francisco?",
        ...             "outputs": [...]
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": [...]
        ...         }
        ...     )
        ... )
        >>> # LLM can use inputs to better judge trajectory appropriateness

    Note:
        Default Ollama model fails fast if unavailable for clear error messages.
        Custom models return error Score for graceful degradation in production.

        LLM evaluation is probabilistic - same trajectory may get different scores
        across runs. For deterministic evaluation, use heuristic matchers instead.

        The inputs parameter is optional but recommended - it provides the LLM
        with task context to better evaluate trajectory appropriateness.
    """
    # Validate before the import guard below, so a missing or unacceptable
    # endpoint is a configuration error either way rather than being masked by a
    # degraded scorer when the SDK happens to be absent.
    endpoint = _validate_judge_endpoint("TrajectoryLLMAsJudge", judge_base_url)

    # Import and create evaluator at function definition time
    try:
        from agentevals.trajectory.llm import (
            TRAJECTORY_ACCURACY_PROMPT,
            create_async_trajectory_llm_as_judge,
        )
    except ImportError:
        # Can't create evaluator without agentevals
        async def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return _create_error_score(
                "TrajectoryLLMAsJudge",
                "agentevals not installed",
                None,
            )

        return error_scorer

    # Use default prompt if not provided
    if prompt is None:
        prompt = TRAJECTORY_ACCURACY_PROMPT

    # Create evaluator (this may fail if model unavailable)
    try:
        evaluator = create_async_trajectory_llm_as_judge(
            prompt=prompt,
            model=model,
            judge=_build_judge(model, endpoint),
        )
    except Exception as model_err:
        # If default Ollama model fails, raise (fail-fast)
        if model == "ollama:gpt-oss:20b":
            raise RuntimeError(
                f"Default Ollama model unavailable: {model_err}. "
                "Install Ollama or specify alternative model "
                "with model= parameter."
            ) from model_err

        # For custom models, return error scorer (graceful degradation)
        captured_err = model_err  # Capture for closure

        async def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return _create_error_score(
                "TrajectoryLLMAsJudge",
                f"Failed to initialize LLM model '{model}': {captured_err}",
                captured_err,
            )

        return error_scorer

    # Return async scorer function
    async def scorer(
        result: TaskResult, expected: ExpectedResult | None = None
    ) -> Score:
        outputs = result.context.get("outputs") if result.context else None

        # Extract reference_outputs from ExpectedResult (semantically correct location)
        reference_outputs = None
        if isinstance(expected, ExpectedResult) and expected.context:
            reference_outputs = expected.context.get("reference_outputs")

        inputs = (
            result.context.get("inputs") if result.context else None
        )  # Optional: provides additional context

        if outputs is None:
            return _create_error_score(
                "TrajectoryLLMAsJudge",
                "'outputs' not found in context",
            )

        if not isinstance(outputs, list):
            return _create_error_score(
                "TrajectoryLLMAsJudge",
                f"'outputs' must be a list, got {type(outputs).__name__}",
            )

        try:
            # Call LLM evaluator with optional inputs parameter (async)
            eval_result = await evaluator(
                inputs=inputs, outputs=outputs, reference_outputs=reference_outputs
            )

            # Add model info to metadata
            metadata_extra = {
                "model_used": model,
                "prompt_template": prompt[:50] + "..." if len(prompt) > 50 else prompt,
            }

            # Normalize result
            score = _normalize_agentevals_result(
                wrapper_name="TrajectoryLLMAsJudge",
                result=eval_result,
                trajectory_length=len(outputs),
                reference_trajectory_length=(
                    len(reference_outputs) if reference_outputs else None
                ),
            )

            # Merge model metadata
            score.metadata.update(metadata_extra)

            return score

        except Exception as e:
            logger.exception("TrajectoryLLMAsJudge evaluation failed")
            return _create_error_score(
                "TrajectoryLLMAsJudge",
                f"LLM evaluation failed: {str(e)}",
                e,
            )

    return scorer


# =============================================================================
# User Story 4: Graph Trajectory Evaluators
# =============================================================================


@scorer_registry.register("GraphTrajectoryStrictMatch")
def GraphTrajectoryStrictMatch() -> Callable[..., Score]:  # noqa: N802
    """Create graph trajectory strict match scorer.

    Strict evaluation for graph-based agent trajectories (LangGraph, StateGraph).
    Validates exact node sequence and transition order. Use this for deterministic
    graph workflows where the execution path must follow a specific route.

    When to use:
    - LangGraph agents with deterministic state transitions
    - Testing that graph agents follow expected node sequences
    - Validating conditional branching logic (different paths for different inputs)
    - Regression testing graph-based workflows

    Graph Format:
        Graph trajectories use dict format with 'steps' key:
        {
            "steps": [
                ["__start__", "agent", "tools", "__end__"],
                ["__start__", "validator", "processor", "saver", "__end__"]
            ]
        }

        Each inner list is a complete path through the graph.
        Multiple paths indicate branching or loops.

    Returns:
        Scorer function conforming to Scorer Protocol

    Examples:
        >>> from agent_evals.adapters.scorers.agentevals import (
        ...     GraphTrajectoryStrictMatch,
        ... )
        >>>
        >>> # Linear graph path
        >>> scorer = GraphTrajectoryStrictMatch()
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": {"steps": [["__start__", "agent", "tools",
        ...                                     "__end__"]]}
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": {"steps": [["__start__", "agent",
        ...                                                "tools", "__end__"]]}
        ...         }
        ...     )
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Conditional branching
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": {"steps": [["__start__", "validator",
        ...                                     "error_handler", "__end__"]]}
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": {"steps": [["__start__", "validator",
        ...                                                "error_handler", "__end__"]]}
        ...         }
        ...     )
        ... )
        >>> print(result.passed)
        True
        >>>
        >>> # Loop detection
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": {"steps": [["__start__", "agent", "tools",
        ...                                     "agent", "tools", "__end__"]]}
        ...         }
        ...     ),
        ...     expected=ExpectedResult(
        ...         expected="",
        ...         context={
        ...             "reference_outputs": {"steps": [["__start__", "agent",
        ...                                                "tools", "agent", "tools",
        ...                                                "__end__"]]}
        ...         }
        ...     )
        ... )
        >>> print(result.passed)
        True

    Note:
        Graph trajectories MUST be dict with 'steps' key.
        Simple list format is NOT supported - will return error Score.

        For message-based trajectories (OpenAI format), use TrajectoryStrictMatch.
    """

    def scorer(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        try:
            from agentevals.graph_trajectory.strict import graph_trajectory_strict_match
        except ImportError as e:
            return _create_error_score(
                "GraphTrajectoryStrictMatch",
                "agentevals not installed",
                e,
            )

        # Extract outputs from task execution
        outputs = result.context.get("outputs") if result.context else None

        # Extract reference_outputs from ExpectedResult (semantically correct location)
        reference_outputs = None
        if isinstance(expected, ExpectedResult) and expected.context:
            reference_outputs = expected.context.get("reference_outputs")

        if outputs is None:
            return _create_error_score(
                "GraphTrajectoryStrictMatch",
                "'outputs' not found in context",
            )

        # Validate graph trajectory format (must be dict with 'steps' key)
        if isinstance(outputs, list):
            return _create_error_score(
                "GraphTrajectoryStrictMatch",
                "Graph trajectories must be dict with 'steps' key, not a simple list. "
                "Use format: {'steps': [['__start__', 'agent', 'tools', '__end__']]}",
            )

        if not isinstance(outputs, dict) or "steps" not in outputs:
            return _create_error_score(
                "GraphTrajectoryStrictMatch",
                "Graph trajectories must be dict with 'steps' key",
            )

        try:
            # Call graph trajectory evaluator (direct function, not factory)
            # Note: graph_trajectory_strict_match does NOT take inputs parameter
            eval_result = graph_trajectory_strict_match(
                outputs=cast("GraphTrajectory", outputs),
                reference_outputs=cast("GraphTrajectory", reference_outputs),
            )

            # Calculate trajectory length from steps
            trajectory_length = None
            if "steps" in outputs:
                trajectory_length = sum(len(step) for step in outputs["steps"])

            reference_trajectory_length = None
            if reference_outputs and "steps" in reference_outputs:
                reference_trajectory_length = sum(
                    len(step) for step in reference_outputs["steps"]
                )

            # Normalize result
            score = _normalize_agentevals_result(
                wrapper_name="GraphTrajectoryStrictMatch",
                result=eval_result,
                trajectory_length=trajectory_length,
                reference_trajectory_length=reference_trajectory_length,
            )

            # Add graph nodes to metadata
            if "steps" in outputs:
                # Flatten steps into single list of nodes
                all_nodes = [node for step in outputs["steps"] for node in step]
                score.metadata["graph_nodes"] = all_nodes

            return score

        except Exception as e:
            logger.exception("GraphTrajectoryStrictMatch evaluation failed")
            return _create_error_score(
                "GraphTrajectoryStrictMatch",
                f"Evaluation failed: {str(e)}",
                e,
            )

    return scorer


@scorer_registry.register("GraphTrajectoryLLMAsJudge")
def GraphTrajectoryLLMAsJudge(  # noqa: N802
    prompt: str | None = None,
    model: str = "ollama:gpt-oss:20b",
    *,
    judge_base_url: str | None = None,
) -> Callable[..., Coroutine[Any, Any, Score]]:
    """Create graph trajectory LLM-as-judge scorer.

    Semantic evaluation for graph-based trajectories using LLM reasoning.
    The LLM evaluates whether the graph execution path achieves the goal
    appropriately, even if the exact node sequence differs from reference.

    When to use:
    - Evaluating LangGraph agents where multiple valid paths exist
    - Testing graph reasoning quality over exact node matching
    - Validating that agent reaches goal via reasonable route
    - Comparing different graph implementations that solve same problem
    - Assessing conditional branching logic qualitatively

    When NOT to use:
    - Deterministic graphs with single valid path (use GraphTrajectoryStrictMatch)
    - High-volume testing where LLM cost/latency is prohibitive
    - Compliance scenarios requiring exact node sequences

    Args:
        prompt: Evaluation prompt template. If None, uses TRAJECTORY_ACCURACY_PROMPT.
            Template should include {outputs} and {reference_outputs} placeholders.
            Customize to focus on graph-specific aspects (efficiency, loop handling).
        model: Model identifier in format "provider:model_name"
            Examples:
            - "ollama:gpt-oss:20b" (default - local, free)
            - "openai:gpt-4o-mini" (faster, paid)
            - "openai:o3-mini" (most capable)
        judge_base_url: Endpoint for the judge model. Required, keyword-only,
            and checked for an accepted transport before use — it must encrypt
            in transit or address only the machine this process runs on. The
            model identifier
            above is a provider nickname rather than an address, so without this
            the endpoint would be resolved by the provider SDK and this library
            would have nothing to check — which is why omitting it is an error
            rather than a warning.
            Example: "https://ollama.example.com/v1"

    Returns:
        Scorer function conforming to Scorer Protocol

    Raises:
        ValueError: If judge_base_url is omitted, or names a transport that is
            not accepted.
        RuntimeError: If default Ollama model is unavailable (fail-fast)

    Examples:
        >>> from agent_evals.adapters.scorers.agentevals import (
        ...     GraphTrajectoryLLMAsJudge,
        ... )
        >>>
        >>> # Local Ollama judge, reached over TLS
        >>> scorer = GraphTrajectoryLLMAsJudge(
        ...     judge_base_url="https://ollama.example.com/v1",
        ... )
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": {
        ...                 "steps": [["__start__", "planner", "executor", "__end__"]]
        ...             },
        ...             "reference_outputs": {
        ...                 "steps": [["__start__", "agent", "tools", "__end__"]]
        ...             }
        ...         }
        ...     )
        ... )
        >>> # LLM evaluates: Different nodes but semantically equivalent path
        >>> print(result.passed)
        True
        >>>
        >>> # Custom prompt for loop evaluation
        >>> loop_prompt = '''
        ... Evaluate if the graph trajectory handles loops appropriately:
        ... Actual: {outputs}
        ... Reference: {reference_outputs}
        ...
        ... Check:
        ... - No infinite loops
        ... - Reasonable iteration count
        ... - Progress toward goal
        ...
        ... Return 1 if acceptable, 0 if problematic.
        ... '''
        >>> loop_scorer = GraphTrajectoryLLMAsJudge(
        ...     prompt=loop_prompt,
        ...     judge_base_url="https://ollama.example.com/v1",
        ... )
        >>>
        >>> # A hosted judge instead of a local one
        >>> premium_scorer = GraphTrajectoryLLMAsJudge(
        ...     model="openai:gpt-4o-mini",
        ...     judge_base_url="https://judge.example.com/v1",
        ... )
        >>>
        >>> # Evaluate branching logic
        >>> result = scorer(
        ...     TaskResult(
        ...         output="",
        ...         context={
        ...             "outputs": {
        ...                 "steps": [[
                    "__start__", "validator", "error_handler", "__end__"
                ]]
        ...             },
        ...             "reference_outputs": {
        ...                 "steps": [[
                    "__start__", "validator", "processor", "__end__"
                ]]
        ...             },
        ...             "inputs": {"has_error": True}  # Context for branching decision
        ...         }
        ...     )
        ... )
        >>> # LLM evaluates: Correct branch taken based on input state

    Note:
        Graph trajectories MUST be dict with 'steps' key.
        The 'inputs' context key can provide state information for evaluation.

        LLM evaluation is probabilistic - use for flexibility, not compliance.
    """
    # Validate before the import guard below, so a missing or unacceptable
    # endpoint is a configuration error either way rather than being masked by a
    # degraded scorer when the SDK happens to be absent.
    endpoint = _validate_judge_endpoint("GraphTrajectoryLLMAsJudge", judge_base_url)

    try:
        from agentevals.graph_trajectory.llm import (
            create_async_graph_trajectory_llm_as_judge,
        )
        from agentevals.trajectory.llm import TRAJECTORY_ACCURACY_PROMPT
    except ImportError:

        async def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return _create_error_score(
                "GraphTrajectoryLLMAsJudge",
                "agentevals not installed",
                None,
            )

        return error_scorer

    if prompt is None:
        prompt = TRAJECTORY_ACCURACY_PROMPT

    try:
        evaluator = create_async_graph_trajectory_llm_as_judge(
            prompt=prompt,
            model=model,
            judge=_build_judge(model, endpoint),
        )
    except Exception as model_err:
        if model == "ollama:gpt-oss:20b":
            raise RuntimeError(
                f"Default Ollama model unavailable: {model_err}. "
                "Install Ollama or specify alternative model."
            ) from model_err

        captured_err = model_err  # Capture for closure

        async def error_scorer(
            result: TaskResult, expected: ExpectedResult | None = None
        ) -> Score:
            return _create_error_score(
                "GraphTrajectoryLLMAsJudge",
                f"Failed to initialize model '{model}': {captured_err}",
                captured_err,
            )

        return error_scorer

    async def scorer(
        result: TaskResult, expected: ExpectedResult | None = None
    ) -> Score:
        outputs = result.context.get("outputs") if result.context else None

        # Extract reference_outputs from ExpectedResult (semantically correct location)
        reference_outputs = None
        if isinstance(expected, ExpectedResult) and expected.context:
            reference_outputs = expected.context.get("reference_outputs")

        inputs = (
            result.context.get("inputs", []) if result.context else []
        )  # Default to empty list - agentevals requires it

        if outputs is None:
            return _create_error_score(
                "GraphTrajectoryLLMAsJudge",
                "'outputs' not found in context",
            )

        if not isinstance(outputs, dict) or "steps" not in outputs:
            return _create_error_score(
                "GraphTrajectoryLLMAsJudge",
                "Graph trajectories must be dict with 'steps' key",
            )

        try:
            # Call evaluator (async)
            eval_result = await evaluator(
                inputs=inputs,
                outputs=outputs,
                reference_outputs=reference_outputs,
            )

            trajectory_length = None
            if "steps" in outputs:
                trajectory_length = sum(len(step) for step in outputs["steps"])

            reference_trajectory_length = None
            if reference_outputs and "steps" in reference_outputs:
                reference_trajectory_length = sum(
                    len(step) for step in reference_outputs["steps"]
                )

            score = _normalize_agentevals_result(
                wrapper_name="GraphTrajectoryLLMAsJudge",
                result=eval_result,
                trajectory_length=trajectory_length,
                reference_trajectory_length=reference_trajectory_length,
            )

            score.metadata.update(
                {
                    "model_used": model,
                    "prompt_template": prompt[:50] + "..."
                    if len(prompt) > 50
                    else prompt,
                }
            )

            if "steps" in outputs:
                all_nodes = [node for step in outputs["steps"] for node in step]
                score.metadata["graph_nodes"] = all_nodes

            return score

        except Exception as e:
            logger.exception("GraphTrajectoryLLMAsJudge evaluation failed")
            return _create_error_score(
                "GraphTrajectoryLLMAsJudge",
                f"LLM evaluation failed: {str(e)}",
                e,
            )

    return scorer
