# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""
Validation tests for backward compatibility with sync run_eval API.

Tests User Story 5: Backward Compatibility
- Validates all existing examples work without modification
- Ensures sync run_eval produces identical results
- Verifies sync API can be called from sync contexts
"""

import pytest

from agent_evals import ExpectedResult, Score, TaskResult, run_eval
from agent_evals.core.types import ExampleData


def test_sync_run_eval_with_simple_scorer():
    """Test sync run_eval API works with simple sync scorer.

    This validates the sync wrapper around async implementation.

    Task: T046, T047
    """

    # Simple sync scorer
    def exact_match(
        result: TaskResult, expected: ExpectedResult | None = None
    ) -> Score:
        expected_value = expected.expected if expected else ""
        matches = result.output == expected_value
        return Score(name="ExactMatch", value=1.0 if matches else 0.0, passed=matches)

    # Simple task
    def task(input_value):
        return TaskResult(output=input_value)

    # Run evaluation using SYNC API
    result = run_eval(
        task=task,
        dataset=[
            ExampleData(input="hello", expected=ExpectedResult(expected="hello")),
            ExampleData(input="world", expected=ExpectedResult(expected="world")),
        ],
        scorers=[exact_match],
    )

    # Verify EvalResult structure matches expected contract
    assert hasattr(result, "experiment_id")
    assert hasattr(result, "experiment_url")
    assert hasattr(result, "platform")
    assert hasattr(result, "scores")
    assert hasattr(result, "examples")
    assert hasattr(result, "summary")
    assert hasattr(result, "duration")

    # Verify results are correct
    assert result.platform == "local"
    assert len(result.examples) == 2
    assert "ExactMatch" in result.scores
    assert result.scores["ExactMatch"] == 1.0  # Both examples match


def test_sync_run_eval_with_multiple_scorers():
    """Test sync run_eval with multiple scorers.

    Task: T046
    """

    # Multiple sync scorers
    def scorer_1(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        return Score(name="Scorer1", value=0.9, passed=True)

    def scorer_2(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        return Score(name="Scorer2", value=0.85, passed=True)

    def scorer_3(result: TaskResult, expected: ExpectedResult | None = None) -> Score:
        return Score(name="Scorer3", value=0.95, passed=True)

    # Task
    def task(input_value):
        return TaskResult(output=f"Output: {input_value}")

    # Run sync evaluation
    result = run_eval(
        task=task,
        dataset=[ExampleData(input="test", expected=ExpectedResult(expected="test"))],
        scorers=[scorer_1, scorer_2, scorer_3],
    )

    # Verify all scorers executed
    assert len(result.examples[0].scores) == 3
    assert "Scorer1" in result.examples[0].scores
    assert "Scorer2" in result.examples[0].scores
    assert "Scorer3" in result.examples[0].scores


def test_sync_run_eval_with_context_passing():
    """Test sync run_eval with context passthrough from dataset.

    Validates that context from dataset's 'context' key and pass-through
    fields are correctly merged and passed to scorers.

    Task: T046
    """
    received_context = {}

    def context_scorer(
        result: TaskResult, expected: ExpectedResult | None = None
    ) -> Score:
        if result.context:
            received_context.update(result.context)
        return Score(name="ContextScorer", value=1.0, passed=True)

    def task(input_value):
        return TaskResult(output=f"Output: {input_value}")

    # Run sync evaluation with custom context
    result = run_eval(
        task=task,
        dataset=[
            ExampleData(
                input="test",
                expected=ExpectedResult(expected="Output: test"),
                output=TaskResult(
                    output="Output: test",
                    context={
                        "custom_field": "custom_value",
                        "extra": "data",
                    },
                ),
            )
        ],
        scorers=[context_scorer],
    )

    # Verify context was passed correctly
    assert len(result.examples) == 1
    assert "ContextScorer" in result.scores

    # Verify context passthrough works (from 'context' key and pass-through fields)
    assert "custom_field" in received_context
    assert "extra" in received_context
    assert received_context["custom_field"] == "custom_value"
    assert received_context["extra"] == "data"


def test_sync_run_eval_produces_identical_structure_to_async():
    """Test sync run_eval produces identical EvalResult structure to async.

    This ensures backward compatibility - existing code expecting certain
    fields in EvalResult continues to work.

    Task: T047
    """

    def simple_scorer(
        result: TaskResult, expected: ExpectedResult | None = None
    ) -> Score:
        return Score(name="SimpleScorer", value=1.0, passed=True)

    def task(input_value):
        return TaskResult(output=input_value)

    # Run sync evaluation
    sync_result = run_eval(
        task=task,
        dataset=[ExampleData(input="test", expected=ExpectedResult(expected="test"))],
        scorers=[simple_scorer],
    )

    # Verify all expected fields exist
    assert isinstance(sync_result.experiment_id, str)
    assert isinstance(sync_result.experiment_url, str)
    assert sync_result.platform == "local"
    assert isinstance(sync_result.scores, dict)
    assert isinstance(sync_result.examples, list)
    assert isinstance(sync_result.summary, dict)
    assert isinstance(sync_result.duration, float)

    # Verify examples structure
    assert len(sync_result.examples) == 1
    example = sync_result.examples[0]
    assert hasattr(example, "input")
    assert hasattr(example, "output")
    assert hasattr(example, "expected")
    assert hasattr(example, "scores")
    assert hasattr(example, "metadata")
    assert hasattr(example, "duration")

    # Verify scores structure
    assert "SimpleScorer" in example.scores
    score = example.scores["SimpleScorer"]
    assert hasattr(score, "name")
    assert hasattr(score, "value")
    assert hasattr(score, "passed")


def test_sync_run_eval_callable_from_sync_context():
    """Test calling sync run_eval from sync context works (no RuntimeError).

    This is critical - users should be able to call run_eval from regular
    Python scripts without async/await.

    Task: T049
    """

    def simple_scorer(
        result: TaskResult, expected: ExpectedResult | None = None
    ) -> Score:
        return Score(name="SyncScorer", value=1.0, passed=True)

    def task(input_value):
        return TaskResult(output=input_value)

    # This should NOT raise RuntimeError about event loop
    try:
        result = run_eval(
            task=task,
            dataset=[ExampleData(input="test")],
            scorers=[simple_scorer],
        )

        # Verify it completed successfully
        assert len(result.examples) == 1
        assert "SyncScorer" in result.examples[0].scores

    except RuntimeError as e:
        # If we get RuntimeError about event loop, that's a failure
        if "event loop" in str(e).lower():
            pytest.fail(f"Sync run_eval should not raise event loop error: {e}")
        raise


def test_sync_run_eval_with_scorer_error_handling():
    """Test sync run_eval handles scorer errors correctly.

    Task: T046
    """

    def failing_scorer(
        result: TaskResult, expected: ExpectedResult | None = None
    ) -> Score:
        raise ValueError("Scorer intentionally failed")

    def passing_scorer(
        result: TaskResult, expected: ExpectedResult | None = None
    ) -> Score:
        return Score(name="PassingScorer", value=1.0, passed=True)

    def task(input_value):
        return TaskResult(output=input_value)

    # Run sync evaluation with failing scorer
    result = run_eval(
        task=task,
        dataset=[ExampleData(input="test")],
        scorers=[failing_scorer, passing_scorer],
    )

    # Verify both scorers were executed (failure isolation)
    assert len(result.examples[0].scores) == 2

    # Verify passing scorer succeeded
    assert "PassingScorer" in result.examples[0].scores
    assert result.examples[0].scores["PassingScorer"].passed is True

    # Verify failing scorer produced Error Score
    assert "failing_scorer" in result.examples[0].scores
    failing_score = result.examples[0].scores["failing_scorer"]
    assert failing_score.passed is False
    assert failing_score.value == 0.0
    assert "error" in failing_score.metadata


def test_every_public_name_resolves():
    """Every name in ``agent_evals.__all__`` must resolve.

    Guards against drift between ``__all__`` and the ``_LAZY`` dispatch
    table in ``agent_evals/__init__.py``: a contributor adding to
    ``__all__`` without wiring a ``_load_*`` helper into ``_LAZY``
    would otherwise produce ``AttributeError`` only at first use.
    """
    import agent_evals

    unresolved = []
    for name in agent_evals.__all__:
        try:
            getattr(agent_evals, name)
        except AttributeError as exc:
            unresolved.append((name, str(exc)))
    assert not unresolved, f"unresolved names in __all__: {unresolved}"


def test_public_api_matches_audit_keep_list():
    """``agent_evals.__all__`` matches the issue-198 KEEP list exactly.

    Companion to the negative drop tests: this positive whitelist catches
    *accidental further shrinkage* of the public surface (e.g., a refactor
    that drops ``BaseAgent`` or ``EvalExample`` without an audit). Adding
    a name here without an issue + rationale fails CONTRIBUTING.md's
    "Adding to ``__all__``" checklist.
    """
    expected_keeps = {
        # Entry points
        "run_eval",
        "run_eval_async",
        "run_benchmark_async",
        # Benchmark return type + ABC
        "BenchmarkResult",
        "BaseAgent",
        # Pydantic value types users construct / annotate
        "Score",
        "EvalResult",
        "EvalExample",
        "ExampleData",
        "TaskResult",
        "ExpectedResult",
        # Framework converters
        "langchain_to_openai",
        "strands_to_openai",
        "mink_to_openai",
        # Encrypted artifact readers
        "read_encrypted_file",
        "read_encrypted_jsonl",
        # Integrity verification
        "IntegrityError",
    }

    import agent_evals

    actual = set(agent_evals.__all__)
    missing = expected_keeps - actual
    extra = actual - expected_keeps
    assert not missing and not extra, (
        f"agent_evals.__all__ drift: missing={sorted(missing)} extra={sorted(extra)}. "
        f"Per CONTRIBUTING.md, additions need an audit + CLAUDE.md rationale; "
        f"removals need an issue + sister-repo evidence."
    )


def test_public_api_drops_unused_surface_per_issue_198():
    """Names dropped by issue #198 must NOT be in agent_evals.__all__.

    Issue #198 audited every name in __all__ against import statements
    across 5 sister projects + 7 .scratch projects. Names with zero
    verified consumers fail the "documented user need" test and were
    dropped from the public surface. Each remains reachable from its
    owning module (core.types, core.ports, adapters.<family>) for the
    rare advanced case; the audit removed only the package-root
    re-export.

    Spec 023's 5 default registries (originally added without
    demonstrated user need) were renamed without the `default_` prefix
    by spec 027; this test asserts they're gone from the public
    surface entirely. The Protocol types (AdapterRegistry,
    PreconditionApplier, BenchmarkConfigReader) are reachable from
    core.ports; concrete adapter classes (StateCheckParser, etc.) are
    reachable from their family modules.
    """
    dropped_names = {
        # Side-effecting wrapper removed by spec 031; run_benchmark_async is now public.
        "run_and_report_async",
        "write_report",
        # Config types users accept as defaults.
        "EvalConfig",
        # Protocols with zero external implementers.
        "AdapterRegistry",
        "PreconditionApplier",
        "BenchmarkConfigReader",
        # Concrete adapter classes dispatched via YAML keys.
        "StateCheckParser",
        "ToolUseCheckParser",
        "StatePreconditionApplier",
        "LocalFileConfigReader",
        # Registry instances with zero external decorator uses.
        "scorer_registry",
        "platform_registry",
        "success_checker_registry",
        "precondition_registry",
        "config_reader_registry",
    }

    import agent_evals

    still_public = sorted(name for name in dropped_names if name in agent_evals.__all__)
    assert not still_public, (
        f"Names dropped by issue #198 are still in __all__: {still_public}. "
        f"Currently in __all__: {agent_evals.__all__}"
    )


def test_dropped_names_unreachable_from_package_root_per_issue_198():
    """Names dropped by issue #198 must raise AttributeError on package-root access.

    Companion to ``test_public_api_drops_unused_surface_per_issue_198``: the
    audit's contract is a *runtime* removal, not just an ``__all__``
    membership change. Without this test, a contributor restoring any of
    these names to ``_LAZY`` (without re-adding to ``__all__``) would
    silently re-create the public surface — ``agent_evals.<name>`` would
    resolve and ``from agent_evals import <name>`` would succeed.
    """
    dropped_names = {
        "run_and_report_async",
        "write_report",
        "EvalConfig",
        "AdapterRegistry",
        "PreconditionApplier",
        "BenchmarkConfigReader",
        "StateCheckParser",
        "ToolUseCheckParser",
        "StatePreconditionApplier",
        "LocalFileConfigReader",
        "scorer_registry",
        "platform_registry",
        "success_checker_registry",
        "precondition_registry",
        "config_reader_registry",
    }

    import agent_evals

    still_resolvable = []
    for name in sorted(dropped_names):
        try:
            getattr(agent_evals, name)
            still_resolvable.append(name)
        except AttributeError:
            pass
    assert not still_resolvable, (
        f"Names dropped by issue #198 still resolve via getattr(agent_evals, ...): "
        f"{still_resolvable}. Likely cause: name was removed from __all__ but "
        f"left in agent_evals/__init__.py:_LAZY."
    )


def test_public_api_drops_register_parse_list_free_functions():
    """The -9 names removed by spec 023 must NOT be importable from agent_evals.

    Zero downstream consumers of these names per the spec 023 audit;
    they were aspirational public API that no one used. The free
    functions exist only as bound methods on the corresponding
    <family>_registry instances.
    """
    removed_names = {
        "register_success_checker",
        "parse_success_checker",
        "list_success_checkers",
        "register_precondition",
        "parse_precondition",
        "list_preconditions",
        "register_config_reader",
        "parse_config_reader",
        "list_config_readers",
    }

    import agent_evals

    for name in removed_names:
        assert name not in agent_evals.__all__, (
            f"{name!r} should be removed from agent_evals.__all__ per "
            f"spec 023. Found in: {agent_evals.__all__}"
        )


def test_run_benchmark_async_is_public_and_pure():
    """031: run_benchmark_async is the public benchmark entry point."""
    import agent_evals

    assert "run_benchmark_async" in agent_evals.__all__
    assert callable(agent_evals.run_benchmark_async)


def test_run_and_report_async_removed():
    """031: the side-effecting wrapper is gone from the public surface."""
    import agent_evals

    assert "run_and_report_async" not in agent_evals.__all__
    with pytest.raises(AttributeError):
        _ = agent_evals.run_and_report_async
