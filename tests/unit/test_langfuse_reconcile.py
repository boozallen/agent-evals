# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Defensive reconciliation for items dropped from ``item_results``.

The safe-task wrapper prevents this in production, but if any path ever
bypasses it (e.g. a ``BaseException`` subclass we deliberately don't
catch), the adapter must still surface the dropped item rather than
silently lose it. Tests construct real ``ExperimentResult`` /
``ExperimentItemResult`` instances so SDK shape drift is visible.
"""

from typing import Any

from langfuse.experiment import ExperimentItemResult, ExperimentResult

from agent_evals.adapters.platforms.langfuse import LangFusePlatform
from agent_evals.core.types import (
    ExampleData,
    ExpectedResult,
    TaskResult,
)


def _item(item: Any, output: Any = None) -> ExperimentItemResult:
    return ExperimentItemResult(
        item=item,
        output=output,
        evaluations=[],
        trace_id=None,
        dataset_run_id=None,
    )


def _experiment_result(
    item_results: list[ExperimentItemResult],
) -> ExperimentResult:
    return ExperimentResult(
        name="t",
        run_name="t-run",
        description="",
        item_results=item_results,
        run_evaluations=[],
        experiment_id="exp-id",
    )


def _dataset(n: int = 2) -> list[ExampleData]:
    return [
        ExampleData(
            input=f"input-{i}",
            expected=ExpectedResult(expected=f"input-{i}"),
        )
        for i in range(n)
    ]


def _langfuse_dataset(dataset: list[ExampleData]) -> list[dict[str, Any]]:
    return [{"input": ex.input, "expected_output": ex.expected} for ex in dataset]


class TestReconcileDroppedItems:
    def test_dropped_failed_item_reconciled_with_error(self):
        adapter = LangFusePlatform()
        ds = _dataset(2)
        lf_ds = _langfuse_dataset(ds)
        # Only the second item made it through (SDK dropped the first).
        platform = _experiment_result([_item(lf_ds[1], output="out-1")])

        examples = adapter._build_examples_from_platform_result(
            platform,
            ds,
            item_errors={id(lf_ds[0]): "boom on input-0"},
            item_durations={id(lf_ds[0]): 0.05, id(lf_ds[1]): 0.15},
            langfuse_dataset=lf_ds,
        )

        by_input = {ex.input: ex for ex in examples}
        assert set(by_input) == {"input-0", "input-1"}
        assert by_input["input-0"].error == "boom on input-0"
        assert by_input["input-1"].error is None

    def test_failures_at_non_terminal_indices_keep_correct_alignment(self):
        """Failure between successes must not shift other items' alignment."""
        adapter = LangFusePlatform()
        ds = _dataset(4)
        lf_ds = _langfuse_dataset(ds)
        platform = _experiment_result(
            [
                _item(lf_ds[1], output="ok-1"),
                _item(lf_ds[3], output="ok-3"),
            ]
        )

        examples = adapter._build_examples_from_platform_result(
            platform,
            ds,
            item_errors={
                id(lf_ds[0]): "fail-0",
                id(lf_ds[2]): "fail-2",
            },
            item_durations={},
            langfuse_dataset=lf_ds,
        )

        errors_by_input = {ex.input: ex.error for ex in examples}
        assert errors_by_input == {
            "input-0": "fail-0",
            "input-1": None,
            "input-2": "fail-2",
            "input-3": None,
        }

    def test_all_items_dropped_all_reconciled(self):
        adapter = LangFusePlatform()
        ds = _dataset(3)
        lf_ds = _langfuse_dataset(ds)
        platform = _experiment_result([])

        examples = adapter._build_examples_from_platform_result(
            platform,
            ds,
            item_errors={
                id(lf_ds[0]): "err-0",
                id(lf_ds[1]): "err-1",
                id(lf_ds[2]): "err-2",
            },
            item_durations={},
            langfuse_dataset=lf_ds,
        )

        assert len(examples) == 3
        assert {ex.error for ex in examples} == {"err-0", "err-1", "err-2"}

    def test_drop_with_no_sidecar_signals_pre_wrapper_failure(self):
        """SDK rejected the item before our wrapper ran (no duration recorded)."""
        adapter = LangFusePlatform()
        ds = _dataset(2)
        lf_ds = _langfuse_dataset(ds)
        platform = _experiment_result([_item(lf_ds[1], output="out-1")])

        examples = adapter._build_examples_from_platform_result(
            platform,
            ds,
            item_errors={},
            item_durations={},
            langfuse_dataset=lf_ds,
        )

        err = next(ex.error for ex in examples if ex.input == "input-0") or ""
        assert "did not produce" in err
        assert "before wrapper ran" in err

    def test_drop_with_duration_but_no_error_signals_base_exception(self):
        """Wrapper ran (duration recorded) but `except Exception` didn't fire —
        e.g. CancelledError or KeyboardInterrupt escaped the wrapper.
        """
        adapter = LangFusePlatform()
        ds = _dataset(2)
        lf_ds = _langfuse_dataset(ds)
        platform = _experiment_result([_item(lf_ds[1], output="out-1")])

        examples = adapter._build_examples_from_platform_result(
            platform,
            ds,
            item_errors={},
            # Duration was recorded by the wrapper's `finally`, but no error.
            item_durations={id(lf_ds[0]): 0.01},
            langfuse_dataset=lf_ds,
        )

        err = next(ex.error for ex in examples if ex.input == "input-0") or ""
        assert "interrupted" in err
        assert "BaseException" in err


class TestReadItemField:
    """``_read_item_field`` handles both dict and object item shapes."""

    def test_reads_from_dict_item(self):
        assert LangFusePlatform._read_item_field({"input": "x"}, "input") == "x"

    def test_reads_from_object_item(self):
        class _Obj:
            input = "y"

        assert LangFusePlatform._read_item_field(_Obj(), "input") == "y"

    def test_missing_field_returns_none(self):
        assert LangFusePlatform._read_item_field({}, "input") is None
        assert LangFusePlatform._read_item_field(object(), "input") is None

    def test_none_item_returns_none(self):
        assert LangFusePlatform._read_item_field(None, "input") is None


class TestTaskResultOutputUnwrapping:
    def test_taskresult_in_item_results_unwrapped_to_string(self):
        """``EvalExample.output`` carries the inner string, not the wrapper."""
        adapter = LangFusePlatform()
        ds = _dataset(1)
        lf_ds = _langfuse_dataset(ds)
        platform = _experiment_result(
            [_item(lf_ds[0], output=TaskResult(output="unwrapped"))]
        )

        examples = adapter._build_examples_from_platform_result(
            platform,
            ds,
            item_errors={},
            item_durations={},
            langfuse_dataset=lf_ds,
        )
        assert examples[0].output == "unwrapped"

    def test_taskresult_with_trajectory_propagates_to_eval_example(self):
        """TaskResult carrying context['outputs'] populates trajectory."""
        adapter = LangFusePlatform()
        ds = _dataset(1)
        lf_ds = _langfuse_dataset(ds)
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        tr = TaskResult.from_messages(messages)
        platform = _experiment_result([_item(lf_ds[0], output=tr)])

        examples = adapter._build_examples_from_platform_result(
            platform,
            ds,
            item_errors={},
            item_durations={},
            langfuse_dataset=lf_ds,
        )
        assert examples[0].trajectory == messages
        assert examples[0].tool_calls == []

    def test_non_taskresult_output_yields_none_trajectory(self):
        """When item_results.output is a bare string (not TaskResult),
        EvalExample.trajectory must be None — pin against future SDK
        shape drift where output is wrapped or stringified differently.
        """
        adapter = LangFusePlatform()
        ds = _dataset(1)
        lf_ds = _langfuse_dataset(ds)
        platform = _experiment_result([_item(lf_ds[0], output="bare-string-output")])

        examples = adapter._build_examples_from_platform_result(
            platform,
            ds,
            item_errors={},
            item_durations={},
            langfuse_dataset=lf_ds,
        )
        assert examples[0].output == "bare-string-output"
        assert examples[0].trajectory is None
        assert examples[0].tool_calls is None
