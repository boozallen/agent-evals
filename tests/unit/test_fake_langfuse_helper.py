# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Composition tests for the FakeLangfuseClient test helper.

These pin the fake's behavior so future rewrites of the langfuse
adapter (or future test rewrites that consume the fake) can rely on a
documented contract. The fake is a test helper, not part of the public
API; these tests are the only place its behavior is specified.
"""

import pytest
from helpers.fake_langfuse import (
    FakeLangfuseClient,
    FakeTraceNotFoundError,
)
from langfuse import Evaluation

from agent_evals.adapters.platforms.langfuse_client import DatasetItemRecord
from agent_evals.core.types import TaskResult


class TestFakeLangfuseClientBasics:
    def test_auth_check_default_returns_true(self):
        fake = FakeLangfuseClient()
        assert fake.auth_check() is True

    def test_auth_check_can_be_overridden_to_false(self):
        fake = FakeLangfuseClient(auth_ok=False)
        assert fake.auth_check() is False


class TestFakeLangfuseClientRunExperiment:
    def test_invokes_task_on_each_data_item(self):
        seen_inputs: list = []

        def task(item):
            seen_inputs.append(item)
            return TaskResult(output="ok")

        fake = FakeLangfuseClient()
        result = fake.run_experiment(
            name="t",
            description="",
            data=[{"input": "a"}, {"input": "b"}],
            task=task,
            evaluators=[],
            metadata={},
        )

        assert seen_inputs == [{"input": "a"}, {"input": "b"}], (
            "fake must invoke task once per data item with the dict-shaped item, "
            "matching the langfuse SDK's task(item={...}) calling shape"
        )
        assert len(result.item_results) == 2

    def test_runs_evaluators_on_outputs(self):
        recorded: list = []

        def scorer_evaluator(*, input, output, expected_output, metadata, **kwargs):
            recorded.append((input, output, expected_output))
            return Evaluation(name="spy", value=1.0, comment="")

        fake = FakeLangfuseClient()
        fake.run_experiment(
            name="t",
            description="",
            data=[{"input": "q", "expected_output": "a"}],
            task=lambda item: TaskResult(output="r"),
            evaluators=[scorer_evaluator],
            metadata={},
        )

        assert len(recorded) == 1
        assert recorded[0][0] == "q"

    def test_records_call_for_inspection(self):
        fake = FakeLangfuseClient()
        fake.run_experiment(
            name="exp-1",
            description="d",
            data=[{"input": "x"}],
            task=lambda item: TaskResult(output="y"),
            evaluators=[],
            metadata={"k": "v"},
        )

        assert len(fake.experiments) == 1
        call = fake.experiments[0]
        assert call["name"] == "exp-1"
        assert call["data"] == [{"input": "x"}]
        assert call["metadata"] == {"k": "v"}

    @pytest.mark.asyncio
    async def test_async_task_is_driven_to_completion(self):
        """An async task callable produces a real result, not an unawaited coroutine."""
        seen: list = []

        async def async_task(item):
            seen.append(item.get("input") if isinstance(item, dict) else item)
            return TaskResult(output="async-result")

        fake = FakeLangfuseClient()
        result = fake.run_experiment(
            name="t",
            description="",
            data=[{"input": "x"}],
            task=async_task,
            evaluators=[],
            metadata={},
        )

        # The fake invokes task(item={...}) — matching the SDK shape — so the
        # async task receives the dict and reads its "input" key.
        assert seen == ["x"]
        assert result.item_results[0].output == TaskResult(output="async-result"), (
            "async task return must be the awaited value, not an unawaited coroutine"
        )

    def test_async_evaluator_is_driven_to_completion(self):
        """An async evaluator's Evaluation reaches the ExperimentItemResult."""

        async def async_evaluator(
            *, input, output, expected_output, metadata, **kwargs
        ):
            return Evaluation(name="async-spy", value=0.5, comment="")

        fake = FakeLangfuseClient()
        result = fake.run_experiment(
            name="t",
            description="",
            data=[{"input": "q", "expected_output": "a"}],
            task=lambda item: TaskResult(output="r"),
            evaluators=[async_evaluator],
            metadata={},
        )

        item = result.item_results[0]
        assert len(item.evaluations) == 1, (
            "async evaluator return must be the awaited Evaluation, not an unawaited coroutine"
        )
        assert item.evaluations[0].name == "async-spy"
        assert item.evaluations[0].value == 0.5

    def test_async_task_exception_propagates_to_caller(self):
        """An async task that raises must surface the exception in the calling thread,
        not silently swallow it inside the worker."""

        async def failing_task(item):
            raise ValueError("kaboom from async task")

        fake = FakeLangfuseClient()
        with pytest.raises(ValueError, match="kaboom from async task"):
            fake.run_experiment(
                name="t",
                description="",
                data=[{"input": "x"}],
                task=failing_task,
                evaluators=[],
                metadata={},
            )


class TestFakeLangfuseClientPortMethods:
    """Pin the flat port surface the fake exposes."""

    def test_get_dataset_items_returns_seeded_records(self):
        item = DatasetItemRecord(
            source_trace_id="trace-1",
            input="hi",
            expected_output="hello",
            metadata={},
        )
        fake = FakeLangfuseClient(dataset_items={"d": [item]})
        assert fake.get_dataset_items("d") == [item]

    def test_get_dataset_items_unknown_name_raises(self):
        fake = FakeLangfuseClient()
        with pytest.raises(KeyError):
            fake.get_dataset_items("does-not-exist")

    def test_get_trace_output_returns_seeded_value(self):
        fake = FakeLangfuseClient(trace_outputs={"trace-1": "result"})
        assert fake.get_trace_output("trace-1") == "result"

    def test_get_trace_output_unknown_id_raises(self):
        """Unknown trace IDs raise ``FakeTraceNotFoundError`` (an ``Exception``
        subclass), which the adapter's bare ``except Exception`` catches
        exactly like the real SDK's network error case.
        """
        fake = FakeLangfuseClient()
        with pytest.raises(FakeTraceNotFoundError, match="404"):
            fake.get_trace_output("missing")

    def test_fake_has_no_list_method(self):
        """The fake mirrors the port's flat shape — no list-shaped method.

        ``api.trace.list()`` returns only the first page of traces in the
        real Langfuse SDK, so any list-based membership check produces
        false negatives once a project has more than one page. Pinning
        the fake's structural absence of ``list`` and ``api`` means a
        regression would have to add the method to the port, the
        wrapper, and this fake — three places, not one.
        """
        fake = FakeLangfuseClient()
        assert not hasattr(fake, "api")
        assert not hasattr(fake, "list")
