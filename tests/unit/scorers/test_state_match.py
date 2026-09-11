# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for StateMatch native scorer."""

from agent_evals.core.types import ExpectedResult, Score, TaskResult


class TestStateMatchFactoryLevel:
    """StateMatch with expected paths declared at factory level."""

    def test_all_paths_match_passes(self):
        from agent_evals.adapters.scorers.state import StateMatch

        scorer = StateMatch(
            expected_state={"lights.kitchen.state": "on", "doors.front": "locked"}
        )
        result = scorer(
            TaskResult(
                output="",
                context={
                    "final_state": {
                        "lights": {"kitchen": {"state": "on"}},
                        "doors": {"front": "locked"},
                    }
                },
            )
        )
        assert isinstance(result, Score)
        assert result.name == "StateMatch"
        assert result.passed is True
        assert result.value == 1.0

    def test_one_path_fails_not_passed(self):
        from agent_evals.adapters.scorers.state import StateMatch

        scorer = StateMatch(
            expected_state={"lights.kitchen.state": "on", "doors.front": "locked"}
        )
        result = scorer(
            TaskResult(
                output="",
                context={
                    "final_state": {
                        "lights": {"kitchen": {"state": "off"}},
                        "doors": {"front": "locked"},
                    }
                },
            )
        )
        assert result.passed is False
        assert result.value == 0.5
        assert "lights.kitchen.state" in result.metadata["failures"]


class TestStateMatchPerExample:
    """StateMatch with per-example expected via ExpectedResult.context."""

    def test_per_example_overrides_factory_default(self):
        from agent_evals.adapters.scorers.state import StateMatch

        scorer = StateMatch(expected_state={"lights.kitchen.state": "on"})
        result = scorer(
            TaskResult(
                output="",
                context={"final_state": {"doors": {"front": "locked"}}},
            ),
            ExpectedResult(
                expected="",
                context={"expected_state": {"doors.front": "locked"}},
            ),
        )
        assert result.passed is True

    def test_per_example_without_factory_default(self):
        from agent_evals.adapters.scorers.state import StateMatch

        scorer = StateMatch()
        result = scorer(
            TaskResult(
                output="",
                context={"final_state": {"alarm": {"armed": True}}},
            ),
            ExpectedResult(
                expected="",
                context={"expected_state": {"alarm.armed": True}},
            ),
        )
        assert result.passed is True


class TestStateMatchErrors:
    """Error-path behavior for StateMatch."""

    def test_missing_expected_returns_error_score(self):
        from agent_evals.adapters.scorers.state import StateMatch

        scorer = StateMatch()
        result = scorer(
            TaskResult(output="", context={"final_state": {}}),
        )
        assert result.passed is False
        assert result.value == 0.0
        assert "no expected_state provided" in result.metadata["error"]

    def test_missing_final_state_returns_error_score(self):
        from agent_evals.adapters.scorers.state import StateMatch

        scorer = StateMatch(expected_state={"x.y": 1})
        result = scorer(TaskResult(output="", context={}))
        assert result.passed is False
        assert "not a dict" in result.metadata["error"]

    def test_final_state_wrong_type_returns_error_score(self):
        from agent_evals.adapters.scorers.state import StateMatch

        scorer = StateMatch(expected_state={"x.y": 1})
        result = scorer(
            TaskResult(output="", context={"final_state": "not a dict"}),
        )
        assert result.passed is False
        assert result.metadata["got_type"] == "str"

    def test_missing_path_reported_as_failure(self):
        from agent_evals.adapters.scorers.state import StateMatch

        scorer = StateMatch(expected_state={"nope.nothere": 1})
        result = scorer(
            TaskResult(output="", context={"final_state": {"x": {"y": 1}}}),
        )
        assert result.passed is False
        assert result.metadata["failures"]["nope.nothere"]["reason"] == "path missing"


class TestStateMatchPackageExport:
    """StateMatch is exported from agent_evals.adapters.scorers."""

    def test_state_match_importable_from_package(self):
        from agent_evals.adapters.scorers import StateMatch

        scorer = StateMatch(expected_state={"x": 1})
        assert callable(scorer)


class TestStateMatchRegistered:
    """StateMatch is accessible via the scorer registry."""

    def test_state_match_resolvable_by_name(self):
        from agent_evals.core._registries import scorer_registry

        factory = scorer_registry.get("StateMatch")
        scorer = factory(expected_state={"x": 1})
        assert callable(scorer)
