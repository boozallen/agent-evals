# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the path-safe identifier constraint on name fields.

Four fields reach a filesystem path segment and are constrained by
``PathSafeIdentifier``: the YAML ``benchmark:`` field, the public
``BenchmarkResult.benchmark``, ``CapabilitySpec.name``, and the local
adapter's ``experiment``. Each is covered here at the layer a user
actually touches, because a constraint that holds on one field and not
another leaves the sink reachable.

Every vector in this module asserts on **all platforms**, including CI's
``ubuntu-latest``: the rejected values are rejected by a character
allow-list rather than by the host's path semantics, so nothing here is
platform-guarded and nothing skips in CI.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from agent_evals.adapters.platforms.local import LocalConfig
from agent_evals.benchmark.config import CapabilitySpec
from agent_evals.benchmark.types import BenchmarkResult

# Values that must never reach a path segment, each with why it is dangerous.
UNSAFE_NAMES = [
    "../../evil",  # plain relative escape
    "..",  # bare parent reference
    ".",  # bare current-directory reference
    "sub/evil",  # forward-slash separator
    "sub\\evil",  # backslash separator (a separator on Windows)
    "C:evil",  # drive-relative: joining discards the parent directory
    "C:/Windows/evil",  # absolute with drive
    "/etc/passwd",  # absolute POSIX
    "",  # empty
    "   ",  # whitespace-only, empty after stripping
    "a" * 200,  # over the length bound
]

# Values that appear in real configs and existing tests; must keep working.
SAFE_NAMES = ["b", "home-v1", "multi_example", "run.42"]


def _capability(name: str = "lights") -> dict:
    return {
        "name": name,
        "scenarios": [
            {
                "id": 1,
                "name": "s",
                "prompt": "turn on the lights",
                "success": {"state": {"lights": "on"}},
            }
        ],
    }


def _config_dict(benchmark: str, capability: str = "lights") -> dict:
    return {
        "benchmark": benchmark,
        "platform": {"name": "local"},
        "capabilities": [_capability(capability)],
    }


def _load(tmp_path: Path, registries: dict, config: dict):
    """Write ``config`` as YAML and load it the way a user would.

    Going through ``load_benchmark`` rather than ``model_validate`` supplies
    the registries the scenario validators require from validation context,
    and exercises the same path a real benchmark file takes.
    """
    from agent_evals.benchmark.loader import load_benchmark

    path = tmp_path / "bench.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_benchmark(path, **registries)


class TestYamlBenchmarkField:
    """``BenchmarkConfig.benchmark`` — the YAML entry point."""

    @pytest.mark.parametrize("bad", UNSAFE_NAMES)
    def test_unsafe_benchmark_name_rejected(
        self, bad: str, tmp_path: Path, default_benchmark_registries: dict
    ) -> None:
        with pytest.raises(ValidationError):
            _load(tmp_path, default_benchmark_registries, _config_dict(bad))

    @pytest.mark.parametrize("good", SAFE_NAMES)
    def test_safe_benchmark_name_accepted(
        self, good: str, tmp_path: Path, default_benchmark_registries: dict
    ) -> None:
        cfg = _load(tmp_path, default_benchmark_registries, _config_dict(good))
        assert cfg.benchmark == good

    def test_surrounding_whitespace_is_stripped(
        self, tmp_path: Path, default_benchmark_registries: dict
    ) -> None:
        cfg = _load(tmp_path, default_benchmark_registries, _config_dict("  home-v1  "))
        assert cfg.benchmark == "home-v1"


class TestCapabilityName:
    """``CapabilitySpec.name`` — composed into the experiment name."""

    @pytest.mark.parametrize("bad", ["x/../../../ESCAPED", "sub/evil", "..", "a\\b"])
    def test_unsafe_capability_name_rejected(
        self, bad: str, tmp_path: Path, default_benchmark_registries: dict
    ) -> None:
        with pytest.raises(ValidationError):
            _load(
                tmp_path,
                default_benchmark_registries,
                _config_dict("home-v1", capability=bad),
            )

    def test_safe_capability_name_accepted(
        self, tmp_path: Path, default_benchmark_registries: dict
    ) -> None:
        cfg = _load(tmp_path, default_benchmark_registries, _config_dict("home-v1"))
        assert cfg.capabilities[0].name == "lights"

    def test_rejected_at_the_model_without_a_config(self) -> None:
        # The field constraint fires before the scenario validators that need
        # registry context, so the name rule is provably on the field itself
        # rather than an artifact of loading.
        with pytest.raises(ValidationError) as exc:
            CapabilitySpec.model_validate(_capability("x/../../../ESCAPED"))
        assert "name" in str(exc.value)


class TestBenchmarkResultField:
    """``BenchmarkResult.benchmark`` — the field the report filename reads.

    Constraining only the YAML field would leave this sink reachable, since
    the filename is built from the result model rather than the config.
    """

    @pytest.mark.parametrize("bad", UNSAFE_NAMES)
    def test_unsafe_name_rejected_at_construction(self, bad: str) -> None:
        # Rejecting at construction makes write_report unreachable with the
        # value; there is no window in which a bad name exists on the model.
        with pytest.raises(ValidationError):
            BenchmarkResult(benchmark=bad)

    def test_safe_name_accepted(self) -> None:
        assert BenchmarkResult(benchmark="home-v1").benchmark == ("home-v1")

    def test_write_report_unreachable_with_traversal_name(self, tmp_path: Path) -> None:
        with pytest.raises(ValidationError):
            BenchmarkResult(benchmark="../../evil").write_report(tmp_path)
        assert not (tmp_path.parent.parent / "evil.md").exists()


class TestLocalExperimentField:
    """``LocalConfig.experiment`` — becomes a directory name."""

    @pytest.mark.parametrize(
        "bad", ["../../evil", "C:/Windows/evil", "C:evil", "sub/evil", ".."]
    )
    def test_unsafe_experiment_rejected(self, bad: str) -> None:
        with pytest.raises(ValidationError):
            LocalConfig(experiment=bad)

    def test_safe_experiment_accepted(self) -> None:
        assert LocalConfig(experiment="run-42").experiment == "run-42"

    def test_omitted_experiment_stays_none(self) -> None:
        # Optionality must survive the constraint; the adapter generates the
        # name at run time rather than at config time.
        assert LocalConfig().experiment is None

    def test_auto_generated_name_satisfies_the_constraint(self) -> None:
        # The adapter composes ``eval-<unix-seconds>`` when experiment is
        # omitted. Feeding that shape back through the field proves the
        # constraint cannot reject the adapter's own generated name.
        generated = f"eval-{int(time.time())}"
        assert LocalConfig(experiment=generated).experiment == generated


class TestComposedExperimentName:
    """The ``model_copy`` bypass at the runner's per-capability sink.

    ``model_copy(update=...)`` installs a value without re-validating, so a
    constraint on ``LocalConfig.experiment`` alone does not cover the
    composed name. These tests exercise the composition the way the runner
    performs it. AC 9.
    """

    def test_model_copy_really_does_bypass_field_validation(self) -> None:
        # Establishes the premise the point-of-use check exists for: if this
        # ever starts raising, Pydantic changed and the check could be
        # reconsidered. Until then a field constraint is provably insufficient.
        cfg = LocalConfig(experiment="safe")
        bypassed = cfg.model_copy(update={"experiment": "x/../../../ESCAPED"})
        assert bypassed.experiment == "x/../../../ESCAPED"

    def test_composed_value_is_rejected_at_the_point_of_use(self) -> None:
        from agent_evals.benchmark.runner import _validated_experiment_name

        # Composed exactly as the runner composes it.
        base_exp = "home-v1"
        capability_name = "x/../../../ESCAPED"
        with pytest.raises(ValueError) as exc:
            _validated_experiment_name(f"{base_exp}-{capability_name}")
        assert "ESCAPED" in str(exc.value), "error must name the offending value"

    def test_legitimate_composed_value_passes_through(self) -> None:
        from agent_evals.benchmark.runner import _validated_experiment_name

        assert _validated_experiment_name("home-v1-lights") == "home-v1-lights"


@pytest.mark.asyncio
async def test_escaping_capability_writes_nothing_outside_output_dir(
    tmp_path: Path,
) -> None:
    """A capability named to escape produces no artifact outside output_dir.

    This is the end-to-end form of the ``model_copy`` bypass: the composed
    experiment name became a directory name at unbounded depth, and the
    vector reproduces on Linux as well as Windows — so this assertion runs
    on CI's only platform. AC 9, task 7.3.
    """
    from agent_evals import BaseAgent, run_benchmark_async
    from agent_evals.core.types import TaskResult

    class _NoopAgent(BaseAgent):
        name = "noop"
        version = "test"

        async def run_case(self, prompt: str, **kwargs) -> TaskResult:
            return TaskResult(output="", context={"final_state": {}})

    output_dir = tmp_path / "out"
    bench = tmp_path / "bench.yaml"
    bench.write_text(
        "\n".join(
            [
                "benchmark: escape-v1",
                "platform:",
                "  name: local",
                "  experiment: t1",
                f"  output_dir: {output_dir.as_posix()}",
                "capabilities:",
                "  - name: x/../../../ESCAPED",
                "    scenarios:",
                "      - id: 1",
                "        name: s",
                "        prompt: hi",
                "        success:",
                "          state:",
                "            lights: on",
            ]
        ),
        encoding="utf-8",
    )

    # Snapshot tmp_path.parent before the run: it's the pytest session's
    # shared tmp root, so other tests running in the same session may
    # legitimately create metadata.json/results.jsonl/summary.json under
    # their own, unrelated tmp_path siblings. Only files newly created by
    # this test's own call are evidence of an escape.
    before = set(tmp_path.parent.rglob("*"))

    with pytest.raises((ValidationError, ValueError)):
        await run_benchmark_async(bench, agent=_NoopAgent)

    after = set(tmp_path.parent.rglob("*")) - before

    # Nothing may appear outside output_dir. The only files under tmp_path
    # should be the config itself plus anything inside output_dir.
    strays = [
        p
        for p in tmp_path.rglob("*")
        if p.is_file() and p != bench and output_dir not in p.parents
    ]
    assert strays == [], f"artifacts written outside output_dir: {strays}"
    for artifact in ("metadata.json", "results.jsonl", "summary.json"):
        new_artifacts = [p for p in after if p.name == artifact]
        assert all(output_dir in p.parents for p in new_artifacts), (
            f"{artifact} escaped output_dir"
        )


def test_remote_experiment_names_stay_unconstrained() -> None:
    """Remote configs accept what the local one rejects.

    The distinction is the presence of a filesystem sink: remote experiment
    names are run labels sent to a service. Constraining them would reject
    names those services accept while preventing nothing. AC 19.
    """
    from agent_evals.core.types import PlatformConfig

    # The base config's experiment field carries no local path meaning.
    assert PlatformConfig(name="mlflow", experiment="team/project").experiment == (
        "team/project"
    )
