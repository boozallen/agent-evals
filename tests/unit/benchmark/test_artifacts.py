# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the _write_report primitive, BenchmarkResult.write_report, and run_benchmark_async."""

import re
import textwrap
from pathlib import Path

import pytest
from foundry_agent_core.encryption import load_encryption_key

from agent_evals import BaseAgent, EvalResult, read_encrypted_file
from agent_evals.benchmark.types import BenchmarkResult
from agent_evals.core.types import TaskResult


def _fake_eval_result(total: int, passed: int) -> EvalResult:
    return EvalResult(
        experiment_id="x",
        experiment_url="file:///tmp/x",
        platform="local",
        scores={"StateMatch": passed / total if total else 0.0},
        pass_rates={"StateMatch": passed / total if total else 0.0},
        examples=[],
        summary={"total_examples": total, "successful_examples": passed},
    )


def _synthetic_result(benchmark: str = "synthetic") -> BenchmarkResult:
    return BenchmarkResult(
        benchmark=benchmark,
        agent_name="test-agent",
        agent_version="1.0",
        eval_results={"lights": _fake_eval_result(total=1, passed=1)},
    )


class TestWriteReport:
    def test_creates_results_dir_if_missing(self, tmp_path: Path):
        from agent_evals.benchmark.artifacts import _write_report

        target = tmp_path / "deep" / "nested" / "results"
        assert not target.exists()

        _write_report(_synthetic_result(), results_dir=target)

        assert target.exists() and target.is_dir()

    def test_returns_path_that_was_written(self, tmp_path: Path):
        from agent_evals.benchmark.artifacts import _write_report

        path = _write_report(_synthetic_result(), results_dir=tmp_path)

        assert path.exists()
        assert path.parent == tmp_path
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        decrypted = read_encrypted_file(path, key)
        assert decrypted is not None
        assert decrypted["content"].startswith("# Benchmark: synthetic")

    def test_default_filename_pattern(self, tmp_path: Path):
        from agent_evals.benchmark.artifacts import _write_report

        path = _write_report(_synthetic_result("home-v1"), results_dir=tmp_path)

        # YYYY-MM-DD_HH-MM-SS_{benchmark}.md
        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_home-v1\.md", path.name
        )

    def test_explicit_filename_overrides_timestamp(self, tmp_path: Path):
        from agent_evals.benchmark.artifacts import _write_report

        path = _write_report(
            _synthetic_result(), results_dir=tmp_path, filename="fixture.md"
        )

        assert path.name == "fixture.md"

    def test_content_matches_to_markdown(self, tmp_path: Path):
        from agent_evals.benchmark.artifacts import _write_report

        result = _synthetic_result()
        path = _write_report(result, results_dir=tmp_path, filename="r.md")

        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        decrypted = read_encrypted_file(path, key)
        assert decrypted is not None
        assert decrypted["content"] == result.to_markdown()


class TestFilenameRejection:
    """The ``filename`` override is rejected, not rewritten.

    Every case here asserts on **all platforms** including CI's
    ``ubuntu-latest``: the rules are applied by this module rather than
    delegated to the host, so nothing is platform-guarded and nothing skips
    in CI. AC 10.
    """

    # Each value with the reason it could redirect the write.
    REDIRECTING = [
        "sub/report.md",  # forward-slash separator
        "sub\\report.md",  # backslash separator (a separator on Windows)
        "../report.md",  # parent traversal
        "..",  # bare parent reference
        ".",  # bare current-directory reference
        "C:evil.md",  # drive-relative: joining discards the output directory
        "C:/Windows/evil.md",  # absolute with drive
        "/etc/passwd",  # absolute POSIX
        "",  # empty
        "   ",  # whitespace-only
        "NUL",  # reserved device name
        "CON",
        "COM1",
        "LPT1",
        "COM9",  # numbered variant
        "nul.md",  # reserved stem with an extension still hits the device
        # Win32 strips trailing spaces and dots from a path component before
        # resolving it, so each of these reaches the device too. Verified on
        # Windows 11: `NUL ` leaves nothing on disk, `CON  ` creates a file
        # named `CON`. A stem check on the raw string admits all of them.
        "NUL ",  # trailing space
        " NUL",  # leading space
        "NUL.",  # trailing dot
        "nul.md.",  # trailing dot after an extension
        "COM1 ",
        "CON  ",  # more than one trailing space
    ]

    @pytest.mark.parametrize("bad", REDIRECTING)
    def test_rejected_with_no_filesystem_side_effect(self, bad: str, tmp_path: Path):
        from agent_evals.benchmark.artifacts import _write_report

        target = tmp_path / "results"
        assert not target.exists()

        with pytest.raises(ValueError):
            _write_report(_synthetic_result(), results_dir=target, filename=bad)

        # Rejection must precede mkdir, so a rejected call leaves no trace.
        assert not target.exists(), "no directory may be created for a rejected name"
        assert list(tmp_path.rglob("*")) == [], "no file may be written"

    def test_not_silently_rewritten_to_final_component(self, tmp_path: Path):
        from agent_evals.benchmark.artifacts import _write_report

        # Rewriting `sub/report.md` to `report.md` would hide the caller's
        # mistake and write somewhere they did not ask for.
        with pytest.raises(ValueError):
            _write_report(
                _synthetic_result(), results_dir=tmp_path, filename="sub/report.md"
            )
        assert not (tmp_path / "report.md").exists()

    def test_ordinary_filename_still_accepted(self, tmp_path: Path):
        from agent_evals.benchmark.artifacts import _write_report

        path = _write_report(
            _synthetic_result(), results_dir=tmp_path, filename="run-42.md"
        )
        assert path.name == "run-42.md"
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        decrypted = read_encrypted_file(path, key)
        assert decrypted is not None
        assert decrypted["content"].startswith("# Benchmark: synthetic")

    def test_dotted_name_that_is_not_a_device_is_accepted(self, tmp_path: Path):
        from agent_evals.benchmark.artifacts import _write_report

        # `console.md` shares a prefix with CON but is not a device name; the
        # check must be on the whole stem, not a prefix match.
        path = _write_report(
            _synthetic_result(), results_dir=tmp_path, filename="console.md"
        )
        assert path.name == "console.md"

    @pytest.mark.parametrize("good", ["nullable.md", "com10.md", "lpt0.md"])
    def test_names_merely_resembling_devices_are_accepted(
        self, good: str, tmp_path: Path
    ):
        from agent_evals.benchmark.artifacts import _write_report

        # The trailing-trim fold must not widen the rule: `nullable.md` starts
        # with NUL, and `com10`/`lpt0` fall outside the COM1-COM9 range that
        # Windows reserves. All three are ordinary filenames.
        path = _write_report(_synthetic_result(), results_dir=tmp_path, filename=good)
        assert path.name == good


class _StateMatchAgent(BaseAgent):
    name = "state-match"
    version = "test"

    async def run_case(self, prompt: str, **kwargs) -> TaskResult:
        state: dict = {"lights": {}}
        if "kitchen" in prompt.lower() and "on" in prompt.lower():
            state["lights"]["kitchen"] = {"state": "on"}
        return TaskResult(output="", context={"final_state": state})


BENCHMARK_YAML = textwrap.dedent(
    """
    benchmark: artifacts-v1
    platform:
      name: local
      experiment: t1
    capabilities:
      - name: lights
        scenarios:
          - id: 1
            name: kitchen on
            prompt: Turn on the kitchen light
            success:
              state:
                lights.kitchen.state: "on"
    """
).strip()


@pytest.fixture
def bench_yaml(tmp_path: Path) -> Path:
    p = tmp_path / "bench.yaml"
    p.write_text(BENCHMARK_YAML)
    return p


class TestRunBenchmarkAsync:
    @pytest.mark.asyncio
    async def test_returns_benchmark_result(self, bench_yaml: Path):
        from agent_evals import run_benchmark_async

        result = await run_benchmark_async(bench_yaml, agent=_StateMatchAgent)

        assert isinstance(result, BenchmarkResult)
        assert result.benchmark == "artifacts-v1"
        assert result.agent_name == "state-match"
        assert result.aggregate_pass_rates["StateMatch"] == 1.0

    @pytest.mark.asyncio
    async def test_is_pure_no_stdout_no_disk(
        self, bench_yaml: Path, tmp_path: Path, capsys
    ):
        from agent_evals import run_benchmark_async

        result = await run_benchmark_async(bench_yaml, agent=_StateMatchAgent)

        assert capsys.readouterr().out == ""  # no auto-print
        assert not (tmp_path / "benchmarks").exists()  # no auto-write
        assert isinstance(result, BenchmarkResult)

    @pytest.mark.asyncio
    async def test_explicit_compose_writes_when_asked(
        self, bench_yaml: Path, tmp_path: Path
    ):
        from agent_evals import run_benchmark_async

        result = await run_benchmark_async(bench_yaml, agent=_StateMatchAgent)
        path = result.write_report(results_dir=tmp_path, filename="out.md")

        assert path == tmp_path / "out.md"
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        decrypted = read_encrypted_file(path, key)
        assert decrypted is not None
        assert "# Benchmark: artifacts-v1" in decrypted["content"]


class TestBenchmarkResultWriteReport:
    def test_method_writes_file_and_returns_path(self, tmp_path: Path):
        result = _synthetic_result("home-v1")

        path = result.write_report(results_dir=tmp_path, filename="r.md")

        assert path == tmp_path / "r.md"
        key = load_encryption_key("FOUNDRY_EVALS_ENCRYPTION_KEY")
        decrypted = read_encrypted_file(path, key)
        assert decrypted is not None
        assert decrypted["content"] == result.to_markdown()

    def test_method_default_filename_pattern(self, tmp_path: Path):
        path = _synthetic_result("home-v1").write_report(results_dir=tmp_path)

        assert re.fullmatch(
            r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}_home-v1\.md", path.name
        )

    def test_method_creates_dir_and_is_silent(self, tmp_path: Path, capsys):
        target = tmp_path / "deep" / "results"

        _synthetic_result().write_report(results_dir=target, filename="r.md")

        assert (target / "r.md").exists()
        assert capsys.readouterr().out == ""  # no print
