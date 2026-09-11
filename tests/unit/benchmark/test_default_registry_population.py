# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Regression tests for registry-population side-effect preservation.

Each path that resolves a registry singleton from ``core._registries`` —
the ``_composition.py`` lazy-getter helpers and the ``agent_evals``
package-root PEP 562 lazy-load — must FIRST import the family package
so the ``@register("...")`` decorators on concrete adapters fire.
Reaching directly into ``core._registries`` does NOT walk the family
package, leaving the registry empty when the consumer runs.

The original failure mode showed up as ``Unknown success checker:
'state'. Available success checkers: none registered`` (and the same
shape for preconditions and config readers).

These tests run in subprocesses so each Python process starts with no
``agent_evals.adapters.*`` modules pre-imported by sibling tests.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_in_subprocess(script: str) -> tuple[int, str, str]:
    """Run ``script`` in a fresh Python process; return ``(rc, stdout, stderr)``.

    Mirrors ``tests/unit/test_lazy_imports.py::_run_in_subprocess``;
    duplicated here to keep this regression test self-contained.
    """
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_success_checker_validator_populates_registry_on_first_use() -> None:
    """``_get_success_checker_registry()`` populates the registry on first call."""
    code, stdout, stderr = _run_in_subprocess(
        """
        import sys
        from agent_evals._composition import _get_success_checker_registry

        # Sanity check: the family package must NOT have been imported by
        # the act of importing _composition itself — only by calling the
        # lazy-getter.
        assert "agent_evals.adapters.success_checkers" not in sys.modules, (
            "family pre-imported before _get_success_checker_registry ran "
            "— test is no longer exercising the lazy-getter path"
        )

        registry = _get_success_checker_registry()
        names = sorted(registry.list())
        assert "state" in names, (
            "state not registered after _get_success_checker_registry: "
            + repr(names)
        )
        # Family must be loaded now — the helper's job.
        assert "agent_evals.adapters.success_checkers" in sys.modules
        print("OK")
        """
    )
    assert code == 0, f"subprocess failed:\nstdout={stdout}\nstderr={stderr}"
    assert "OK" in stdout, stdout


def test_precondition_validator_populates_registry_on_first_use() -> None:
    """``_get_precondition_registry()`` populates the registry on first call."""
    code, stdout, stderr = _run_in_subprocess(
        """
        import sys
        from agent_evals._composition import _get_precondition_registry

        assert "agent_evals.adapters.preconditions" not in sys.modules, (
            "family pre-imported before _get_precondition_registry ran"
        )

        registry = _get_precondition_registry()
        names = sorted(registry.list())
        assert "state" in names, (
            "state not registered: " + repr(names)
        )
        assert "agent_evals.adapters.preconditions" in sys.modules
        print("OK")
        """
    )
    assert code == 0, f"subprocess failed:\nstdout={stdout}\nstderr={stderr}"
    assert "OK" in stdout, stdout


def test_config_reader_fallback_populates_registry_on_first_use() -> None:
    """``_get_config_reader_registry()`` populates the registry on first call."""
    code, stdout, stderr = _run_in_subprocess(
        """
        import sys
        from agent_evals._composition import _get_config_reader_registry

        assert "agent_evals.adapters.benchmark_config_readers" not in sys.modules, (
            "family pre-imported before _get_config_reader_registry ran"
        )

        registry = _get_config_reader_registry()
        names = sorted(registry.list())
        assert "file" in names, (
            "file not registered: " + repr(names)
        )
        assert "agent_evals.adapters.benchmark_config_readers" in sys.modules
        print("OK")
        """
    )
    assert code == 0, f"subprocess failed:\nstdout={stdout}\nstderr={stderr}"
    assert "OK" in stdout, stdout


def test_composition_module_populates_platform_registry() -> None:
    """``_get_platform_registry()`` populates the registry on first call."""
    code, stdout, stderr = _run_in_subprocess(
        """
        import sys
        from agent_evals._composition import _get_platform_registry

        # Sanity check: the family package must NOT have been imported by
        # the act of importing _composition itself — only by calling the
        # lazy-getter.
        assert "agent_evals.adapters.platforms" not in sys.modules, (
            "family pre-imported before _get_platform_registry ran "
            "— test is no longer exercising the lazy-getter path"
        )

        registry = _get_platform_registry()
        platforms = sorted(registry.list())
        assert platforms, f'platform_registry empty: {platforms}'
        assert 'local' in platforms, f'local not registered: {platforms}'
        # Family must be loaded now — the helper's job.
        assert "agent_evals.adapters.platforms" in sys.modules
        print('OK', platforms)
        """
    )
    assert code == 0, f"subprocess failed:\nstdout={stdout}\nstderr={stderr}"
    assert "OK" in stdout, stdout
