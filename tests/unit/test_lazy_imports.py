# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""``import agent_evals`` is a public-surface-only import.

Pins two properties:

1. ``import agent_evals`` does not pull ``core.runner``, the optional
   platform SDKs, or the converter libraries into ``sys.modules``.
2. The lazy attributes (``run_eval``, ``run_eval_async``, the
   converters) still resolve via PEP 562 ``__getattr__`` and trigger
   the underlying load on first access.

The lazy-load assertions run in a subprocess so unrelated tests cannot
pre-populate ``sys.modules`` and mask a regression.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap


def _run_in_subprocess(script: str) -> tuple[int, str, str]:
    proc = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        check=False,
    )
    return proc.returncode, proc.stdout, proc.stderr


def test_import_agent_evals_does_not_load_runner_or_optional_sdks():
    code, stdout, stderr = _run_in_subprocess(
        """
        import sys
        import agent_evals  # noqa: F401

        forbidden = [
            "agent_evals.core.runner",
            "agent_evals._composition",
            "mlflow",
            "braintrust",
            "langfuse",
            "langchain_core",
            "strands",
        ]
        loaded = [m for m in forbidden if m in sys.modules]
        if loaded:
            print("FAIL:" + ",".join(loaded))
            sys.exit(1)
        print("OK")
        """
    )
    assert code == 0, f"stdout={stdout!r} stderr={stderr!r}"
    assert "OK" in stdout, stdout


def test_run_eval_lazy_load_pulls_composition_root():
    code, stdout, stderr = _run_in_subprocess(
        """
        import sys
        import agent_evals

        assert "agent_evals._composition" not in sys.modules, "premature load"
        fn = agent_evals.run_eval
        assert "agent_evals._composition" in sys.modules, "did not load"
        assert "agent_evals.core.runner" in sys.modules, "runner not loaded"
        assert callable(fn)
        print("OK")
        """
    )
    assert code == 0, f"stdout={stdout!r} stderr={stderr!r}"
    assert "OK" in stdout, stdout


def test_run_eval_async_lazy_load_pulls_composition_root():
    code, stdout, stderr = _run_in_subprocess(
        """
        import sys
        import agent_evals

        assert "agent_evals._composition" not in sys.modules
        fn = agent_evals.run_eval_async
        assert "agent_evals._composition" in sys.modules
        assert callable(fn)
        print("OK")
        """
    )
    assert code == 0, f"stdout={stdout!r} stderr={stderr!r}"
    assert "OK" in stdout, stdout


def test_langchain_converter_lazy_loads_only_on_access():
    import pytest

    pytest.importorskip("langchain_core", reason="langchain extra not installed")
    code, stdout, stderr = _run_in_subprocess(
        """
        import sys
        import agent_evals

        assert "langchain_core" not in sys.modules, "premature langchain load"
        fn = agent_evals.langchain_to_openai
        assert "langchain_core" in sys.modules, "langchain did not load on access"
        assert callable(fn)
        print("OK")
        """
    )
    assert code == 0, f"stdout={stdout!r} stderr={stderr!r}"
    assert "OK" in stdout, stdout


def test_unknown_attribute_still_raises_attribute_error():
    import pytest

    import agent_evals

    missing = "this_is_not_a_real_symbol"
    with pytest.raises(AttributeError, match=missing):
        getattr(agent_evals, missing)


def test_every_lazy_attr_is_resolvable():
    """Every name registered in ``_LAZY`` must resolve via getattr.

    Insurance against typos in module paths inside loader functions —
    a typo would silently fail at runtime when a user accesses the
    attribute, but this test catches it early. As ``_LAZY`` grows
    with each adapter-family migration this test scales for free.
    """
    import agent_evals

    lazy_attrs = agent_evals._LAZY

    failures = []
    for name in lazy_attrs:
        try:
            obj = getattr(agent_evals, name)
            assert obj is not None, f"{name} resolved to None"
        except (AttributeError, ImportError) as exc:
            failures.append(f"{name}: {exc}")

    assert not failures, "Lazy-load failures:\n" + "\n".join(failures)


def test_run_benchmark_async_lazy_loads_from_composition():
    """``agent_evals.run_benchmark_async`` resolves from ``_composition.py``."""
    code, stdout, stderr = _run_in_subprocess(
        """
        from agent_evals._composition import run_benchmark_async
        assert run_benchmark_async.__module__ == "agent_evals._composition", (
            f"expected agent_evals._composition, got {run_benchmark_async.__module__}"
        )
        print("OK")
        """
    )
    assert code == 0, f"subprocess failed:\nstdout={stdout}\nstderr={stderr}"
    assert "OK" in stdout, stdout


def test_from_agent_evals_import_run_eval_does_not_load_benchmark_registries():
    """``from agent_evals import run_eval`` doesn't preload benchmark-only registries.

    Importing ``run_eval`` loads ``_composition`` (which *defines* the
    benchmark-family lazy-getters) but does not *call* them, so the
    benchmark-only adapter families stay out of ``sys.modules``.
    """
    code, stdout, stderr = _run_in_subprocess(
        """
        import sys
        import agent_evals
        _ = agent_evals.run_eval

        forbidden = [
            "agent_evals.adapters.preconditions",
            "agent_evals.adapters.success_checkers",
            "agent_evals.adapters.benchmark_config_readers",
        ]
        for module in forbidden:
            assert module not in sys.modules, f"{module} loaded too early"
        print("OK")
        """
    )
    assert code == 0, f"subprocess failed:\nstdout={stdout}\nstderr={stderr}"
    assert "OK" in stdout, stdout
