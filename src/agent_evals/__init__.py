# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Platform-agnostic LLM evaluation library — public re-export surface.

The composition root lives in ``agent_evals._composition`` and is loaded
lazily on first attribute access (PEP 562). Importing this module pulls
in the user-facing Pydantic types and ``BaseAgent``; entry-point
functions, converters, and ``BenchmarkResult`` lazy-load via PEP 562
``__getattr__`` on first attribute access. ``core.runner``, the platform
SDKs, and the optional converter libraries are never pulled in by
``import agent_evals`` alone.
"""

from collections.abc import Callable
from importlib.metadata import version as _get_version
from typing import TYPE_CHECKING, Any

from agent_evals.benchmark.agent import BaseAgent
from agent_evals.core.types import (
    EvalExample,
    EvalResult,
    ExampleData,
    ExpectedResult,
    Score,
    TaskResult,
)

# Re-exported here so type checkers and IDEs resolve them without
# triggering the lazy load that ``__getattr__`` performs at runtime.
if TYPE_CHECKING:
    from agent_evals._composition import (
        run_benchmark_async,
        run_eval,
        run_eval_async,
    )
    from agent_evals._readers import (
        IntegrityError,
        read_encrypted_file,
        read_encrypted_jsonl,
    )
    from agent_evals.benchmark.types import BenchmarkResult

__version__ = _get_version("agent-evals")


__all__ = [
    "run_eval",
    "run_eval_async",
    "run_benchmark_async",
    "BenchmarkResult",
    "BaseAgent",
    "Score",
    "EvalResult",
    "EvalExample",
    "ExampleData",
    "TaskResult",
    "ExpectedResult",
    "langchain_to_openai",
    "strands_to_openai",
    "mink_to_openai",
    "read_encrypted_file",
    "read_encrypted_jsonl",
    "IntegrityError",
]


def _load_run_eval() -> Any:
    from agent_evals._composition import run_eval

    return run_eval


def _load_run_eval_async() -> Any:
    from agent_evals._composition import run_eval_async

    return run_eval_async


def _load_run_benchmark_async() -> Any:
    from agent_evals._composition import run_benchmark_async

    return run_benchmark_async


def _load_langchain_to_openai() -> Any:
    from agent_evals.adapters.converters import langchain_to_openai

    return langchain_to_openai


def _load_strands_to_openai() -> Any:
    from agent_evals.adapters.converters import strands_to_openai

    return strands_to_openai


def _load_mink_to_openai() -> Any:
    from agent_evals.adapters.converters import mink_to_openai

    return mink_to_openai


def _load_read_encrypted_file() -> Any:
    from agent_evals._readers import read_encrypted_file

    return read_encrypted_file


def _load_read_encrypted_jsonl() -> Any:
    from agent_evals._readers import read_encrypted_jsonl

    return read_encrypted_jsonl


def _load_integrity_error() -> Any:
    from agent_evals._readers import IntegrityError

    return IntegrityError


def _load_benchmark_result() -> Any:
    from agent_evals.benchmark.types import BenchmarkResult

    return BenchmarkResult


def _load_compile_check() -> Any:
    from agent_evals._composition import compile_check

    return compile_check


def _load_compiled_check() -> Any:
    from agent_evals._composition import CompiledCheck

    return CompiledCheck


_LAZY: dict[str, Callable[[], Any]] = {
    # Public entry points (defined in _composition.py).
    "run_eval": _load_run_eval,
    "run_eval_async": _load_run_eval_async,
    "run_benchmark_async": _load_run_benchmark_async,
    # Framework-output normalizers (defined in adapters.converters).
    "langchain_to_openai": _load_langchain_to_openai,
    "strands_to_openai": _load_strands_to_openai,
    "mink_to_openai": _load_mink_to_openai,
    # Encrypted artifact readers (defined in _readers.py).
    "read_encrypted_file": _load_read_encrypted_file,
    "read_encrypted_jsonl": _load_read_encrypted_jsonl,
    "IntegrityError": _load_integrity_error,
    # Benchmark public surface.
    "BenchmarkResult": _load_benchmark_result,
    # Internal aliases for benchmark layer (underscore-prefixed, NOT in __all__).
    # Benchmark cannot import _composition directly per Contract 3; these
    # PEP-562 entries let `from agent_evals import _compile_check, _CompiledCheck`
    # resolve through the package root without violating architecture boundaries.
    "_compile_check": _load_compile_check,
    "_CompiledCheck": _load_compiled_check,
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        return _LAZY[name]()
    raise AttributeError(f"module 'agent_evals' has no attribute {name!r}")
