# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin _composition.py's top-level imports to wiring-only modules.

Top-level adapter imports break the lazy-import contract. Function-body
imports are explicitly allowed and required for the side-effect pattern
inside the ``_get_<family>_registry`` helpers.
"""

from __future__ import annotations

import ast

# Forbidden at module scope; function-body imports are allowed.
_FORBIDDEN_TOP_LEVEL_PREFIXES = (
    "agent_evals.adapters",
    "agent_evals.core._registries",
)


def _read_composition_source() -> str:
    import agent_evals._composition as comp

    source_path = comp.__file__
    assert source_path is not None
    with open(source_path, encoding="utf-8") as fh:
        return fh.read()


def test_composition_top_level_imports_only_wiring_modules() -> None:
    """Walk Module.body only — function bodies are out of scope.

    The forbidden prefixes are ``agent_evals.adapters`` (concrete adapter
    families) and ``agent_evals.core._registries`` (registry singletons).
    Both must stay inside the lazy-getter helpers; hoisting either to
    module scope makes importing ``_composition`` eagerly walk every
    adapter family.
    """
    source = _read_composition_source()
    tree = ast.parse(source)

    forbidden_at_module_scope: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                for prefix in _FORBIDDEN_TOP_LEVEL_PREFIXES:
                    if alias.name.startswith(prefix):
                        forbidden_at_module_scope.append(f"import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for prefix in _FORBIDDEN_TOP_LEVEL_PREFIXES:
                if module.startswith(prefix):
                    names = ", ".join(a.name for a in node.names)
                    forbidden_at_module_scope.append(f"from {module} import {names}")

    assert not forbidden_at_module_scope, (
        "_composition.py has forbidden top-level imports — these break "
        "the lazy-import contract. Move them inside the relevant function "
        "(e.g., _get_<family>_registry). Forbidden statements found:\n  "
        + "\n  ".join(forbidden_at_module_scope)
    )
