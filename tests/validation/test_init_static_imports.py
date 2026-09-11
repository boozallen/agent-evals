# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pin the no-eager-load property of ``agent_evals/__init__.py``.

``_composition`` must be reachable only via PEP 562 ``__getattr__``
(function-local) or ``if TYPE_CHECKING:`` (static-only). A bare
top-level ``from agent_evals._composition import ...`` would defeat
the lazy-load — ``import agent_evals`` would suddenly pull in
``core.runner`` and the platform SDKs. import-linter doesn't catch
this because the whitelist accepts the (TYPE_CHECKING) edge; this
AST test catches a regression that would silently slip through.
"""

import ast
from pathlib import Path


def test_init_does_not_statically_import_composition():
    init_path = Path("src/agent_evals/__init__.py")
    tree = ast.parse(init_path.read_text())

    # Walk the top-level statements (not inside functions/methods)
    static_imports = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            if node.module == "agent_evals._composition":
                static_imports.append(node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "agent_evals._composition":
                    static_imports.append(node)

    assert not static_imports, (
        "agent_evals/__init__.py must not statically import _composition. "
        "PEP 562 __getattr__ is the only allowed reference; otherwise "
        "_composition would be eagerly loaded and the import-linter "
        "ignore_imports whitelist would hide an unintended layering edge."
    )
