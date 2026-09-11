# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Framework-output normalizers.

Each framework gets one free function that converts its native message
shape into OpenAI-format message dicts. Callers then build a TaskResult
from that list — typically via `TaskResult.from_messages(...)` for the
common case, or the plain `TaskResult(...)` constructor when a
non-default `output` string is required.

How this folder differs from its siblings
------------------------------------------
`adapters/platforms/` and `adapters/scorers/` implement ports
(`Platform`, `Scorer` Protocols) and are resolved at
runtime by name through a registry. Classic ports-and-adapters shape.

This folder does NOT implement a Protocol, has no registry, and is
never resolved at runtime. Each file exports a single pure function
that transforms one data shape into another. Callers import the
function by name at write time.

Both patterns still qualify as "adapters" in Uncle Bob's wider Clean
Architecture sense — they translate between external-world data shapes
and the library's internal shape — so they remain in `adapters/`. The
narrower "adapter = Protocol implementer" reading is a local convention
this folder intentionally does not follow. See the "Why There Is No
Converter Protocol" section in `docs/architecture.md` for the rationale.

See docs/converters.md for the full contributor and consumer guide.

Lazy loading
------------
The LangChain and Strands normalizers import their optional libraries
(`langchain_core`, `strands`) at module load. To avoid forcing every
consumer to install both extras just to `import agent_evals`, these
modules are loaded LAZILY via PEP 562 module-level `__getattr__`.
The mink-sdk normalizer has no optional dependency (conversion
is shape-driven) and could be eager-imported, but is loaded via the
same mechanism for uniformity.
"""

from typing import Any

__all__ = [
    "langchain_to_openai",
    "mink_to_openai",
    "strands_to_openai",
]


def __getattr__(name: str) -> Any:
    if name == "langchain_to_openai":
        from agent_evals.adapters.converters.langchain import langchain_to_openai

        return langchain_to_openai

    if name == "strands_to_openai":
        from agent_evals.adapters.converters.strands import strands_to_openai

        return strands_to_openai

    if name == "mink_to_openai":
        from agent_evals.adapters.converters.mink import (
            mink_to_openai,
        )

        return mink_to_openai

    raise AttributeError(
        f"module 'agent_evals.adapters.converters' has no attribute {name!r}"
    )
