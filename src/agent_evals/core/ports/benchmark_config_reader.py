# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark-config-reader port.

A BenchmarkConfigReader fetches the text content of a benchmark-config
URI. The benchmark loader resolves the URI's scheme (the part before
``://``) against the reader registry and dispatches to the matching
reader.

Each registered reader handles exactly one scheme (``file``, future
``s3``, ``https``, ...). Readers are stateless; they bind no YAML
payload, so concrete implementations are typically plain classes (no
Pydantic). See ``docs/architecture.md`` "Benchmark Config Readers" for
the design rationale (notably why this family has no cross-family edges
— readers don't compose anything).
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class BenchmarkConfigReader(Protocol):
    """Protocol every concrete benchmark-config reader must satisfy.

    The Protocol is structural; the
    ``config_reader_registry.register`` decorator additionally
    enforces the method at registration time so registry-path readers
    fail fast on a missing method.
    """

    def read(self, uri: str) -> str:
        """Return the text content of ``uri``.

        ``uri`` is expected to use the scheme this reader is registered
        under (e.g., ``file://...`` for ``LocalFileConfigReader``). A
        reader receiving a URI for a different scheme should raise; the
        registry is the primary dispatch point but defense-in-depth
        catches misuse.

        Errors raised by ``read()`` (file not found, network failure,
        permission denied, ...) propagate to the loader, which wraps
        them with the URI of the parent file that referenced this one
        so users see both ends of a broken link.
        """
        ...
