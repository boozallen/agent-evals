# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Benchmark config readers — pluggable URI-dispatched config sources.

Concrete reader classes (``LocalFileConfigReader``) register against
``config_reader_registry`` at import time. The benchmark loader
receives the registry by injection (with the default supplied by the
composition root) and dispatches URIs through it.
"""

from agent_evals.adapters.benchmark_config_readers.local_file import (
    LocalFileConfigReader,
)
from agent_evals.core._registries import config_reader_registry

__all__ = [
    "LocalFileConfigReader",
    "config_reader_registry",
]
