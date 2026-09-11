# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""State success-check parser: YAML key ``state``.

The YAML payload IS the expected-state dict (no inner field).
``to_spec()`` wraps it into a ``StateCheck`` DTO.
"""

from __future__ import annotations

from typing import Any

from pydantic import RootModel

from agent_evals.core._registries import success_checker_registry
from agent_evals.core.checks import StateCheck


@success_checker_registry.register("state")
class StateCheckParser(RootModel[dict[str, Any]]):
    """Validates that the payload is a dict; non-dict payloads fail at
    config-load time.
    """

    def to_spec(self) -> StateCheck:
        return StateCheck(expected_state=self.root)


__all__ = ["StateCheckParser"]
