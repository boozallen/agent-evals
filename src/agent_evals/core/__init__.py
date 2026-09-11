# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Core evaluation types and Protocols.

Deliberately omits ``run_eval`` / ``run_eval_async`` — those live at the
package root (``from agent_evals import run_eval``) where the
composition root injects the default adapter factory. Importing the
runner here would drag it into every ``import agent_evals``.
"""

from agent_evals.core.ports import Scorer
from agent_evals.core.types import EvalConfig, EvalExample, EvalResult, Score

__all__ = [
    # Core types
    "Score",
    "EvalResult",
    "EvalExample",
    "EvalConfig",
    # Protocols
    "Scorer",
]
