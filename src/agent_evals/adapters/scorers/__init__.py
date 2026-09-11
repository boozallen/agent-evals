# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Scorer adapters for external evaluation libraries.

Available scorer adapters:
- autoevals: Wrappers for autoevals library (25+ scorers)
- agentevals: Wrappers for agentevals trajectory evaluators
"""

from agent_evals.adapters.scorers.agentevals import (
    GraphTrajectoryLLMAsJudge,
    GraphTrajectoryStrictMatch,
    TrajectoryLLMAsJudge,
    TrajectoryStrictMatch,
    TrajectorySubsetMatch,
    TrajectorySupsetMatch,
    TrajectoryUnorderedMatch,
)
from agent_evals.adapters.scorers.autoevals import (
    Battle,
    ClosedQA,
    EmbeddingSimilarity,
    ExactMatch,
    Factuality,
    Humor,
    JSONDiff,
    Levenshtein,
    ListContains,
    LLMClassifier,
    Moderation,
    NumericDiff,
    Possible,
    Security,
    Sql,
    Summary,
    Translation,
    ValidJSON,
)
from agent_evals.adapters.scorers.state import StateMatch
from agent_evals.adapters.scorers.tool_calls import ToolCallExactMatch

__all__ = [
    # Autoevals - String scorers
    "Levenshtein",
    "EmbeddingSimilarity",
    # Autoevals - Number scorers
    "NumericDiff",
    # Autoevals - Value scorers
    "ExactMatch",
    # Autoevals - List scorers
    "ListContains",
    # Autoevals - JSON scorers
    "JSONDiff",
    "ValidJSON",
    # Autoevals - LLM scorers
    "Factuality",
    "ClosedQA",
    "Humor",
    "Battle",
    "Security",
    "Summary",
    "Translation",
    "Sql",
    "Possible",
    "LLMClassifier",
    "Moderation",
    # AgentEvals - Trajectory evaluators
    "TrajectoryStrictMatch",
    "TrajectoryUnorderedMatch",
    "TrajectorySubsetMatch",
    "TrajectorySupsetMatch",
    "TrajectoryLLMAsJudge",
    # AgentEvals - Graph trajectory evaluators
    "GraphTrajectoryStrictMatch",
    "GraphTrajectoryLLMAsJudge",
    # Native scorers
    "StateMatch",
    "ToolCallExactMatch",
]
