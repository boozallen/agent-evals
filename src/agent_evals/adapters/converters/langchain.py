# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""LangChain / LangGraph message normalization.

Exposes a single function that converts LangChain's native message objects
(HumanMessage, AIMessage, ToolMessage, etc.) into OpenAI-format message
dicts. Callers then wrap the result in a TaskResult — typically via
TaskResult.from_messages(...).

Requires the `langchain` extra:
    pip install "agent-evals[langchain]"
"""

from typing import Any

try:
    from langchain_core.messages.utils import convert_to_openai_messages
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "LangChain converter requires the 'langchain' extra. "
        "Install with: pip install 'agent-evals[langchain]' "
        "(or uv add 'agent-evals[langchain]')."
    ) from _err


def langchain_to_openai(messages: Any) -> list[dict]:
    """Normalize LangChain messages to OpenAI-format message dicts.

    Args:
        messages: The list of LangChain BaseMessage objects (e.g., what
            LangGraph's `agent.invoke(...)["messages"]` returns).

    Returns:
        A list of OpenAI-format message dicts suitable for
        `TaskResult.from_messages(...)`.
    """
    return convert_to_openai_messages(messages)
