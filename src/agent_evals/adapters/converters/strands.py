# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Strands (AWS Bedrock content-block format) message normalization.

Strands conversations come back as `list[Message]` where Message is a
TypedDict with `role` and `content: list[ContentBlock]`. A ContentBlock
is a tagged union — one of `{"text": ...}`, `{"toolUse": {...}}`,
`{"toolResult": {...}}`, or other media types.

This module flattens that shape to OpenAI format:
  - Text blocks -> content string on the containing message
  - toolUse blocks -> tool_calls entry on the containing assistant message
  - toolResult blocks (carried on "user" role in Strands) -> separate
    "tool" role messages with tool_call_id linking back to the toolUse

Requires the `strands` extra:
    pip install "agent-evals[strands]"

The extra is required even though the conversion itself only uses stdlib
dicts: we want the `ImportError` to fire at the module's import site with
a helpful message, so users see "install the strands extra" rather than
some downstream failure when their Strands agent tries to run.
"""

import json
from typing import Any

try:
    import strands  # noqa: F401 -- presence check
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "Strands converter requires the 'strands' extra. "
        "Install with: pip install 'agent-evals[strands]' "
        "(or uv add 'agent-evals[strands]')."
    ) from _err


def _content_list_to_text(content: list) -> str:
    """Collapse a list of ToolResultContent items into a plain string."""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            parts.append(str(item["text"]))
        elif isinstance(item, dict) and "json" in item:
            parts.append(json.dumps(item["json"]))
        else:
            parts.append(str(item))
    return "\n".join(parts)


def strands_to_openai(messages: Any) -> list[dict]:
    """Normalize a Strands message history into OpenAI-format message dicts.

    Args:
        messages: `list[Message]` from a Strands agent
            (`list(agent.messages)` after invoking the agent, or the
            equivalent from a session store).

    Returns:
        A list of OpenAI-format message dicts suitable for
        `TaskResult.from_messages(...)`.

    Raises:
        TypeError: If `messages` is not a list.
    """
    if not isinstance(messages, list):
        raise TypeError(
            f"strands_to_openai expects a list of Strands Message dicts; "
            f"got {type(messages).__name__}"
        )

    out: list[dict] = []

    for msg in messages:
        role = msg["role"]
        content_blocks = msg.get("content", [])

        text_parts: list[str] = []
        tool_uses: list[dict] = []
        tool_results: list[dict] = []

        for block in content_blocks:
            if not isinstance(block, dict):
                continue
            if "text" in block:
                text_parts.append(str(block["text"]))
            elif "toolUse" in block:
                tu = block["toolUse"]
                tool_uses.append(
                    {
                        "id": tu.get("toolUseId", ""),
                        "type": "function",
                        "function": {
                            "name": tu.get("name", ""),
                            "arguments": json.dumps(tu.get("input", {})),
                        },
                    }
                )
            elif "toolResult" in block:
                tr = block["toolResult"]
                tool_results.append(
                    {
                        "tool_call_id": tr.get("toolUseId", ""),
                        "content": _content_list_to_text(tr.get("content", [])),
                    }
                )

        # Bedrock "user" with ONLY toolResults -> N "tool" messages in OpenAI.
        if role == "user" and tool_results and not text_parts:
            for tr in tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": tr["content"],
                    }
                )
            continue

        openai_msg: dict[str, Any] = {
            "role": role,
            "content": "\n".join(text_parts) if text_parts else "",
        }
        if tool_uses:
            openai_msg["tool_calls"] = tool_uses
        out.append(openai_msg)

        # Edge case: user message mixing text + toolResult -> emit tool msgs after.
        if role == "user" and tool_results and text_parts:
            for tr in tool_results:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": tr["tool_call_id"],
                        "content": tr["content"],
                    }
                )

    return out
