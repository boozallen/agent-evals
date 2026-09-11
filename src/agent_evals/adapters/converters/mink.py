# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""mink-sdk tool_history normalization.

Normalizes a `tool_history` list (as returned by
`mink_sdk.AgenticAgent.chat(...)["tool_history"]`) into OpenAI-format
message dicts. The caller extracts the list from the `chat()` envelope and
decides separately what to do with the `response` string — typically
passing it as `output=` to the `TaskResult(...)` constructor.

This module does NOT import `mink_sdk` — conversion is shape-driven.
The SDK is distributed via an internal package index, not public PyPI, and
the conversion only needs the well-known return shape; pulling the SDK in
just to type-check would force every consumer of agent-evals to also
configure the internal index. Validation happens on the dict shape at
runtime.

Each `tool_history` entry has the shape:
    {"call": {"name": str, "args": dict},
     "result": {"name": str, "output": Any} | None}

For each entry this produces:
  - One `{"role": "assistant", "tool_calls": [...]}` message.
  - One `{"role": "tool", "tool_call_id": ..., "content": ...}` message,
    only if `result` is not None (an in-flight call emits the assistant
    message alone).
"""

import json
from typing import Any, cast


def mink_to_openai(tool_history: Any) -> list[dict]:
    """Normalize a mink-sdk tool_history list to OpenAI-format dicts.

    Args:
        tool_history: The list under `AgenticAgent.chat(...)["tool_history"]`.
            Each entry must have a `call` sub-dict and a `result` sub-dict
            (or None for in-flight entries).

    Returns:
        A list of OpenAI-format message dicts suitable for
        `TaskResult.from_messages(...)` or `TaskResult(context={"outputs": ...})`.

    Raises:
        TypeError: If `tool_history` is not a list.
    """
    if not isinstance(tool_history, list):
        raise TypeError(
            f"mink_to_openai expects a list of tool_history "
            f"entries; got {type(tool_history).__name__}"
        )

    msgs: list[dict] = []
    for i, entry in enumerate(tool_history):
        # tool_history is typed Any; narrow each entry to a dict for .get() calls.
        entry_dict = cast(dict[str, Any], entry if isinstance(entry, dict) else {})
        call = entry_dict.get("call") or {}
        result = entry_dict.get("result")
        tool_call_id = f"tc_{i}"
        msgs.append(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.get("name", ""),
                            "arguments": json.dumps(call.get("args", {})),
                        },
                    }
                ],
            }
        )
        if result is not None:
            msgs.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": str(result.get("output", "")),
                }
            )
    return msgs
