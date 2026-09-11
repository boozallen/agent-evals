# Framework converters

Converters bridge your agent's native output (LangChain messages, Strands content blocks, mink-sdk tool history) into the `list[dict]` of OpenAI-format messages that `agent-evals` scorers expect. Each supported framework gets a single free function whose only job is message-shape normalization — no `TaskResult` wrapping, no scorer-specific state handling.

Call a converter inside your `BaseAgent.run_case` method, then wrap the result with `TaskResult.from_messages(...)`.

---

## Consumer quick-start

### LangChain / LangGraph

```python
from agent_evals import BaseAgent, TaskResult
from agent_evals.adapters.converters.langchain import langchain_to_openai


class MyAgent(BaseAgent):
    name = "my-langgraph"
    version = "0.1.0"

    async def setup(self) -> None:
        self._agent = build_my_langgraph_agent()

    async def run_case(self, prompt: str, **context) -> TaskResult:
        raw = await self._agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        return TaskResult.from_messages(
            langchain_to_openai(raw["messages"]),
            final_state=snapshot_world(),
        )
```

Install the extra:

```bash
uv add "agent-evals[langchain]"
```

### Strands

```python
from agent_evals import BaseAgent, TaskResult
from agent_evals.adapters.converters.strands import strands_to_openai
from strands import Agent


class MyAgent(BaseAgent):
    name = "my-strands"
    version = "0.1.0"

    async def setup(self) -> None:
        self._model = build_my_strands_model()

    async def run_case(self, prompt: str, **context) -> TaskResult:
        strands_agent = Agent(model=self._model, tools=[...])
        # Strands is async-native; await invoke_async to cooperate with the event loop.
        await strands_agent.invoke_async(prompt)
        return TaskResult.from_messages(
            strands_to_openai(strands_agent.messages),
            final_state=snapshot_world(),
        )
```

Install the extra:

```bash
uv add "agent-evals[strands]"
```

### Already in OpenAI format

No converter needed. Pass the dicts straight into `TaskResult.from_messages(...)`:

```python
async def run_case(self, prompt: str, **context) -> TaskResult:
    messages = await call_my_openai_agent(prompt)
    return TaskResult.from_messages(messages, final_state=snapshot_world())
```

### mink-sdk

The `AgenticAgent.chat(...)` return has two pieces: a `response` string and a `tool_history` list. The converter normalizes just the list; the string is the caller's concern. Because `output` is non-default here, use the plain `TaskResult(...)` constructor rather than `TaskResult.from_messages(...)`:

```python
from agent_evals import BaseAgent, TaskResult
from agent_evals.adapters.converters.mink import mink_to_openai


class MyAgent(BaseAgent):
    name = "my-mink-agent"

    async def setup(self) -> None:
        self._agent_ctx = AgenticAgent(config)
        self._agent = await self._agent_ctx.__aenter__()

    async def teardown(self) -> None:
        await self._agent_ctx.__aexit__(None, None, None)

    async def run_case(self, prompt: str, **context) -> TaskResult:
        result = await self._agent.chat(
            prompt, thread_id=f"bench-{uuid.uuid4().hex[:12]}"
        )
        return TaskResult(
            output=result["response"],
            context={
                "outputs": mink_to_openai(result["tool_history"]),
                "final_state": {"drone": ws.get_namespace("drone")},
            },
        )
```

No optional extra is required. The converter does not import `mink_sdk` — conversion is shape-driven from the documented `chat()` return dict.

---

## The two-step pattern

All converters follow the same two-step pattern:

```python
# Step 1: normalize framework-native messages to OpenAI-format dicts
openai_messages = <framework>_to_openai(raw_messages)

# Step 2: build a TaskResult using either the helper or the constructor
# Default-output case — use the classmethod:
TaskResult.from_messages(openai_messages, **context)

# Non-default output — use the constructor:
TaskResult(
    output=my_custom_output,
    context={"outputs": openai_messages, **context},
)
```

`TaskResult.from_messages` applies the library's two conventions:
- Normalized messages land under `context["outputs"]` — the key all trajectory scorers read.
- `output` defaults to the last assistant message's text content — so text scorers (`Levenshtein`, `ExactMatch`, `Factuality`) work without extra setup.

`**context` is the uniform extensibility slot. Scorer-specific data — `final_state=` for `StateMatch`, `reference_outputs=` for trajectory comparison, or anything a custom scorer reads — goes here. The user at the call site is the only place that knows which scorers will run, so scorer-vocabulary keys live at the call site.

---

## Where converters are useful

- **Benchmarks** (`run_benchmark_async`) — call inside `BaseAgent.run_case`. The most common first-time-user foot-gun (hand-rolling trajectory conversion) goes away.
- **Evals** (`run_eval_async`) — same converter; call inside whatever callable you hand to `run_eval_async`.
- **Historical data evaluation** — if you have stored LangChain/Strands transcripts, pass them through the converter to score them.
- **Custom scorers you write** — any time you're building a `TaskResult` by hand, the converter is the primitive.

---

## How to write a new converter (contributors)

A converter is a single free function that takes the framework's native message shape and returns a `list[dict]` of OpenAI-format messages. No class, no decorator, no Protocol.

### Template

```python
# src/agent_evals/adapters/converters/<framework>.py
"""<Framework> message normalization."""

from typing import Any


def <framework>_to_openai(messages: Any) -> list[dict]:
    """Normalize <framework> messages to OpenAI-format message dicts.

    Args:
        messages: The framework's native message representation.

    Returns:
        A list of OpenAI-format message dicts suitable for
        `TaskResult.from_messages(...)`.

    Raises:
        TypeError: If `messages` is the wrong shape.
    """
    # Your framework-specific conversion. The return value must be a list
    # of OpenAI-format message dicts:
    #   {"role": "user"|"assistant"|"tool", "content": str, ...}
    #   - Assistant messages with tool calls: add "tool_calls": [...]
    #   - Tool-result messages: {"role": "tool", "tool_call_id": str, "content": str}
    ...
```

### Checklist before you ship

- [ ] Your function takes one argument (the native messages) and returns `list[dict]`.
- [ ] It raises `TypeError` if given the wrong shape (silent wrong-score is worse than a clear error).
- [ ] Tests exist in `tests/unit/adapters/converters/test_<framework>.py` covering: text messages, tool calls, tool results, shape-error behavior, and an integration assertion that `TaskResult.from_messages(<framework>_to_openai(...))` produces the expected `TaskResult` shape.
- [ ] You re-exported `<framework>_to_openai` from `src/agent_evals/adapters/converters/__init__.py` (add a lazy `__getattr__` branch if the module has an optional dependency).
- [ ] You added the function name to `src/agent_evals/__init__.py`'s `__all__` and `__getattr__` if you want users to import it from the package root.
- [ ] If your framework requires an external library, add it as an optional extra in `pyproject.toml` under `[project.optional-dependencies].<framework>` and gate the import at the top of your module with a helpful `ImportError`.

### Optional dependencies

If your framework requires a native library, follow the pattern:

```toml
# pyproject.toml
[project.optional-dependencies]
myframework = [
    "myframework-sdk>=1.0.0",
]
```

```python
# converter file — at module top level
try:
    import myframework_sdk  # noqa: F401 -- presence check
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "<framework> converter requires the 'myframework' extra. "
        "Install with: pip install 'agent-evals[myframework]'."
    ) from _err
```

Users who install `agent-evals` without your extra can still use every other converter; they only hit the error when they try to import yours.

---

## Error messages you might see

### `ImportError: LangChain converter requires the 'langchain' extra`

You imported the converter module without installing the extra. Fix:

```bash
uv add "agent-evals[langchain]"
```

### `TypeError: strands_to_openai expects a list of Strands Message dicts; got str`

You passed the wrong shape of data. Each converter expects its framework's native message representation, not a raw string.

---

## FAQ

### Do I need to use a converter?

No. If your agent already returns OpenAI-format message dicts, call `TaskResult.from_messages(...)` directly. The converter exists only to do framework-specific message-shape translation.

### My framework isn't in the supported list. What do I do?

Two options:
- **Write your own converter** following the template above. One function, tests, done. If you open a PR, everyone benefits.
- **Do ad-hoc conversion inline** in your `run_case` method. Works fine for single-use cases; doesn't scale across multiple benchmarks.

### Can I have multiple converters for the same framework?

Yes — they're just functions. Import whichever one you want at the call site. The library doesn't resolve converters by name, so there's no naming conflict.

### Does `agent-evals` run my agent?

No. You write a `BaseAgent` subclass whose `run_case` invokes your agent however your framework wants, then pass its output through the converter. The library calls `run_case` with each test's prompt and scores the `TaskResult` you return. Framework invocation stays entirely in your code — the converter only handles the output shape.
