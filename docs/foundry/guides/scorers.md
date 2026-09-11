# Scorers

Scorers evaluate your agent's outputs. Mix and match scorers from different libraries or write your own.

## Native Scorers

Native scorers ship in `agent_evals` directly and have no external dependencies beyond Pydantic.

### StateMatch

Validates that an agent's final world state matches expected dotted-path values. Useful when your agent mutates a world (smart-home simulator, game state, database fixture) and you want to assert specific cells ended up correct — independent of the agent's prose output or tool-call order.

**Agent contract:** return `TaskResult(context={"final_state": <dict>})`. `StateMatch` traverses `final_state` via dotted paths and compares each leaf to the expected value.

**Two ways to supply expectations:**

- **Factory-level** (shared across all examples): pass `expected_state=` when constructing the scorer.
- **Per-example** (different expectations per test): attach `ExpectedResult.context["expected_state"]` to each `ExampleData`. Per-example always overrides the factory default when present. This is the path the benchmark runner uses.

```python
import asyncio

from agent_evals import run_eval_async, TaskResult
from agent_evals.adapters.scorers import StateMatch
from agent_evals.core.types import ExampleData, ExpectedResult


def light_agent(input_):
    prompt = input_["prompt"] if isinstance(input_, dict) else input_
    state = {"lights": {}}
    if "kitchen" in prompt.lower() and "on" in prompt.lower():
        state["lights"]["kitchen"] = {"state": "on"}
    return TaskResult(output="", context={"final_state": state})


dataset = [
    ExampleData(
        input="Turn on the kitchen light",
        expected=ExpectedResult(
            expected="",
            context={"expected_state": {"lights.kitchen.state": "on"}},
        ),
    ),
]


async def main():
    result = await run_eval_async(
        task=light_agent,
        dataset=dataset,
        scorers=[StateMatch()],  # no factory default — each test supplies its own
    )
    print(result.pass_rates)  # {"StateMatch": 1.0}


if __name__ == "__main__":
    asyncio.run(main())
```

**What the score tells you:**

- `value` = fraction of declared paths that matched (e.g. `0.5` when 1 of 2 paths fail)
- `passed` = `True` only when every path matched
- `metadata["failures"]` = per-path `{expected, actual, reason?}` — including `"path missing"` when a dotted path doesn't exist in the state dict at all
- `metadata["paths_checked"]` = total count of paths evaluated

**When StateMatch shines:**
- End-to-end smart-home / IoT regression testing (lights, doors, thermostats)
- Database-fixture assertions after a pipeline run
- World-model validation in game/simulation agents
- Anywhere tool-call trajectory matters less than the *outcome* of those calls

StateMatch is the default scorer pattern for YAML benchmarks — see [Benchmarks](benchmarks.md) for how per-test `expected_state` in YAML flows into this scorer automatically.

### ToolCallExactMatch

Assert the agent made exactly this ordered sequence of tool calls — no more, no less. Compares **tool calls only** (not whole message lists), pairwise, in order.

**Agent contract:** return `TaskResult(context={"outputs": <messages>})`. Reference goes in `ExpectedResult.context["reference_outputs"]`. Each message's `tool_calls` entries can be either OpenAI wire shape (`{"function": {"name": ..., "arguments": "<json>"}}`) or natural flat shape (`{"name": ..., "args": {...}}`) — both normalize to the same `(name, args)` pair.

**Factory kwarg:**
- `tool_args_match_mode: "exact" | "subset" | "superset" | "ignore"` (default `"exact"`). Direction matches the `match_calls` vocabulary so the same word means the same containment regardless of axis.
  - `"exact"` — actual args dict equals reference args dict (same keys, same values).
  - `"subset"` — actual ⊆ reference (every kv the agent passes is allowed by the reference; **no extras** beyond what reference lists).
  - `"superset"` — reference ⊆ actual (every reference kv is present and equal in actual; **extras OK**).
  - `"ignore"` — skip arg comparison entirely; only names and order must match.

**YAML usage:**

```yaml
capabilities:
  - id: land-where-asked
    prompt: "Fly to (1, 2) and land."
    scorers:
      - type: ToolCallExactMatch
        tool_args_match_mode: exact
    reference_outputs:
      - role: assistant
        tool_calls:
          - name: fly_to
            args: { lat: 1, lon: 2 }
      - role: assistant
        tool_calls:
          - name: land
            args: { soft: true }
```

Non-tool-call messages (user prompts, tool responses, plain assistant text) in either actual or reference are filtered out before comparison, so you can write the reference with only the expected tool calls.

**The five `ToolCall*` natives at a glance:**

| Scorer | Order | Extras allowed? | Omissions allowed? |
|---|---|---|---|
| `ToolCallExactMatch` | required | no | no |
| `ToolCallSupersetMatch` | required (subseq) | yes | no |
| `ToolCallSubsetMatch` | required (subseq) | no | yes |
| `ToolCallUnorderedMatch` | not required | no | no |
| `ToolCallAnyMatch` | n/a (existence) | yes | n/a |

All five strip non-tool-call frames before comparison, so YAML references list only the expected calls without tool-response or prose padding. The corresponding `agentevals` trajectory scorers (`TrajectoryStrictMatch`, `TrajectorySupsetMatch`, `TrajectorySubsetMatch`, `TrajectoryUnorderedMatch`) compare full OpenAI message lists and remain available for direct callers who want to assert on user/assistant prose frames; `tool_use:` benchmark success checkers no longer route through them.

### ToolCallSupersetMatch

Assert that **every** reference call appears in actual, in order; intervening or trailing extras are tolerated. Use this when the YAML lists the *required minimum* set of calls the agent must make and you don't care what else it does.

**Factory kwarg:**
- `tool_args_match_mode: "exact" | "subset" | "superset" | "ignore"` (default `"exact"`). Same uniform direction as the other natives.

### ToolCallSubsetMatch

Assert that **every** actual call appears in reference, in order. The agent may skip reference calls but cannot introduce calls not in reference. Use this when reference lists the *full set of permitted* calls and the agent should pick a valid subsequence.

**Factory kwarg:** same as above.

### ToolCallUnorderedMatch

Assert the same multiset of calls regardless of order. Counts must match. Use this when the agent must call exactly the same tools as reference but no specific ordering is required.

**Factory kwarg:** same as above.

### ToolCallAnyMatch

Assert the agent made **at least one** call matching **any** entry in the reference list. Disjunctive existence check: extras are allowed, silence (zero matching calls) fails. Use this when the YAML lists a set of *acceptable* tool-call shapes and the agent must make at least one matching call — for example, a status query that accepts `device_type=lights` or `device_type=all`.

**Agent contract:** identical to `ToolCallExactMatch` — `TaskResult(context={"outputs": <messages>})` and `ExpectedResult.context["reference_outputs"]`.

**Factory kwarg:**
- `tool_args_match_mode: "exact" | "subset" | "superset" | "ignore"` (default `"exact"`). Same semantics as `ToolCallExactMatch`.

**Why it exists:** the other four list-level modes don't express "at least one of these alternatives, with silence-fails." `subset` lets silence pass (empty trajectory ⊆ any reference); `superset` ANDs across reference entries (every entry required). `ToolCallAnyMatch` fills the disjunctive gap. Reach it from a benchmark by setting `match_calls: any_of` on the `tool_use:` success checker.

---

## Autoevals Scorers

[Autoevals](https://github.com/braintrustdata/autoevals) provides production-ready scorers for string matching, LLM-as-judge evaluation, and more.

**Available Categories:**
- **Heuristic** (4 scorers): Fast, deterministic - `Levenshtein`, `ExactMatch`, `NumericDiff`, `JSONDiff`
- **Embedding** (1 scorer): Semantic similarity - `EmbeddingSimilarity`
- **LLM-as-Judge** (10 scorers): Use LLMs to evaluate - `Factuality`, `ClosedQA`, `Humor`, `Security`, `Summary`, `Translation`, `Sql`, `Possible`, `Battle`, `LLMClassifier`
- **Composite** (2 scorers): Complex evaluations - `ListContains`, `ValidJSON`

### Example: Combining Multiple Scorers

The following example demonstrates combining an Autoevals scorer with a custom scorer we defined to validate business logic.

```python
import asyncio
import json

from agent_evals import run_eval_async, TaskResult
from agent_evals.core.types import ExampleData, ExpectedResult, Score
from agent_evals.adapters.scorers.autoevals import ValidJSON

# 1. Define your agent
async def calculator_agent(query: str) -> TaskResult:
    """Agent that performs calculations and returns JSON results."""
    calculations = {
        "What is 2 + 2?": '{"result": "4"}',
        "What is 10 * 5?": '{"result": "48"}',  # Wrong answer!
        "What is 100 / 4?": '{"result": "25", invalid json',  # Malformed JSON
    }
    return TaskResult(output=calculations.get(query, '{"error": "unknown"}'))

# 2. Create a custom scorer for business logic validation
async def calculation_correct(result: TaskResult, expected: ExpectedResult | None = None, **context) -> Score:
    """Validates that the calculation result is mathematically correct.

    Note: **context required by Scorer Protocol for forward compatibility.
    """
    try:
        result_json = json.loads(result.output)
        actual = result_json.get("result")
        expected_value = expected.expected if expected else None
        is_correct = actual == expected_value
        return Score(name="calculation_correct", value=1.0 if is_correct else 0.0, passed=is_correct)
    except (json.JSONDecodeError, KeyError, TypeError):
        return Score(name="calculation_correct", value=0.0, passed=False)

# 3. Create test cases
dataset = [
    ExampleData(input="What is 2 + 2?", expected=ExpectedResult(expected="4")),
    ExampleData(input="What is 10 * 5?", expected=ExpectedResult(expected="50")),
    ExampleData(input="What is 100 / 4?", expected=ExpectedResult(expected="25")),
]

# 4. Configure scorers - each evaluates a different layer
scorers = [
    ValidJSON(),                    # Structural: is it valid JSON?
    calculation_correct,            # Business logic: is the math correct?
]

async def main():
    # 5. Run evaluation
    from agent_evals.adapters.platforms.local import LocalConfig
    result = await run_eval_async(
        task=calculator_agent,
        dataset=dataset,
        scorers=scorers,
        platform=LocalConfig(experiment="multi_scorer"),
    )

    # Show results for each scorer
    print("=== Evaluation Results ===\n")
    for scorer_name in ["ValidJSON", "calculation_correct"]:
        print(f"{scorer_name}:")
        print(f"  Pass Rate: {result.pass_rates[scorer_name]:.0%}")
        print(f"  Avg Score: {result.scores[scorer_name]:.2f}\n")

    print("Per-example breakdown:")
    for i, example in enumerate(result.examples, 1):
        print(f"\nExample {i}: {example.output}")
        for scorer_name in ["ValidJSON", "calculation_correct"]:
            score = example.scores[scorer_name]
            status = "✓" if score.passed else "✗"
            print(f"  {scorer_name}: {status} ({score.value:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
```

**Output**:
```
=== Evaluation Results ===

ValidJSON:
  Pass Rate: 67%
  Avg Score: 0.67

calculation_correct:
  Pass Rate: 33%
  Avg Score: 0.33

Per-example breakdown:

Example 1: {"result": "4"}
  ValidJSON: ✓ (1.00)
  calculation_correct: ✓ (1.00)

Example 2: {"result": "48"}
  ValidJSON: ✓ (1.00)
  calculation_correct: ✗ (0.00)

Example 3: {"result": "25", invalid json
  ValidJSON: ✗ (0.00)
  calculation_correct: ✗ (0.00)
```

**What this shows**:
- **Multiple scorers** evaluate different layers: structure and business logic
- **ValidJSON** (autoevals): Checks if output is valid JSON - Example 3 fails (malformed)
- **calculation_correct** (custom): Checks if math is correct - Example 1 passes (4 = 4), Example 2 fails (48 ≠ 50)
- **Function-based scorer**: Simpler than classes for straightforward validation logic
- **Use case**: Layer validation from structure → business logic, mixing built-in and custom scorers
- **Note**: `**context` is required by the Scorer Protocol for forward compatibility, even when not used

**Learn more**: [Autoevals documentation →](https://www.braintrust.dev/docs/reference/autoevals) - Reference for every Autoevals scorer, organized by category (heuristic, embedding, LLM-as-judge, composite)

---

## Agentevals Scorers

[Agentevals](https://github.com/langchain-ai/agentevals) provides scorers for evaluating multi-step agent trajectories, tool calls, and execution paths.

**Available Scorers:**
- **Trajectory Matching** (No API key required): `TrajectoryStrictMatch`, `TrajectoryUnorderedMatch`, `TrajectorySubsetMatch`, `TrajectorySupsetMatch`
- **Trajectory LLM Evaluation** (Requires API key or Ollama): `TrajectoryLLMAsJudge`, `GraphTrajectoryLLMAsJudge`

### Configuring the judge endpoint

`TrajectoryLLMAsJudge` and `GraphTrajectoryLLMAsJudge` send prompts, trajectories, and tool-call payloads to a judge model, so both require `judge_base_url` naming where that model is:

```python
from agent_evals.adapters.scorers.agentevals import TrajectoryLLMAsJudge

scorer = TrajectoryLLMAsJudge(
    model="openai:gpt-4o-mini",
    judge_base_url="https://judge.example.com/v1",
)
```

`model` selects the provider and model; `judge_base_url` pins where it is reached. The two are separate because `model` is a provider nickname (`"ollama:gpt-oss:20b"`, `"openai:gpt-4o-mini"`), not an address — omitting the endpoint would leave it to be resolved inside the provider package, where this library never sees it. Omitting it therefore raises `ValueError`, and so does an endpoint that would carry those payloads over a network in cleartext — a judge on your own machine is fine, so `judge_base_url="http://localhost:11434/v1"` is accepted while the same scheme on a remote host is not. See the transport-security section of [platforms.md](platforms.md).

### Example: Trajectory Matching

This example demonstrates evaluating a **real agent** with actual tool calls using LangGraph:

```python
import asyncio
import json

from agent_evals import run_eval_async, TaskResult
from agent_evals.core.types import ExampleData, ExpectedResult
from agent_evals.adapters.scorers.agentevals import TrajectoryStrictMatch
from langchain.agents import create_agent
from langchain_core.messages.utils import convert_to_openai_messages
from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import MemorySaver

# 1. Define tools for your agent
@tool
def lookup_employee_by_id(employee_id: int) -> str:
    """Look up employee name by ID."""
    employees = {1001: "Alice Smith", 1002: "Bob Johnson"}
    return employees.get(employee_id, "Employee not found")

@tool
def get_hr_data(employee_id: int) -> dict:
    """Get HR data for an employee by ID."""
    hr_data = {
        1001: {"salary": 85000, "department": "Engineering", "start_date": "2020-01-15"},
        1002: {"salary": 72000, "department": "Marketing", "start_date": "2021-06-01"},
    }
    return hr_data.get(employee_id, {})

# 2. Create agent wrapper that returns trajectory
def hr_agent_wrapper(query: str) -> TaskResult:
    """Run HR agent and return message trajectory."""
    # Create LangGraph agent
    checkpointer = MemorySaver()
    graph = create_agent(
        model=ChatOllama(model="gpt-oss:20b"),
        tools=[lookup_employee_by_id, get_hr_data],
        checkpointer=checkpointer,
    )

    # Execute agent
    result = graph.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config={"configurable": {"thread_id": "1"}},
    )

    messages = convert_to_openai_messages(result["messages"])
    return TaskResult(output="", context={"outputs": messages})

# 3. Define expected trajectory
dataset = [
    ExampleData(
        input="Lookup the employee name and HR data for employee ID 1001",
        expected=ExpectedResult(
            expected="",
            context={
                "reference_outputs": [
                    {"role": "user", "content": "Lookup the employee name and HR data for employee ID 1001"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "lookup_employee_by_id", "arguments": json.dumps({"employee_id": 1001})}}],
                    },
                    {"role": "tool", "content": "Alice Smith"},
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [{"function": {"name": "get_hr_data", "arguments": json.dumps({"employee_id": 1001})}}],
                    },
                    {"role": "tool", "content": str({"salary": 85000, "department": "Engineering", "start_date": "2020-01-15"})},
                    {"role": "assistant", "content": "Alice Smith works in Engineering"},
                ]
            },
        ),
    )
]

async def main():
    # 4. Run evaluation
    result = await run_eval_async(
        task=hr_agent_wrapper,
        dataset=dataset,
        scorers=[TrajectoryStrictMatch()],
    )

    score = result.examples[0].scores["TrajectoryStrictMatch"]
    print(f"Result: {'✓ PASSED' if score.passed else '✗ FAILED'}")
    print(f"Score: {score.value}")


if __name__ == "__main__":
    asyncio.run(main())
```

**What this shows**:
- **Real Agent Pattern**: Agent executes live with actual LLM and tool calls
- **Multi-step trajectory**: Agent calls `lookup_employee_by_id` first, then `get_hr_data`
- **TrajectoryStrictMatch** validates tool calls (names, arguments, order)
- Agent wrapper returns trajectory in `TaskResult.context["outputs"]`
- Compares actual trajectory to `ExpectedResult.context["reference_outputs"]`

**Learn more**: [agentevals documentation →](https://github.com/langchain-ai/agentevals) - Trajectory matching modes (strict, unordered, subset, superset), LLM-as-judge, and full integration details
