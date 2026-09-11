# Benchmarks

Benchmarks run one `run_eval_async` per capability against a shared `BaseAgent`, producing an aggregate `BenchmarkResult` across all capabilities. They are defined in YAML and executed from a small Python entry point.

**One blessed pattern:** you subclass `agent_evals.BaseAgent`, implement `run_case(prompt, **context)`, and optionally override `setup` / `teardown`. The runner owns the lifecycle — it instantiates the class, awaits `setup` before the case loop, and awaits `teardown` in a `finally` block. No callable shim, no boilerplate around `asyncio.run`.

## Scope — what benchmarks are for

The benchmark feature is deliberately narrow. It's built for agents whose correctness is about **one or both** of:

1. **Trajectory correctness** — did the agent call the right tools, with the right arguments, in the right order?
2. **State correctness** — did the world end up the way it should after the agent ran?

That's it. Benchmarks exist because those two assertions need capability-level grouping, per-dimension slicing, and macro-averaged aggregation — structure that ad-hoc evaluation doesn't provide.

Benchmarks here are **domain-specific evaluation suites**, not public yardsticks like MMLU or Terminal-Bench. They are built to approximate your agent's expected production input distribution so you can hill-climb against it as you change models, prompts, and tools. The capability partition gives you a **diagnostic vector** — when the aggregate score drops, you can see which scenario class regressed and act on it — rather than a single opaque scalar. Scores are meant to be compared run-to-run within your project, not against another team's numbers.

## What benchmarks are NOT for

If your agent is a plain prompt → text function and your correctness criterion is about the text itself (semantic similarity, factuality, string match, JSON validity), the benchmark feature is the wrong tool. Use `run_eval_async` directly — a simpler API without the capability/trajectory/state shape. See the [main README](https://github.com/boozallen/agent-evals/blob/main/README.md) for the `run_eval_async` quick start.

## Designing the test set

Because a benchmark is a hill-climbing instrument, the dataset's job is to make score deltas trustworthy and actionable. Three concerns matter more than test count:

1. **Distributional faithfulness.** The set should look like the inputs the agent actually sees. Sample from production traces, support tickets, or error-analysis sessions; use synthetic generation to fill gaps you've identified, not to substitute for real signal. A benchmark composed of "tests I thought of off the top of my head" hill-climbs the agent toward your imagination, not your users.
2. **Score stability.** A 5pp swing between runs is meaningless if run-to-run variance is also 5pp. Set `runs_per_scenario` (see [`execution`](#execution-optional-object)) high enough that the standard deviation per capability is materially smaller than the deltas you care about — typically 3+ when comparing model or prompt versions, 1 during fast iteration.
3. **Capability boundaries that map to action.** Two capabilities earn their separation when a regression in each would lead to a different investigation. If `qa` regressing and `planning` regressing would both send you to "look at the system prompt," they're probably one capability with dimensions, not two. The taxonomy is load-bearing for diagnostics, so let actionability — not topical neatness — draw the lines.

## Benchmark vs. eval

| | Benchmark | Eval (`run_eval_async`) |
|---|---|---|
| **Shape** | Many single-turn tests, sliced by dimensions for breadth coverage | Targeted scenarios, can be multi-turn |
| **Correctness** | Tool trajectory, world state, or both | Any: text, trajectory, state, custom |
| **Config** | YAML | Python |
| **Use case** | Domain hill-climbing instrument across capability axes | Adversarial scalpel for specific failure modes |

Benchmarks are single-turn in v1. Multi-turn scenarios (building context across prompts, mid-session state injection, replanning) belong in Python evals.

## Recommended project layout

Keep everything benchmark-related together in a single `benchmarks/` folder at your project root:

```
my-project/
├── benchmarks/
│   ├── home-v1.yaml         # benchmark config (stem matches `benchmark:` field)
│   ├── agent.py             # your BaseAgent subclass
│   ├── run.py               # entry point
│   └── results/             # library-written reports (gitignore if you prefer)
│       └── 2026-05-04_14-30-21_home-v1.md
└── ...                      # rest of your project
```

Run it with `uv run python benchmarks/run.py` (or `python benchmarks/run.py`).

**Naming conventions:**
- **YAML filename stem equals the `benchmark:` field** — `home-v1.yaml` contains `benchmark: home-v1`. Keeps filename and content in sync.
- **`BaseAgent` subclass in `benchmarks/agent.py`** — keeps the benchmark-specific glue separate from your production agent code. If your agent is trivially wrappable, you can put `agent.py` + `run.py` in one file; the `run.py` / `agent.py` split is the recommended default.
- **Reports go in `benchmarks/results/`** — timestamped, per-benchmark-id. Add to `.gitignore` if you don't want them committed.

This convention isn't enforced by the library — you can structure your project however you like. The library's reporter (`result.to_markdown()`) returns a string; you decide where it goes.

## Quick Start

### 1. Write the YAML

Save as `benchmarks/home-v1.yaml`:

```yaml
benchmark: home-v1
description: Home-automation regression suite

platform:
  name: local
  experiment: v1

execution:
  runs_per_scenario: 1
  n_parallel_runs: 1

capabilities:
  - name: lights
    scenarios:
      - id: 1
        name: kitchen on
        prompt: Turn on the kitchen light
        dimensions: { phrasing: imperative, depth: literal }   # each dimension key becomes a "By <key>" section in the report
        success:
          state:
            lights.kitchen.state: "on"

  - name: doors
    scenarios:
      - id: 2
        name: lock front door
        prompt: Lock the front door
        dimensions: { phrasing: imperative, depth: literal }
        success:
          state:
            doors.front: locked
          tool_use:
            match_calls: exact     # optional, default "exact"
            match_args: subset     # optional, default "exact"
            calls:
              - name: lock_door
                args: { door: front }

      - id: 3
        name: kitchen off when on
        prompt: Turn off the kitchen light
        dimensions: { phrasing: imperative, depth: literal }
        precondition:                     # seeded *before* the agent runs
          state:
            lights.kitchen.state: "on"    # default is off; flip it on first
        success:                          # asserted *after* the agent runs
          state:
            lights.kitchen.state: "off"
```

The `precondition:` block on scenario 3 above mirrors the structure of `success:`: same registry-dispatched key/payload shape, applied symmetrically before vs. after the run. Without the precondition, "turn off the kitchen light" would tautologically pass — the default state already matches the goal. Seeding `on` first makes the scenario actually exercise the agent.

Each scenario's `success:` block declares **what success means** for that
scenario; the library decides which scorers run from the success-checker
keys you list. There is no capability-level scorer list anymore — each
scenario opts into the checkers it needs, so heterogeneous scenarios
inside one capability are first-class.

### 2. Write the agent

Save as `benchmarks/agent.py`:

```python
from agent_evals import BaseAgent, TaskResult
from agent_evals.adapters.converters.langchain import langchain_to_openai

from my_project.agent import build_langgraph_agent, snapshot_world  # your code


class HomeAgent(BaseAgent):
    name = "home-langgraph"
    version = "0.1.0"

    async def setup(self) -> None:
        # Expensive, once per benchmark run.
        self._agent = build_langgraph_agent()

    async def run_case(self, prompt: str, **context) -> TaskResult:
        raw = await self._agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        # Normalize framework output to OpenAI-format dicts, then wrap
        # in a TaskResult. See converters.md for the full list.
        return TaskResult.from_messages(
            langchain_to_openai(raw["messages"]),
            final_state=snapshot_world(),
        )

    async def teardown(self) -> None:
        # Optional — override when you need to close HTTP clients,
        # terminate subprocesses, or exit framework context managers.
        return None
```

### 3. Write the entry point

Save as `benchmarks/run.py`:

```python
import asyncio

from agent_evals import run_benchmark_async

from agent import HomeAgent


async def main():
    result = await run_benchmark_async("benchmarks/home-v1.yaml", agent=HomeAgent)
    print(result.to_markdown())                       # show the report
    path = result.write_report("benchmarks/results")  # write a timestamped copy
    print(f"Report written to {path}")


if __name__ == "__main__":
    asyncio.run(main())
```

`run_benchmark_async` runs the benchmark and returns the `BenchmarkResult` — no printing, no disk I/O. Call `result.write_report(dir)` to persist a timestamped Markdown copy, and `print(result.to_markdown())` to echo it. Bind the result to a variable if you need to add a CI gate or further analysis (see [Inspecting the result](#inspecting-the-result)).

### 4. Run it

```bash
uv run python benchmarks/run.py
```

### Calling from inside an existing event loop

FastAPI handlers, Jupyter cells with top-level `await`, or any other `async def` context `await` directly instead of calling `asyncio.run`:

```python
@app.post("/run-benchmark")
async def trigger():
    result = await run_benchmark_async("benchmarks/home-v1.yaml", agent=HomeAgent)
    return {"pass_rate": result.aggregate_pass_rates.get("StateMatch", 0.0)}
```

The benchmark API is async-only: `run_benchmark_async` (the public entry point) and `BenchmarkResult.write_report` (the disk sink). Scripts wrap the call with `asyncio.run(...)` at the entry point; everywhere else you're already in an event loop and `await` directly.

### Inspecting the result

`run_benchmark_async` returns the `BenchmarkResult` — use it for CI gates:

```python
async def main():
    result = await run_benchmark_async("benchmarks/home-v1.yaml", agent=HomeAgent)
    if result.aggregate_pass_rates.get("StateMatch", 0.0) < 0.9:
        raise SystemExit("Regression: StateMatch below 90%")


if __name__ == "__main__":
    asyncio.run(main())
```

`run_benchmark_async` is pure (no I/O, no stdout). `result.write_report(dir)` is the disk sink. `BenchmarkResult.to_markdown()` is the renderer.

---

## YAML Schema

Top-level fields are flat (promptfoo-style). The only nested container is `execution`, because its fields are tightly coupled.

### `benchmark` (required, string)

Stable identifier for the benchmark — used in platform experiment names and `BenchmarkResult.benchmark`.

### `description` (optional, string)

Human-readable summary. Not functionally significant.

### `platform` (required, scalar or object)

Either a scalar shorthand:

```yaml
platform: local
```

…or the full object form:

```yaml
platform:
  name: local
  experiment: my-run
```

`name` selects the platform adapter (`local`, `braintrust`, `mlflow`, `langfuse`). The remaining fields are validated against that adapter's typed config; unknown keys or keys that belong to a different platform raise a `ValidationError` at load time. The benchmark runner appends `-<capability_name>` to the `experiment` value so each capability becomes its own experiment on the platform.

**Per-platform fields:**

| Platform | Required | Optional |
|---|---|---|
| `local` | — | `experiment` (auto-named if omitted), `output_dir` |
| `mlflow` | `experiment` | — (tracking URI comes from `MLFLOW_TRACKING_URI` env var) |
| `braintrust` | `experiment`, `project` | `metadata` |
| `langfuse` | `experiment` | `description`, `metadata` |

> **mlflow note:** set the `MLFLOW_TRACKING_URI` environment variable before running; the `tracking_uri` field is no longer accepted in config.

**Example (braintrust):**

```yaml
platform:
  name: braintrust
  project: home-bench
  experiment: v1
```

### `execution` (optional, object)

| Field | Default | Meaning |
|---|---|---|
| `runs_per_scenario` | `1` | Times each scenario runs. Raise for variance data (pass rate + stddev). |
| `n_parallel_runs` | `1` | How many runs execute simultaneously inside a single scorer-group. |

**Safe-by-default:** `n_parallel_runs=1` matches `EvalConfig.max_concurrent_tests=1` — the framework can't detect whether your agent mutates shared state, so serial is the safe assumption. Raise it when your agent is stateless *and* your LLM setup can tolerate the parallelism.

**Per-group, not per-benchmark:** `n_parallel_runs` caps runs in flight *within one scorer-group*. Capabilities run sequentially, and groups within a capability also run sequentially (one `run_eval_async` call per group). The maximum LLM calls in flight at any moment therefore equals `n_parallel_runs` — it is **not** multiplied across groups or capabilities. If you set `n_parallel_runs: 5`, you will see at most 5 concurrent agent invocations regardless of how many groups your scenarios fan out into.

### `capabilities` (required, list)

Each capability is a named bundle of scenarios. The loader composes the per-scorer dataset groups inside the capability from each scenario's declared `success:` block — there is no capability-level scorer list.

```yaml
capabilities:
  - name: <string>            # stable identifier (becomes experiment suffix)
    scenarios:
      - id: <int>              # stable scenario ID — must be unique within its capability; cross-capability duplicates are allowed
        name: <string>         # human-readable
        prompt: <string>       # the agent input
        dimensions:            # optional, dict[str, str]; each key becomes a "By <key>" report section
          <key>: <value>
        success:               # required, non-empty
          <checker-key>:       # one or more registered success checkers
            <checker payload>
        precondition:          # optional; default {} = no preconditions
          <applier-key>:       # zero or more registered precondition appliers
            <applier payload>
```

#### Multi-file composition

For larger benchmarks, capabilities can live in their own files and be composed from a parent. Entries under `capabilities:` accept a URI string (in addition to inline objects); the loader fetches and substitutes the referenced content *before* validation.

```yaml
# benchmarks/home-v1.yaml  (the parent)
benchmark: home-v1
platform: { name: local, experiment: v1 }
capabilities:
  - file://capabilities/lights.yaml
  - file://capabilities/doors.yaml
```

```yaml
# benchmarks/capabilities/lights.yaml  (one capability per file)
name: lights
scenarios:
  - id: 1
    name: kitchen on
    prompt: Turn on the kitchen light
    success:
      state:
        lights.kitchen.state: "on"
```

Children are bare `CapabilitySpec` payloads (`name:` + `scenarios:`); top-level fields like `benchmark:` or `platform:` belong to the parent. Mixed inline + `file://` entries are supported, in source order:

```yaml
capabilities:
  - name: experimental             # iterating? keep it inline
    scenarios: [ ... ]
  - file://capabilities/lights.yaml
  - file://capabilities/doors.yaml
```

A child file may itself be a YAML list, in which case its entries splice into the parent list (each child entry becomes one parent entry); otherwise the child substitutes (one URI string → one parent entry).

**Path resolution.** `file://` references resolve against the *parent file's directory*, not the working directory. `file://capabilities/lights.yaml` from `benchmarks/home-v1.yaml` finds `benchmarks/capabilities/lights.yaml`, regardless of where the entry-point script ran from.

**Containment.** An include may only name a file at or below the directory of the config that references it. A reference resolving outside that directory raises `BenchmarkConfigReadError` at load time, before the file is read. This applies to absolute URIs as well: `file:///abs/path` works when it resolves inside the parent file's directory and raises when it does not — the rule is containment, not a ban on the absolute form. Containment is evaluated on the decoded, resolved filesystem path, so percent-encoded spellings (`%2e%2e`, `..%2f`) are treated identically to their plain equivalents. Backslashes are rejected in include URIs on every platform, since they are a separator on some and an ordinary character on others.

**Cycles and duplicates.** Recursive include cycles (A includes B includes A) raise at load time with all cycle members named in the message. Same URI referenced multiple times in one resolution pass is read once (in-pass memoization). Equivalent spellings of one target collapse to a single key: `file://sub/../child.yaml` and `file://child.yaml` are the same file, so they share one cache entry and cannot be used to slip past cycle detection.

**Pluggable storage.** The dispatch is by URI scheme — `file:` is shipped today; future readers (`s3:`, `https:`) become available when their adapters land. To register a custom reader, see "Adding a new benchmark config reader" below.

#### Adding a new benchmark config reader

A reader fetches the bytes of a benchmark-config URI for one URI scheme. Register a reader to enable that scheme in your benchmark YAML. Neither the Protocol nor the registry instance is in the public re-export surface — both are reachable from their owning modules:

```python
from agent_evals.core.ports import BenchmarkConfigReader  # Protocol (for type annotation)
from agent_evals.core._registries import config_reader_registry


@config_reader_registry.register("s3")
class S3ConfigReader:
    """Fetch benchmark-config text from s3:// URIs."""

    def read(self, uri: str) -> str:
        # Parse uri, fetch from S3, return text content as UTF-8 string.
        ...
```

Once registered, `s3://my-bucket/benchmarks/home-v1.yaml` (and references inside resolved files) become valid include URIs. The `@config_reader_registry.register` decorator enforces the `read(uri)` method at decoration time; missing it fails fast at module import.

#### `success:` block

Each scenario must declare at least one success checker. The keys are registered names (resolved through `agent_evals.adapters.success_checkers`); each value is the checker's YAML payload, validated by the checker's own Pydantic model. The current registered success checkers are:

| Key | Bound scorer | Payload | What it asserts |
|---|---|---|---|
| `state` | `StateMatch` | `{<dotted-path>: <expected>, ...}` | The agent left the world in the expected state. |
| `tool_use` | `ToolCallExactMatch` (default) / `ToolCallSupersetMatch` / `ToolCallSubsetMatch` / `ToolCallUnorderedMatch` / `ToolCallAnyMatch` | `{calls: [{name, args}, ...], match_calls?: "exact"\|"subset"\|"superset"\|"unordered"\|"any_of", match_args?: "exact"\|"subset"\|"superset"\|"ignore"}` | The agent emitted the expected tool calls. Two strictness knobs: `match_calls` controls how the *list* of calls is compared (default `"exact"`); `match_args` controls how each call's *args dict* is compared (default `"exact"`). The bound scorer is selected by `match_calls`. |

The `match_calls` values map to:

- `"exact"` (default) — same calls, same count, same order, pairwise. Same behavior as before this knob existed.
- `"superset"` — the agent must include these calls; extras are allowed. Use this when the agent legitimately calls additional tools beyond the assertion (e.g., a status query that pulls extra context).
- `"subset"` — the agent's calls must be a subset of the reference; the agent may skip listed calls but cannot add extras. Note: empty actual ⊆ any reference, so a silent agent passes.
- `"unordered"` — same calls, any order.
- `"any_of"` — at least one actual call must match at least one entry in `calls:`. Extras are allowed; silence (zero matching calls) fails. Use this for **disjunctive correctness** — the agent has multiple acceptable shapes, and you want to enumerate them as alternatives rather than pick one. Differs from `subset` because silence fails; differs from `superset` because reference entries are alternatives (OR), not requirements (AND).

The `match_args` values map to (direction matches `match_calls` so the same word means the same containment regardless of axis):

- `"exact"` (default) — actual args dict equals reference args dict.
- `"subset"` — actual ⊆ reference (every kv the agent passes is allowed by reference; **no extras**).
- `"superset"` — reference ⊆ actual (every reference kv is present and equal in actual; **extras OK**). Use this when the assertion is "agent must pass at least these args, but can add more."
- `"ignore"` — skip arg comparison entirely; only the tool name (and list-level structure) matters.

```yaml
# Superset example: assert the agent locks the front door, but allow it
# to pass extra knobs (force, override, etc.) without failing the score.
success:
  tool_use:
    match_calls: exact
    match_args: superset
    calls:
      - {name: lock_door, args: {door: front}}
```

> **Orthogonality.** `match_calls` and `match_args` compose independently. `match_args` always describes per-call args containment in a uniform direction (`subset = actual ⊆ reference`, `superset = reference ⊆ actual`, `exact = equality`, `ignore = no comparison`); `match_calls` only governs the list-level relation. The same `match_args` word means the same containment under every list-level mode.

```yaml
# Disjunction example: status query that accepts either a typed lookup or "all"
success:
  tool_use:
    match_calls: any_of
    match_args: exact
    calls:
      - {name: get_device_status, args: {device_type: lights, room: bedroom}}
      - {name: get_device_status, args: {device_type: all,    room: bedroom}}
```

A scenario can declare both `state:` and `tool_use:` — they are independent assertions. The bound scorer for each success checker runs only against scenarios whose `success:` block names that checker; cross-scenario leakage of "scorer ran but had no reference data" is not possible.

Single-field success checkers (like `state`) take their payload directly — the checker *is* its dict. Multi-field success checkers (like `tool_use`, which has `calls` plus optional `match_calls` and `match_args`) wrap their payload in a mapping so the additional fields have somewhere to live.

> **Vocabulary note.** The YAML key `tool_use:` and the parser class `ToolUseCheckParser` use Anthropic-native vocabulary; the OpenAI message-format field on each tool-call dict (`tool_calls` inside `reference_outputs:`) and the bound scorer (`ToolCallExactMatch`) use OpenAI-native vocabulary. The seam is intentional — see the parser class docstring or `docs/architecture.md` "Success-Check DTOs at the Composition Root" for why.

#### Adding a new success checker

If neither `state` nor `tool_use` matches your assertion shape, add a new check type. The pattern is: **DTO in `core/` + parser in `adapters/success_checkers/` + one `case` arm in `_composition.compile_check`**. The canonical rationale is in `docs/architecture.md` "Success-Check DTOs at the Composition Root".

**Step 1 — Declare a frozen dataclass in `core/checks.py`** (alongside `StateCheck` and `ToolUseCheck`):

```python
# core/checks.py
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class ToolArgSubsetCheck:
    """Assert each actual tool call's args is a subset of a reference dict."""
    args: dict[str, Any]
```

**Step 2 — Extend the `CheckSpec` union** in the same file:

```python
CheckSpec = StateCheck | ToolUseCheck | ToolArgSubsetCheck
```

**Step 3 — Add a `case` arm in `_composition.compile_check`**:

```python
# _composition.py  (inside compile_check)
case ToolArgSubsetCheck():
    from agent_evals.adapters.scorers.tool_calls import ToolCallExactMatch
    scorer = ToolCallExactMatch(tool_args_match_mode="subset")
    expected_context = {
        "reference_outputs": [
            {"role": "assistant",
             "tool_calls": [{"name": "*", "args": spec.args}]}
        ]
    }
    return CompiledCheck(scorer=scorer, expected_context=expected_context, scorer_config={})
```

`assert_never` exhaustiveness at the end of `compile_check` means forgetting this step is a type-check failure, not a silent runtime bug.

**Step 4 — Write a Pydantic parser in `adapters/success_checkers/tool_arg_subset.py`**:

```python
from typing import Any
from pydantic import RootModel
from agent_evals.core._registries import success_checker_registry
from agent_evals.core.checks import ToolArgSubsetCheck


@success_checker_registry.register("tool_arg_subset")
class ToolArgSubsetParser(RootModel[dict[str, Any]]):
    """YAML parser for ``tool_arg_subset:`` success blocks.

    Validates the payload and produces a ``ToolArgSubsetCheck`` DTO.
    Scorer construction happens at the composition root (``compile_check``),
    not here — this class is pure YAML-validation infrastructure.
    """

    def to_spec(self) -> ToolArgSubsetCheck:
        return ToolArgSubsetCheck(args=self.root)

__all__ = ["ToolArgSubsetParser"]
```

**Step 5 — Add an eager import + `__all__` entry in `adapters/success_checkers/__init__.py`**:

```python
from agent_evals.adapters.success_checkers.tool_arg_subset import ToolArgSubsetParser

__all__ = [..., "ToolArgSubsetParser"]
```

Register before `run_benchmark_async` is called — typically at module-import time in a file your entry point imports. Once registered, YAML can use the key:

```yaml
success:
  tool_arg_subset:
    args: { room: kitchen }
```

**Testing the new type:** add a parametric case to `tests/unit/test_compile_check.py` covering the new `compile_check` dispatch, and parser-side validation tests in `tests/unit/adapters/success_checkers/test_tool_arg_subset.py`.

**Why this works:** the DTO is consumer-facing data; the parser is YAML-validation infrastructure; the composition root is the only place that knows both. `assert_never` exhaustiveness in `compile_check` ensures forgetting step 3 fails type-checking, not runtime. All adapter families remain peer-isolated — parsers don't import scorers, so there is no cross-family edge to reason about.

#### `precondition:` block

The optional `precondition:` block declares state to seed *before* the agent runs. It mirrors `success:` on the wire: each key names a registered precondition applier (resolved through `agent_evals.adapters.preconditions`); each value is the applier's YAML payload, validated by the applier's own Pydantic model. Omit the block (or leave it empty) for scenarios that don't need seeded state — unlike `success:`, an empty `precondition:` block is allowed.

| Key | Payload | What it does |
|---|---|---|
| `state` | `{<dotted-path>: <value>, ...}` | Deep-merges the payload into the agent's state surface before the prompt runs. Dotted paths expand into nested dicts (`{"a.b.c": v}` → `{"a": {"b": {"c": v}}}`); override always wins on collision. |

Preconditions exist so that the post-run `success:` assertion can detect a real change. If your scenario goal already matches the agent's default state, the success check is a tautology and tells you nothing about the model. Seeding the *opposite* state via `precondition:` makes the scenario actually exercise the agent — see scenario 3 in the [Quick Start](#quick-start) for the canonical pre/post pairing.

Agents opt in by accepting a `preconditions` keyword argument on `run_case`. Agents that ignore the kwarg silently drop the precondition (and the scenario silently regresses to a tautology). Two ways to consume preconditions inside `run_case`:

```python
async def run_case(self, prompt: str, preconditions=(), **_) -> TaskResult:
    for applier in preconditions:
        applier.apply(self._world.state)   # mutates in place
    # ... then run the agent against the (now-seeded) world.
```

The `target` you pass to `applier.apply(target)` is the agent's choice. For `state`-shaped preconditions it should be the dict the agent's tools read and write — typically the same dict you snapshot into `TaskResult.final_state`.

#### Adding a new precondition type

If `state` doesn't match the surface you need to seed (e.g., browser cookies, a database fixture, a vector-store corpus), register a new precondition applier. A precondition applier is one Pydantic model with one method, `apply(target)`, that performs the side effect. The class IS the applier — Pydantic validates the YAML payload at construction; `apply()` mutates the target. The registry decorator binds the YAML key to the class:

```python
from typing import Any
from pydantic import BaseModel, ConfigDict
from agent_evals.core.ports import PreconditionApplier  # Protocol (for type annotation)
from agent_evals.core._registries import precondition_registry


@precondition_registry.register("cookies")
class CookiePreconditionApplier(BaseModel):
    """Seed browser cookies before the agent runs."""

    model_config = ConfigDict(extra="forbid")

    domain: str
    values: dict[str, str]

    def apply(self, target: dict[str, Any]) -> None:
        # ``target`` is the agent's cookie surface — whatever shape the
        # agent passes to ``applier.apply(...)`` inside ``run_case``.
        target.setdefault(self.domain, {}).update(self.values)


# Statically asserts the class satisfies the PreconditionApplier Protocol.
_: type[PreconditionApplier] = CookiePreconditionApplier
```

Single-field appliers (like `state`) take their payload directly — the applier *is* its dict. Multi-field appliers (like the `cookies` example above) wrap their payload in a mapping so the additional fields have somewhere to live. This mirrors the `state` vs `tool_use` shape on the success-checker side.

Register before `run_benchmark_async` is called — typically at module-import time in a file your entry point imports. Once registered, YAML can use the key:

```yaml
precondition:
  cookies:
    domain: example.com
    values: { session: abc123 }
```

When you add a second applier shape that needs a different surface from the agent's state dict, `run_case` will need to dispatch on `type(applier)` to pick the right surface — see `docs/architecture.md` section 7 for the deferred design tradeoff, the rationale for the one-Protocol shape, and the two ways forward.

---

## Writing a `BaseAgent`

Subclass `agent_evals.BaseAgent` and implement `run_case`. Optionally override `setup` and `teardown`. Declare `name` and `version` as class attributes so reports can identify which agent produced them.

```python
from agent_evals import BaseAgent, TaskResult


class MyAgent(BaseAgent):
    name = "my-agent"
    version = "0.1.0"

    async def setup(self) -> None:
        # Runs once before the case loop. Build your framework agent,
        # open HTTP clients, start subprocesses, load fixtures here.
        ...

    async def run_case(self, prompt: str, **context) -> TaskResult:
        # Runs once per case. Must return a TaskResult. Populate
        # context["outputs"] for trajectory scorers and context["final_state"]
        # for StateMatch -- most framework users do this in one line via a
        # converter. Accept `**context` for forward compatibility; the
        # runner may pass per-case metadata in future releases.
        ...

    async def teardown(self) -> None:
        # Guaranteed to run in a finally block, even if setup or run_case raised.
        # Close HTTP clients, terminate subprocesses, exit context managers.
        # Subclass is responsible for tolerating partial initialization.
        ...
```

**Lifecycle guarantees:**
- `setup` is awaited exactly once before any `run_case` call.
- `teardown` is awaited in a `finally` block, even if `setup` raised partway or `run_case` raised mid-loop.
- `run_case` is called once per case (sequentially or concurrently per the YAML's `n_parallel_runs`).

**About `**context`:** the runner passes no context kwargs today; the slot exists so future per-case metadata (e.g., `run_index` when multi-run variance measurement lands) can be delivered without a signature change. Include `**context` in your subclass signature even if you don't read from it — copy-paste the blessed pattern and you're forward-compatible.

### Using a framework? Skip the hand-written shim

If you're on LangChain / LangGraph, Strands, mink-sdk, or holding OpenAI-format messages already, pair a framework [normalizer](converters.md) with `TaskResult.from_messages(...)` inside `run_case` instead of hand-writing the trajectory conversion:

```python
from agent_evals import BaseAgent, TaskResult
from agent_evals.adapters.converters.langchain import langchain_to_openai


class HomeAgent(BaseAgent):
    name = "home"
    version = "0.1.0"

    async def setup(self) -> None:
        self._agent = build_langgraph_agent()

    async def run_case(self, prompt: str, **context) -> TaskResult:
        raw = await self._agent.ainvoke(
            {"messages": [{"role": "user", "content": prompt}]}
        )
        return TaskResult.from_messages(
            langchain_to_openai(raw["messages"]),
            final_state=snapshot_world(),
        )
```

The normalizer handles framework-native → OpenAI-format message translation; `TaskResult.from_messages(...)` lands the messages at `context["outputs"]` and derives `output` from the last assistant message. Additional keyword arguments (like `final_state=...`) merge into `context` alongside `outputs`. See [Framework converters](converters.md) for the full list and how to write one for a framework we don't yet support.

### What `TaskResult` needs for each scorer

| Scorer | Put data at |
|---|---|
| `StateMatch` | `TaskResult.context["final_state"]` — a dict the scorer will traverse via dotted paths |
| `ToolCall*` natives, `Trajectory*` agentevals wrappers | `TaskResult.context["outputs"]` — list of message dicts with `role` / `content` / `tool_calls` (OpenAI message-format field, not the YAML `tool_use:` key) |

Converters populate these keys for you. If you're not using a converter:

```python
async def run_case(self, prompt: str, **context) -> TaskResult:
    trajectory, final_state = await _execute(prompt)
    return TaskResult(
        output="",  # free-form text if your scorers need it
        context={
            "outputs": trajectory,       # for trajectory scorers
            "final_state": final_state,  # for StateMatch
        },
    )
```

### Remote agents and the observability boundary

The framework scores only what `BaseAgent.run_case` returns. If your agent is behind an HTTP endpoint and you want `ToolCallExactMatch` or `StateMatch` to work, your application must expose that data. How to expose it is an application design concern; the framework does not introspect, intercept, or infer agent internals.

Three common shapes:

**1. In-process (LangGraph, Strands, etc.)** — snapshot a module variable and hand it to a converter. (Shown above.)

**2. Remote, structured response** — endpoint returns messages + state:

```python
class RemoteStructuredAgent(BaseAgent):
    name = "prod-agent"
    version = "api-v3"

    async def setup(self) -> None:
        self._http = httpx.AsyncClient(base_url=os.environ["AGENT_URL"], timeout=60)

    async def teardown(self) -> None:
        await self._http.aclose()

    async def run_case(self, prompt: str, **context) -> TaskResult:
        resp = await self._http.post("/chat", json={
            "prompt": prompt,
            "session_id": f"bench-{uuid.uuid4().hex[:12]}",
        })
        data = resp.json()
        # Messages are already OpenAI-format dicts — no normalizer needed,
        # just hand them straight to TaskResult.from_messages.
        return TaskResult.from_messages(data["messages"], final_state=data["state"])
```

**3. Remote, requires a follow-up state call** — bundle both calls into `run_case`:

```python
async def run_case(self, prompt: str, **context) -> TaskResult:
    chat = await self._http.post("/chat", json={"prompt": prompt, "session_id": sid})
    state = await self._http.get(f"/session/{sid}/state")
    return TaskResult.from_messages(
        chat.json()["messages"],
        final_state=state.json(),
    )
```

**4. Pure black-box text** — if your remote agent returns only text, you can only use text-based scorers (`LLMRubric`, string match, latency/cost checks). Structured scorers like `StateMatch` cannot apply without structured data:

```python
async def run_case(self, prompt: str, **context) -> TaskResult:
    resp = await self._http.post("/chat", json={"prompt": prompt})
    return TaskResult(output=resp.json()["answer"])
```

The rule: **if you want to score it, `run_case` must return it.**

---

## `BenchmarkResult`

```python
class BenchmarkResult:
    benchmark: str
    agent_name: str | None         # from BaseAgent.name
    agent_version: str | None      # from BaseAgent.version
    eval_results: dict[str, EvalResult]  # keyed by capability name

    @property
    def aggregate_scores: dict[str, float]     # macro-avg by scorer name
    @property
    def aggregate_pass_rates: dict[str, float] # macro-avg by scorer name
```

Aggregation is **macro-average across capabilities** — each capability contributes equally regardless of how many tests it contains. If you need a micro-average (weighted by test count), compute it yourself from `eval_results`:

```python
totals = sum(er.summary["total_examples"] for er in result.eval_results.values())
weighted = sum(
    er.pass_rates["StateMatch"] * er.summary["total_examples"]
    for er in result.eval_results.values()
)
micro_avg = weighted / totals
```

Per-capability `EvalResult` objects retain every `EvalExample` with its scores, output, and metadata — drill into `result.eval_results["lights"].examples` for row-level detail.

### Human-readable report

Call `result.to_markdown()` to get a self-contained Markdown report:

```python
result = await run_benchmark_async("benchmarks/home-v1.yaml", agent=HomeAgent)
print(result.to_markdown())
```

Writing to disk (following the recommended `benchmarks/results/` convention):

```python
from datetime import datetime
from pathlib import Path

results_dir = Path(__file__).parent / "results"
results_dir.mkdir(exist_ok=True)
report_path = results_dir / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{result.benchmark}.md"
report_path.write_text(result.to_markdown())
```

The report includes:
- **Score** — headline pass rate across all examples
- **Run info** — platform, capabilities, runs per scenario, total examples
- **Breakdown** — per-capability AND per-dimension rollups (every dimension key you added becomes an axis)
- **Failures (N)** — dedicated section listing failing tests with scorer reasoning, so you know what broke without opening JSONL
- **Tasks** — per-capability table with pass/fail (or Pass% + Std if `runs_per_scenario > 1`), plus a link to the raw artifacts

What shows up in the Breakdown section: `By capability` is always present (structural — every benchmark groups by its capabilities). The `By <dimension>` sections come from the per-scenario `dimensions:` field — declare `dimensions: { phrasing: imperative, depth: literal }` on your scenarios and you get `### By phrasing` and `### By depth` automatically, no reporter configuration needed. The reporter discovers axes by walking metadata, so a typo (`phrasng:`) silently creates a stray section instead of failing — keep dimension keys consistent across scenarios.

---

## Example metadata

Each `ExampleData` the loader produces carries this metadata (available on `example.metadata` in the result):

```python
{
  "scenario_id": int,       # from scenario.id
  "scenario_name": str,     # from scenario.name
  "capability": str,        # from capability.name
  "run_index": int,         # 0..runs_per_scenario-1
  **scenario.dimensions,    # every key from scenario.dimensions verbatim
}
```

Use dimensions for per-axis analysis on your platform — e.g., filter by `phrasing=imperative` vs `phrasing=question` to compare robustness.

---

## Scorer name reference (in-scope only)

Scorers are reached indirectly: each `success:` block key (`state`, `tool_use`, ...) names a registered parser, which produces a `CheckSpec` that `compile_check` maps to a scorer. The names below are the scorer identifiers that show up in `result.pass_rates` / `result.aggregate_pass_rates` and in the markdown report's per-scorer columns — useful to know when reading reports or writing CI gates.

The scorers below are the ones the benchmark feature is built around — all three families assert either tool trajectory or world state. Other scorers in the library (text match, LLM-as-judge on raw text, embedding similarity, JSON validity, etc.) are valid *at the registry level*, but no built-in success checker binds them today. For text-shaped scoring, use `run_eval_async` directly; for new structured assertions, register a new success checker (see [Adding a new success checker](#adding-a-new-success-checker) above).

### Native

| Name | Purpose | Key kwargs |
|---|---|---|
| `StateMatch` | Dotted-path world-state assertions on `TaskResult.context["final_state"]` | Reads `expected_state` from `ExpectedResult.context` (populated by the `state` success checker) |
| `ToolCallExactMatch` | Same calls, same order, pairwise (default `tool_use:` dispatch) | `tool_args_match_mode: "exact" \| "subset" \| "superset" \| "ignore"` |
| `ToolCallSupersetMatch` | Every reference call appears in actual, in order; extras allowed | `tool_args_match_mode` |
| `ToolCallSubsetMatch` | Every actual call appears in reference, in order; agent may skip | `tool_args_match_mode` |
| `ToolCallUnorderedMatch` | Same multiset of calls, any order | `tool_args_match_mode` |
| `ToolCallAnyMatch` | At least one actual call matches at least one reference entry; silence fails | `tool_args_match_mode` |

### Agentevals — trajectory match (sync, full-message-list comparison)

These compare full OpenAI message lists (every user/assistant/tool frame). `tool_use:` no longer routes through them; the native `ToolCall*` family above handles all five `match_calls` modes. Use these wrappers directly when you need to assert on assistant prose between tool calls.

| Name | Purpose | Key kwargs |
|---|---|---|
| `TrajectoryStrictMatch` | Exact message sequence (same messages, same order) | `tool_args_match_mode: "exact" \| "subset"` |
| `TrajectoryUnorderedMatch` | Same messages, any order | `tool_args_match_mode` |
| `TrajectorySubsetMatch` | Actual messages ⊆ reference | `tool_args_match_mode` |
| `TrajectorySupsetMatch` | Actual messages ⊇ reference | `tool_args_match_mode` |
| `GraphTrajectoryStrictMatch` | Exact graph node sequence (LangGraph-style agents) | — |

### Agentevals — LLM-as-judge (async, requires API key or Ollama)

Use when strict/subset matching is too rigid and you need semantic judgment about the trajectory. Costs per test run.

| Name | Purpose |
|---|---|
| `TrajectoryLLMAsJudge` | LLM evaluates trajectory quality against reference |
| `GraphTrajectoryLLMAsJudge` | LLM evaluates graph trajectory |

See [Scorers](scorers.md) for detailed per-scorer documentation including all factory kwargs.

---

## Troubleshooting

**`ValueError: Unknown success checker: 'stat'. Available success checkers: ['state', 'tool_use']`**
Typo in a `success:` block key. Copy the exact checker name from the [`success:` block](#success-block) table above. (Checker keys are not the same as scorer names — `state` binds to `StateMatch`, `tool_use` binds to `ToolCallExactMatch`.)

**`pydantic.ValidationError: 1 validation error ... execution.max_parallel_scorers: Extra inputs are not permitted`**
`max_parallel_scorers` isn't a real field — you probably meant `n_parallel_runs`.

**`asyncio.run() cannot be called from a running event loop`**
You're calling `asyncio.run(run_benchmark_async(...))` from inside a function that's already running in an event loop (FastAPI handler, Jupyter with top-level `await`, async framework entry point). Use `await run_benchmark_async(...)` instead — you're already in a loop, so you don't need to create one.

**Agent runs sequentially even though `n_parallel_runs: 10` is set**
Check that your agent is actually parallel-safe. If `BaseAgent.run_case` mutates shared state (e.g., a module-level world object), parallel runs will corrupt each other even when the benchmark dispatches them in parallel. Either make `run_case` stateless (per-call state lives on local variables) or leave `n_parallel_runs: 1`.

**`TypeError: agent must be a BaseAgent subclass or instance`**
You passed a plain callable. The benchmark runner requires `agent=` to be a `BaseAgent` subclass (or instance). Wrap your existing agent function in a subclass that implements `run_case` — see [Writing a `BaseAgent`](#writing-a-baseagent).

**`TypeError: Can't instantiate abstract class X with abstract method run_case`**
Your subclass didn't implement `run_case`. Add it. This error fires before `setup` runs, so you won't accidentally spin up HTTP clients or subprocesses only to find you forgot the core method.

---

## Relationship to the scorer registry

Benchmark YAML reaches scorers through the success-checker registry and the composition root. Each registered parser's `to_spec()` method produces a `CheckSpec` DTO; `_composition.compile_check` maps the spec to a scorer callable. Every built-in scorer factory is decorated with `@scorer_registry.register("FactoryName")`, so the same identifier serves as the Python symbol *and* as the name surfaced in `result.pass_rates` keys and report columns.

You can register custom scorers (for use inside `compile_check` when you extend the `CheckSpec` union, or for direct use with `run_eval_async`) by decorating your factory:

```python
from agent_evals.core._registries import scorer_registry
from agent_evals.core.types import Score


@scorer_registry.register("MyCustomScorer")
def MyCustomScorer(threshold: float = 0.5):
    def scorer(result, expected=None):
        ...
        return Score(name="MyCustomScorer", value=v, passed=v >= threshold)
    return scorer
```

The registration must happen before `run_benchmark_async` is called — typically at module-import time in a file your entry point imports.
