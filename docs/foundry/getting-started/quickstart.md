---
sidebar_position: 1
---

# Quickstart

Agent Evals is a platform-agnostic agent evaluation library. Call `run_eval` /
`run_eval_async` with a task, dataset, scorers, and a `platform` string
(`"local"`, `"braintrust"`, `"mlflow"`, `"langfuse"`); adapters handle
everything platform-specific.

## Install

Requires Python 3.14+ and [uv](https://docs.astral.sh/uv/).

```bash
uv init --python 3.14
```

Install as a dev dependency (evals are test-time infrastructure, not runtime code):

```bash
uv add --dev "agent-evals @ git+https://github.com/boozallen/agent-evals"
```

Or pinned to a specific release:

```bash
uv add --dev "agent-evals @ git+https://github.com/boozallen/agent-evals@v1.0.0"
```

**Optional extras**: `braintrust`, `langchain`, and `strands` — each pulls in
that framework's SDK so the corresponding adapter or converter can be used.

```bash
uv add --dev "agent-evals[braintrust] @ git+https://github.com/boozallen/agent-evals"
uv add --dev "agent-evals[strands] @ git+https://github.com/boozallen/agent-evals"
```

## Two ways to evaluate

- **Eval** (`run_eval` / `run_eval_async`) — a single-scenario regression test
  against a dataset of examples, scored by one or more scorers. Reach for this
  when you're testing one specific behavior or failure mode.
- **Benchmark** (`run_benchmark_async`) — breadth-coverage testing across many
  single-turn scenarios, sliced by capability. Subclass `agent_evals.BaseAgent`
  and drive it from a YAML config.

See the full walkthroughs, including a worked example agent, in
[Eval Quickstart](https://github.com/boozallen/agent-evals/blob/main/README.md#eval-quickstart)
and
[Benchmark Quickstart](https://github.com/boozallen/agent-evals/blob/main/README.md#benchmark-quickstart).

## Next steps

- **[Architecture](../guides/architecture.md)** — hexagonal design, ports and adapters, rationale
- **[Benchmarks](../guides/benchmarks.md)** — YAML benchmark schema and `run_benchmark_async`
- **[Scorers](../guides/scorers.md)** — native, Autoevals, and AgentEvals scorers
- **[Platforms](../guides/platforms.md)** — Braintrust, MLflow, LangFuse, and local adapters
- **[Converters](../guides/converters.md)** — framework message normalizers
- **[Writing a new adapter](../guides/new-adapter-dev-guide.md)** — step-by-step guide
