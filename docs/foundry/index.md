---
sidebar_position: 1
---

# Agent Evals

A platform-agnostic agent evaluation library. Call `run_eval` /
`run_eval_async` with a task, dataset, scorers, and a `platform` string
(`"local"`, `"braintrust"`, `"mlflow"`, `"langfuse"`); adapters handle
everything platform-specific. For breadth-coverage benchmarking across many
single-turn scenarios sliced by capability, subclass `agent_evals.BaseAgent`
and call `run_benchmark_async`.

This is a **library**, not an application — you depend on its public API
from your own code.

- **[Quickstart](./getting-started/quickstart.md)** — install and run your first eval
- **[Architecture](./guides/architecture.md)** — hexagonal design, ports and adapters, rationale
- **[Benchmarks](./guides/benchmarks.md)** — YAML benchmark schema and `run_benchmark_async`
- **[Scorers](./guides/scorers.md)** — native, Autoevals, and AgentEvals scorers
- **[Platforms](./guides/platforms.md)** — Braintrust, MLflow, LangFuse, and local adapters
- **[Converters](./guides/converters.md)** — framework message normalizers
- **[Writing a new adapter](./guides/new-adapter-dev-guide.md)** — step-by-step guide
