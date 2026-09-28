---
sidebar_position: 1
---

# Agent Evals

## Make agent readiness measurable

Agent Evals helps teams determine whether an AI agent continues to behave as
expected as prompts, models, tools, and data change. Instead of relying on a
successful demo or a one-time review, teams define the behaviors that matter
and evaluate them consistently across every version.

Engineers get fast feedback about regressions. Mission owners get documented,
repeatable evidence to support readiness decisions. Evaluations can check
whether the agent gave the expected answer, used the right tools, or left a
system in the expected state.

## Evals and benchmarks

- **Evals** protect against a specific failure mode. When a team fixes a
  problem, an eval helps ensure that it does not return.
- **Benchmarks** measure behavior across many scenarios, grouped by capability,
  so teams can see where the agent improved and where it regressed.

Teams can run these checks during development or use them as automated quality
gates before release.

## Technical overview

Agent Evals is a Python library that works with any Python agent. It can run
locally or integrate with Braintrust, MLflow, and LangFuse. Use `run_eval` or
`run_eval_async` for targeted evaluations and `run_benchmark_async` for
capability benchmarks.

## Explore the documentation

- **[Quickstart](./getting-started/quickstart.md)** — install and run your first eval
- **[Architecture](./guides/architecture.md)** — hexagonal design, ports and adapters, rationale
- **[Benchmarks](./guides/benchmarks.md)** — YAML benchmark schema and `run_benchmark_async`
- **[Scorers](./guides/scorers.md)** — native, Autoevals, and AgentEvals scorers
- **[Platforms](./guides/platforms.md)** — Braintrust, MLflow, LangFuse, and local adapters
- **[Converters](./guides/converters.md)** — framework message normalizers
- **[Writing a new adapter](./guides/new-adapter-dev-guide.md)** — step-by-step guide
