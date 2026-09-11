# agent-evals

Platform-agnostic agent evaluation library. Users call `run_eval` / `run_eval_async` with a task, dataset, scorers, and a `platform` string (`"local"`, `"braintrust"`, `"mlflow"`, `"langfuse"`); adapters handle everything platform-specific. For breadth-coverage benchmarking across many single-turn tests sliced by capability and per-scenario dimensions, users subclass `agent_evals.BaseAgent` (implementing `run_case`, optionally overriding `setup` / `teardown`) and call `run_benchmark_async(yaml_path, agent=MyAgent)`, which instantiates the class, runs `setup`, orchestrates one `run_eval_async` per capability, runs `teardown` in a `finally` block, and returns a `BenchmarkResult`. Users then call `print(result.to_markdown())` to echo the report and `result.write_report(dir)` to persist a timestamped Markdown copy.

This is a **library**, not an application. Users depend on its public API from their own code, so stability constraints are stricter than a typical app.

## Commands

Everything runs through `uv`. Python 3.14+.

```bash
uv sync --all-extras                     # install deps + dev group + all extras
uv run pytest                            # all tests
uv run pytest tests/unit -x --tb=short   # fast inner loop (matches pre-commit)
uv run ruff check . && uv run ruff format .
uv run ty check                          # type check (Astral's ty, not mypy)
uv run lint-imports                      # architecture boundary enforcement
uv run pre-commit run --all-files        # everything at once
```

Main branch is `develop` (not `main`). Open PRs against `develop`.

## Architecture — non-negotiable rules

Hexagonal (ports & adapters). `import-linter` enforces these in CI; violating them fails the build.

- **`core/` never imports from `adapters/`.** Core defines Protocol ports (`core/ports/platform.py`, `core/ports/scorer.py`) and orchestrates; adapters implement the ports.
- **`core/` holds types that layers below the feature need.** Feature-private types (used only by the feature itself, even across its consumers) stay with the feature. Example: `BaseAgent` lives in `benchmark/agent.py` because only the benchmark feature uses it; `CheckSpec` lives in `core/checks.py` because adapter-layer parsers produce it and the benchmark loader consumes it (cross-layer use). When in doubt: check whether anything below the feature's home layer imports the type — if no, the type belongs with the feature.
- **`core/` describes itself in its own terms, not its callers'.** Docstrings, inline comments, and identifier prose in `core/` reference the Protocols and types defined in `core.ports` — not the families, modules, or external packages that consume or extend them. A `core/` docstring that enumerates "the five families," names a specific `adapters/<x>/__init__.py`, lists optional dependencies (`mlflow`, `braintrust`), or cites spec/issue numbers ties core's documentation to its consumers — adding a sixth family or renaming an external module then forces edits to a module that has no behavioral change. Generic prose belongs in `core/`; specifics belong in the consumer modules. Same architectural principle as the runtime rule above, applied to prose instead of code edges.
- **`adapters/platforms/` and `adapters/scorers/` are independent** — neither imports the other.
- **`benchmark/` is an application layer above adapters.** Its `load_benchmark` accepts `AdapterRegistry[T]` instances for the success-checker, precondition, and config-reader families; the platform registry is consumed by `core/runner.py` (injected by `_composition.py`); the scorer registry is referenced indirectly by `compile_check` calls — but never injected through any runner port. All five families implement the same `AdapterRegistry[T]` Protocol; the module-level instances are `<family>_registry` exports in `core/_registries.py`. None of these registry instances are in `agent_evals.__all__` (per issue #198 audit); advanced users who need to register custom adapters import them from `core/_registries.py` directly. `benchmark/` imports value types from `core.types`, Protocol types from `core.ports` (per DIP — application depends on abstractions), DTOs from `core.checks`, and the public re-exports from `agent_evals` — but never from `core.runner`, specific adapter modules, or `_composition` directly. Contract enforced by `import-linter` (Contract 3, `Benchmark may import only stable abstractions`). All 5 registries wire through `_composition.py` lazy-getter helpers; `benchmark/{loader,config,includes}.py` accept registries as required kwargs, and `_composition.py` is the production caller that supplies defaults.
- **`adapters/converters/` holds framework-output normalizers** — one free function per framework (`langchain_to_openai`, `strands_to_openai`, `mink_to_openai`), each a pure `list[dict]` transformer. No Protocol, no registry: users import the function by name and call it directly. All adapter families are strictly peer-isolated. No cross-family edges exist; success-check construction lives at the composition root via `compile_check` (see `_composition.py`), not as direct imports between adapter families. Reverse edges are also forbidden by import-linter Contract 2. `adapters/preconditions/` is a fifth peer-isolated adapter family with **no** cross-family edges (preconditions don't compose scorers or anything else); both directions are forbidden by import-linter. `adapters/benchmark_config_readers/` is a sixth peer-isolated adapter family, also with **no** cross-family edges (readers don't compose anything — they fetch bytes for a URI). See `docs/foundry/guides/architecture.md` for the rationale on each.
- **Use `typing.Protocol`, not `ABC`**, for interfaces. Duck-typed scorers (plain callables) are a feature.
  - **Exception: `agent_evals.BaseAgent`** (in `benchmark/agent.py`) is a deliberate ABC — it's a user-facing *template*, not a port. The benchmark runner requires subclasses to implement `run_case`; `setup` / `teardown` have no-op defaults. `@abstractmethod` gives instantiation-time failure on missing `run_case`, which matters because `setup` may do expensive work. This exception applies only to user-facing templates, never to internal interfaces.
- **Extend via the `context` dict on `TaskResult` / `ExpectedResult`**, not by adding parameters to port methods. Scorers read their inputs from `result.context.get("outputs")`, `expected.context.get("reference_outputs")`, etc. — that's the one extensibility path. Scorer kwargs (`def scorer(output, expected, **kwargs)`) are NOT plumbed through any platform adapter; only the context dict on the Pydantic types survives round-trip. The minimal interface is load-bearing.
- **Platform task-call contract.** All platform adapters invoke the user task as `task(input_value)`, where `input_value` is the bare `ExampleData.input` from the dataset. Adapters that integrate with SDKs requiring a different call shape (e.g., langfuse's `task(item={...})`) MUST unwrap the SDK shape inside the adapter before calling the user's task. The contract lives on the `Platform` Protocol docstring (`core/ports/platform.py`) and is mechanically pinned by `tests/validation/test_platform_contract.py`.
- **Async-first.** `run_eval_async` and `run_benchmark_async` are the public async entry points; both are in `agent_evals.__all__`. `run_eval` is a thin sync wrapper kept for the `run_eval_async` family only (historical compatibility). The benchmark layer is async-only — `run_benchmark_async` has no sync wrapper; scripts wrap the call in `asyncio.run(...)` at their entry point, async contexts `await` directly. New I/O paths should be async.
- **Composition root is `src/agent_evals/__init__.py`** — it's the only place that wires the adapter factory into core. Don't import adapters directly from core modules.
- **Public API surface (14 names; per-name rationale below).** `agent_evals.__all__` is the documented contract. Each name passes the test "a user, on the documented path, has to write this name in their own source code." Drops require evidence of zero consumer use across `agent-evals-examples`, `agent-evals-cicd-example`, `agent-benchmark-example`, `agent-evals-skills`, `skills-dev-agent-evals`, `insider-threat-eval-service`, `agentic-aiops/gng-eval-harness`, AND `.scratch/` projects. All are under `boozallen` except `gng-eval-harness`, which is a different org — search both when auditing. Reconciled against the org on 2026-08-13, when it was missing `agent-benchmark-example` and `gng-eval-harness`; re-derive it (`gh search repos agent-evals`) rather than trusting it when the audit is load-bearing. Deliberately excluded: `agent-evals-action-test` is a stale (Nov 2025) fork that vendors `src/agent_evals` and self-references `agent-evals[examples]` as its own extra with no index source — it consumes nothing published, so counting its imports as consumer use would be counting this library against itself. Package is pre-1.0 (`0.x`) — surface-area changes do not require a major version bump.

  | Name | Why it's public |
  |---|---|
  | `run_eval`, `run_eval_async`, `run_benchmark_async` | Primary entry points users call |
  | `BenchmarkResult` | Return type of `run_benchmark_async`; users call `.to_markdown()` / `.write_report()` on it |
  | `BaseAgent` | Required ABC users subclass for benchmarks |
  | `Score`, `EvalResult`, `EvalExample` | Return-shape types users annotate / inspect |
  | `ExampleData`, `ExpectedResult`, `TaskResult` | Input/output value types users construct (incl. `TaskResult.from_messages`) |
  | `langchain_to_openai`, `strands_to_openai`, `mink_to_openai` | Documented framework converters; users with those frameworks call directly |

  **What's NOT public** (per issue #198 audit): `write_report` (the free-function name; the public form is `BenchmarkResult.write_report`), `ExecutionConfig`, `PlatformName`, `AdapterRegistry`, `StateCheckParser`, `ToolUseCheckParser`, `PreconditionApplier`, `StatePreconditionApplier`, `BenchmarkConfigReader`, `LocalFileConfigReader`, `scorer_registry`, `platform_registry`, `success_checker_registry`, `precondition_registry`, `config_reader_registry` (15 names total). Each remains reachable from its owning module (`core.types`, `core.ports`, `adapters.<family>`, `_composition`, `core._registries`) for advanced users — only the package-root re-export was dropped. The exact set is mechanically pinned by `tests/validation/test_backward_compatibility.py:test_public_api_drops_unused_surface_per_issue_198` and its runtime-drop companion.

  **Tests are not application code.** Tests may import from underscore-prefixed modules (`agent_evals._composition`, `agent_evals.core._registries`) when verifying internal behavior. The leading underscore is a directive to *application* code, not to tests. Production code outside the composition root never imports `_composition` directly — that's enforced mechanically by `import-linter` Contract 3 (`Benchmark may import only stable abstractions`).

  **Underscore-prefixed `_LAZY` entries (`_compile_check`, `_CompiledCheck`) are NOT public API.** They exist solely to give `benchmark/` a Contract-3-compliant re-export channel for `_composition.py` symbols. They never appear in `__all__`, never appear in user-facing docs, and may be renamed without an RFC.

  Adding a name to `__all__` requires the PR to state (a) the documented user-need, (b) that no alternative public path serves it, and (c) the rationale to extend the table above.

Architectural decisions live in `docs/foundry/guides/architecture.md`; read it before any structural change.

## Layout

```
src/agent_evals/
├── __init__.py                # composition root + public API
├── core/                      # orchestration, types, ports — no adapter imports
│   ├── runner.py              # _run_eval_async_internal (the real engine)
│   ├── types.py               # Pydantic v2 models: Score, ExampleData, etc.
│   └── ports/                 # Platform, Scorer, PreconditionApplier, BenchmarkConfigReader protocols
├── adapters/
│   ├── platforms/                  # braintrust, mlflow, langfuse, local, registry
│   ├── scorers/                    # autoevals, agentevals, state (StateMatch), registry
│   ├── success_checkers/           # success-check parsers (StateCheckParser, ToolUseCheckParser; produce CheckSpec DTOs)
│   ├── preconditions/              # precondition-applier adapters (state, registry)
│   ├── benchmark_config_readers/   # benchmark-config readers (local_file, registry; URI-dispatched)
│   └── converters/                 # langchain, strands, mink (free-function message normalizers)
└── benchmark/                 # application layer: YAML-driven multi-capability runner
    ├── agent.py               # BaseAgent ABC — user-facing template for agents under test
    ├── config.py              # Pydantic YAML schema (BenchmarkConfig, ScenarioSpec, ...)
    ├── loader.py              # load_benchmark + compile_capability (per-scorer-set groups)
    ├── runner.py              # BenchmarkRunner, _run_benchmark_async_internal, _merge_eval_results
    ├── artifacts.py           # _write_report (pure I/O primitive; public form is BenchmarkResult.write_report)
    ├── report.py              # build_markdown_report (pure renderer)
    └── types.py               # BenchmarkResult (macro-avg aggregation)

tests/
├── unit/          # fast, no external services — run these in the inner loop
├── integration/   # may hit real platforms
└── validation/    # contract tests guarding backward compatibility
```

## Types and conventions

- **All core types are Pydantic v2** (`core/types.py`). Don't hand-roll dataclasses for anything that crosses the public API.
- **Datasets must be `list[ExampleData]`**, not `list[dict]`. Plain dicts fail validation — see the test in `tests/unit/test_example_data_validation.py`. When writing or updating tests, construct `ExampleData(input=..., expected=ExpectedResult(expected=...))`.
- **Scorers are callables** `(output: TaskResult, expected: ExpectedResult, **context) -> Score`. Sync or async both work. Duck-typing means plain callables work everywhere.
- **All five adapter families share one dispatch shape: `AdapterRegistry[T]`.** Each family's `registry.py` exports a `<family>_registry` instance; concrete adapters decorate against it (`@scorer_registry.register("Name")`, `@platform_registry.register("name")`, `@success_checker_registry.register("state")`, etc.). The benchmark layer accepts a registry per family as a keyword parameter and lazy-loads the module-level instance when the caller passes `None`. Tests construct fresh `_<Family>Registry()` instances and inject them — they never mutate the module-level instances. See `docs/foundry/guides/architecture.md` "Adapter Registry Injection" for the full pattern.
- **Scorer factories register via `@scorer_registry.register("Name")`** so they're resolvable by name from YAML benchmark configs. The decorator is optional for user-defined scorers (duck typing still works for Python callers), but required for YAML benchmark use.
- **Platform adapters register via `@platform_registry.register("name")`** and must implement `evaluate`, `aevaluate`, and `pull_traces`.
- **Benchmark YAML's `type:` strings must match the `@scorer_registry.register("Name")` identifier** — the same name is used as the Python export and the YAML identifier. When adding a new scorer, decorate the factory and the YAML key comes along for free.
- When mocking `Score` in tests, set `reasoning=None` and `metadata={}` explicitly — Pydantic validation is strict.

## Examples

Examples live in a **separate repo** (`agent-evals-examples`). Don't add example files to this repo; add them there and link from docs if needed.

## Commit style

Conventional commits: `feat(scope): subject`, `fix(scope): subject`, `docs:`, `refactor:`, `test:`, `chore:`. Scopes usually map to a layer (`core`, `adapters`, `scorers`) or an adapter name (`braintrust`, `mlflow`).

## Further reading

- `README.md` — user-facing quick start
- `CONTRIBUTING.md` — full developer workflow and architectural compliance checklist
- `docs/foundry/guides/architecture.md` — design rationale and extension points
- `docs/foundry/guides/scorers.md` — all available scorers (native `StateMatch`, autoevals, agentevals)
- `docs/foundry/guides/benchmarks.md` — YAML benchmark schema, `run_benchmark_async`, and the scorer registry
- `docs/foundry/guides/converters.md` — framework normalizers (`langchain_to_openai`, `strands_to_openai`, `mink_to_openai`) paired with `TaskResult.from_messages(...)`, and how to write a new one
- `docs/foundry/guides/new-adapter-dev-guide.md` — step-by-step for new platform/scorer adapters
