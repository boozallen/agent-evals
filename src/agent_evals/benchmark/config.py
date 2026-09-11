# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Pydantic models for benchmark YAML config."""

from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationInfo,
    model_validator,
)

from agent_evals.core.checks import CheckSpec
from agent_evals.core.ports import (
    AdapterRegistry,
    Parser,
    Platform,
    PreconditionApplier,
)
from agent_evals.core.types import PathSafeIdentifier, PlatformConfig


class ScenarioSpec(BaseModel):
    """One scenario inside a capability.

    The ``success:`` block declares per-scenario success checkers; keys
    name registered success-checker types (see
    ``agent_evals.adapters.success_checkers``).

    The optional ``precondition:`` block declares per-scenario state to
    seed *before* the agent runs; keys name registered precondition-
    applier types (see ``agent_evals.adapters.preconditions``). Empty or
    omitted means "no preconditions for this scenario."

    Each key in ``dimensions`` becomes a ``By <key>`` section in the
    benchmark report — these are diagnostic slicing axes, not free-form
    labels. A typo creates a stray section, so keep keys consistent
    across scenarios.
    """

    id: int
    name: str
    prompt: str
    dimensions: dict[str, str] = Field(default_factory=dict)
    success: dict[str, Any] = Field(...)
    precondition: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="forbid")

    # Pre-parsed CheckSpec instances. The Pydantic validators below
    # invoke each registered parser's to_spec() at config-load time so
    # the loader can read DTOs without re-parsing.
    _parsed_check_specs: list[CheckSpec] = PrivateAttr(default_factory=list)
    # PreconditionApplier (Protocol); benchmark layer can't import the
    # Protocol type directly per import-linter Contract 3.
    _parsed_preconditions: list = PrivateAttr(default_factory=list)

    @property
    def parsed_check_specs(self) -> list[CheckSpec]:
        return self._parsed_check_specs

    @property
    def parsed_preconditions(self) -> list:
        return self._parsed_preconditions

    @model_validator(mode="after")
    def _validate_success_block(self, info: ValidationInfo) -> ScenarioSpec:
        """Parse each ``success:`` entry through the success-checker registry,
        producing a CheckSpec via the parser's `to_spec()`.

        Done at YAML-load (not in the loader) so typos surface early
        with the registry's "Available success checkers:" diagnostic, and
        so the loader can read pre-parsed CheckSpec instances from
        ``parsed_check_specs``.

        The success-checker registry is required and read from
        ``info.context["success_checker_registry"]``. ``load_benchmark``
        supplies it; the public ``run_benchmark_async`` is the typical
        production caller.
        """
        if not self.success:
            raise ValueError(
                f"Scenario {self.id!r} ({self.name!r}): `success:` block "
                "must declare at least one success checker. See "
                "docs/benchmarks.md."
            )
        if info.context is None or info.context.get("success_checker_registry") is None:
            raise ValueError(
                "success_checker_registry must be provided in validation context. "
                "Use load_benchmark(..., success_checker_registry=...) or call "
                "run_benchmark_async."
            )
        registry: AdapterRegistry[type[Parser[CheckSpec]]] = info.context[
            "success_checker_registry"
        ]
        parsed: list[CheckSpec] = []
        for key, payload in self.success.items():
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Scenario {self.id!r}: success checker {key!r} payload "
                    f"must be a mapping; got {type(payload).__name__}."
                )
            try:
                cls = registry.get(key)
            except ValueError as exc:
                raise ValueError(
                    f"Scenario {self.id!r} ({self.name!r}): {exc}"
                ) from exc
            parsed.append(cls(**payload).to_spec())
        self._parsed_check_specs = parsed
        return self

    @model_validator(mode="after")
    def _validate_precondition_block(self, info: ValidationInfo) -> ScenarioSpec:
        """Parse each ``precondition:`` entry through the precondition registry.

        Kept separate from ``_validate_success_block`` so error messages
        stay scoped (a precondition typo doesn't surface as a "success"
        failure) and so each block's allow-empty rule reads cleanly.
        Unlike ``success:``, an empty ``precondition:`` block is allowed.

        The precondition registry is required and read from
        ``info.context["precondition_registry"]``. ``load_benchmark``
        supplies it.
        """
        if not self.precondition:
            return self
        if info.context is None or info.context.get("precondition_registry") is None:
            raise ValueError(
                "precondition_registry must be provided in validation context. "
                "Use load_benchmark(..., precondition_registry=...) or call "
                "run_benchmark_async."
            )
        registry: AdapterRegistry[type[PreconditionApplier]] = info.context[
            "precondition_registry"
        ]
        parsed: list = []
        for key, payload in self.precondition.items():
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Scenario {self.id!r}: precondition {key!r} payload "
                    f"must be a mapping; got {type(payload).__name__}."
                )
            # Wrap ONLY the registry.get() lookup; see
            # _validate_success_block for the pydantic.ValidationError
            # rationale.
            try:
                cls = registry.get(key)
            except ValueError as exc:
                raise ValueError(
                    f"Scenario {self.id!r} ({self.name!r}): {exc}"
                ) from exc
            parsed.append(cls(**payload))
        self._parsed_preconditions = parsed
        return self


class CapabilitySpec(BaseModel):
    """One capability: a named bundle of scenarios.

    There is no capability-level scorer list; each scenario's
    ``success:`` block declares which success checkers apply to it.
    """

    # Path-safe because the runner composes this into the per-capability
    # experiment name, which becomes a directory name for local runs.
    name: PathSafeIdentifier
    scenarios: list[ScenarioSpec]

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="after")
    def _validate_scenario_ids_unique(self) -> CapabilitySpec:
        """Reject duplicate ``id:`` within one capability.

        The markdown report groups examples by ``scenario_id`` per-capability
        (``report.py:_group_examples_by_scenario_id``). Two scenarios sharing
        ``id:`` collapse into one row with ``scenario_name`` from whichever
        scenario was processed last and aggregate metrics averaged across
        unrelated runs — a silent failure. Cross-capability duplicates
        are fine because the report iterates per-capability.
        """
        seen: dict[int, str] = {}
        for scenario in self.scenarios:
            if scenario.id in seen:
                raise ValueError(
                    f"Capability {self.name!r}: scenario id {scenario.id} "
                    f"is duplicated (used by {seen[scenario.id]!r} and "
                    f"{scenario.name!r}). Within a capability, scenario "
                    f"ids must be unique; cross-capability duplicates are "
                    f"allowed."
                )
            seen[scenario.id] = scenario.name
        return self


class ExecutionBlock(BaseModel):
    """Per-benchmark concurrency and replication knobs."""

    runs_per_scenario: int = Field(default=1, ge=1)
    """Number of times each scenario runs.

    Each scenario contributes ``runs_per_scenario`` runs to the execution
    pool. Higher values produce variance data (pass rate + stddev); lower
    values are faster. Default 1 for fast iteration; raise to 3+ for a
    release benchmark where you care about confidence.
    """

    n_parallel_runs: int = Field(default=1, ge=1)
    """How many runs execute simultaneously within a single scorer-group.

    A scenario expands into ``runs_per_scenario`` runs; all scenarios'
    runs form one flat pool per scorer-group, and the platform adapter is
    expected to keep ``n_parallel_runs`` of them in flight at once — drawn
    across the pool regardless of which scenario each run belongs to. With
    one scenario and ``runs_per_scenario`` > 1, that means multiple runs
    of the *same* scenario can execute concurrently.

    Enforced as a hard cap by the ``local`` platform (via an
    ``asyncio.Semaphore``). Platform adapters that delegate concurrency to
    their own SDK (``braintrust``, ``mlflow``, ``langfuse``) accept the
    value but may not honor it exactly.

    Capabilities run sequentially, and groups within a capability also
    run sequentially. ``n_parallel_runs`` therefore equals the maximum
    runs in flight at any moment overall — it is not multiplied across
    groups.

    Defaults to 1 (serial) because the framework cannot detect whether
    your agent mutates shared state. Raise when:
      - Your agent is stateless (each invocation independent), AND
      - Your LLM setup can tolerate the parallelism.
    """

    model_config = ConfigDict(extra="forbid")


class BenchmarkConfig(BaseModel):
    """Top-level benchmark YAML schema.

    The agent callable is not part of the YAML; it's passed to
    ``run_benchmark_async(yaml, agent=...)`` from the user's entry point.

    The ``platform`` field holds a resolved ``PlatformConfig`` subclass
    instance. Resolution (raw YAML value → name lookup via the injected
    registry → config_class → typed instance) happens in the
    ``_resolve_platform`` before-validator, which reads
    ``platform_registry`` from the Pydantic validation context supplied
    by ``load_benchmark``. Programmatic construction that passes an
    already-typed ``PlatformConfig`` instance skips resolution
    automatically.
    """

    # Path-safe because this name propagates to ``BenchmarkResult.benchmark``,
    # from which the report filename is built.
    benchmark: PathSafeIdentifier
    description: str | None = None
    platform: PlatformConfig
    execution: ExecutionBlock = Field(default_factory=ExecutionBlock)
    capabilities: list[CapabilitySpec]

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _resolve_platform(cls, data: Any, info: ValidationInfo) -> Any:
        """Resolve the raw ``platform:`` value into a typed ``PlatformConfig`` instance.

        Reads the raw value (a string shorthand, a mapping, or ``None``)
        from the incoming data dict and the ``platform_registry`` from
        the validation context, then resolves the named adapter and
        instantiates its ``config_class``.

        Normalises scalar shorthand (``platform: local``) to
        ``{"name": "local"}`` before dispatch. Unknown keys surface as a
        Pydantic ``ValidationError`` because every config class carries
        ``extra="forbid"``.

        Skips resolution when ``data["platform"]`` is already a
        ``PlatformConfig`` instance (programmatic construction path).

        The ``platform_registry`` is required and read from
        ``info.context["platform_registry"]``. ``load_benchmark``
        supplies it; the public ``run_benchmark_async`` is the typical
        production caller.
        """
        if not isinstance(data, dict):
            return data
        if "platform" not in data:
            return data
        platform_raw = data["platform"]
        # Skip resolution for already-typed instances (programmatic construction
        # or re-validation after the before-validator has already run).
        if isinstance(platform_raw, PlatformConfig):
            return data
        if info.context is None or info.context.get("platform_registry") is None:
            raise ValueError(
                "platform_registry must be provided in validation context. "
                "Use load_benchmark(..., platform_registry=...) or call "
                "run_benchmark_async."
            )
        registry: AdapterRegistry[type[Platform]] = info.context["platform_registry"]
        # Normalise scalar shorthand: `platform: local` → `{name: local}`
        if isinstance(platform_raw, str):
            platform_block: dict[str, Any] = {"name": platform_raw}
        elif isinstance(platform_raw, dict):
            platform_block = dict(platform_raw)
        else:
            raise ValueError(
                f"benchmark `platform:` must be a string or mapping; "
                f"got {type(platform_raw).__name__!r}"
            )
        name = platform_block.get("name")
        if not name:
            raise ValueError(
                "benchmark `platform:` block requires a `name:` field naming the "
                "platform adapter (e.g. `name: local`)"
            )
        try:
            adapter_cls = registry.get(name)
        except ValueError as exc:
            raise ValueError(f"benchmark platform block: {exc}") from exc
        config_cls = getattr(adapter_cls, "config_class", None)
        if config_cls is None:
            raise ValueError(
                f"benchmark platform block: adapter {name!r} has no `config_class` "
                "attribute; cannot build typed config"
            )
        # Let Pydantic ValidationError from config_cls(**...) propagate as-is;
        # wrap other exceptions with benchmark context.
        try:
            data["platform"] = config_cls(**platform_block)
        except Exception as exc:
            from pydantic import ValidationError

            if isinstance(exc, ValidationError):
                raise
            raise type(exc)(f"benchmark platform block: {exc}") from exc
        return data
