# New Adapter Development Guide

This guide provides detailed instructions for developing new platform and scorer adapters for the agent-evals library. Each adapter type serves a distinct purpose in the evaluation pipeline and follows specific protocols to ensure seamless integration.

## Table of Contents

- [Overview](#overview)
- [Platform Adapters](#platform-adapters)
  - [Platform Adapter Port Contract](#platform-adapter-port-contract)
  - [Platform Implementation Guide](#platform-implementation-guide)
  - [Platform Reference Implementations](#platform-reference-implementations)
- [Scorer Adapters](#scorer-adapters)
  - [Scorer Adapter Port Contract](#scorer-adapter-port-contract)
  - [Scorer Implementation Guide](#scorer-implementation-guide)
  - [Scorer Reference Implementations](#scorer-reference-implementations)
- [Testing Your Adapters](#testing-your-adapters)

## Overview

The agent-evals library uses **hexagonal architecture** (ports and adapters) to enable extensible evaluation platforms and scoring systems. There are two types of adapters:

- **Platform Adapters**: Control WHERE and HOW evaluations run (Braintrust, MLflow, LangFuse, Local)
- **Scorer Adapters**: Provide Scorers for Evaluating Agent Outputs (AgentEvals, Autoevals, Custom)

Both adapter types conform to the port contracts, acting as hexagonal adapters that bridge the evaluation runner to external platforms and scoring libraries. The contract is defined using a Protocol that establishes the required signature and methods for each adapter type.

## Platform Adapters

Platform adapters orchestrate evaluation execution and integrate with external evaluation platforms (e.g.  Braintrust, MLflow)  or handle local execution. They control the evaluation strategy and convert between unified types and platform-specific formats.

### Platform Adapter Port Contract

Platform adapters must implement the contract defined by the `Platform` Protocol below, see [src/agent_evals/core/ports/platform.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/core/ports/platform.py) for additional details:

```python
class Platform(Protocol):
    def evaluate(
        self,
        task: Callable,
        dataset: list[ExampleData],
        evaluators: list[Callable],
        config: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Synchronous evaluation (backward compatibility)."""
        ...

    async def aevaluate(
        self,
        task: Callable | None,
        dataset: list[ExampleData],
        evaluators: list[Callable],
        config: dict[str, Any] | None = None,
    ) -> EvalResult:
        """Asynchronous evaluation (primary interface)."""
        ...

    def pull_traces(
        self,
        config: dict[str, Any] | None = None,
        filter: str | None = None,
    ) -> list[ExampleData]:
        """Extract historical evaluation data from platform."""
        ...

    # Private conversion methods
    def _convert_to_platform_scorer(self, evaluators: list[Callable]) -> list[Callable[..., Any]]:
        """Convert Agent Evals scorers to platform-specific format."""
        ...

    def _convert_to_platform_dataset(self, dataset: list[ExampleData]) -> Any:
        """Convert Agent Evals dataset to platform-specific format."""
        ...

    def _convert_from_platform_result(self, platform_result: Any) -> EvalResult:
        """Convert platform result back to unified EvalResult."""
        ...
```

### Platform Implementation Guide

#### 1. Adapter Discovery Mechanism

The agent-evals framework uses a registration-based discovery system for adapters:

```python
# In your adapter file (e.g., [src/agent_evals/adapters/platforms/my_platform.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/adapters/platforms/my_platform.py))
from agent_evals.core._registries import platform_registry

@platform_registry.register("my_platform")  # This name will be used in config
class MyPlatform:
    # ... implementation
```

**How Registration Works:**
1. The `@platform_registry.register()` decorator adds your platform to a global registry
2. When users pass a typed `platform=MyPlatformConfig(...)` object, the framework resolves the adapter by `config.name` at the composition root
3. The platform is instantiated and used for evaluation

**Define your config class first** — every platform adapter must have a companion `PlatformConfig` subclass:

```python
from typing import ClassVar, Literal
from agent_evals.core.types import PlatformConfig

class MyPlatformConfig(PlatformConfig):
    """Config for MyPlatform adapter. Users pass platform=MyPlatformConfig(...)."""
    name: Literal["my_platform"] = "my_platform"
    experiment: str
    # add any other platform-specific fields here
    # unknown fields raise at load time (extra="forbid" inherited from PlatformConfig)

@register_adapter("my_platform")
class MyPlatformAdapter:
    config_class: ClassVar[type[MyPlatformConfig]] = MyPlatformConfig
    # ... rest of implementation
```

Users then call:

```python
from agent_evals.adapters.platforms.my_platform import MyPlatformConfig
result = await run_eval_async(task, dataset, scorers, platform=MyPlatformConfig(experiment="nightly-regression"))
```

##### Making Your Platform Adapter Available

1. **Create adapter file**: [src/agent_evals/adapters/platforms/my_platform.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/adapters/platforms/my_platform.py)
2. **Add to package**: Include in [src/agent_evals/adapters/platforms/__init__.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/adapters/platforms/__init__.py)
3. **Update setup**: Add optional dependencies to [pyproject.toml](https://github.com/boozallen/agent-evals/blob/main/pyproject.toml)
4. **Update docs**: Add to platform list in README.md
5. **Handle optional dependencies gracefully**: Include proper import error handling at the top of your adapter file:
   ```python
   try:
       import my_platform_sdk
   except ImportError as e:
       raise ImportError(
           "My Platform adapter requires the my_platform_sdk package. "
           "Install it with: pip install my_platform_sdk"
       ) from e
   ```

#### 2. Public Evaluation Methods

Platform adapters must implement two evaluation methods that handle different execution contexts and requirements.

##### The evaluate() Method - Synchronous Evaluation

The `evaluate()` method provides backward compatibility for synchronous evaluation workflows. This method is typically used in simple scripts or environments where async/await syntax is not practical.

**Implementation Pattern:** Almost all platform adapters implement `evaluate()` as a simple wrapper that calls `asyncio.run()` on the async `aevaluate()` method. This avoids code duplication while maintaining both sync and async interfaces.

##### The aevaluate() Method - Asynchronous Evaluation (Primary Interface)

The `aevaluate()` method is the primary evaluation interface and handles the core evaluation orchestration. This method can be used for both live evaluation (with a task function) and historical data scoring (task=None).

**Key Implementation Flow:**

1. **Configuration Validation** via a platform-specific helper method that:
   - Validates required config fields (API keys, project names, tracking URIs)
   - Sets up platform clients or connections
   - Normalizes configuration values for consistency

2. **Data Format Conversion** using the required private methods:
   - `_convert_to_platform_dataset()` transforms Agent Evals `ExampleData` to platform format
   - `_convert_to_platform_scorer()` wraps Agent Evals scorers to match platform expectations

3. **Platform Evaluation Execution** which varies significantly by platform:
   - **SDK Delegation** (Braintrust, MLflow, LangFuse): Call platform's evaluation API
   - **Internal Orchestration** (Local): Execute task and scorers directly within the adapter

   **Task-call contract:** Your adapter MUST invoke the user task as `task(input_value)` with the bare `ExampleData.input` value. If the platform SDK delivers items in a wrapper shape (e.g., langfuse passes `{"input": ..., "expected_output": ..., "metadata": ...}`), unwrap the SDK shape inside the adapter — typically inside a wrapper function passed into the SDK — before calling the user's task. Never push the SDK shape through to the user's signature; that breaks platform portability and fails `tests/validation/test_platform_contract.py`. See `LangFusePlatform._make_safe_task` for the canonical pattern. The contract lives on the `Platform` Protocol docstring as the single source of truth.

4. **Result Conversion** via `_convert_from_platform_result()` to transform platform-specific results into unified `EvalResult` objects

**Examples of Platform-Specific Considerations for Evaluation Execution:**

- **Braintrust**: Uses `braintrust.Eval()` with converted data and scorers
- **MLflow**: Uses `mlflow.genai.evaluate()` with decorated scorers
- **LangFuse**: Uses `client.run_experiment()` with evaluator objects
- **Local**: Implements complete evaluation loop internally (task execution, scorer runs, aggregation)

Each platform has different requirements for scorer signatures, data formats, and result structures that must be handled in the conversion methods.

#### 3. Public Historical Data Method

The `pull_traces()` method enables scoring past workflows by retrieving historical evaluation data from the platform and converting it to Agent Evals format. This allows users to apply new scorers to previously executed evaluations without re-running the original tasks.

**Implementation Flow:**

1. **Configuration Validation** for trace export, which typically requires:
   - Platform identifiers (project IDs, experiment names, dataset names)
   - Authentication credentials (same as evaluation methods)
   - Optional filtering parameters (time ranges, score thresholds)

2. **Platform Trace Retrieval** using platform-specific data export mechanisms:
   - **Braintrust**: 'Exports traces from Braintrust REST API via BTQL query'
   - **MLflow**: Uses `mlflow.search_traces()` to query experiment traces
   - **LangFuse**: Uses `client.get_dataset()` and trace API to fetch historical runs

3. **Trace Format Conversion** using `_convert_traces()` transform platform-specific trace data into `ExampleData` objects:
   - Extract original inputs from trace records
   - Extract task outputs from trace execution history
   - Extract expected results from original evaluation setup
   - Preserve metadata for context and debugging

**Examples of Platform-Specific Data Access Patterns:**

Each platform provides different mechanisms for accessing historical data:

- **Braintrust**: Exports traces from Braintrust REST API via BTQL query
- **MLflow**: Query traces by experiment ID, returns pandas DataFrame with structured trace data
- **LangFuse**: Access dataset items and linked traces through REST API

The filter parameter enables selective data retrieval based on platform-specific query capabilities (score ranges, time periods, metadata tags).

#### 4. Private Conversion Methods

Platform adapters must implement four private methods that handle data conversion between Agent Evals unified types and platform-specific formats. These methods are called by the public evaluation methods to ensure seamless integration.

##### _convert_to_platform_scorer() - Scorer Format Adaptation

This method wraps Agent Evals scorer functions to match the platform's expected scorer interface. Each platform has different requirements for scorer signatures and return values.

**Examples of Platform-Specific Scorer Adaptations:**
- **Braintrust**: Scorers must return float values (0.0-1.0)
- **MLflow**: Scorers must return `Feedback` objects with specific metadata structure
- **LangFuse**: Scorers must return `Evaluation` objects with score and comment fields
- **Local**: Scorers can return Score objects directly (no conversion needed)

##### _convert_to_platform_dataset() - Dataset Format Transformation

This method transforms Agent Evals `ExampleData` objects into the format expected by the platform's evaluation APIs.

**Examples of Platform-Specific Dataset Formats:**
- **Braintrust**: List of dicts with `input`, `expected`, `metadata` keys
- **MLflow**: List of dicts formatted for `mlflow.genai.evaluate()` data parameter
- **LangFuse**: List of dicts matching LangFuse experiment data schema
- **Local**: Preserves Agent Evals format for internal processing

##### _convert_from_platform_result() - Result Unification

This method transforms platform-specific evaluation results back into unified `EvalResult` objects that provide consistent interfaces across all platforms.

**Examples of Platform-Specific Result Parsing:**
- **Braintrust**: Extract from `braintrust.AsyncEval` result object with `.results` attribute
- **MLflow**: Query traces and assessments using MLflow tracking APIs
- **LangFuse**: Parse `ExperimentResult` object with run statistics

Each conversion method requires platform-specific parsing logic to handle the unique data structures and APIs of different evaluation platforms.

##### _convert_traces() - Historical Data Tranformation

This method transforms the traces pulled by the platform's unique trace data structure to 'ExampleData' objects for use as evaluation test cases when working with historical data.

**Examples of Platform-Specific Result Parsing:**
- **Braintrust**: Extract from JSON result objects returned from API
- **MLflow**: Retrieve values from Pandas DataFrame returned by `search_traces()`
- **LangFuse**: Parse `Trace` object returned by `langfuse.api.trace.get()`

#### 5. Other Adapter-Specific Helper Methods

Beyond the required Protocol methods, adapters may need additional helper methods to handle platform-specific operations. These helper methods are referenced in the public methods but must be implemented according to each platform's unique requirements and live within the adapter.

**Common Helper Methods:**

```python
def _validate_and_setup_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
    """Validate configuration and set up platform client.

    This method should validate both config parameters and required environment variables.
    Document required environment variables clearly in your adapter's docstring:

    Environment Variables:
        MY_PLATFORM_API_KEY: API key for My Platform (required)
        MY_PLATFORM_BASE_URL: Custom base URL (optional, defaults to production)
        MY_PLATFORM_TIMEOUT: Request timeout in seconds (optional, defaults to 30)

    Example:
        export MY_PLATFORM_API_KEY=your_api_key_here
        export MY_PLATFORM_TIMEOUT=60
    """
    if not config:
        raise ValueError("Configuration required")

    # Platform-specific required fields (examples):
    # Braintrust: ["project", "experiment"]
    # MLflow: ["experiment"]  (tracking_uri comes from MLFLOW_TRACKING_URI env var)
    # LangFuse: ["experiment"]
    required_fields = ["experiment"]  # Customize per platform

    for field in required_fields:
        if not getattr(config, field, None):
            raise ValueError(f"Missing required config field: {field}")

    # Platform-specific client setup (implement per platform):
    # - Braintrust: Validate BRAINTRUST_API_KEY environment variable
    # - MLflow: Read os.environ["MLFLOW_TRACKING_URI"], call mlflow.set_tracking_uri() and mlflow.set_experiment()
    # - LangFuse: Initialize client with get_client() and auth_check()
    platform_client_initializer(config)

    return config

def _report_score(self, evaluator: Callable, score: Score) -> Any:
    """Send one score to the platform SDK."""
    # Don't hand-roll name derivation — call the shared helper. A scorer built
    # by a factory is a closure literally named `scorer`, so `__name__` alone
    # collapses every such scorer onto one name.
    name = infer_scorer_name(evaluator)
    ...
```

`infer_scorer_name` lives in `adapters/platforms/utils.py` and is shared by every adapter that needs it — do not re-implement it.

**If your adapter caches per-scorer state by name** (a declared threshold, for
example), key the cache by the name the score is *reported to the platform
under*, not by the name `infer_scorer_name` derives from the callable. Those are
two independent derivations and nothing forces them to agree: the reconstruction
path only sees what the SDK hands back. Getting this wrong is silent — the
lookup misses and falls through to a default. See `CHANGELOG.md` for background on this exact failure mode.

Each platform adapter will require different combinations of these helpers and possibly others, dependent on the platform's API design and capabilities. Refer to the existing adapter implementations for concrete examples of platform-specific helper methods.

#### 6. Validating Caller-Supplied Endpoints

If your adapter resolves an endpoint itself — from a config key, an environment
variable, or a constructor argument — validate it before handing it to the SDK:

```python
from agent_evals.core.transport import validate_secure_transport

tracking_uri = os.getenv("MY_PLATFORM_HOST", "")
validate_secure_transport("MY_PLATFORM_HOST", tracking_uri)
```

`validate_secure_transport` is an allowlist: encrypted schemes (`https`, `wss`),
schemes that transmit nothing (`file`, `sqlite`), local filesystem paths,
cleartext only to a loopback host, and an unset value. Anything else raises
`ValueError`. There is no opt-out, and the function performs no I/O.

Three rules for placement:

- **Validate where the value is resolved, not where it is used.** Both entry
  points into `MlflowPlatform` (the environment variable and the config dict)
  converge on one site in the adapter. A check inside the session wrapper would
  be skippable, because a caller can inject their own session.
- **Put the call before any `try` block that re-raises.** `LangfusePlatform`
  wraps SDK construction in a handler that reports everything as a credentials
  error; a `ValueError` raised inside it would send an operator hunting for a bad
  API key.
- **An endpoint reached through an injected collaborator is out of scope.** If
  the caller builds their own client and hands it to you, its endpoint is theirs
  to validate. Say so in the adapter's docstring.

A hardcoded `https` constant needs no call — see `BTQL_ENDPOINT` in the
braintrust adapter.

If your adapter opens sockets, also call
`log_transport_posture(logger, "my_platform")` from
`adapters/platforms/transport_posture.py` at the end of `__init__`, so one
`INFO` record records the resolved posture. Pass your registered platform name
as the second argument: it goes into the message body, so the line identifies
itself even under a log format that drops the logger name. Adapters that touch
no network (`LocalPlatform`) do not call it at all.

### Platform Reference Implementations

Study these implementations for complete examples:

#### Braintrust Adapter ([src/agent_evals/adapters/platforms/braintrust.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/adapters/platforms/braintrust.py))
- **Pattern**: SDK delegation to `braintrust.AsyncEval()`
- **Key Features**: API key validation, auto scorer conversion, result parsing
- **Study Focus**: Clean SDK integration, scorer wrapping patterns

#### MLflow Adapter ([src/agent_evals/adapters/platforms/mlflow.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/adapters/platforms/mlflow.py))
- **Pattern**: SDK delegation to `mlflow.genai.evaluate()`
- **Key Features**: Tracking URI setup, scorer decoration, trace export
- **Study Focus**: Dynamic scorer decoration, pandas trace handling

#### LangFuse Adapter ([src/agent_evals/adapters/platforms/langfuse.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/adapters/platforms/langfuse.py))
- **Pattern**: SDK delegation to `langfuse.run_experiment()`
- **Key Features**: Client initialization, evaluation objects, trace queries
- **Study Focus**: Async evaluation, historical data handling

#### Local Adapter ([src/agent_evals/adapters/platforms/local.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/adapters/platforms/local.py))
- **Pattern**: Internal orchestration with file storage
- **Key Features**: Complete evaluation pipeline, JSONL storage, concurrent execution
- **Study Focus**: Task execution, internal orchestration, result aggregation

## Scorer Adapters

Scorer adapters are hexagonal adapters that bridge the core evaluation runner to external scoring libraries. Scorer adapters wrap external scoring libraries to return unified `Score` objects and conform their callables to the scorer port contract.

### Scorer Adapter Port Contract

Scorer adapters must implement the contract defined by the `Scorer` Protocol below, see [src/agent_evals/core/ports/scorer.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/core/ports/scorer.py) for additional details:

```python
from collections.abc import Awaitable
from typing import Protocol, runtime_checkable
from agent_evals.core.types import TaskResult, ExpectedResult, Score

@runtime_checkable
class Scorer(Protocol):
    def __call__(
        self,
        result: TaskResult,
        expected: ExpectedResult | None = None,
    ) -> Score | Awaitable[Score]:
        """Score the task result.

        Args:
            result: TaskResult with .output (str) and .context (dict)
            expected: Optional ExpectedResult with .expected and .context

        Returns:
            Score object with evaluation results
        """
        ...
```

### Scorer Implementation Guide

#### 1. Simple Custom Function Scorers

For writing simple custom scoring logic, implement as a function:

```python
from agent_evals.core.types import TaskResult, ExpectedResult, Score

def my_simple_scorer(
    result: TaskResult,
    expected: ExpectedResult | None = None
) -> Score:
    """Simple example scorer."""
    output_text = result.output
    context = result.context or {}

    # Scoring logic
    score_value = len(output_text) / 100.0  # Example: length-based scoring
    passed = score_value >= 0.5

    return Score(
        name="length_scorer",
        value=score_value,
        passed=passed,
        metadata={"output_length": len(output_text)},
        reasoning=f"Output length: {len(output_text)} characters"
    )
```

#### 2. Configurable Scorers

For scorers that need configuration, use a factory function returning a
closure. Every scorer shipped in `adapters/scorers/` uses this pattern:

```python
from collections.abc import Callable

from agent_evals.core.types import ExpectedResult, Score, TaskResult


def LengthScorer(threshold: float = 0.5, name: str = "length") -> Callable:
    """Factory returning a scorer configured with a threshold."""

    async def scorer(
        result: TaskResult, expected: ExpectedResult | None = None, **context
    ) -> Score:
        value = len(result.output) / 100.0
        return Score(
            name=name,
            value=value,
            passed=value >= threshold,
            metadata={"threshold": threshold},
            reasoning=f"Length ratio {value:.2f} vs threshold {threshold}",
        )

    return scorer


# Usage: call the factory, pass the result to run_eval_async
scorers = [LengthScorer(threshold=0.8)]
```

The closure captures configuration, and `scorer` may be `def` or `async def` —
use `async def` when the scorer awaits I/O, such as an LLM judge. The closure
also holds anything the scorer needs to reuse across calls (a client to keep
open, a cache to keep warm), so configuration and state both live in the
factory's scope.

Any callable satisfies the `Scorer` Protocol, so a class with `__call__` works
too, but prefer the factory above: it is the pattern the shipped scorers and
examples use, and it keeps registration via
`@scorer_registry.register("Name")` uniform.

#### 3. Context-Based Scoring

Most scorers that need the output of an agent will access the output field of TaskResult. For scorers that need more complex data, access is provided through the context parameter:

```python
def trajectory_scorer(
    result: TaskResult,
    expected: ExpectedResult | None = None
) -> Score:
    """Score based on agent trajectory."""
    context = result.context or {}

    # Extract trajectory data
    trajectory = context.get('trajectory', [])
    tool_calls = context.get('tool_calls', [])

    if not trajectory:
        return Score(
            name="trajectory_efficiency",
            value=0.0,
            passed=False,
            reasoning="No trajectory data available"
        )

    # Score based on trajectory length
    step_count = len(trajectory)
    efficiency = max(0.0, 1.0 - (step_count - 5) * 0.1)

    return Score(
        name="trajectory_efficiency",
        value=efficiency,
        passed=step_count <= 10,
        metadata={
            "step_count": step_count,
            "tool_call_count": len(tool_calls)
        },
        reasoning=f"Agent completed task in {step_count} steps"
    )
```

#### 4. External Library Wrappers

When integrating existing scoring libraries, you need to create wrapper functions that convert between the library's format and Agent Evals' `Score` objects. This pattern involves:

- **Input Conversion**: Transform `TaskResult` and `ExpectedResult` into the format expected by the external library
- **Library Invocation**: Call the external scorer with converted inputs
- **Output Conversion**: Convert the library's result back to a `Score` object
- **Error Handling**: Gracefully handle failures and return appropriate error scores
- **Endpoint Validation**: If the wrapper takes an endpoint for the external
  library to call, validate it — see below

#### 5. Validating Caller-Supplied Endpoints

A scorer factory that accepts an endpoint must validate it before passing it on,
the same way platform adapters do:

```python
from agent_evals.core.transport import validate_secure_transport

@scorer_registry.register("MyJudge")
def MyJudge(*, judge_base_url: str, **kwargs) -> Scorer:
    validate_secure_transport("judge_base_url", judge_base_url)
    ...
```

Two things specific to scorers:

- **Validate in every factory, not only in a shared kwargs helper.** Three of the
  twelve autoevals factories build their arguments directly and never reach the
  shared helper, so a single call there would have left them uncovered.
  `tests/validation/test_transport_enforcement.py` enumerates the scorer registry
  and fails the build if a factory declaring `base_url` does not reject cleartext.
- **Take a URL, not a provider nickname.** A string like `"ollama:gpt-oss:20b"`
  is resolved to an endpoint downstream, where nothing can inspect it — there is
  no value in hand to validate. Require the endpoint as its own parameter and
  bind it to the provider yourself. That guard test cannot see a parameter named
  anything other than `base_url`, so a differently-named one is on you.

Scorers do **not** log transport posture; that happens once per platform-adapter
construction.

### Scorer Reference Implementations

Study these implementations for complete examples:

#### Autoevals Adapters ([src/agent_evals/adapters/scorers/autoevals.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/adapters/scorers/autoevals.py))
- **Pattern**: Wrapper functions around autoevals library
- **Key Features**: Async LLM scorers, sync heuristic scorers, threshold handling
- **Study Focus**: Library wrapping patterns, async/sync detection, error handling

#### AgentEvals Adapters ([src/agent_evals/adapters/scorers/agentevals.py](https://github.com/boozallen/agent-evals/blob/main/src/agent_evals/adapters/scorers/agentevals.py))
- **Pattern**: Trajectory evaluation wrappers
- **Key Features**: Context-based scoring, trajectory comparison, LLM judges
- **Study Focus**: Context extraction, trajectory handling, async LLM evaluation

## Testing Your Adapters

Create comprehensive tests for your platform adapters. For reference implementations of test files for specific Platform Adapters, see:

- [Braintrust Platform Tests](https://github.com/boozallen/agent-evals/blob/main/tests/integration/test_braintrust_platform_live.py)
- [MLflow Platform Tests](https://github.com/boozallen/agent-evals/blob/main/tests/integration/test_mlflow_platform_live.py)
- [LangFuse Platform Tests](https://github.com/boozallen/agent-evals/blob/main/tests/integration/test_langfuse_platform.py)
- [Local Platform Tests](https://github.com/boozallen/agent-evals/blob/main/tests/integration/test_local_platform.py)
- [Platform Async Methods Tests](https://github.com/boozallen/agent-evals/blob/main/tests/integration/test_platform_async_methods.py)

For reference implementations of test files for Scorer Adapters, see:

- [Core Scorer Tests](https://github.com/boozallen/agent-evals/blob/main/tests/unit/scorers/test_scorer_registry.py)
- [Scorer Failure Isolation Tests](https://github.com/boozallen/agent-evals/blob/main/tests/unit/test_scorer_failure_isolation.py)
- [Autoevals Wrapper Tests](https://github.com/boozallen/agent-evals/blob/main/tests/integration/test_autoevals_wrappers.py)
- [AgentEvals Wrapper Tests](https://github.com/boozallen/agent-evals/blob/main/tests/integration/test_agentevals_wrappers.py)
- [Mixed Scorers Tests](https://github.com/boozallen/agent-evals/blob/main/tests/integration/test_mixed_scorers.py)
- [Autoevals Contract Tests](https://github.com/boozallen/agent-evals/blob/main/tests/validation/test_autoevals_contract.py)
- [AgentEvals Contract Tests](https://github.com/boozallen/agent-evals/blob/main/tests/validation/test_agentevals_contract.py)
