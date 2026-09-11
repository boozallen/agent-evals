# Platforms

Agent-evals supports multiple evaluation platforms. Switch platforms by changing one parameter.

## Local (Default)

Results saved to local filesystem (`~/.agent-evals/experiments/`). No account or API key needed.

```python
from agent_evals import run_eval_async
from agent_evals.adapters.platforms.local import LocalConfig

result = await run_eval_async(
    task=my_agent,
    dataset=dataset,
    scorers=[...],
    platform=LocalConfig(experiment="sentiment-classifier-v2"),
)
```

`experiment` is optional — if omitted, a timestamped name is generated automatically. `LocalConfig` has no `project` field.

## Braintrust

[Braintrust](https://www.braintrust.dev/) cloud platform includes rich visualization, experiment tracking, and collaboration features. Requires API key.

```bash
export BRAINTRUST_API_KEY=<your-key>
```

```python
from agent_evals import run_eval_async
from agent_evals.adapters.platforms.braintrust import BraintrustConfig

result = await run_eval_async(
    task=my_agent,
    dataset=dataset,
    scorers=[...],
    platform=BraintrustConfig(project="support-agent", experiment="sentiment-classifier-v2"),
)
```

Both `project` and `experiment` are required. Optional: `metadata={...}` for extra key-value tags on the run.

## MLflow

Requires an MLflow tracking server to be running. If setup is needed, check [MlFlow's Quickstart](https://mlflow.org/docs/latest/genai/eval-monitor/quickstart/) guide and setup the local backend, either with docker or running the server via CLI.

The tracking server URL is read from the `MLFLOW_TRACKING_URI` environment variable — it is **not** a config field.

```bash
export MLFLOW_TRACKING_URI=https://mlflow.example.com
```

The value must be an encrypted endpoint, a local store, or a loopback address — see [Transport security](#transport-security). A tracking server on your own machine works as `http://localhost:5000`, and a local store such as `./mlruns` or `sqlite:///mlflow.db` needs no server at all.

```python
from agent_evals import run_eval_async
from agent_evals.adapters.platforms.mlflow import MlflowConfig

result = await run_eval_async(
    task=my_agent,
    dataset=dataset,
    scorers=[...],
    platform=MlflowConfig(experiment="sentiment-classifier-v2"),
)
```

`experiment` is required.

## LangFuse

[LangFuse](https://langfuse.com/) is an open-source LLM observability platform with experiment tracking, tracing, and scoring. Can be self-hosted or used via their cloud offering. Requires API keys.

```bash
export LANGFUSE_PUBLIC_KEY=<your-public-key>
export LANGFUSE_SECRET_KEY=<your-secret-key>
export LANGFUSE_HOST=https://cloud.langfuse.com  # or your self-hosted URL
```

```python
from agent_evals import run_eval_async
from agent_evals.adapters.platforms.langfuse import LangfuseConfig

result = await run_eval_async(
    task=my_agent,
    dataset=dataset,
    scorers=[...],
    platform=LangfuseConfig(experiment="sentiment-classifier-v2", description="Nightly regression run"),
)
```

`experiment` is required and maps to the run name shown in the LangFuse UI. `description` and `metadata={...}` are optional.

### LangFuse `pull_traces` sources

`pull_traces` exports platform history as a dataset you can re-evaluate. LangFuse has two sources, selected explicitly by `config["source"]`:

```python
adapter = LangFusePlatform()

# Default: a curated LangFuse dataset. Items usually carry an expected_output.
dataset = adapter.pull_traces(config={"dataset": "my-dataset"})

# Raw execution traces, narrowed by session, user, and count.
recent = adapter.pull_traces(
    config={"source": "traces", "session_id": "abc-123", "limit": 20},
    filter="metadata.environment = prod",
)
```

The source is **never inferred**. `session_id` was an accepted-and-ignored key before trace selection existed, so a caller passing one today is receiving dataset results; inferring the trace path from its presence would silently reroute them to a different data source on upgrade. Passing both `source: "traces"` and `dataset` raises rather than preferring one.

`config` keys for `source: "traces"`:

| Key | Meaning |
|---|---|
| `session_id` | Restrict to one session |
| `user_id` | Restrict to one user |
| `limit` | Total **most-recent** records to return, across pages |
| `expected_metadata_key` | Metadata key to read an expected value from |

Two things about raw traces differ from a dataset, and both are deliberate:

- **`limit` is a total budget, not a page size** — the opposite of LangFuse's own per-page `limit`. Listing requests are ordered `timestamp.desc`, which is also what gives positional pagination a defined order so a record cannot fall between two pages. The order is single-field, so records sharing an identical timestamp (plausible under batch ingestion) can reorder between requests; narrow the query rather than relying on a boundary if you need exactness.
- **A raw trace carries no answer key**, so examples come back with `expected=None` unless you nominate a metadata field via `expected_metadata_key`. No field is inferred — a metadata key that resembles an answer key may be an unrelated annotation. Comparison scorers report the absence by name (see [scorers.md](scorers.md)) rather than scoring against an empty string.

## Where `pull_traces` filtering is applied

The three adapters that implement `pull_traces` do not all apply filtering in the same place, and the difference is visible in what crosses the network:

| Adapter | Server-side | Applied by this library |
|---|---|---|
| Braintrust | `filter` (embedded in the BTQL query) | — |
| MLflow | `filter` (forwarded to `search_traces`) | — |
| LangFuse | `session_id`, `user_id`, `limit` | `filter` |

LangFuse is the one that differs, and not by choice: its `api.trace.list` accepts a `filter` of its own, but the SDK documents that when supplied it **takes precedence over** the `userId`, `sessionId`, `name`, `tags`, `version`, `release`, `environment`, and timestamp query parameters. Forwarding a clause through it would not merely fail to parse — it would silently discard the session and user selection and return a plausible result set answering a question nobody asked.

Because it is applied client-side, the LangFuse clause form is this library's own small grammar rather than LangFuse's: one or more `<field> <op> <value>` comparisons joined by `AND`, with operators `=`, `!=`, `>`, `>=`, `<`, `<=`. Fields are `id`, `session_id`, `user_id`, `input`, `output`, and `metadata.<key>`. Anything outside that — `OR`, `LIKE`, parentheses, an unknown field — raises a `ValueError` naming the token it could not interpret, so the limitation surfaces at the call rather than as a wrong result set in your numbers. Score thresholds (`scores.Factuality > 0.8`, valid in Braintrust's BTQL) are rejected with that reason: a LangFuse trace listing returns score *IDs*, not values.

A `filter` passed with the LangFuse dataset source also raises. Accepting it and dropping it is the same defect wearing a friendlier face.

## Transport security

Endpoints you supply are checked before anything connects to them. A value that would put prompts, model outputs, or test data on a network unencrypted is rejected with a `ValueError` at construction time, naming the setting and the offending value. If that value embeds a password — as an MLflow database backend store does, `postgresql://<user>:<password>@<host>:5432/<db>` — the password is masked in the message, so an uncaught traceback in a CI log does not carry it.

**What is validated:**

| Setting | Where it comes from |
|---|---|
| `MLFLOW_TRACKING_URI` | Environment variable, or `config["tracking_uri"]` for `pull_traces` |
| `LANGFUSE_HOST` | Environment variable |
| `base_url` | Keyword argument on any autoevals scorer factory |
| `judge_base_url` | Required keyword argument on the agentevals LLM-judge scorer factories |

Braintrust's API endpoint is a fixed `https` address this library controls, so there is nothing to validate and nothing a caller can downgrade.

**What is accepted:**

- Encrypted network schemes: `https://`, `wss://`
- Cleartext `http://` and `ws://` **only when the host is loopback** — `http://localhost:5000`, `http://127.0.0.1:11434`, `http://[::1]:5000`
- Non-network schemes that transmit nothing: `file://`, `sqlite://`
- Local filesystem paths — relative (`./mlruns`, `mlruns`), absolute (`/var/mlruns`), Windows drive paths (`C:\mlruns`, `d:/runs`), and UNC paths (`\\server\share\mlruns`)
- MLflow's `databricks` sentinel, which resolves from ambient Databricks configuration
- An unset or empty value, where the setting's own existing "required" behaviour applies instead

Everything else is rejected. This is an allowlist, so an unrecognised scheme fails rather than being assumed safe. Four consequences worth knowing:

- **Loopback is decided from the value, never resolved.** The name `localhost` is accepted, as is any address in `127.0.0.0/8` or IPv6 `::1`. Any other hostname is rejected even if it happens to resolve to a loopback address today — including `app.localhost`, which RFC 6761 reserves for loopback. Write the address literal instead.
- **`http://` to anything else is still rejected**, however local it looks: `http://mlflow.internal`, `http://10.0.0.5:5000`, and `http://localhost.example.com` all fail.
- **A bare host and port is rejected.** `localhost:5000` names no scheme, so nothing here can tell which transport you meant. Write it out as `http://localhost:5000`.
- **A malformed URL fails loudly.** A typo like `htp://host` is rejected rather than being quietly treated as a directory named `htp:`.

**Confirming the posture from a log.** Each platform adapter that opens a socket records the transport posture once, at `INFO`, when it is constructed, attributed to that adapter's own logger (`agent_evals.adapters.platforms.mlflow`, and so on). The record names the platform in the message body as well as on the logger, so a run that builds two adapters produces two distinguishable lines even under a format string that drops the logger name. `INFO` rather than `DEBUG` so that a run auditing its own logs at a production level still finds the statement.

**There is no opt-out.** No environment variable, argument, or configuration key relaxes this check. The loopback allowance above is not a switch — it is a property of the value, decided the same way on every machine, and it never admits cleartext to a host other than your own. To reach a *remote* server that speaks only cleartext, terminate TLS in front of it; there is no accepted value for it here.

**One boundary this library cannot cover.** If you pass your own `session=` to `MLflowPlatform`, `client=` to `LangFusePlatform`, or `client=` to an autoevals scorer factory, you own that collaborator's transport. This library validates the values it resolves and passes itself; it does not inspect what an injected object connects to. A `base_url=` passed to a scorer factory *is* validated — a `base_url` you set on your own SDK client before injecting it is not.
