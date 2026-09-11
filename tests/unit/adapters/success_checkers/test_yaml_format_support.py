# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Lock the YAML format flexibility we rely on.

Both success checkers advertise "block or flow style works" in the
docs. That flexibility comes for free from yaml.safe_load — block and
flow are equivalent syntaxes for the same compound data — but it's a
contract worth testing explicitly so a future parser swap or schema
change doesn't silently change which forms parse.

Each test loads a benchmark YAML through the full pipeline (the same
``load_benchmark`` used by ``run_benchmark_async``) and asserts the
parsed success checker is identical regardless of which YAML style was
used.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from agent_evals.benchmark.loader import compile_capability, load_benchmark
from agent_evals.core.checks import StateCheck, ToolCall, ToolUseCheck


def _load(tmp_path: Path, yaml_text: str, default_benchmark_registries: dict):
    """Write `yaml_text` (already-dedented) to a temp file and load it."""
    p = tmp_path / "bench.yaml"
    p.write_text(yaml_text)
    return load_benchmark(p, **default_benchmark_registries)


EXPECTED_CALLS = [{"name": "lock_door", "args": {"door": "front", "locked": True}}]
EXPECTED_STATE = {"doors.front": "locked"}


def _assert_canonical(config) -> None:
    """The single scenario must yield canonical state + tool_use specs."""
    scenario = config.capabilities[0].scenarios[0]
    by_class = {type(c).__name__: c for c in scenario.parsed_check_specs}
    assert set(by_class) == {"StateCheck", "ToolUseCheck"}

    assert isinstance(by_class["StateCheck"], StateCheck)
    assert by_class["StateCheck"].expected_state == EXPECTED_STATE

    tcc = by_class["ToolUseCheck"]
    assert isinstance(tcc, ToolUseCheck)
    expected_typed = (
        ToolCall(name="lock_door", args={"door": "front", "locked": True}),
    )
    assert tcc.calls == expected_typed
    assert tcc.match_args == "exact"
    assert tcc.match_calls == "exact"


# ---------------------------------------------------------------------------
# 1. Block-style — the canonical form documented in docs/benchmarks.md.
# ---------------------------------------------------------------------------


def test_block_style_yaml_parses_to_canonical(
    tmp_path: Path, default_benchmark_registries: dict
) -> None:
    yaml_text = textwrap.dedent("""\
        benchmark: format-v1
        platform: local
        capabilities:
          - name: doors
            scenarios:
              - id: 1
                name: lock front
                prompt: Lock the front door
                success:
                  tool_use:
                    calls:
                      - name: lock_door
                        args:
                          door: front
                          locked: true
                  state:
                    doors.front: locked
    """)
    _assert_canonical(_load(tmp_path, yaml_text, default_benchmark_registries))


# ---------------------------------------------------------------------------
# 2. Flow-style — single-line {} mappings and [] sequences. YAML grammar
#    treats this as equivalent to the block form; we lock the contract.
# ---------------------------------------------------------------------------


def test_flow_style_yaml_parses_identically(
    tmp_path: Path, default_benchmark_registries: dict
) -> None:
    yaml_text = textwrap.dedent("""\
        benchmark: format-v1
        platform: local
        capabilities:
          - name: doors
            scenarios:
              - id: 1
                name: lock front
                prompt: Lock the front door
                success:
                  tool_use:
                    calls: [ { name: lock_door, args: { door: front, locked: true } } ]
                  state: { doors.front: locked }
    """)
    _assert_canonical(_load(tmp_path, yaml_text, default_benchmark_registries))


# ---------------------------------------------------------------------------
# 3. Mixed — block list of calls but flow-style args. The most common
#    real-world shape (block where structure helps, flow where it doesn't).
# ---------------------------------------------------------------------------


def test_mixed_block_calls_with_flow_args(
    tmp_path: Path, default_benchmark_registries: dict
) -> None:
    yaml_text = textwrap.dedent("""\
        benchmark: format-v1
        platform: local
        capabilities:
          - name: doors
            scenarios:
              - id: 1
                name: lock front
                prompt: Lock the front door
                success:
                  tool_use:
                    calls:
                      - name: lock_door
                        args: { door: front, locked: true }
                  state:
                    doors.front: locked
    """)
    _assert_canonical(_load(tmp_path, yaml_text, default_benchmark_registries))


# ---------------------------------------------------------------------------
# 4. Quoting: bare ``on`` is YAML's boolean True (the "Norway problem");
#    quoted ``"on"`` stays a string. State values that look like booleans
#    must be quoted, which we tell users in docs/benchmarks.md.
# ---------------------------------------------------------------------------


def test_quoted_on_off_state_values_stay_strings(
    tmp_path: Path, default_benchmark_registries: dict
) -> None:
    """Quoted ``"on"`` / ``"off"`` are preserved as strings, not coerced to bools."""
    yaml_text = textwrap.dedent("""\
        benchmark: format-v1
        platform: local
        capabilities:
          - name: lights
            scenarios:
              - id: 1
                name: kitchen on
                prompt: Turn on the kitchen light
                success:
                  state:
                    lights.kitchen.state: "on"
                    lights.bedroom.state: "off"
    """)
    config = _load(tmp_path, yaml_text, default_benchmark_registries)
    state_crit = next(
        c
        for c in config.capabilities[0].scenarios[0].parsed_check_specs
        if isinstance(c, StateCheck)
    )
    assert state_crit.expected_state == {
        "lights.kitchen.state": "on",
        "lights.bedroom.state": "off",
    }
    # Defensive: explicitly verify the values are strings, not booleans.
    assert state_crit.expected_state["lights.kitchen.state"] is not True


def test_unquoted_on_demonstrates_yaml_norway_trap(
    tmp_path: Path, default_benchmark_registries: dict
) -> None:
    """Bare ``on`` parses as True — locks the YAML behavior our docs warn about.

    This test isn't asserting "the library handles this correctly"; it
    documents the YAML parser behavior. If this ever started failing
    (i.e., bare ``on`` started parsing as the string ``"on"``), we'd
    have to rewrite the YAML quoting paragraph in docs/benchmarks.md.
    """
    yaml_text = textwrap.dedent("""\
        benchmark: format-v1
        platform: local
        capabilities:
          - name: lights
            scenarios:
              - id: 1
                name: kitchen on
                prompt: Turn on the kitchen light
                success:
                  state:
                    lights.kitchen.state: on
    """)
    config = _load(tmp_path, yaml_text, default_benchmark_registries)
    state_crit = next(
        c
        for c in config.capabilities[0].scenarios[0].parsed_check_specs
        if isinstance(c, StateCheck)
    )
    assert state_crit.expected_state["lights.kitchen.state"] is True


# ---------------------------------------------------------------------------
# 5. ``dimensions:`` is a per-scenario dict whose keys land in
#    ``ExampleData.metadata`` and become "By <key>" sections in the report.
#    The block/flow equivalence is the same yaml.safe_load freebie as the
#    success-block tests above; locking it keeps a future schema or parser
#    swap from silently changing which forms parse.
# ---------------------------------------------------------------------------


def _example_metadata(config) -> dict:
    """Compile the single scenario and return the example's metadata dict."""
    [(dataset, _scorers)] = compile_capability(config.capabilities[0], config.execution)
    return dataset[0].metadata


def test_block_style_dimensions_landed_in_metadata(
    tmp_path: Path, default_benchmark_registries: dict
) -> None:
    yaml_text = textwrap.dedent("""\
        benchmark: dimensions-v1
        platform: local
        capabilities:
          - name: lights
            scenarios:
              - id: 1
                name: kitchen on
                prompt: Turn on the kitchen light
                dimensions:
                  phrasing: imperative
                  depth: literal
                success:
                  state:
                    lights.kitchen.state: "on"
    """)
    metadata = _example_metadata(
        _load(tmp_path, yaml_text, default_benchmark_registries)
    )
    assert metadata["phrasing"] == "imperative"
    assert metadata["depth"] == "literal"
    assert metadata["capability"] == "lights"


def test_flow_style_dimensions_landed_in_metadata(
    tmp_path: Path, default_benchmark_registries: dict
) -> None:
    yaml_text = textwrap.dedent("""\
        benchmark: dimensions-v1
        platform: local
        capabilities:
          - name: lights
            scenarios:
              - id: 1
                name: kitchen on
                prompt: Turn on the kitchen light
                dimensions: { phrasing: imperative, depth: literal }
                success:
                  state: { lights.kitchen.state: "on" }
    """)
    metadata = _example_metadata(
        _load(tmp_path, yaml_text, default_benchmark_registries)
    )
    assert metadata["phrasing"] == "imperative"
    assert metadata["depth"] == "literal"
    assert metadata["capability"] == "lights"


# ---------------------------------------------------------------------------
# 6. Both strictness knobs round-trip through both YAML styles, and the
#    legacy `match:` alias keeps writing into `match_args`.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["exact", "subset", "ignore"])
def test_match_args_roundtrips_through_yaml(
    tmp_path: Path, mode: str, default_benchmark_registries: dict
) -> None:
    yaml_text = textwrap.dedent(f"""\
        benchmark: match-v1
        platform: local
        capabilities:
          - name: doors
            scenarios:
              - id: 1
                name: lock front
                prompt: Lock the front door
                success:
                  tool_use:
                    match_args: {mode}
                    calls:
                      - name: lock_door
                        args: {{ door: front }}
    """)
    config = _load(tmp_path, yaml_text, default_benchmark_registries)
    tcc = next(
        c
        for c in config.capabilities[0].scenarios[0].parsed_check_specs
        if isinstance(c, ToolUseCheck)
    )
    assert tcc.match_args == mode


def test_legacy_match_field_in_yaml_is_rejected(
    tmp_path: Path, default_benchmark_registries: dict
) -> None:
    """Old YAML using `match:` no longer loads; `extra="forbid"` rejects it."""
    yaml_text = textwrap.dedent("""\
        benchmark: legacy-v1
        platform: local
        capabilities:
          - name: doors
            scenarios:
              - id: 1
                name: lock front
                prompt: Lock the front door
                success:
                  tool_use:
                    match: subset
                    calls:
                      - name: lock_door
                        args: { door: front }
    """)
    p = tmp_path / "bench.yaml"
    p.write_text(yaml_text)
    with pytest.raises(ValueError):  # noqa: PT011
        load_benchmark(p, **default_benchmark_registries)


@pytest.mark.parametrize("mode", ["exact", "subset", "superset", "unordered"])
def test_match_calls_roundtrips_through_yaml(
    tmp_path: Path, mode: str, default_benchmark_registries: dict
) -> None:
    yaml_text = textwrap.dedent(f"""\
        benchmark: match-v1
        platform: local
        capabilities:
          - name: doors
            scenarios:
              - id: 1
                name: lock front
                prompt: Lock the front door
                success:
                  tool_use:
                    match_calls: {mode}
                    calls:
                      - name: lock_door
                        args: {{ door: front }}
    """)
    config = _load(tmp_path, yaml_text, default_benchmark_registries)
    tcc = next(
        c
        for c in config.capabilities[0].scenarios[0].parsed_check_specs
        if isinstance(c, ToolUseCheck)
    )
    assert tcc.match_calls == mode
