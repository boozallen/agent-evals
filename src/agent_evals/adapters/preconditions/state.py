# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""State precondition applier: YAML key ``state``.

Seeds scenario state into the agent-supplied target dict before the
agent runs. The YAML payload IS the seed-state dict (no inner field).

Semantics:

* **Dotted-path expansion.** Top-level keys containing ``.`` split into
  nested dicts: ``{"a.b.c": v}`` → ``{"a": {"b": {"c": v}}}``. This
  matches the dotted-path-key convention used elsewhere in the codebase
  for state-shaped payloads, so writers use one mental model. Nested
  dict values are NOT recursively expanded — only top-level keys split.
  Recursion happens via the deep-merge step.
* **Deep merge — override wins.** After expansion, the payload is
  deep-merged into the target dict. Both sides dict → recurse; either
  side non-dict → the override (payload) replaces. Mirrors Liquid AI's
  ``_deep_merge``.
* **In-place mutation.** ``apply()`` mutates the passed-in target dict
  and returns ``None``.
"""

from __future__ import annotations

from typing import Any

from pydantic import RootModel

from agent_evals.core._registries import precondition_registry


@precondition_registry.register("state")  # ty: ignore[invalid-argument-type]
class StatePreconditionApplier(RootModel[dict[str, Any]]):
    """Seed agent state by deep-merging the payload into ``target``.

    The YAML payload IS the seed-state dict. Top-level keys with ``.``
    expand into nested dicts before the merge.
    """

    def apply(self, target: dict) -> None:
        """Expand dotted paths in the bound payload, then deep-merge into target.

        Mutates ``target`` in place; returns ``None``.
        """
        expanded = _expand_dotted_paths(self.root)
        _deep_merge(target, expanded)


def _expand_dotted_paths(d: dict) -> dict:
    """Expand top-level dotted-path keys into nested dicts.

    ``{"a.b.c": v}`` becomes ``{"a": {"b": {"c": v}}}``. Plain keys (no
    dot) pass through unchanged. Nested dict values are not recursively
    expanded — recursion happens via deep-merge.
    """
    out: dict = {}
    for key, value in d.items():
        if isinstance(key, str) and "." in key:
            parts = key.split(".")
            cursor = out
            for part in parts[:-1]:
                # If a prior key already placed a non-dict here, the
                # override-wins policy of the merge step would clobber
                # it; here we just build the path eagerly.
                existing = cursor.get(part)
                if not isinstance(existing, dict):
                    existing = {}
                    cursor[part] = existing
                cursor = existing
            cursor[parts[-1]] = value
        else:
            out[key] = value
    return out


def _deep_merge(target: dict, override: dict) -> None:
    """Deep-merge ``override`` into ``target`` in place. Override wins.

    * Both values dict → recurse.
    * Either side non-dict (or key absent in target) → override replaces.
    """
    for key, override_value in override.items():
        target_value = target.get(key)
        if isinstance(target_value, dict) and isinstance(override_value, dict):
            _deep_merge(target_value, override_value)
        else:
            target[key] = override_value


__all__ = ["StatePreconditionApplier"]
