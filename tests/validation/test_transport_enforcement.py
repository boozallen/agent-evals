# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Open/Closed guard: every registered scorer taking an endpoint validates it.

Per-call-site enforcement is open to extension and closed to nothing — the next
factory added is a coin flip. That is not a hypothetical: the deficiency this
change remediates existed because three of the twelve endpoint-accepting
factories bypassed the shared kwargs helper, and nothing tied "new factory with
an endpoint parameter" to "validates that parameter."

So this guard selects by signature inspection, not by a list of names. Adding a
thirteenth factory that declares ``base_url`` makes this test fail until that
factory validates.

**What this guard does NOT cover.** The agentevals LLM-judge factories
(``TrajectoryLLMAsJudge``, ``GraphTrajectoryLLMAsJudge``) declare no ``base_url``
and are therefore silently skipped by the filter below. Their endpoint parameter
is named ``judge_base_url``, because the ``model`` argument beside it is a
provider nickname rather than an address. A green run here is **not** evidence
their endpoints are validated; the coverage for those two is
``tests/unit/test_agentevals_transport_validation.py``.
"""

from __future__ import annotations

import inspect

import pytest

# Importing the family package is what populates the production registry via the
# concrete modules' @register decorators.
import agent_evals.adapters.scorers  # noqa: F401
from agent_evals.core._registries import scorer_registry

# The endpoint parameter this guard selects on. Named once so the filter and the
# failure messages cannot drift apart.
ENDPOINT_PARAM = "base_url"

# AST-enumerated on develop: 12 public autoevals factories declare base_url.
# Non-vacuity floor, not an exact count — a new factory should raise this, and
# raising it is a deliberate edit, whereas the registry silently failing to
# populate (an import error in an extras-gated module, say) would otherwise make
# this coverage test pass by examining nothing at all.
MINIMUM_EXPECTED = 12

# A cleartext endpoint on a *remote* host. Loopback cleartext is accepted by
# design — it reaches no network — so a loopback value here would assert the
# opposite of the rule and fail every factory that is behaving correctly.
CLEARTEXT_ENDPOINT = "http://scorer.example.com:11434/v1"


def _endpoint_accepting_factories() -> list[tuple[str, object]]:
    """Read the production registry; never register into it or mutate it.

    Per the repo's registry-injection convention, tests that need to *change* a
    registry build their own instance. This guard's whole purpose is to inspect
    what production actually has, so it reads the module-level instance and
    leaves it untouched.
    """
    selected = []
    for name in scorer_registry.list():
        factory = scorer_registry.get(name)
        try:
            signature = inspect.signature(factory)
        except TypeError, ValueError:  # pragma: no cover - defensive
            continue
        if ENDPOINT_PARAM in signature.parameters:
            selected.append((name, factory))
    return selected


def test_guard_examines_a_plausible_number_of_factories():
    """Non-vacuity. A coverage test that examines nothing must not pass."""
    selected = _endpoint_accepting_factories()

    assert len(selected) >= MINIMUM_EXPECTED, (
        f"expected at least {MINIMUM_EXPECTED} registered factories declaring "
        f"{ENDPOINT_PARAM!r}, found {len(selected)}: "
        f"{sorted(name for name, _ in selected)}. Either the registry failed to "
        f"populate (making the enforcement check below vacuous), or factories "
        f"were removed — in which case lower MINIMUM_EXPECTED deliberately."
    )


@pytest.mark.parametrize(
    ("name", "factory"),
    _endpoint_accepting_factories(),
    ids=lambda value: value if isinstance(value, str) else "",
)
def test_every_endpoint_accepting_factory_rejects_cleartext(name, factory):
    """Selected by signature, so a newly added factory is covered automatically."""
    with pytest.raises(ValueError, match=ENDPOINT_PARAM) as exc_info:
        factory(**{ENDPOINT_PARAM: CLEARTEXT_ENDPOINT})

    assert "https" in str(exc_info.value), (
        f"{name} rejected the cleartext endpoint but the error does not say what "
        f"would be accepted, leaving a user no way forward."
    )
