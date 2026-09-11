# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""The transport posture is recorded once per network-adapter construction.

Two properties pull in opposite directions, so both are pinned here: an operator
must be able to confirm from a log scan that endpoint validation was in force for
a run, and a library must not chatter. Hence one record per platform-adapter
construction, at ``INFO`` so it survives a production log configuration, and
never per scorer.

Attribution is asserted twice over, because the two mechanisms fail separately.
The helper takes the caller's logger, so records name the constructing adapter
and users' existing per-adapter filters keep working; and the platform name is
in the message body, so the line still identifies itself under a format string
that drops the logger name. A shared module logger, or a message identical
across adapters, would break one of those silently.
"""

from __future__ import annotations

import logging

import pytest

from agent_evals.adapters.platforms.braintrust import BraintrustPlatform
from agent_evals.adapters.platforms.langfuse import LangFusePlatform
from agent_evals.adapters.platforms.local import LocalPlatform
from agent_evals.adapters.platforms.mlflow import MLflowPlatform
from agent_evals.adapters.scorers.autoevals import Factuality

pytestmark = pytest.mark.requires_braintrust

# The adapters that open sockets, paired with the logger each must attribute to
# and the platform name each must name in the message body.
NETWORK_ADAPTERS = [
    pytest.param(
        MLflowPlatform,
        "agent_evals.adapters.platforms.mlflow",
        "mlflow",
        id="mlflow",
    ),
    pytest.param(
        LangFusePlatform,
        "agent_evals.adapters.platforms.langfuse",
        "langfuse",
        id="langfuse",
    ),
    pytest.param(
        BraintrustPlatform,
        "agent_evals.adapters.platforms.braintrust",
        "braintrust",
        id="braintrust",
    ),
]

PARAMS = ("adapter_cls", "logger_name", "platform_name")


@pytest.mark.parametrize(PARAMS, NETWORK_ADAPTERS)
def test_exactly_one_posture_record_per_construction(
    adapter_cls, logger_name, platform_name, caplog
):
    """One statement per adapter, not one per operation it later performs."""
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        adapter_cls()

    records = [r for r in caplog.records if r.name == logger_name]
    assert len(records) == 1, "exactly one posture record per construction"
    assert "enforced" in records[0].getMessage()


@pytest.mark.parametrize(PARAMS, NETWORK_ADAPTERS)
def test_posture_is_recorded_at_info(adapter_cls, logger_name, platform_name, caplog):
    """``INFO``, so an audit of a production run can find it.

    Production log configurations routinely sit at ``INFO`` or above. A record
    below that level is not a record an auditor can find, so the level is pinned
    exactly rather than as an upper bound.
    """
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        adapter_cls()

    records = [r for r in caplog.records if r.name == logger_name]
    assert len(records) == 1
    assert records[0].levelno == logging.INFO


@pytest.mark.parametrize(PARAMS, NETWORK_ADAPTERS)
def test_record_attributes_to_the_constructing_adapter(
    adapter_cls, logger_name, platform_name, caplog
):
    """The helper takes the caller's logger, so filters by adapter keep working."""
    with caplog.at_level(logging.DEBUG):
        adapter_cls()

    assert any(r.name == logger_name for r in caplog.records), (
        f"no posture record attributed to {logger_name}; a shared module logger "
        f"would break users' per-adapter log configuration"
    )


@pytest.mark.parametrize(PARAMS, NETWORK_ADAPTERS)
def test_message_body_names_the_platform(
    adapter_cls, logger_name, platform_name, caplog
):
    """The line identifies itself without the logger name.

    A format string that omits ``%(name)s`` is common, and under one the records
    from two adapters in the same run would be byte-identical.
    """
    with caplog.at_level(logging.DEBUG, logger=logger_name):
        adapter_cls()

    records = [r for r in caplog.records if r.name == logger_name]
    assert len(records) == 1
    assert platform_name in records[0].getMessage(), (
        f"posture record does not name {platform_name!r}; with the logger name "
        f"stripped, this line would be indistinguishable from another adapter's"
    )


def test_platform_names_make_the_records_distinguishable(caplog):
    """Two adapters in one run must not produce the same line.

    Asserted on the messages alone, with logger names discarded, because that is
    what a reader of a flat log has.
    """
    with caplog.at_level(logging.DEBUG):
        for adapter_cls, _, _ in (p.values for p in NETWORK_ADAPTERS):
            adapter_cls()

    messages = [
        r.getMessage()
        for r in caplog.records
        if "ransport validation" in r.getMessage()
    ]
    assert len(messages) == len(NETWORK_ADAPTERS)
    assert len(set(messages)) == len(NETWORK_ADAPTERS)


def test_local_platform_logs_no_posture(caplog):
    """LocalPlatform opens no socket, so a transport line there asserts nothing."""
    with caplog.at_level(logging.DEBUG):
        LocalPlatform()

    posture_records = [
        r for r in caplog.records if "ransport validation" in r.getMessage()
    ]
    assert posture_records == []


def test_scorer_construction_logs_no_posture(caplog):
    """Per-scorer posture logging is noise.

    A benchmark run builds dozens of scorers; one posture record each would bury
    the adapter-level statement that is actually worth reading.
    """
    with caplog.at_level(logging.DEBUG):
        Factuality(base_url="https://judge.example.com/v1")

    posture_records = [
        r for r in caplog.records if "ransport validation" in r.getMessage()
    ]
    assert posture_records == []
