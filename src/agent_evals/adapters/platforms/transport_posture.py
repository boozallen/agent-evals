# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""One-line transport-posture record for platform adapters that open sockets.

An operator reading logs should be able to confirm that endpoint validation was
in force for a given run without reading the code that ran. This emits that
statement once, at platform-adapter construction.

Construction, not per scorer: a benchmark run builds dozens of scorers, and a
record per scorer would be noise that buries the one fact worth reading.
"""

from __future__ import annotations

import logging


def log_transport_posture(logger: logging.Logger, platform: str) -> None:
    """Emit one record stating that endpoint validation is enforced.

    Takes the caller's ``logger`` rather than owning a module-level one, so the
    record attributes to the adapter that constructed it
    (``agent_evals.adapters.platforms.mlflow``, say) and users' existing
    per-adapter log filters and levels keep applying.

    ``platform`` is named in the message body as well, so the line stands on its
    own: a run that builds more than one adapter would otherwise produce records
    that differ only in the logger name, which is dropped by many format strings
    and by anyone reading a flat log.

    ``INFO``, not ``DEBUG``: the record exists to be found in an audit of a
    production run, and production log configurations routinely sit at ``INFO``
    or above, where a ``DEBUG`` record does not exist to be found.

    Args:
        logger: The constructing adapter's logger.
        platform: The registered name of the constructing adapter, as it appears
            in a ``platform=`` argument (``"mlflow"``, ``"langfuse"``, ...).
    """
    logger.info(
        "Transport validation is enforced for platform %s: caller-supplied "
        "endpoints must transmit over an encrypted transport, transmit nothing, "
        "or address only the machine this process runs on.",
        platform,
    )
