# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Parser port: payload → typed value.

A ``Parser[T]`` validates a payload (typically a YAML fragment, via
Pydantic) and produces a typed value ``T`` via ``to_spec()``. Registries
enforce method presence at registration time; the Protocol is structural
so any object with a matching ``to_spec`` satisfies it.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

T_co = TypeVar("T_co", covariant=True)


@runtime_checkable
class Parser(Protocol[T_co]):
    """Structural contract for adapter-family parsers.

    Implementers validate a constructor payload (typically a YAML
    fragment, via a Pydantic ``BaseModel``) and expose ``to_spec()``
    to produce the typed value the application layer consumes.
    """

    def to_spec(self) -> T_co: ...
