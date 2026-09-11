# Copyright 2026 Booz Allen Hamilton Inc.
# SPDX-License-Identifier: Apache-2.0
"""Platform adapters for logging evaluation results.

Available adapters:
- local: Always available (no dependencies)
- braintrust: Requires 'braintrust' package (uv add braintrust)
- mlflow: Requires 'mlflow' package (uv add mlflow)
- langfuse: Requires 'langfuse' package (uv add langfuse)

Public API:
- platform_registry: AdapterRegistry[type[Platform]] instance.
  Concrete adapters register against it; the runner resolves names
  through it. The composition root injects this default into the
  public ``run_eval`` / ``run_eval_async`` API.
"""

import importlib
import logging
from typing import TYPE_CHECKING, Any

# Always import local adapter (always available)
from agent_evals.adapters.platforms import local  # noqa: F401
from agent_evals.core._registries import platform_registry

logger = logging.getLogger(__name__)

# Wire the data-driven _lazy_package so _PlatformRegistry._trigger_lazy
# knows which package to PEP-562-walk for optional-extras adapters
# (mlflow, braintrust, langfuse). Without this write, the lazy trigger
# is a no-op and only the eagerly-imported `local` adapter is reachable.
platform_registry._lazy_package = "agent_evals.adapters.platforms"

__all__ = [
    "local",
    "platform_registry",
]

if TYPE_CHECKING:
    from agent_evals.adapters.platforms import braintrust as braintrust


def __getattr__(name: str) -> Any:
    """Lazy-load optional platform adapters.

    Optional adapters are imported only when accessed, and raise
    helpful errors if their dependencies aren't installed.

    Args:
        name: Adapter module name

    Returns:
        The requested adapter module

    Raises:
        ImportError: If adapter requires uninstalled package
        AttributeError: If adapter name is unknown
    """
    # Known optional adapters (name -> package_name)
    optional_adapters = {
        "braintrust": "braintrust",
        "mlflow": "mlflow",
        "langfuse": "langfuse",
    }

    if name in optional_adapters:
        package = optional_adapters[name]
        try:
            module = importlib.import_module(f"agent_evals.adapters.platforms.{name}")
            if name not in __all__:
                __all__.append(name)
            logger.info("Loaded optional platform adapter: %s", name)
            return module
        except ImportError as e:
            logger.warning(
                "Failed to load optional adapter %s: package %s not installed",
                name,
                package,
            )
            raise ImportError(
                f"The '{name}' adapter requires the '{package}' package.\n"
                f"Install it with: uv add 'agent-evals[{name}]'"
            ) from e

    logger.warning("Unknown platform adapter requested: %s", name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
