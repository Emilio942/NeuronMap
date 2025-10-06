"""Compatibility layer for legacy `config` package imports.

Historically the project exposed configuration helpers from a top-level
`config` package. The modern code lives in `src.utils.config_manager`,
but several modules (web APIs, legacy scripts) still import from
`config.config_manager`. This lightweight wrapper keeps those imports
working without duplicating logic.
"""

from src.utils.config_manager import (
    ConfigManager,
    get_config,
    get_config_manager,
    reset_config_manager,
    setup_global_config,
)

__all__ = [
    "ConfigManager",
    "get_config",
    "get_config_manager",
    "reset_config_manager",
    "setup_global_config",
]
