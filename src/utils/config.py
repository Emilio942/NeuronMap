"""Compatibility layer for the legacy `utils.config` API.

This module re-exports the canonical configuration system implemented in
``utils.config_manager`` so existing imports keep working while the project
uses a single ConfigManager implementation under the hood.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from .config_manager import (
    AnalysisConfig,
    ConfigManager,
    ConfigValidationError,
    DataConfig,
    Environment,
    ExperimentConfig,
    LogLevel,
    ModelConfig,
    NeuronMapConfig,
    SystemConfig,
    VisualizationConfig,
    WebConfig,
    get_config as _get_config,
    get_config_manager as _get_config_manager,
    reset_config_manager as _reset_core_manager,
    setup_global_config as _setup_core_manager,
)


class EnvironmentType(str, Enum):
    """Backwards compatible alias for environment names."""

    DEVELOPMENT = Environment.DEVELOPMENT.value
    TESTING = Environment.TESTING.value
    PRODUCTION = Environment.PRODUCTION.value


class DeviceType(str, Enum):
    """Simple device enum maintained for compatibility."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


class ModelType(str, Enum):
    """Legacy model architecture identifiers."""

    GPT = "gpt"
    BERT = "bert"
    T5 = "t5"
    LLAMA = "llama"


def get_config_manager(
    config_dir: Optional[str] = None,
    environment: Optional[str] = None,
) -> ConfigManager:
    """Return the shared ConfigManager instance.

    If ``config_dir`` is provided a fresh manager is created via
    :func:`setup_global_config` to keep a single authoritative instance.
    When only ``environment`` is provided the existing manager is reused
    with the requested environment activated.
    """

    if config_dir is not None:
        return setup_global_config(config_dir=config_dir, environment=environment)

    manager = _get_config_manager()
    if environment is not None:
        manager.set_environment(environment)
    return manager


def reset_config_manager() -> None:
    """Reset the global ConfigManager instance."""

    _reset_core_manager()


def setup_global_config(
    config_dir: Optional[str] = None,
    environment: Optional[str] = None,
) -> ConfigManager:
    """Initialize and validate the global configuration manager."""

    return _setup_core_manager(config_dir=config_dir, environment=environment)


def get_config(config_path: Optional[str] = None):
    """Return the active configuration object or load from a specific file."""

    return _get_config(config_path)


# Re-export commonly used classes and helpers for convenience
__all__ = [
    "AnalysisConfig",
    "ConfigManager",
    "ConfigValidationError",
    "DataConfig",
    "Environment",
    "EnvironmentType",
    "ExperimentConfig",
    "DeviceType",
    "LogLevel",
    "ModelConfig",
    "ModelType",
    "NeuronMapConfig",
    "SystemConfig",
    "VisualizationConfig",
    "WebConfig",
    "get_config",
    "get_config_manager",
    "reset_config_manager",
    "setup_global_config",
]