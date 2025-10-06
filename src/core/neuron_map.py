"""Fallback NeuronMap core module.

This lightweight placeholder prevents import errors when optional
core components are unavailable. It provides a minimal surface that
can be extended with real functionality when the full core package is
present.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class NeuronMapConfig:
    """Minimal configuration stub."""

    name: str = "default"
    options: Dict[str, Any] | None = None


class NeuronMap:
    """Minimal stub implementation used for development-only workflows.

    The real project defines an extensive orchestration layer for model
    loading, activation analysis, and reporting. During lightweight web UI
    prototyping we only need a sentinel object to keep optional imports
    satisfied. The methods below can be expanded with proper behaviors
    once the full core module is restored.
    """

    def __init__(self, config: Optional[NeuronMapConfig] = None) -> None:
        self.config = config or NeuronMapConfig()

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow configuration snapshot."""

        return {
            "name": self.config.name,
            "options": self.config.options or {},
        }

    # ------------------------------------------------------------------
    # Compatibility helpers
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Placeholder for the real load routine."""

    def shutdown(self) -> None:
        """Placeholder for cleanup logic."""


__all__ = ["NeuronMap", "NeuronMapConfig"]
