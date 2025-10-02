"""Deprecated compatibility module for NeuronMap configuration.

All functionality has moved to :mod:`utils.config_manager`.  This module now
re-exports the canonical types to preserve backwards compatibility for any
legacy imports.
"""

from __future__ import annotations

from .config_manager import *  # noqa: F401,F403

__all__ = [name for name in globals().keys() if not name.startswith("_")]
