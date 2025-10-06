"""Compatibility wrapper for the deprecated `config.config_manager` module.

The canonical implementation lives in `src.utils.config_manager`.  This
module simply re-exports the public API so legacy imports continue to work.
"""

from src.utils.config_manager import *  # noqa: F401,F403
