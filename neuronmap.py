#!/usr/bin/env python3
"""Main entry point for NeuronMap CLI."""

__all__ = ["main", "__version__"]
__version__ = "1.0.0"

import sys
import logging
from pathlib import Path

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

try:
    from src.cli.intervention_cli import main
except ImportError:
    # Fallback for different directory structures
    import importlib.util
    cli_path = current_dir / "src" / "cli" / "intervention_cli.py"
    spec = importlib.util.spec_from_file_location("intervention_cli", cli_path)
    cli_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli_module)
    main = cli_module.main

if __name__ == "__main__":
    from src.utils.structured_logging import get_logger

    base_logger = get_logger()
    base_logger.logger.setLevel(logging.INFO)
    logging.getLogger("neuronmap.cli.entry").setLevel(logging.INFO)
    base_logger.log_system_event(
        event_type="entry_point_invoked",
        message="NeuronMap entry CLI invoked"
    )
    main()
