"""Lightweight smoke tests to ensure core entry points work."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]


@pytest.mark.unit
def test_config_manager_validation():
    """ConfigManager should load default configs without errors."""
    from src.utils.config import ConfigManager

    manager = ConfigManager()
    assert manager.is_valid(), f"Config validation errors: {manager.get_validation_errors()}"


@pytest.mark.unit
def test_package_imports():
    """Top-level neuronmap package should import without side effects."""
    module = __import__("neuronmap")
    assert hasattr(module, "__version__"), "Expected __version__ attribute on neuronmap package"


@pytest.mark.integration
def test_cli_help_invocation():
    """Running the CLI with --help should succeed and print usage text."""
    result = subprocess.run(
        [sys.executable, "neuronmap-cli.py", "--help"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "NeuronMap:" in result.stdout, result.stdout
