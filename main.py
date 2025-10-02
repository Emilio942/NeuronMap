"""
Main entry point for NeuronMap CLI system.

This module provides a unified CLI interface that integrates:
- Intervention analysis commands (argparse-based)
- Circuit discovery commands (click-based)
- SAE and abstraction commands (click-based)
"""

import argparse
import sys
import os
import logging
from typing import List, Optional

import click
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.cli.intervention_cli import InterventionCLI
from src.cli.circuits_commands import circuits_cli
from src.cli.sae_commands import sae_cli
from src.cli.zoo_commands import zoo
from src.utils.structured_logging import get_logger


_BASE_LOGGER = get_logger()
_DEFAULT_LOG_LEVEL = os.environ.get("NEURONMAP_LOG_LEVEL", "INFO")
_BASE_LOGGER.logger.setLevel(getattr(logging, _DEFAULT_LOG_LEVEL.upper(), logging.INFO))
_MAIN_LOGGER = logging.getLogger("neuronmap.cli.main")
_MAIN_LOGGER.setLevel(_BASE_LOGGER.logger.level)

_BASE_LOGGER.log_system_event(
    event_type="main_cli_initialized",
    message="Primary CLI entrypoint initialized",
    metadata={"log_level": _DEFAULT_LOG_LEVEL},
    level=_DEFAULT_LOG_LEVEL
)


COMMAND_METADATA = {
    "generate": {
        "help": "Generate questions and experiment configurations",
        "description": "Create synthetic question sets and export configuration templates.",
    },
    "extract": {
        "help": "Extract activations from models",
        "description": "Run activation extraction pipelines for selected models and datasets.",
    },
    "analyze": {
        "help": "Analyze activations and attention patterns",
        "description": "Launch analysis routines including attention, activation, and circuit metrics.",
    },
    "visualize": {
        "help": "Visualize NeuronMap results",
        "description": "Render dashboards, charts, and interactive visualizations of computed metrics.",
    },
}


def build_arg_parser() -> argparse.ArgumentParser:
    """Create a legacy argparse parser for compatibility tests."""

    parser = argparse.ArgumentParser(
        prog="neuronmap",
        description="Legacy NeuronMap CLI compatible with argparse-based integrations.",
    )

    subparsers = parser.add_subparsers(dest="command", metavar="command")

    for command, meta in COMMAND_METADATA.items():
        subparser = subparsers.add_parser(command, help=meta["help"], description=meta["description"])
        subparser.add_argument("--config", help="Optional path to configuration file", default=None)
        subparser.add_argument("--output", help="Optional output directory", default=None)

    return parser


def run_legacy_cli(argv: Optional[List[str]] = None) -> int:
    """Parse legacy argparse commands and delegate to the modern click CLI."""

    parser = build_arg_parser()
    try:
        args, remaining = parser.parse_known_args(argv)
    except SystemExit as exc:  # Handles --help / -h
        return exc.code or 0

    if not args.command:
        parser.print_help()
        return 0

    # Provide guidance for migrated commands while preserving legacy behaviour.
    click.echo(
        f"The '{args.command}' command is now handled by the unified click CLI. "
        "Forwarding to the new interface..."
    )

    # Reconstruct argv for click invocation: [command, ...remaining]
    rewritten_args = [args.command] + remaining
    try:
        main.main(args=rewritten_args, standalone_mode=False)  # type: ignore[attr-defined]
    except SystemExit as exc:  # pragma: no cover - click handles exiting
        return exc.code or 0
    return 0


@click.group()
@click.version_option(version='0.1.0')
@click.pass_context
def main(ctx):
    """
    NeuronMap: Neural Network Interpretability Toolkit
    
    A comprehensive toolkit for analyzing neural network internals including:
    - Model Surgery & Path Analysis
    - Circuit Discovery & Analysis  
    - SAE Training & Feature Analysis
    - Abstraction Tracking
    """
    if ctx.obj is None:
        ctx.obj = {}


@main.command('surgery')
@click.pass_context
def surgery_command(ctx):
    """Run model surgery and path analysis commands."""
    # Delegate to the intervention CLI
    try:
        cli = InterventionCLI()
        cli.run()
    except SystemExit:
        # Handle graceful exit from intervention CLI
        pass


# Register the click-based command groups
main.add_command(circuits_cli)
main.add_command(sae_cli)
main.add_command(zoo)


@main.command('help')
@click.pass_context
def help_command(ctx):
    """Show detailed help information."""
    click.echo("NeuronMap: Neural Network Interpretability Toolkit")
    click.echo("=" * 50)
    click.echo()
    click.echo("Available command groups:")
    click.echo()
    click.echo("🔧 surgery    - Model surgery and path analysis")
    click.echo("               • Ablation experiments")
    click.echo("               • Path patching analysis")
    click.echo("               • Causal effect calculation")
    click.echo()
    click.echo("🔍 circuits   - Circuit discovery and analysis")
    click.echo("               • Find induction heads")
    click.echo("               • Find copying heads")
    click.echo("               • Analyze attention compositions")
    click.echo("               • Analyze neuron-head interactions")
    click.echo()
    click.echo("🧠 sae        - SAE training and feature analysis")
    click.echo("               • Train sparse auto-encoders")
    click.echo("               • Analyze SAE features")
    click.echo("               • Find max activating examples")
    click.echo("               • Track abstraction evolution")
    click.echo()
    click.echo("🏛️ zoo        - Analysis Zoo artifact management")
    click.echo("               • Login/logout to the zoo")
    click.echo("               • Push artifacts to share")
    click.echo("               • Pull artifacts from community")
    click.echo("               • Search for artifacts")
    click.echo()
    click.echo("Usage examples:")
    click.echo("  neuronmap surgery --help")
    click.echo("  neuronmap circuits find-induction-heads --help")
    click.echo("  neuronmap sae train --help")
    click.echo("  neuronmap zoo login")
    click.echo("  neuronmap zoo search --query 'gpt2 sae'")
    click.echo()
    click.echo("For detailed help on any command group:")
    click.echo("  neuronmap <group> --help")


if __name__ == '__main__':
    legacy_commands = set(COMMAND_METADATA.keys())
    argv = sys.argv[1:]

    if any(arg in legacy_commands for arg in argv):
        sys.exit(run_legacy_cli(argv))

    main()
