#!/usr/bin/env python3
"""
NeuronMap CLI Main Entry Point

This script provides the main CLI interface for NeuronMap, with subcommands for:
- Analysis Zoo artifact management (zoo)
- Circuit discovery and analysis (circuits)
- Model interventions and path analysis (analyze)
"""

import click
from .zoo_commands import zoo
from .circuits_commands import circuits_cli
from .sae_commands import sae_cli

@click.group()
@click.version_option(version='1.0.0')
def cli():
    """NeuronMap - Advanced Neural Network Interpretability Toolkit"""
    pass

# Register subcommands
cli.add_command(zoo)
cli.add_command(circuits_cli, name='circuits')
cli.add_command(sae_cli, name='sae')

if __name__ == '__main__':
    cli()
