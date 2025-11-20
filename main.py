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


@main.command('generate')
@click.option('--model', default='gpt2', help='Model to generate questions for')
@click.option('--count', default=10, help='Number of questions to generate')
@click.option('--output', help='Output file path')
@click.option('--use-ollama/--no-ollama', default=False, help='Use Ollama for generation')
@click.option('--domain', help='Domain for questions (e.g. science, technology)')
@click.option('--difficulty', help='Difficulty range (e.g. 1-5 or 5)')
@click.option('--topic', multiple=True, help='Specific topics to include')
def generate_command(model, count, output, use_ollama, domain, difficulty, topic):
    """Generate synthetic questions for analysis."""
    from src.data_generation.question_generator import QuestionGenerator
    
    click.echo(f"Generating {count} questions for {model}...")
    generator = QuestionGenerator(config={'model': model})
    
    # Parse difficulty
    difficulty_range = None
    if difficulty:
        try:
            if '-' in difficulty:
                start, end = map(int, difficulty.split('-'))
                difficulty_range = (start, end)
            else:
                val = int(difficulty)
                difficulty_range = (val, val)
        except ValueError:
            click.echo("Invalid difficulty format. Use 'min-max' or 'value'.", err=True)
            return

    questions = generator.generate_questions(
        count=count,
        use_ollama=use_ollama,
        domain=domain,
        difficulty_range=difficulty_range,
        topics=list(topic) if topic else None,
        structured=True
    )
    
    if output:
        import json
        with open(output, 'w') as f:
            json.dump(questions, f, indent=2)
        click.echo(f"Questions saved to {output}")
    else:
        for q in questions:
            click.echo(f"- {q['text']}")


@main.command('extract')
@click.option('--model', required=True, help='Model name')
@click.option('--input', required=True, help='Input text or file')
@click.option('--layers', help='Comma-separated list of layers')
@click.option('--output', required=True, help='Output file path')
def extract_command(model, input, layers, output):
    """Extract activations from a model."""
    from src.analysis.activation_extractor import ActivationExtractor
    
    click.echo(f"Extracting activations from {model}...")
    
    # Handle input file or text
    input_text = input
    if os.path.exists(input):
        with open(input, 'r') as f:
            input_text = f.read()
            
    extractor = ActivationExtractor(model_name_or_config=model, target_layer=layers)
    # Note: ActivationExtractor API might need adjustment based on actual usage
    # For now assuming basic usage
    click.echo("Extraction logic placeholder - connecting to backend...")
    # TODO: Connect to actual extraction logic
    click.echo(f"Activations saved to {output}")


@main.command('visualize')
@click.option('--input', required=True, help='Input activations file')
@click.option('--type', type=click.Choice(['heatmap', 'pca', 'tsne']), default='heatmap')
@click.option('--output', required=True, help='Output image path')
def visualize_command(input, type, output):
    """Visualize analysis results."""
    from src.visualization.visualizer import ActivationVisualizer
    
    click.echo(f"Generating {type} visualization from {input}...")
    visualizer = ActivationVisualizer(output_dir=os.path.dirname(output))
    # TODO: Load data and call visualizer methods
    click.echo(f"Visualization saved to {output}")


@main.group('analyze')
def analyze_group():
    """Analyze activations and attention patterns."""
    pass

@analyze_group.command('ablate')
@click.option('--model', required=True, help='Model name')
@click.option('--prompt', required=True, help='Input prompt')
@click.option('--layer', required=True, help='Target layer')
@click.option('--neurons', help='Comma-separated list of neurons')
@click.option('--output', help='Output directory')
def ablate_command(model, prompt, layer, neurons, output):
    """Run ablation analysis."""
    from src.analysis.model_integration import get_model_manager
    from src.analysis.intervention_cache import InterventionCache
    import json
    
    click.echo(f"Running ablation on {model} layer {layer}...")
    
    neuron_list = [int(x.strip()) for x in neurons.split(',')] if neurons else None
    
    model_manager = get_model_manager()
    cache = InterventionCache()
    
    results = model_manager.run_ablation_analysis(
        model_name=model,
        prompt=prompt,
        layer_name=layer,
        neuron_indices=neuron_list,
        cache=cache
    )
    
    if output:
        os.makedirs(output, exist_ok=True)
        with open(os.path.join(output, 'ablation_results.json'), 'w') as f:
            json.dump(results, f, indent=2)
        click.echo(f"Results saved to {output}")
    else:
        click.echo(json.dumps(results, indent=2))


@analyze_group.command('patch')
@click.option('--model', required=True, help='Model name')
@click.option('--source', required=True, help='Source prompt')
@click.option('--target', required=True, help='Target prompt')
@click.option('--layer', required=True, help='Target layer')
@click.option('--output', help='Output directory')
def patch_command(model, source, target, layer, output):
    """Run path patching analysis."""
    # Placeholder for patching logic
    click.echo(f"Running patching on {model}...")


@main.command('surgery')
@click.pass_context
def surgery_command(ctx):
    """Run model surgery and path analysis commands (Legacy)."""
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
    click.echo("🔧 analyze    - Analysis tools (ablate, patch)")
    click.echo("🔧 surgery    - Legacy surgery tools")
    click.echo("🔍 circuits   - Circuit discovery and analysis")
    click.echo("🧠 sae        - SAE training and feature analysis")
    click.echo("🏛️ zoo        - Analysis Zoo artifact management")
    click.echo("📝 generate   - Generate synthetic questions")
    click.echo("📥 extract    - Extract activations")
    click.echo("📊 visualize  - Visualize results")
    click.echo()


if __name__ == '__main__':
    main()
