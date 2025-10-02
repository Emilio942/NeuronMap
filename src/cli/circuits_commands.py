"""
CLI commands for circuit discovery and analysis.

This module implements CLI commands for finding and analyzing neural circuits
including induction heads, copying heads, and attention head compositions.
"""

from ..analysis.circuits import (
    NeuralCircuit,
    AttentionHeadCompositionAnalyzer,
    InductionHeadScanner,
    CopyingHeadScanner,
    NeuronToHeadAnalyzer,
    CircuitVerifier
)
import click
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
import sys
import torch
import transformers

from ..utils.structured_logging import get_logger

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))


logger = logging.getLogger("neuronmap.cli.circuits")


def setup_logging(verbose: bool):
    """Set up logging configuration."""
    base_logger = get_logger()
    level = logging.DEBUG if verbose else logging.INFO
    base_logger.logger.setLevel(level)
    logger.setLevel(level)

    base_logger.log_system_event(
        event_type="circuits_cli_logging_initialized",
        message="Circuits CLI logging configured",
        metadata={"verbose": verbose},
        level="DEBUG" if verbose else "INFO"
    )


def load_model_and_tokenizer(model_name: str, device: str = "auto"):
    """Load a model and tokenizer."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    click.echo(f"Loading model {model_name} on {device}...")

    try:
        model = transformers.AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            output_attentions=True
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model.to(device)
        model.eval()

        click.echo(f"✓ Model loaded successfully")
        return model, tokenizer
    except Exception as e:
        click.echo(f"❌ Error loading model: {e}")
        raise


@click.group(name='circuits')
@click.pass_context
def circuits_cli(ctx):
    """Circuit discovery and analysis commands."""
    if ctx.obj is None:
        ctx.obj = {}


@circuits_cli.command('find-induction-heads')
@click.option('--model', '-m', required=True, help='Model name to analyze')
@click.option('--threshold', '-t', default=0.1,
              help='Minimum induction score threshold')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'graphml', 'both']),
              default='json', help='Output format')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def find_induction_heads(ctx, model: str, threshold: float, output: Optional[str],
                         output_format: str, device: str, verbose: bool):
    """
    Find induction heads in a transformer model.

    Induction heads are attention heads that implement the pattern:
    [A][B] ... [A] -> [B]

    They are crucial for in-context learning and sequence completion.
    """
    setup_logging(verbose)

    try:
        # Load model and tokenizer
        model_obj, tokenizer = load_model_and_tokenizer(model, device)

        # Create scanner
        scanner = InductionHeadScanner(model_obj, tokenizer)

        click.echo(f"🔍 Scanning {model} for induction heads...")
        click.echo(f"📊 Using threshold: {threshold}")

        # Create test prompt
        test_prompt = scanner.create_induction_test_prompt()

        # Run scan
        candidates = scanner.scan_for_induction_heads(test_prompt, threshold=threshold)

        if not candidates:
            click.echo("❌ No induction heads found with the given threshold.")
            return

        click.echo(f"✅ Found {len(candidates)} induction head candidates")

        # Create circuit
        circuit = scanner.create_induction_circuit(
            threshold=threshold, input_ids=test_prompt)

        # Prepare results
        results = {
            'model_name': model,
            'threshold': threshold,
            'candidates': [(layer, head, float(score)) for layer, head, score in candidates],
            'circuit_summary': circuit.get_circuit_statistics(),
            'metadata': {
                'scan_type': 'induction_heads',
                'num_candidates': len(candidates),
                'test_prompt_length': test_prompt.size(1)
            }
        }

        # Display summary
        click.echo("\n📈 Summary:")
        for i, (layer, head, score) in enumerate(candidates[:5]):  # Show top 5
            click.echo(f"  {i + 1}. Layer {layer}, Head {head} (score: {score:.3f})")

        if len(candidates) > 5:
            click.echo(f"  ... and {len(candidates) - 5} more")

        # Save results
        if output:
            output_path = Path(output)
        else:
            output_path = Path(f"induction_heads_{model.replace('/', '_')}")

        if output_format in ['json', 'both']:
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            click.echo(f"💾 Results saved to: {json_path}")

        if output_format in ['graphml', 'both']:
            graphml_path = output_path.with_suffix('.graphml')
            circuit.export_to_graphml(str(graphml_path))
            click.echo(f"🌐 Circuit graph saved to: {graphml_path}")

        # Display circuit statistics
        stats = circuit.get_circuit_statistics()
        click.echo(f"\n🔗 Circuit Statistics:")
        click.echo(f"  Components: {stats['total_components']}")
        click.echo(f"  Connections: {stats['total_connections']}")
        click.echo(f"  Circuit depth: {stats['circuit_depth']} layers")

    except Exception as e:
        click.echo(f"❌ Error during induction head scan: {e}")
        if verbose:
            raise


@circuits_cli.command('find-copying-heads')
@click.option('--model', '-m', required=True, help='Model name to analyze')
@click.option('--threshold', '-t', default=0.3, help='Minimum copying score threshold')
@click.option('--concentration-threshold', default=2.0,
              help='Minimum concentration threshold')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'graphml', 'both']),
              default='json', help='Output format')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def find_copying_heads(
        ctx,
        model: str,
        threshold: float,
        concentration_threshold: float,
        output: Optional[str],
        output_format: str,
        device: str,
        verbose: bool):
    """
    Find copying/saliency heads in a transformer model.

    Copying heads primarily copy information from important positions
    like the first token, end token, or other salient positions.
    """
    setup_logging(verbose)

    try:
        # Load model and tokenizer
        model_obj, tokenizer = load_model_and_tokenizer(model, device)

        # Create scanner
        scanner = CopyingHeadScanner(model_obj, tokenizer)

        click.echo(f"🔍 Scanning {model} for copying heads...")
        click.echo(f"📊 Using thresholds: copying={
                   threshold}, concentration={concentration_threshold}")

        # Create test prompt (using tokenizer to create a diverse prompt)
        test_text = "The quick brown fox jumps over the lazy dog. This sentence contains many different words."
        test_prompt = tokenizer(test_text, return_tensors="pt")[
            "input_ids"].to(model_obj.device)

        # Run scan
        candidates = scanner.scan_for_copying_heads(
            test_prompt,
            copying_threshold=threshold,
            concentration_threshold=concentration_threshold
        )

        if not candidates:
            click.echo("❌ No copying heads found with the given thresholds.")
            return

        click.echo(f"✅ Found {len(candidates)} copying head candidates")

        # Prepare results
        results = {
            'model_name': model,
            'copying_threshold': threshold,
            'concentration_threshold': concentration_threshold,
            'candidates': [
                {
                    'layer': layer,
                    'head': head,
                    'scores': scores
                } for layer, head, scores in candidates
            ],
            'metadata': {
                'scan_type': 'copying_heads',
                'num_candidates': len(candidates),
                'test_prompt_length': test_prompt.size(1)
            }
        }

        # Display summary
        click.echo("\n📈 Summary:")
        for i, (layer, head, scores) in enumerate(candidates[:5]):  # Show top 5
            main_score = max(scores.get('first_token', 0), scores.get('last_token', 0))
            click.echo(
                f"  {
                    i +
                    1}. Layer {layer}, Head {head} (main score: {
                    main_score:.3f})")
            click.echo(f"      First token: {scores.get('first_token', 0):.3f}, "
                       f"Last token: {scores.get('last_token', 0):.3f}, "
                       f"Concentration: {scores.get('concentration', 0):.3f}")

        if len(candidates) > 5:
            click.echo(f"  ... and {len(candidates) - 5} more")

        # Save results
        if output:
            output_path = Path(output)
        else:
            output_path = Path(f"copying_heads_{model.replace('/', '_')}")

        if output_format in ['json', 'both']:
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            click.echo(f"💾 Results saved to: {json_path}")

    except Exception as e:
        click.echo(f"❌ Error during copying head scan: {e}")
        if verbose:
            raise


@circuits_cli.command('analyze-composition')
@click.option('--model', '-m', required=True, help='Model name to analyze')
@click.option('--layer1', default=0, help='Source layer for composition analysis')
@click.option('--layer2', default=1, help='Target layer for composition analysis')
@click.option('--threshold', '-t', default=0.3,
              help='Minimum composition score threshold')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'graphml', 'both']),
              default='json', help='Output format')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def analyze_composition(
        ctx,
        model: str,
        layer1: int,
        layer2: int,
        threshold: float,
        output: Optional[str],
        output_format: str,
        device: str,
        verbose: bool):
    """
    Analyze attention head composition between layers.

    Composition analysis determines which heads in later layers "read" from
    the outputs of heads in earlier layers.
    """
    setup_logging(verbose)

    try:
        # Load model and tokenizer
        model_obj, tokenizer = load_model_and_tokenizer(model, device)

        # Create analyzer
        analyzer = AttentionHeadCompositionAnalyzer(model_obj, tokenizer)

        click.echo(f"🔍 Analyzing attention head composition in {model}")
        click.echo(f"📊 Layer {layer1} -> Layer {layer2}, threshold: {threshold}")

        # Create test prompt
        test_text = "The quick brown fox jumps over the lazy dog"
        test_prompt = tokenizer(test_text, return_tensors="pt")[
            "input_ids"].to(model_obj.device)

        # Run composition analysis
        compositions = analyzer.analyze_layer_compositions(
            test_prompt, layer1, layer2, threshold)

        if not compositions:
            click.echo("❌ No significant compositions found with the given threshold.")
            return

        click.echo(f"✅ Found {len(compositions)} significant compositions")

        # Build full circuit
        circuit = analyzer.build_composition_circuit(test_prompt, threshold)

        # Prepare results
        results = {
            'model_name': model,
            'layer1': layer1,
            'layer2': layer2,
            'threshold': threshold,
            'compositions': [
                {
                    'head1': head1,
                    'head2': head2,
                    'score': float(score)
                } for head1, head2, score in compositions
            ],
            'circuit_summary': circuit.get_circuit_statistics(),
            'metadata': {
                'analysis_type': 'attention_head_composition',
                'num_compositions': len(compositions)
            }
        }

        # Display summary
        click.echo("\n📈 Summary:")
        for i, (head1, head2, score) in enumerate(compositions[:5]):  # Show top 5
            click.echo(f"  {i + 1}. Head {head1} -> Head {head2} (score: {score:.3f})")

        if len(compositions) > 5:
            click.echo(f"  ... and {len(compositions) - 5} more")

        # Save results
        if output:
            output_path = Path(output)
        else:
            output_path = Path(
                f"composition_{
                    model.replace(
                        '/',
                        '_')}_L{layer1}_L{layer2}")

        if output_format in ['json', 'both']:
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            click.echo(f"💾 Results saved to: {json_path}")

        if output_format in ['graphml', 'both']:
            graphml_path = output_path.with_suffix('.graphml')
            circuit.export_to_graphml(str(graphml_path))
            click.echo(f"🌐 Circuit graph saved to: {graphml_path}")

        # Display circuit statistics
        stats = circuit.get_circuit_statistics()
        click.echo(f"\n🔗 Circuit Statistics:")
        click.echo(f"  Components: {stats['total_components']}")
        click.echo(f"  Connections: {stats['total_connections']}")
        click.echo(f"  Circuit depth: {stats['circuit_depth']} layers")

    except Exception as e:
        click.echo(f"❌ Error during composition analysis: {e}")
        if verbose:
            raise


@circuits_cli.command('analyze-neuron-head')
@click.option('--model', '-m', required=True, help='Model name to analyze')
@click.option('--max-layers', default=3, help='Maximum number of layers to analyze')
@click.option('--threshold', '-t', default=0.1,
              help='Minimum influence score threshold')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'graphml', 'both']),
              default='json', help='Output format')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def analyze_neuron_head(
        ctx,
        model: str,
        max_layers: int,
        threshold: float,
        output: Optional[str],
        output_format: str,
        device: str,
        verbose: bool):
    """
    Analyze neuron-to-head influence connections.

    This analysis quantifies how MLP neurons in one layer influence
    attention heads in subsequent layers.
    """
    setup_logging(verbose)

    try:
        # Load model and tokenizer
        model_obj, tokenizer = load_model_and_tokenizer(model, device)

        # Create analyzer
        analyzer = NeuronToHeadAnalyzer(model_obj, tokenizer)

        click.echo(f"🔍 Analyzing neuron-to-head influences in {model}")
        click.echo(f"📊 Max layers: {max_layers}, threshold: {threshold}")

        # Create test prompt
        test_text = "The quick brown fox jumps over the lazy dog"
        test_prompt = tokenizer(test_text, return_tensors="pt")[
            "input_ids"].to(model_obj.device)

        # Build neuron-head circuit
        circuit = analyzer.build_neuron_head_circuit(test_prompt, max_layers, threshold)

        if len(circuit._components) == 0:
            click.echo("❌ No significant neuron-to-head influences found.")
            return

        click.echo(f"✅ Found circuit with {len(circuit._components)} components")

        # Prepare results
        results = {
            'model_name': model,
            'max_layers': max_layers,
            'threshold': threshold,
            'circuit_summary': circuit.get_circuit_statistics(),
            'components': {
                comp_id: comp.to_dict()
                for comp_id, comp in circuit._components.items()
            },
            'metadata': {
                'analysis_type': 'neuron_to_head_influence',
                'num_components': len(circuit._components)
            }
        }

        # Display summary
        stats = circuit.get_circuit_statistics()
        click.echo(f"\n🔗 Circuit Statistics:")
        click.echo(f"  Components: {stats['total_components']}")
        click.echo(f"  Connections: {stats['total_connections']}")
        click.echo(f"  Component types: {stats['components_by_type']}")

        # Save results
        if output:
            output_path = Path(output)
        else:
            output_path = Path(f"neuron_head_{model.replace('/', '_')}")

        if output_format in ['json', 'both']:
            json_path = output_path.with_suffix('.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            click.echo(f"💾 Results saved to: {json_path}")

        if output_format in ['graphml', 'both']:
            graphml_path = output_path.with_suffix('.graphml')
            circuit.export_to_graphml(str(graphml_path))
            click.echo(f"🌐 Circuit graph saved to: {graphml_path}")

    except Exception as e:
        click.echo(f"❌ Error during neuron-head analysis: {e}")
        if verbose:
            raise


@circuits_cli.command('verify-circuit')
@click.option('--model', '-m', required=True, help='Model name to use for verification')
@click.option('--circuit-file', '-c', required=True, type=click.Path(exists=True),
              help='Path to circuit file (JSON or GraphML)')
@click.option('--prompt', '-p', required=True, help='Test prompt for verification')
@click.option('--target-position', default=-
              1, help='Target token position (-1 for last)')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--device', default='auto', help='Device to use (auto, cpu, cuda)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.pass_context
def verify_circuit(
        ctx,
        model: str,
        circuit_file: str,
        prompt: str,
        target_position: int,
        output: Optional[str],
        device: str,
        verbose: bool):
    """
    Verify a circuit's functional importance using causal interventions.

    This command loads a circuit and systematically ablates its components
    to measure their causal effect on model outputs.
    """
    setup_logging(verbose)

    try:
        # Load model and tokenizer
        model_obj, tokenizer = load_model_and_tokenizer(model, device)

        # Load circuit
        circuit_path = Path(circuit_file)
        if circuit_path.suffix == '.json':
            circuit = NeuralCircuit.from_json_file(circuit_file)
        elif circuit_path.suffix == '.graphml':
            circuit = NeuralCircuit.from_graphml_file(circuit_file)
        else:
            raise ValueError(f"Unsupported circuit file format: {circuit_path.suffix}")

        click.echo(f"📁 Loaded circuit: {circuit.circuit_id}")
        click.echo(f"🔬 Components: {len(circuit._components)}")

        # Create verifier
        verifier = CircuitVerifier(model_obj, tokenizer)

        click.echo(f"🔍 Verifying circuit with prompt: '{prompt}'")
        click.echo(f"📊 Target position: {target_position}")

        # Run verification
        results = verifier.verify_circuit(circuit, prompt, target_position)

        click.echo(f"\n✅ Verification complete!")
        click.echo(f"📈 Overall verification score: {results['verification_score']:.4f}")
        click.echo(f"🎯 Max component effect: {results['max_component_effect']:.4f}")
        click.echo(f"🧮 Components tested: {results['total_components_tested']}")

        # Show top affecting components
        component_effects = results['component_effects']
        sorted_effects = sorted(
            component_effects.items(),
            key=lambda x: x[1]['effect_size'],
            reverse=True
        )

        click.echo(f"\n🏆 Top 5 most important components:")
        for i, (comp_id, effect_data) in enumerate(sorted_effects[:5]):
            click.echo(f"  {i + 1}. {comp_id}")
            click.echo(f"      Effect size: {effect_data['effect_size']:.4f}")
            click.echo(f"      Type: {effect_data['component_type']}")
            click.echo(
                f"      Layer {
                    effect_data['layer']}, Position {
                    effect_data['position']}")

        # Save results
        if output:
            output_path = Path(output)
        else:
            output_path = Path(f"verification_{circuit.circuit_id}.json")

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        click.echo(f"\n💾 Verification results saved to: {output_path}")

    except Exception as e:
        click.echo(f"❌ Error during circuit verification: {e}")
        if verbose:
            raise


# Main CLI entry point for circuits commands
if __name__ == '__main__':
    circuits_cli()
