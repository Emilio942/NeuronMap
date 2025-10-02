"""
CLI commands for SAE (Sparse Auto-Encoder) training, feature analysis, and abstraction tracking.

This module implements CLI commands for:
- C1: SAE training pipeline
- C2: SAE feature extraction and analysis
- C3: Max activating examples analysis
- C4: Abstraction tracking and analysis
"""

from ..utils.config import AnalysisConfig
from ..analysis.model_integration import ModelManager
from ..analysis.abstraction_tracker import AbstractionTracker
from ..analysis.sae_model_hub import SAEModelHub
from ..analysis.sae_feature_analysis import SAEFeatureExtractor, FeatureAnalysis, MaxActivatingExamplesFinder
from ..analysis.sae_training import SAETrainer, SAEConfig
import click
import logging
import json
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any, List
import sys
import torch
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ..utils.structured_logging import get_logger

logger = logging.getLogger("neuronmap.cli.sae")


def setup_logging(verbose: bool):
    """Set up logging configuration."""
    base_logger = get_logger()
    level = logging.DEBUG if verbose else logging.INFO
    base_logger.logger.setLevel(level)
    logger.setLevel(level)

    base_logger.log_system_event(
        event_type="sae_cli_logging_initialized",
        message="SAE CLI logging configured",
        metadata={"verbose": verbose},
        level="DEBUG" if verbose else "INFO"
    )


@click.group(name='sae')
@click.pass_context
def sae_cli(ctx):
    """SAE training, feature analysis, and abstraction tracking commands."""
    if ctx.obj is None:
        ctx.obj = {}


@sae_cli.command('train')
@click.option('--model', '-m', required=True, help='Model name to train SAE on')
@click.option('--layer', '-l', required=True, type=int, help='Layer to train SAE on')
@click.option('--component', '-c', default='mlp',
              type=click.Choice(['mlp', 'attention', 'residual']),
              help='Model component to train SAE on')
@click.option('--dict-size', '-d', default=8192, type=int, help='SAE dictionary size')
@click.option('--sparsity', '-s', default=0.01, type=float,
              help='Target sparsity coefficient')
@click.option('--learning-rate', '-lr', default=1e-4, type=float, help='Learning rate')
@click.option('--batch-size', '-b', default=32, type=int, help='Batch size')
@click.option('--epochs', '-e', default=100, type=int, help='Number of epochs')
@click.option('--output-dir', '-o', type=click.Path(),
              help='Output directory for trained SAE')
@click.option('--device', default='auto', help='Device to use (cpu/cuda/auto)')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def train_sae(ctx, model: str, layer: int, component: str, dict_size: int,
              sparsity: float, learning_rate: float, batch_size: int, epochs: int,
              output_dir: Optional[str], device: str, verbose: bool):
    """Train a Sparse Auto-Encoder on model activations."""
    setup_logging(verbose)

    try:
        click.echo(f"🔧 Training SAE on {model} layer {layer} ({component})")

        # Setup basic configuration
        # Note: Using simple dict config since SAEConfig is available in sae_training
        click.echo(f"🔧 Training SAE on {model} layer {layer} ({component})")

        try:
            # Initialize trainer with basic configuration
            trainer_config = {
                'model_name': model,
                'layer': layer,
                'component': component,
                'dict_size': dict_size,
                'sparsity_coefficient': sparsity,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'epochs': epochs,
                'device': device if device != 'auto' else (
                    'cuda' if torch.cuda.is_available() else 'cpu')}

            # Create SAE config
            # input_dim is a placeholder; it will be determined during activation collection
            sae_config = SAEConfig(
                model_name=model,
                layer=layer,
                component=component,
                input_dim=768,  # Placeholder, will be updated by SAETrainer
                hidden_dim=dict_size,
                sparsity_penalty=sparsity,
                learning_rate=learning_rate,
                batch_size=batch_size,
                num_epochs=epochs,
                output_dir=output_dir if output_dir else "outputs/sae_models",
                model_name_prefix=f"{model}_{layer}_{component}_sae"
            )

            # Initialize trainer
            trainer = SAETrainer(sae_config, device=trainer_config['device'])

            # Train SAE (this is synchronous, not async)
            results = trainer.train(model_manager=ModelManager(device=trainer_config['device']))

            click.echo(f"✅ SAE training completed successfully!")

        except Exception as e:
            logger.error(f"SAE training failed: {e}")
            click.echo(f"❌ Error during training: {e}", err=True)
            sys.exit(1)

        # Save results
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            # Save trained SAE
            sae_path = output_path / f"sae_{model}_{layer}_{component}.pkl"
            trainer.save_sae(str(sae_path))

            # Save training results
            results_path = output_path / \
                f"training_results_{model}_{layer}_{component}.json"
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=2)

            click.echo(f"✅ SAE saved to {sae_path}")
            click.echo(f"📊 Training results saved to {results_path}")

        # Display summary
        click.echo(f"\n📈 Training Summary:")
        click.echo(f"  Final Loss: {results.get('final_loss', 'N/A'):.6f}")
        click.echo(f"  Final Sparsity: {results.get('final_sparsity', 'N/A'):.4f}")
        click.echo(f"  Training Time: {results.get('training_time', 'N/A'):.2f}s")
        click.echo(f"  Dictionary Size: {dict_size}")
        click.echo(f"  Active Features: {results.get('active_features', 'N/A')}")

    except Exception as e:
        logger.error(f"SAE training failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


    @sae_cli.command('analyze-features')
    @click.option('--sae-path', '-s', required=True, type=click.Path(exists=True),
                  help='Path to trained SAE model')
    @click.option('--model', '-m', required=True, help='Original model name')
    @click.option('--top-features', '-t', default=50, type=int,
                  help='Number of top features to analyze')
    @click.option('--max-examples', '-e', default=10, type=int,
                  help='Maximum activating examples per feature')
    @click.option('--output', '-o', type=click.Path(), help='Output file path')
    @click.option('--format', 'output_format',
                  type=click.Choice(['json', 'html', 'both']),
                  default='json', help='Output format')
    @click.option('--threshold', default=0.5, type=float,
                  help='Minimum activation threshold')
    @click.option('--dataset', default='openwebtext', help='Dataset for finding examples')
    @click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
    @click.pass_context
    def analyze_features(ctx, sae_path: str, model: str, top_features: int,
                         max_examples: int, output: Optional[str], output_format: str,
                         threshold: float, dataset: str, verbose: bool):
        """Analyze SAE features and find maximally activating examples."""
        setup_logging(verbose)

        try:
            click.echo(f"🔍 Analyzing SAE features from {sae_path}")

            try:
                # Load SAE from hub
                hub = SAEModelHub()
                sae_info = hub.load_sae(sae_path)

                if not sae_info:
                    click.echo(f"❌ Could not load SAE from {sae_path}", err=True)
                    sys.exit(1)

                # Initialize feature extractor
                sae_model = sae_info['model']
                sae_config = sae_info['config']
                extractor = SAEFeatureExtractor(sae_model, sae_config)

                # Initialize model manager
                model_manager = ModelManager(device=sae_info['device'])

                # Initialize max activating examples finder
                examples_finder = MaxActivatingExamplesFinder(
                    model_manager=model_manager,
                    sae_extractor=extractor
                )

                # Load texts using OpenWebTextLoader
                data_loader = OpenWebTextLoader(num_samples=sae_config.max_sequences)
                texts = data_loader.stream_texts()

                # Perform analysis
                analysis_result = examples_finder.analyze_all_features(
                    texts=texts,
                    feature_ids=list(range(min(top_features, sae_config.hidden_dim))),
                    top_k_examples=max_examples
                )

                click.echo(f"✅ Feature analysis completed!")

            except Exception as e:
                logger.error(f"Feature analysis failed: {e}")
                click.echo(f"❌ Error during analysis: {e}", err=True)
                sys.exit(1)

            # Generate output
            timestamp = Path().cwd().name
            default_output = f"sae_feature_analysis_{analysis_result.model_name}_layer{analysis_result.layer_index}"

            if output_format in ['json', 'both']:
                json_path = Path(output or f"{default_output}.json")
                with open(json_path, 'w') as f:
                    json.dump(analysis_result.to_dict(), f, indent=2, default=str)
                click.echo(f"📊 JSON results saved to {json_path}")

            if output_format in ['html', 'both']:
                html_path = Path(output or f"{default_output}.html")
                # Generate simple HTML report
                html_content = f"""
                <html>
                <head><title>SAE Feature Analysis Report</title></head>
                <body>
                <h1>SAE Feature Analysis Report</h1>
                <p>Model: {analysis_result.model_name}, Layer: {analysis_result.layer_index}</p>
                <p>Top Features: {len(analysis_result.feature_analyses)}</p>
                <h2>Features:</h2>
                <ul>
                """
                for feature in analysis_result.feature_analyses[:10]:  # Show top 10
                    html_content += f"<li>Feature {feature.feature_id}: {feature.max_activation:.4f} max activation</li>"
                html_content += "</ul></body></html>"

                with open(html_path, 'w') as f:
                    f.write(html_content)
                click.echo(f"📄 HTML report saved to {html_path}")

            # Display summary
            click.echo(f"\n🎯 Feature Analysis Summary:")
            click.echo(f"  Total Features Analyzed: {len(analysis_result.feature_analyses)}")
            click.echo(f"  Mean Sparsity: {analysis_result.global_statistics.get('mean_sparsity', 'N/A'):.4f}")
            click.echo(f"  Mean Max Activation: {analysis_result.global_statistics.get('mean_max_activation', 'N/A'):.4f}")

            # Show top features
            if analysis_result.feature_analyses:
                click.echo(f"\n🏆 Top 5 Features:")
                for i, feature in enumerate(analysis_result.feature_analyses[:5]):
                    click.echo(f"  {i + 1}. Feature {feature.feature_id}: "
                               f"Score {feature.max_activation:.4f}, "
                               f"Examples: {len(feature.max_activating_examples)}")

        except Exception as e:
            logger.error(f"Feature analysis failed: {e}")
            click.echo(f"❌ Error: {e}", err=True)
            sys.exit(1)


@sae_cli.command('find-examples')
@click.option('--sae-path', '-s', required=True, type=click.Path(exists=True),
              help='Path to trained SAE model')
@click.option('--feature-ids', '-f', help='Comma-separated feature IDs to analyze')
@click.option('--max-examples', '-e', default=20, type=int,
              help='Maximum examples per feature')
@click.option('--min-activation', default=1.0, type=float,
              help='Minimum activation threshold')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def find_examples(
        ctx,
        sae_path: str,
        feature_ids: Optional[str],
        max_examples: int,
        min_activation: float,
        output: Optional[str],
        verbose: bool):
    """Find examples that maximally activate specific SAE features."""
    # Placeholder for find_examples logic
    click.echo("Find examples command is not yet implemented.")


@sae_cli.command('track-abstractions')
@click.option('--model', '-m', required=True, help='Model name to analyze')
@click.option('--prompt', '-p', required=True, help='Input prompt to analyze')
@click.option('--token-indices', '-t', type=str, help='Comma-separated token indices to track (e.g., "0,1,5")')
@click.option('--output', '-o', type=click.Path(), help='Output file path')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'html', 'both']),
              default='json', help='Output format')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def track_abstractions(ctx, model: str, prompt: str, token_indices: Optional[str],
                       output: Optional[str],
                       output_format: str, verbose: bool):
    """Track how abstractions evolve across model layers."""
    setup_logging(verbose)

    try:
        click.echo(f"🔍 Tracking abstractions for model {model}")
        click.echo(f"📝 Prompt: {prompt[:100]}...")

        # Initialize model manager
        from ..core.model_manager import ModelManager
        from ..analysis.abstraction_tracker import AbstractionTracker
        
        model_manager = ModelManager()
        model_manager.load_model(model)

        # Initialize abstraction tracker
        tracker = AbstractionTracker(model_manager)

        # Parse token indices
        parsed_token_indices = None
        if token_indices:
            parsed_token_indices = [int(x.strip()) for x in token_indices.split(',')]

        # For now, just show placeholder
        click.echo("� Abstraction tracking feature is under development")
        click.echo("✅ This will track concept evolution across layers")

    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@sae_cli.command('list-models')
@click.option('--hub-path', default='~/.neuronmap/sae_models',
              help='Path to SAE model hub')
@click.option('--model-filter', help='Filter by original model name')
@click.option('--layer-filter', type=int, help='Filter by layer')
@click.option('--component-filter',
              type=click.Choice(['mlp', 'attention', 'residual']),
              help='Filter by component type')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def list_models(ctx, hub_path: str, model_filter: Optional[str],
                layer_filter: Optional[int], component_filter: Optional[str],
                verbose: bool):
    """List available SAE models in the model hub."""
    setup_logging(verbose)

    try:
        click.echo(f"📚 Listing SAE models from {hub_path}")

        # Initialize model hub
        hub = SAEModelHub(hub_path)

        # List models with filters
        models = hub.list_models(
            base_model_name=model_filter,
            layer_index=layer_filter,
            tags=None
        )

        if not models:
            click.echo("🤷 No SAE models found matching the criteria")
            return

        click.echo(f"\n🔍 Found {len(models)} SAE models:")

        for model_info in models:
            click.echo(f"\n📦 {model_info['name']}")
            click.echo(f"  Model: {model_info['original_model']}")
            click.echo(f"  Layer: {model_info['layer']}")
            click.echo(f"  Component: {model_info['component']}")
            click.echo(f"  Dictionary Size: {model_info['dict_size']}")
            click.echo(f"  Created: {model_info['created_at']}")
            click.echo(f"  Path: {model_info['path']}")

            if verbose and model_info.get('metrics'):
                metrics = model_info['metrics']
                click.echo(f"  Final Loss: {metrics.get('final_loss', 'N/A')}")
                click.echo(f"  Final Sparsity: {metrics.get('final_sparsity', 'N/A')}")
                click.echo(
                    f"  Active Features: {
                        metrics.get(
                            'active_features',
                            'N/A')}")

    except Exception as e:
        logger.error(f"Model listing failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@sae_cli.command('export-features')
@click.option('--sae-path', '-s', required=True, type=click.Path(exists=True),
              help='Path to trained SAE model')
@click.option('--output', '-o', type=click.Path(), required=True,
              help='Output file path')
@click.option('--format', 'output_format',
              type=click.Choice(['json', 'csv', 'npy']),
              default='json', help='Export format')
@click.option('--include-weights', is_flag=True,
              help='Include SAE weights in export')
@click.option('--include-biases', is_flag=True,
              help='Include SAE biases in export')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
@click.pass_context
def export_features(ctx, sae_path: str, output: str, output_format: str,
                    include_weights: bool, include_biases: bool, verbose: bool):
    """Export SAE features and weights to various formats."""
    setup_logging(verbose)

    try:
        click.echo(f"📤 Exporting SAE features from {sae_path}")

        # Initialize model hub for loading
        hub = SAEModelHub()

        # Load SAE
        sae_info = hub.load_sae(sae_path)

        # Prepare export data
        export_data = {
            'metadata': sae_info['metadata'],
            'features': sae_info.get('features', [])
        }

        if include_weights and 'weights' in sae_info:
            export_data['weights'] = sae_info['weights']

        if include_biases and 'biases' in sae_info:
            export_data['biases'] = sae_info['biases']

        # Export based on format
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == 'json':
            with open(output_path, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)

        elif output_format == 'csv':
            import pandas as pd
            # Export metadata and feature info as CSV
            df = pd.DataFrame(export_data['features'])
            df.to_csv(output_path, index=False)

        elif output_format == 'npy':
            import numpy as np
            # Export weights as numpy array
            if 'weights' in export_data:
                np.save(output_path, export_data['weights'])
            else:
                raise ValueError("No weights available for numpy export")

        click.echo(f"✅ Features exported to {output_path}")

        # Display export summary
        click.echo(f"\n📊 Export Summary:")
        click.echo(f"  Format: {output_format}")
        click.echo(f"  Features: {len(export_data.get('features', []))}")
        click.echo(f"  Includes Weights: {'✅' if include_weights else '❌'}")
        click.echo(f"  Includes Biases: {'✅' if include_biases else '❌'}")

    except Exception as e:
        logger.error(f"Feature export failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    sae_cli()
