"""
NeuronMap CLI - Command Line Interface for Model Surgery & Path Analysis

This module implements the CLI commands for the intervention system,
including C1: CLI-Befehl `analyze:ablate` and C2: CLI-Befehl `analyze:patch`
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

import torch

from ..analysis.interventions import run_with_ablation, run_with_patching, calculate_causal_effect
from ..analysis.intervention_cache import InterventionCache, create_clean_run_cache
from ..analysis.intervention_config import (
    ConfigurationManager,
    AblationExperimentConfig,
    PathPatchingExperimentConfig,
    generate_config_template,
    validate_config_file
)
from ..analysis.base_model_handler import ModelFactory
from ..utils.error_handling import NeuronMapError
from ..utils.structured_logging import get_logger


logger = logging.getLogger("neuronmap.cli.intervention")


class InterventionCLI:
    """Command Line Interface for intervention experiments."""

    def __init__(self):
        self.parser = self._create_parser()
        self.cache = None
        self.model_factory = ModelFactory()

    def _create_parser(self) -> argparse.ArgumentParser:
        """Create the main argument parser."""
        parser = argparse.ArgumentParser(
            prog="neuronmap",
            description="NeuronMap: Neural Network Interpretability Toolkit",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        # Global options
        parser.add_argument(
            "--verbose", "-v",
            action="store_true",
            help="Enable verbose logging"
        )
        parser.add_argument(
            "--config",
            type=Path,
            help="Path to global configuration file"
        )
        parser.add_argument(
            "--cache-dir",
            type=Path,
            help="Override cache directory"
        )

        # Create subcommands
        subparsers = parser.add_subparsers(
            dest="command",
            help="Available commands",
            metavar="COMMAND"
        )

        # Add intervention analysis commands
        self._add_analyze_commands(subparsers)

        # Add utility commands
        self._add_utility_commands(subparsers)

        return parser

    def _add_analyze_commands(self, subparsers):
        """Add analysis subcommands."""
        analyze_parser = subparsers.add_parser(
            "analyze",
            help="Run analysis experiments",
            description="Perform various neural network analysis experiments"
        )

        analyze_subparsers = analyze_parser.add_subparsers(
            dest="analyze_command",
            help="Analysis types",
            metavar="TYPE"
        )

        # C1: CLI-Befehl `analyze:ablate`
        self._add_ablate_command(analyze_subparsers)

        # C2: CLI-Befehl `analyze:patch`
        self._add_patch_command(analyze_subparsers)

    def _add_ablate_command(self, subparsers):
        """Add ablation analysis command."""
        ablate_parser = subparsers.add_parser(
            "ablate",
            help="Run ablation experiments",
            description="Analyze neural network behavior by selectively disabling components")

        # Required arguments
        ablate_parser.add_argument(
            "--model",
            required=True,
            help="Model name or path (e.g., 'gpt2', 'bert-base-uncased')"
        )
        ablate_parser.add_argument(
            "--prompt",
            required=True,
            help="Input prompt to analyze"
        )
        ablate_parser.add_argument(
            "--layer",
            required=True,
            help="Layer name to ablate (e.g., 'transformer.h.8.mlp')"
        )

        # Optional arguments
        ablate_parser.add_argument(
            "--neurons",
            type=str,
            help="Comma-separated list of neuron indices to ablate (default: all)"
        )
        ablate_parser.add_argument(
            "--config-file",
            type=Path,
            help="YAML configuration file for complex experiments"
        )
        ablate_parser.add_argument(
            "--output",
            type=Path,
            default=Path("./outputs/ablation"),
            help="Output directory"
        )
        ablate_parser.add_argument(
            "--output-format",
            choices=["json", "csv", "yaml"],
            default="json",
            help="Output format"
        )
        ablate_parser.add_argument(
            "--device",
            default="auto",
            help="Device to run on (auto, cpu, cuda, cuda:0, etc.)"
        )
        ablate_parser.add_argument(
            "--save-activations",
            action="store_true",
            help="Save intermediate activations"
        )
        ablate_parser.add_argument(
            "--baseline-comparison",
            action="store_true",
            default=True,
            help="Compare against baseline (no ablation)"
        )

    def _add_patch_command(self, subparsers):
        """Add path patching command."""
        patch_parser = subparsers.add_parser(
            "patch",
            help="Run path patching experiments",
            description="Analyze causal relationships through activation patching"
        )

        # Config-based approach (recommended)
        patch_parser.add_argument(
            "--config",
            type=Path,
            required=True,
            help="YAML configuration file for patching experiment"
        )

        # Output options
        patch_parser.add_argument(
            "--output",
            type=Path,
            help="Override output directory from config"
        )
        patch_parser.add_argument(
            "--output-format",
            choices=["json", "csv", "yaml"],
            default="json",
            help="Output format"
        )
        patch_parser.add_argument(
            "--device",
            help="Override device from config"
        )
        patch_parser.add_argument(
            "--experiment-id",
            help="Override experiment ID from config"
        )

    def _add_utility_commands(self, subparsers):
        """Add utility commands."""
        # Config generation
        config_parser = subparsers.add_parser(
            "generate-config",
            help="Generate configuration templates"
        )
        config_parser.add_argument(
            "type",
            choices=["ablation", "patching"],
            help="Type of configuration to generate"
        )
        config_parser.add_argument(
            "--output",
            type=Path,
            help="Output file path"
        )

        # Config validation
        validate_parser = subparsers.add_parser(
            "validate-config",
            help="Validate configuration files"
        )
        validate_parser.add_argument(
            "config_file",
            type=Path,
            help="Configuration file to validate"
        )
        validate_parser.add_argument(
            "type",
            choices=["ablation", "patching"],
            help="Type of configuration"
        )

        # Cache management
        cache_parser = subparsers.add_parser(
            "cache",
            help="Manage intervention cache"
        )
        cache_subparsers = cache_parser.add_subparsers(
            dest="cache_command",
            help="Cache operations"
        )

        # Cache info
        cache_subparsers.add_parser(
            "info",
            help="Show cache information"
        )

        # Cache cleanup
        cleanup_parser = cache_subparsers.add_parser(
            "cleanup",
            help="Clean up cache"
        )
        cleanup_parser.add_argument(
            "--experiment-id",
            help="Clean specific experiment"
        )
        cleanup_parser.add_argument(
            "--all",
            action="store_true",
            help="Clean all cache entries"
        )

    def _setup_environment(self, args):
        """Setup logging and environment based on arguments."""
        base_logger = get_logger()
        level = logging.DEBUG if args.verbose else logging.INFO
        base_logger.logger.setLevel(level)
        logger.setLevel(level)

        base_logger.log_system_event(
            event_type="intervention_cli_environment_setup",
            message="Intervention CLI environment initialized",
            metadata={
                "verbose": args.verbose,
                "cache_dir": str(args.cache_dir) if getattr(args, "cache_dir", None) else None
            },
            level="DEBUG" if args.verbose else "INFO"
        )

        # Initialize cache
        cache_dir = args.cache_dir if hasattr(
            args, 'cache_dir') and args.cache_dir else None
        self.cache = InterventionCache(cache_dir=cache_dir)

    def run_ablation_experiment(self, args) -> Dict[str, Any]:
        """Run ablation experiment - implements C1: CLI-Befehl `analyze:ablate`."""
        logger.info(f"Starting ablation experiment on {args.model}")

        # Load configuration if provided
        if args.config_file:
            config = ConfigurationManager.load_ablation_config(args.config_file)
            # CLI args override config
            model_name = args.model
            prompts = [args.prompt]
            layer_name = args.layer
            output_dir = args.output
        else:
            # Use CLI arguments directly
            model_name = args.model
            prompts = [args.prompt]
            layer_name = args.layer
            output_dir = args.output

        # Parse neuron indices
        neuron_indices = None
        if args.neurons:
            try:
                neuron_indices = [int(x.strip()) for x in args.neurons.split(',')]
            except ValueError:
                raise NeuronMapError(
                    "Invalid neuron indices format. Use comma-separated integers.")

        # Setup device
        device = torch.device(args.device if args.device != "auto" else
                              ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load model
        logger.info(f"Loading model: {model_name}")
        model = self.model_factory.create_model(model_name, device=device)
        tokenizer = self.model_factory.get_tokenizer(model_name)

        results = []

        for prompt in prompts:
            logger.info(f"Processing prompt: {prompt[:50]}...")

            # Tokenize input
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
            input_tensor = inputs.input_ids.to(device)

            # Run baseline (no ablation) if requested
            baseline_result = None
            if args.baseline_comparison:
                logger.info("Running baseline (no ablation)")
                model.eval()
                with torch.no_grad():
                    baseline_output = model(input_tensor)
                baseline_result = {
                    'output': baseline_output,
                    'prompt': prompt
                }

            # Run ablation
            logger.info(f"Running ablation on layer: {layer_name}")
            ablation_result = run_with_ablation(
                model=model,
                input_tensor=input_tensor,
                layer_name=layer_name,
                neuron_indices=neuron_indices,
                return_activations=args.save_activations
            )

            # Calculate effect if baseline available
            effect_size = None
            if baseline_result:
                effect_size = calculate_causal_effect(
                    clean_output=baseline_result['output'],
                    corrupted_output=ablation_result['output'],
                    # Same as corrupted for ablation
                    patched_output=ablation_result['output']
                )

            # Collect results
            result = {
                'prompt': prompt,
                'model': model_name,
                'layer': layer_name,
                'ablated_neurons': neuron_indices,
                'effect_size': effect_size,
                'baseline_output': baseline_result,
                'ablation_output': ablation_result
            }

            if args.save_activations and 'activations' in ablation_result:
                result['activations'] = {
                    k: v.tolist() if isinstance(v, torch.Tensor) else v
                    for k, v in ablation_result['activations'].items()
                }

            results.append(result)

        # Save results
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.output_format == "json":
            output_file = output_dir / "ablation_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

        logger.info(f"Ablation experiment completed. Results saved to {output_dir}")

        # C3: Ausgabeformatierung - Clear CLI output
        self._print_ablation_summary(results)

        return results

    def _print_ablation_summary(self, results: List[Dict[str, Any]]):
        """Print clear summary of ablation results."""
        print("\n" + "=" * 60)
        print("ABLATION EXPERIMENT RESULTS")
        print("=" * 60)

        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Prompt: {result['prompt'][:60]}...")
            print(f"  Model: {result['model']}")
            print(f"  Ablated Layer: {result['layer']}")
            print(f"  Ablated Neurons: {result['ablated_neurons'] or 'ALL'}")

            if result['effect_size'] is not None:
                print(f"  Effect Size: {result['effect_size']:.4f}")

                # Interpret effect size
                if result['effect_size'] > 0.5:
                    interpretation = "LARGE effect - this component is critical"
                elif result['effect_size'] > 0.2:
                    interpretation = "MEDIUM effect - this component is important"
                elif result['effect_size'] > 0.05:
                    interpretation = "SMALL effect - this component has minor influence"
                else:
                    interpretation = "MINIMAL effect - this component is not critical"

                print(f"  Interpretation: {interpretation}")

        print("\n" + "=" * 60)

    def run_patching_experiment(self, args) -> Dict[str, Any]:
        """Run path patching experiment - implements C2: CLI-Befehl `analyze:patch`."""
        logger.info("Starting path patching experiment")

        # Load configuration
        config = ConfigurationManager.load_patching_config(args.config)

        # Apply CLI overrides
        if args.output:
            config.output.output_dir = args.output
        if args.device:
            config.model.device = args.device
        if args.experiment_id:
            config.cache.experiment_id = args.experiment_id

        # Setup device
        device = torch.device(config.model.device if config.model.device != "auto" else
                              ("cuda" if torch.cuda.is_available() else "cpu"))

        # Load model
        logger.info(f"Loading model: {config.model.name}")
        model = self.model_factory.create_model(config.model.name, device=device)
        tokenizer = self.model_factory.get_tokenizer(config.model.name)

        results = []

        # Process each prompt pair
        for clean_prompt, corrupted_prompt in zip(
            config.inputs.clean_prompts,
            config.inputs.corrupted_prompts
        ):
            logger.info(f"Processing: '{clean_prompt}' vs '{corrupted_prompt}'")

            # Tokenize inputs
            clean_inputs = tokenizer(clean_prompt, return_tensors="pt", truncation=True)
            corrupted_inputs = tokenizer(
                corrupted_prompt, return_tensors="pt", truncation=True)

            clean_tensor = clean_inputs.input_ids.to(device)
            corrupted_tensor = corrupted_inputs.input_ids.to(device)

            # Extract layer names for patching
            patch_specs = []
            for target in config.patch_targets:
                # Simple implementation - assumes explicit layer names
                layer_names = target.layer_selection.names or []
                for layer_name in layer_names:
                    patch_specs.append((layer_name, None))  # None = patch all neurons

            # Run path patching
            patching_result = run_with_patching(
                model=model,
                clean_input=clean_tensor,
                corrupted_input=corrupted_tensor,
                patch_specs=patch_specs
            )

            # Calculate causal effects
            effect_size = calculate_causal_effect(
                clean_output=patching_result['clean_output'],
                corrupted_output=patching_result['corrupted_output'],
                patched_output=patching_result['patched_output']
            )

            result = {
                'clean_prompt': clean_prompt,
                'corrupted_prompt': corrupted_prompt,
                'patch_specs': patch_specs,
                'effect_size': effect_size,
                'patching_result': patching_result
            }

            results.append(result)

        # Save results
        output_dir = config.output.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        if args.output_format == "json":
            output_file = output_dir / "patching_results.json"
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)

        logger.info(
            f"Path patching experiment completed. Results saved to {output_dir}")

        # Print summary
        self._print_patching_summary(results)

        return results

    def _print_patching_summary(self, results: List[Dict[str, Any]]):
        """Print clear summary of patching results."""
        print("\n" + "=" * 60)
        print("PATH PATCHING EXPERIMENT RESULTS")
        print("=" * 60)

        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Clean Prompt: {result['clean_prompt']}")
            print(f"  Corrupted Prompt: {result['corrupted_prompt']}")
            print(f"  Patched Layers: {[spec[0] for spec in result['patch_specs']]}")
            print(f"  Causal Effect: {result['effect_size']:.4f}")

            # Interpret causal effect
            if result['effect_size'] > 0.7:
                interpretation = "STRONG causal relationship - patching mostly recovers clean behavior"
            elif result['effect_size'] > 0.4:
                interpretation = "MODERATE causal relationship - patching partially recovers behavior"
            elif result['effect_size'] > 0.1:
                interpretation = "WEAK causal relationship - minimal recovery from patching"
            else:
                interpretation = "NO causal relationship - patching has little effect"

            print(f"  Interpretation: {interpretation}")

        print("\n" + "=" * 60)

    def handle_utility_commands(self, args):
        """Handle utility commands."""
        if args.command == "generate-config":
            template = generate_config_template(args.type)

            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w') as f:
                    f.write(template)
                print(f"Configuration template saved to {args.output}")
            else:
                print(template)

        elif args.command == "validate-config":
            is_valid = validate_config_file(args.config_file, args.type)
            if is_valid:
                print(f"✓ Configuration file {args.config_file} is valid")
                return True
            else:
                print(f"✗ Configuration file {args.config_file} is invalid")
                return False

        elif args.command == "cache":
            if args.cache_command == "info":
                info = self.cache.get_cache_info()
                print("Cache Information:")
                print(f"  Memory cache entries: {info['memory_cache_size']}")
                print(f"  Memory usage: {info['memory_usage_mb']:.1f}MB / {info['memory_limit_mb']:.1f}MB")
                print(f"  Disk cache entries: {info['disk_cache_size']}")
                print(f"  Disk usage: {info['total_disk_size_mb']:.1f}MB")
                print(f"  Cache directory: {info['cache_dir']}")

            elif args.cache_command == "cleanup":
                if args.all:
                    self.cache.clear_memory_cache()
                    self.cache.clear_disk_cache()
                    print("Cleared all cache entries")
                elif args.experiment_id:
                    self.cache.clear_disk_cache(args.experiment_id)
                    print(f"Cleared cache for experiment: {args.experiment_id}")

    def run(self, args_list: Optional[List[str]] = None):
        """Main entry point for CLI."""
        args = self.parser.parse_args(args_list)

        # Setup environment
        self._setup_environment(args)

        try:
            if args.command == "analyze":
                if args.analyze_command == "ablate":
                    return self.run_ablation_experiment(args)
                elif args.analyze_command == "patch":
                    return self.run_patching_experiment(args)
                else:
                    self.parser.error(
                        "Please specify an analysis type (ablate or patch)")

            elif args.command in ["generate-config", "validate-config", "cache"]:
                return self.handle_utility_commands(args)

            else:
                self.parser.print_help()

        except KeyboardInterrupt:
            logger.info("Interrupted by user")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Error: {e}")
            if args.verbose:
                raise
            sys.exit(1)


def main():
    """Entry point for the neuronmap CLI."""
    cli = InterventionCLI()
    cli.run()


if __name__ == "__main__":
    main()
