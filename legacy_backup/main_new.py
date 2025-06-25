#!/usr/bin/env python3
"""
NeuronMap - Modular Neural Network Analysis System
=================================================

This is the main entry point for the NeuronMap neural network analysis system.
It provides a command-line interface for running various analysis tasks.
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from utils.config_manager import ConfigManager, get_config
from analysis.activation_analyzer import ActivationAnalyzer
from analysis.multi_model_analyzer import MultiModelAnalyzer
from analysis.advanced_analyzer import AdvancedAnalyzer
from analysis.advanced_analytics import AdvancedAnalyticsEngine
from data_processing.question_loader import QuestionLoader
from visualization.activation_visualizer import ActivationVisualizer

def setup_logging(config):
    """Setup logging based on configuration."""
    level = getattr(logging, config.experiment.logging_level.upper())
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('neuronmap.log', mode='a')
        ]
    )
    return logging.getLogger(__name__)

def create_argument_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="NeuronMap Neural Network Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --analyze                           # Run basic activation analysis
  %(prog)s --config configs/custom.yaml       # Use custom configuration
  %(prog)s --model gpt2 --device cuda         # Override model settings
  %(prog)s --list-layers                      # List available model layers
  %(prog)s --visualize                        # Generate visualizations
        """
    )
    
    # Configuration
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (YAML/JSON)'
    )
    
    # Analysis options
    parser.add_argument(
        '--analyze', '-a',
        action='store_true',
        help='Run activation analysis'
    )
    
    parser.add_argument(
        '--multi-model', '-m',
        action='store_true',
        help='Run multi-model comparative analysis'
    )
    
    parser.add_argument(
        '--models',
        nargs='+',
        help='List of models for multi-model analysis (e.g., gpt2 distilgpt2)'
    )
    
    parser.add_argument(
        '--advanced', 
        action='store_true',
        help='Run advanced analysis (clustering, dimensionality, statistics)'
    )
    
    parser.add_argument(
        '--advanced-analytics',
        action='store_true',
        help='Run comprehensive advanced analytics (attention flow, gradient attribution, cross-layer analysis)'
    )
    
    parser.add_argument(
        '--visualize', '-v',
        action='store_true',
        help='Generate visualizations'
    )
    
    parser.add_argument(
        '--list-layers',
        action='store_true',
        help='List all available layers in the model'
    )
    
    # Model overrides
    parser.add_argument(
        '--model',
        type=str,
        help='Model name to use (overrides config)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cpu', 'cuda'],
        help='Device to use (overrides config)'
    )
    
    parser.add_argument(
        '--input-file',
        type=str,
        help='Input file with questions (overrides config)'
    )
    
    parser.add_argument(
        '--output-file',
        type=str,
        help='Output file for results (overrides config)'
    )
    
    # Analysis parameters
    parser.add_argument(
        '--target-layers',
        nargs='+',
        help='Target layers to analyze (overrides config)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        help='Batch size for processing (overrides config)'
    )
    
    # Verbosity
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Minimize output'
    )
    
    return parser

def apply_cli_overrides(config, args):
    """Apply command line argument overrides to configuration."""
    updates = {}
    
    # Model overrides
    if args.model:
        updates.setdefault('model', {})['name'] = args.model
    
    if args.device:
        updates.setdefault('model', {})['device'] = args.device
    
    if args.target_layers:
        updates.setdefault('model', {})['target_layers'] = args.target_layers
    
    if args.batch_size:
        updates.setdefault('model', {})['batch_size'] = args.batch_size
    
    # Data overrides
    if args.input_file:
        updates.setdefault('data', {})['input_file'] = args.input_file
    
    if args.output_file:
        updates.setdefault('data', {})['output_file'] = args.output_file
    
    # Logging overrides
    if args.verbose:
        updates.setdefault('experiment', {})['logging_level'] = 'DEBUG'
    elif args.quiet:
        updates.setdefault('experiment', {})['logging_level'] = 'WARNING'
    
    if updates:
        config_manager = ConfigManager()
        config_manager.update_config(**updates)
        return config_manager.get_config()
    
    return config

def list_model_layers(config, logger):
    """List all available layers in the model."""
    logger.info("Loading model to list available layers...")
    
    try:
        analyzer = ActivationAnalyzer(config)
        layers = analyzer.get_model_layers()
        
        print("\n" + "="*60)
        print("AVAILABLE MODEL LAYERS")
        print("="*60)
        
        for i, layer_name in enumerate(layers, 1):
            print(f"{i:3d}. {layer_name}")
        
        print("="*60)
        print(f"Total: {len(layers)} layers found")
        print("\nYou can use these layer names in your configuration file")
        print("under 'model.target_layers' or with --target-layers argument.")
        
    except Exception as e:
        logger.error(f"Failed to load model and list layers: {e}")
        return False
    
    return True

def run_advanced_analysis(config, logger):
    """Run advanced analysis methods."""
    logger.info("Starting advanced analysis...")
    
    try:
        # Load questions
        logger.info("Loading questions...")
        question_loader = QuestionLoader(config)
        questions = question_loader.load_questions()
        
        if not questions:
            logger.error("No questions loaded. Cannot proceed.")
            return False
        
        # Limit questions for advanced analysis 
        max_questions = min(len(questions), 10)
        questions = questions[:max_questions]
        logger.info(f"Using {len(questions)} questions for advanced analysis")
        
        # Get activations first
        logger.info("Extracting activations...")
        analyzer = ActivationAnalyzer(config)
        results = analyzer.analyze_questions(questions)
        
        # Extract activation arrays
        layer_activations = {}
        for result in results:
            if result.get('success', False):
                for layer_name, activation in result.get('activations', {}).items():
                    if layer_name not in layer_activations:
                        layer_activations[layer_name] = []
                    layer_activations[layer_name].append(activation)
        
        # Convert to numpy arrays
        import numpy as np
        for layer_name in layer_activations:
            layer_activations[layer_name] = np.array(layer_activations[layer_name])
        
        if not layer_activations:
            logger.error("No activations extracted for advanced analysis")
            return False
        
        logger.info(f"Extracted activations from {len(layer_activations)} layers")
        
        # Initialize advanced analyzer
        advanced_analyzer = AdvancedAnalyzer(config)
        analyses = {}
        
        # Use first layer for individual analyses
        first_layer = list(layer_activations.keys())[0]
        first_activations = layer_activations[first_layer]
        
        logger.info(f"Running advanced analysis on layer: {first_layer}")
        logger.info(f"Activation shape: {first_activations.shape}")
        
        # Clustering analysis
        logger.info("Running clustering analysis...")
        analyses['clustering'] = advanced_analyzer.cluster_activations(first_activations)
        
        # Dimensionality analysis
        logger.info("Running dimensionality analysis...")
        analyses['dimensionality'] = advanced_analyzer.dimensionality_analysis(first_activations)
        
        # Statistical analysis
        logger.info("Running statistical analysis...")
        analyses['statistics'] = advanced_analyzer.statistical_analysis(first_activations, first_layer)
        
        # Similarity analysis (if multiple layers)
        if len(layer_activations) > 1:
            logger.info("Running similarity analysis...")
            analyses['similarity'] = advanced_analyzer.activation_similarity_analysis(layer_activations)
        
        # Generate report
        logger.info("Generating advanced analysis report...")
        advanced_analyzer.generate_advanced_report(analyses)
        
        # Log summary
        logger.info("Advanced Analysis Summary:")
        if 'clustering' in analyses:
            clustering = analyses['clustering']
            logger.info(f"  Clustering: {clustering.get('n_clusters', 'N/A')} clusters found")
        
        if 'dimensionality' in analyses:
            dim = analyses['dimensionality']
            logger.info(f"  Effective dimensionality: {dim.get('effective_dimensionality_95', 'N/A')}")
            logger.info(f"  Sparsity: {dim.get('sparsity', 0):.2%}")
        
        logger.info("Advanced analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Advanced analysis failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def run_multi_model_analysis(config, logger, model_names=None):
    """Run multi-model comparative analysis."""
    logger.info("Starting multi-model analysis...")
    
    try:
        # Default models if none specified
        if not model_names:
            model_names = ["distilgpt2", "distilbert-base-uncased"]
            logger.info(f"Using default models: {model_names}")
        
        # Load questions
        logger.info("Loading questions...")
        question_loader = QuestionLoader(config)
        questions = question_loader.load_questions()
        
        if not questions:
            logger.error("No questions loaded. Cannot proceed.")
            return False
        
        # Limit questions for multi-model analysis (to manage time/resources)
        max_questions = min(len(questions), 5)  # Limit to 5 for demo
        questions = questions[:max_questions]
        logger.info(f"Using {len(questions)} questions for multi-model analysis")
        
        # Initialize multi-model analyzer
        logger.info("Initializing multi-model analyzer...")
        analyzer = MultiModelAnalyzer(config)
        
        # Add models
        successful_models = []
        for model_name in model_names:
            logger.info(f"Adding model: {model_name}")
            if analyzer.add_model(model_name):
                successful_models.append(model_name)
            else:
                logger.warning(f"Failed to add model: {model_name}")
        
        if not successful_models:
            logger.error("No models successfully loaded")
            return False
        
        # Run analysis
        logger.info(f"Running analysis across {len(successful_models)} models...")
        results = analyzer.analyze_multiple_models(questions, successful_models)
        
        if not results:
            logger.error("Multi-model analysis failed to produce results")
            return False
        
        # Generate comparison
        logger.info("Generating model comparison...")
        comparison = analyzer.compare_models(results)
        
        # Save results
        logger.info("Saving multi-model results...")
        analyzer.save_multi_model_results(results, comparison)
        
        # Print summary
        logger.info("Multi-model Analysis Summary:")
        logger.info(f"  Models analyzed: {len(results)}")
        for model_name, model_data in results.items():
            stats = model_data.get('statistics', {})
            success_rate = stats.get('success_rate', 0)
            analysis_time = stats.get('analysis_time', 0)
            logger.info(f"  {model_name}: {success_rate:.1%} success, {analysis_time:.1f}s")
        
        logger.info("Multi-model analysis completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Multi-model analysis failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def run_analysis(config, logger):
    """Run the main activation analysis."""
    logger.info("Starting activation analysis...")
    
    try:
        # Load questions
        logger.info("Loading questions...")
        question_loader = QuestionLoader(config)
        questions = question_loader.load_questions()
        
        if not questions:
            logger.error("No questions loaded. Cannot proceed.")
            return False
        
        logger.info(f"Loaded {len(questions)} questions")
        
        # Run analysis
        logger.info("Initializing activation analyzer...")
        analyzer = ActivationAnalyzer(config)
        
        logger.info("Starting activation extraction...")
        results = analyzer.analyze_questions(questions)
        
        if not results:
            logger.error("Analysis failed to produce results.")
            return False
        
        # Save results
        logger.info("Saving results...")
        analyzer.save_results(results)
        
        logger.info(f"Analysis completed successfully! Results saved to {config.data.output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def run_visualization(config, logger):
    """Generate visualizations of the results."""
    logger.info("Starting visualization generation...")
    
    try:
        if not Path(config.data.output_file).exists():
            logger.error(f"Results file {config.data.output_file} not found. Run analysis first.")
            return False
        
        visualizer = ActivationVisualizer(config)
        visualizer.generate_all_visualizations()
        
        logger.info("Visualizations generated successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def run_advanced_analytics(config, logger):
    """Run comprehensive advanced analytics including attention flow, gradient attribution, etc."""
    logger.info("Starting comprehensive advanced analytics...")
    
    try:
        # First run basic analysis to get activations
        logger.info("Running basic analysis to extract activations...")
        
        # Load questions
        question_loader = QuestionLoader(config)
        questions = question_loader.load_questions()
        
        if not questions:
            logger.error("No questions loaded. Cannot proceed.")
            return False
        
        logger.info(f"Loaded {len(questions)} questions")
        
        # Initialize activation analyzer
        analyzer = ActivationAnalyzer(config)
        
        # Extract activations for advanced analytics
        results = analyzer.analyze_questions(questions)
        
        if not results:
            logger.error("Failed to extract activations for advanced analytics")
            return False
        
        # Convert results to activation format for advanced analytics
        layer_activations = {}
        for result in results:
            if 'activations' in result:
                for layer_name, activation in result['activations'].items():
                    if layer_name not in layer_activations:
                        layer_activations[layer_name] = []
                    layer_activations[layer_name].append(activation)
        
        # Convert to numpy arrays
        import numpy as np
        for layer_name in layer_activations:
            layer_activations[layer_name] = np.array(layer_activations[layer_name])
        
        if not layer_activations:
            logger.error("No activations extracted for advanced analytics")
            return False
        
        logger.info(f"Extracted activations from {len(layer_activations)} layers")
        
        # Initialize advanced analytics engine
        analytics_engine = AdvancedAnalyticsEngine(analyzer.model_adapter, config)
        
        # Run comprehensive analysis
        comprehensive_results = analytics_engine.run_comprehensive_analysis(questions, layer_activations)
        
        # Save results
        output_path = Path(config.data.outputs_dir) / "advanced_analytics"
        analytics_engine.save_advanced_results(comprehensive_results, output_path)
        
        logger.info(f"Advanced analytics completed successfully! Results saved to {output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Advanced analytics failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return False

def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # Apply CLI overrides
        config = apply_cli_overrides(config, args)
        
        # Setup logging
        logger = setup_logging(config)
        
        logger.info("="*60)
        logger.info("NeuronMap Neural Network Analysis System")
        logger.info("="*60)
        logger.info(f"Configuration: {args.config or 'default'}")
        logger.info(f"Model: {config.model.name}")
        logger.info(f"Device: {config.model.device}")
        logger.info(f"Input file: {config.data.input_file}")
        logger.info(f"Output file: {config.data.output_file}")
        
        # Validate paths
        if not config_manager.validate_paths():
            logger.error("Path validation failed. Check directory structure.")
            return 1
        
        # Execute requested actions
        success = True
        
        if args.list_layers:
            success = list_model_layers(config, logger)
        
        elif args.multi_model:
            success = run_multi_model_analysis(config, logger, args.models)
            
            # Auto-generate visualizations if requested
            if success and args.visualize:
                success = run_visualization(config, logger)
        
        elif args.advanced:
            success = run_advanced_analysis(config, logger)
            
            # Auto-generate visualizations if requested
            if success and args.visualize:
                success = run_visualization(config, logger)
        
        elif args.advanced_analytics:
            success = run_advanced_analytics(config, logger)
            
            # Auto-generate visualizations if requested
            if success and args.visualize:
                success = run_visualization(config, logger)
        
        elif args.analyze:
            success = run_analysis(config, logger)
            
            # Auto-generate visualizations if requested
            if success and args.visualize:
                success = run_visualization(config, logger)
        
        elif args.visualize:
            success = run_visualization(config, logger)
        
        else:
            # Default: run analysis
            logger.info("No specific action requested. Running default analysis...")
            success = run_analysis(config, logger)
        
        if success:
            logger.info("="*60)
            logger.info("✅ Operation completed successfully!")
            logger.info("="*60)
            return 0
        else:
            logger.error("="*60)
            logger.error("❌ Operation failed!")
            logger.error("="*60)
            return 1
    
    except KeyboardInterrupt:
        print("\n🛑 Operation cancelled by user")
        return 1
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
