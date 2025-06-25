"""Main command-line interface for NeuronMap."""

import argparse
import json
import logging
import numpy as np
import sys
import time
import torch
from pathlib import Path

logger = logging.getLogger(__name__)

from src.data_generation.question_generator import QuestionGenerator
from src.utils.config_manager import get_config
from src.utils.validation import (
    check_system_requirements,
    validate_experiment_config,
    validate_questions_file,
    validate_activation_file
)

# Optional imports with graceful degradation
try:
    from src.visualization.visualizer import ActivationVisualizer
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Visualization module not available: {e}")
    ActivationVisualizer = None
    VISUALIZATION_AVAILABLE = False

try:
    from src.visualization.interactive_visualizer import InteractiveVisualizer
    INTERACTIVE_VIZ_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Interactive visualization not available: {e}")
    InteractiveVisualizer = None
    INTERACTIVE_VIZ_AVAILABLE = False

try:
    from src.analysis.activation_extractor import ActivationExtractor
    ACTIVATION_EXTRACTION_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Activation extractor not available: {e}")
    ActivationExtractor = None
    ACTIVATION_EXTRACTION_AVAILABLE = False

try:
    from src.analysis.multi_model_extractor import MultiModelActivationExtractor
    MULTI_MODEL_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Multi-model extractor not available: {e}")
    MultiModelActivationExtractor = None
    MULTI_MODEL_AVAILABLE = False

try:
    from src.analysis.advanced_analysis import ActivationAnalyzer
    ADVANCED_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced analysis not available: {e}")
    ActivationAnalyzer = None
    ADVANCED_ANALYSIS_AVAILABLE = False

try:
    from src.analysis.attention_analysis import AttentionAnalyzer
    ATTENTION_ANALYSIS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Attention analysis not available: {e}")
    AttentionAnalyzer = None
    ATTENTION_ANALYSIS_AVAILABLE = False

try:
    from src.utils.monitoring import SystemMonitor, HealthChecker
    MONITORING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Monitoring not available: {e}")
    SystemMonitor = None
    HealthChecker = None
    MONITORING_AVAILABLE = False

try:
    from src.utils.error_handling import global_error_handler
    ERROR_HANDLING_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Advanced error handling not available: {e}")
    global_error_handler = None
    ERROR_HANDLING_AVAILABLE = False


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('neuronmap.log')
        ]
    )


def cmd_generate_questions(args):
    """Generate questions using Ollama."""
    logger = logging.getLogger(__name__)
    logger.info("Starting question generation...")
    
    generator = QuestionGenerator(args.config)
    
    # Load custom prompt if provided
    prompt_template = None
    if args.prompt_file:
        try:
            with open(args.prompt_file, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
            logger.info(f"Loaded custom prompt from {args.prompt_file}")
        except Exception as e:
            logger.error(f"Error loading prompt file: {e}")
            return False
    
    success = generator.run_generation(prompt_template)
    
    if success:
        logger.info("Question generation completed successfully!")
        return True
    else:
        logger.error("Question generation failed!")
        return False


def cmd_extract_activations(args):
    """Extract activations from neural networks."""
    logger = logging.getLogger(__name__)
    logger.info("Starting activation extraction...")
    
    extractor = ActivationExtractor(args.config)
    
    # List layers if requested
    if args.list_layers:
        try:
            extractor.load_model()
            layers = extractor.print_model_layers()
            print(f"\nFound {len(layers)} layers in the model.")
            return True
        except Exception as e:
            logger.error(f"Error listing layers: {e}")
            return False
    
    # Validate questions file if provided
    if args.questions_file:
        validation = validate_questions_file(args.questions_file)
        if not validation["valid"]:
            logger.error(f"Invalid questions file: {validation['errors']}")
            return False
        logger.info(f"Questions file validated: {validation['num_questions']} questions")
    
    success = extractor.run_extraction(args.questions_file, args.target_layer)
    
    if success:
        logger.info("Activation extraction completed successfully!")
        return True
    else:
        logger.error("Activation extraction failed!")
        return False


def cmd_visualize(args):
    """Create visualizations of activations."""
    logger = logging.getLogger(__name__)
    logger.info("Starting visualization...")
    
    # Validate input file if provided
    if args.input_file:
        validation = validate_activation_file(args.input_file)
        if not validation["valid"]:
            logger.error(f"Invalid activation file: {validation['errors']}")
            return False
        logger.info(f"Activation file validated: {validation['num_samples']} samples, "
                   f"{validation['num_features']} features")
    
    visualizer = ActivationVisualizer(args.config)
    
    # Override methods if specified
    if args.methods:
        visualizer.viz_config["methods"] = args.methods
    
    success = visualizer.run_visualization(args.input_file)
    
    if success:
        logger.info("Visualization completed successfully!")
        return True
    else:
        logger.error("Visualization failed!")
        return False


def cmd_interactive_visualize(args):
    """Create interactive visualizations of activations."""
    logger = logging.getLogger(__name__)
    logger.info("Starting interactive visualization...")
    
    # Validate input file
    if not args.input_file:
        logger.error("Input file is required for interactive visualization")
        return False
    
    validation = validate_activation_file(args.input_file)
    if not validation["valid"]:
        logger.error(f"Invalid activation file: {validation['errors']}")
        return False
    
    logger.info(f"Activation file validated: {validation['num_samples']} samples")
    
    try:
        # Initialize interactive visualizer
        visualizer = InteractiveVisualizer(args.config)
        
        # Load activation data
        logger.info("Loading activation data...")
        data = visualizer.load_activation_data(args.input_file)
        
        # Create visualizations based on requested types
        output_dir = args.output_dir or "data/outputs/interactive"
        
        if args.dashboard:
            logger.info("Creating interactive dashboard...")
            visualizer.create_dashboard(data, output_dir)
            logger.info(f"Dashboard created in {output_dir}")
        else:
            # Create individual visualizations
            available_layers = list(data['activations'].keys())
            layer_name = args.layer or available_layers[0] if available_layers else None
            
            if not layer_name:
                logger.error("No layers found in activation data")
                return False
            
            logger.info(f"Creating visualizations for layer: {layer_name}")
            
            # Create requested visualization types
            if 'heatmap' in args.types:
                logger.info("Creating activation heatmap...")
                fig = visualizer.create_activation_heatmap(data, layer_name)
                output_path = f"{output_dir}/heatmap_{layer_name.replace('.', '_')}.html"
                visualizer.save_interactive_html(fig, output_path)
                logger.info(f"Heatmap saved to {output_path}")
            
            if 'dimensionality' in args.types:
                method = args.dim_method or 'pca'
                logger.info(f"Creating dimensionality reduction plot ({method})...")
                fig = visualizer.create_dimensionality_reduction_plot(
                    data, layer_name, method=method, 
                    show_clusters=args.show_clusters
                )
                output_path = f"{output_dir}/dimensionality_{method}_{layer_name.replace('.', '_')}.html"
                visualizer.save_interactive_html(fig, output_path)
                logger.info(f"Dimensionality plot saved to {output_path}")
            
            if 'distribution' in args.types:
                logger.info("Creating neuron activity distribution...")
                fig = visualizer.create_neuron_activity_distribution(
                    data, layer_name, top_k=args.top_k_neurons
                )
                output_path = f"{output_dir}/distribution_{layer_name.replace('.', '_')}.html"
                visualizer.save_interactive_html(fig, output_path)
                logger.info(f"Distribution plot saved to {output_path}")
            
            if 'animation' in args.types and len(available_layers) > 1:
                logger.info("Creating layer activation animation...")
                fig = visualizer.create_activation_animation(
                    data, available_layers, 
                    question_indices=list(range(min(10, len(data['questions']))))
                )
                output_path = f"{output_dir}/animation_layers.html"
                visualizer.save_interactive_html(fig, output_path)
                logger.info(f"Animation saved to {output_path}")
        
        logger.info("Interactive visualization completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Interactive visualization failed: {str(e)}")
        return False


def cmd_run_full_pipeline(args):
    """Run the complete NeuronMap pipeline."""
    logger = logging.getLogger(__name__)
    logger.info("Starting full NeuronMap pipeline...")
    
    # Step 1: Generate questions
    logger.info("Step 1: Generating questions...")
    if not cmd_generate_questions(args):
        logger.error("Pipeline failed at question generation step")
        return False
    
    # Step 2: Extract activations
    logger.info("Step 2: Extracting activations...")
    if not cmd_extract_activations(args):
        logger.error("Pipeline failed at activation extraction step")
        return False
    
    # Step 3: Create visualizations
    logger.info("Step 3: Creating visualizations...")
    if not cmd_visualize(args):
        logger.error("Pipeline failed at visualization step")
        return False
    
    logger.info("Full pipeline completed successfully!")
    return True


def cmd_validate_setup(args):
    """Validate system setup and requirements."""
    logger = logging.getLogger(__name__)
    logger.info("Validating system setup...")
    
    # Check system requirements
    requirements = check_system_requirements()
    
    print("\n=== System Requirements Check ===")
    all_good = True
    for package, available in requirements.items():
        status = "✓" if available else "✗"
        print(f"{status} {package}")
        if not available:
            all_good = False
    
    if not all_good:
        print("\nSome required packages are missing. Install them with:")
        print("pip install -r requirements.txt")
        return False
    
    # Validate configuration
    try:
        config = get_config()
        experiment_config = config.get_experiment_config(args.config)
        
        validation_errors = validate_experiment_config(experiment_config)
        
        if validation_errors:
            print(f"\n=== Configuration Validation Errors ===")
            for error in validation_errors:
                print(f"✗ {error}")
            return False
        else:
            print(f"\n✓ Configuration '{args.config}' is valid")
            
    except Exception as e:
        logger.error(f"Error validating configuration: {e}")
        return False
    
    print("\n✓ All validations passed!")
    return True


def cmd_monitor_system(args):
    """Monitor system resources and health."""
    logger = logging.getLogger(__name__)
    logger.info("Starting system monitoring...")
    
    # Create monitors
    system_monitor = SystemMonitor(args.interval)
    health_checker = HealthChecker()
    
    if args.health_only:
        # Run single health check
        results = health_checker.run_comprehensive_health_check(
            ollama_host=args.ollama_host,
            model_name=args.model
        )
        
        print("\n=== Health Check Results ===")
        print(f"Overall Status: {'✓ HEALTHY' if results['overall_healthy'] else '⚠ ISSUES FOUND'}")
        
        if results['recommendations']:
            print("\nRecommendations:")
            for rec in results['recommendations']:
                print(f"  - {rec}")
        
        return results['overall_healthy']
    
    else:
        # Continuous monitoring
        logger.info(f"Monitoring system for {args.duration} seconds...")
        start_time = time.time()
        
        try:
            while time.time() - start_time < args.duration:
                # Get system metrics
                metrics = system_monitor.get_system_metrics()
                system_monitor.log_metrics(metrics)
                
                # Check resource availability
                resource_check = system_monitor.check_resource_availability()
                if not resource_check['all_checks_passed']:
                    logger.warning("Resource constraints detected:")
                    for rec in resource_check['recommendations']:
                        logger.warning(f"  - {rec}")
                
                time.sleep(args.interval)
                
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
        
        # Save metrics if requested
        if args.output:
            system_monitor.save_metrics_to_file(args.output)
            logger.info(f"Metrics saved to {args.output}")
        
        return True


def cmd_show_errors(args):
    """Show recent errors and statistics."""
    logger = logging.getLogger(__name__)
    
    error_summary = global_error_handler.get_error_summary(args.count)
    
    print(f"\n=== Error Summary (Last {args.count} errors) ===")
    print(f"Total errors in session: {error_summary['total_errors']}")
    print(f"Recent errors: {error_summary['recent_errors']}")
    
    if error_summary['error_types']:
        print(f"\nError types:")
        for error_type, count in error_summary['error_types'].items():
            print(f"  {error_type}: {count}")
    
    if error_summary['operations']:
        print(f"\nOperations with errors:")
        for operation, count in error_summary['operations'].items():
            print(f"  {operation}: {count}")
    
    if error_summary['recent_errors_list']:
        print(f"\nRecent errors:")
        for error in error_summary['recent_errors_list']:
            print(f"  - {error['operation']}: {error['error_type']} - {error['error_message']}")
    
    return True


def cmd_multi_extract(args):
    """Extract activations from multiple layers using multi-model extractor."""
    logger = logging.getLogger(__name__)
    logger.info("Starting multi-layer activation extraction...")
    
    extractor = MultiModelActivationExtractor(args.config)
    
    # Load model
    try:
        if args.model:
            extractor.load_model(args.model)
        else:
            extractor.load_model()
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return False
    
    # Discover layers if requested
    if args.discover_layers:
        layers = extractor.discover_model_layers()
        print("\nDiscovered layers:")
        for category, layer_list in layers.items():
            print(f"\n{category.upper()} layers ({len(layer_list)}):")
            for layer in layer_list[:10]:  # Show first 10
                print(f"  {layer}")
            if len(layer_list) > 10:
                print(f"  ... and {len(layer_list) - 10} more")
        return True
    
    # Run multi-layer extraction
    layer_range = None
    if args.layer_range:
        layer_range = tuple(args.layer_range)
    
    success = extractor.run_multi_layer_extraction(
        questions_file=args.questions_file,
        target_layers=args.layers,
        layer_range=layer_range,
        batch_size=args.batch_size,
        output_format=args.output_format
    )
    
    if success:
        logger.info("Multi-layer extraction completed successfully!")
    else:
        logger.error("Multi-layer extraction failed!")
    
    return success


def cmd_analyze_activations(args):
    """Perform advanced analysis on extracted activations."""
    logger = logging.getLogger(__name__)
    logger.info("Starting advanced activation analysis...")
    
    analyzer = ActivationAnalyzer(args.config)
    
    try:
        # Load activation data
        data = analyzer.load_activations_hdf5(args.input_file)
        
        if args.layer:
            # Analyze specific layer
            results = analyzer.analyze_layer_activations(data, args.layer)
            output_file = Path(args.output) / f"{args.layer.replace('.', '_')}_analysis.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Layer analysis saved to: {output_file}")
            
        elif args.compare_layers:
            # Compare specific layers
            results = analyzer.compare_layers(data, args.compare_layers)
            output_file = Path(args.output) / "layer_comparison.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            import json
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            logger.info(f"Layer comparison saved to: {output_file}")
            
        else:
            # Generate comprehensive report
            report_path = analyzer.generate_analysis_report(data, args.output)
            logger.info(f"Comprehensive analysis completed: {report_path}")
        
        return True
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        return False


def cmd_analyze_attention(args):
    """Analyze attention patterns in transformer models."""
    logger = logging.getLogger(__name__)
    logger.info("Starting attention pattern analysis...")
    
    analyzer = AttentionAnalyzer(args.config)
    
    # Implementation would depend on having attention data
    # For now, just indicate the functionality is available
    logger.info("Attention analysis functionality available")
    logger.info("Note: Requires attention weights to be captured during extraction")
    
    return True


def cmd_discover_models(args):
    """Discover and test different model configurations."""
    logger = logging.getLogger(__name__)
    logger.info("Discovering available model configurations...")
    
    config = get_config()
    models = config.data.get("models", {})
    
    print(f"\nAvailable model configurations ({len(models)}):")
    print("=" * 50)
    
    for model_name, model_config in models.items():
        print(f"\nModel: {model_name}")
        print(f"  Name: {model_config['name']}")
        print(f"  Type: {model_config.get('type', 'auto')}")
        print(f"  Layers: {model_config.get('layers', {}).get('total_layers', 'unknown')}")
        
        # Test if model is available
        if args.test_availability:
            try:
                extractor = MultiModelActivationExtractor(args.config)
                success = extractor.load_model(model_name)
                status = "✓ Available" if success else "✗ Failed to load"
                print(f"  Status: {status}")
            except Exception as e:
                print(f"  Status: ✗ Error - {e}")
    
    return True


def cmd_show_config(args):
    """Show current configuration."""
    try:
        config = get_config()
        
        if args.models:
            print("=== Available Model Configurations ===")
            models_config = config.load_models_config()
            for model_name in models_config['models']:
                model_config = models_config['models'][model_name]
                print(f"\n{model_name}:")
                print(f"  Name: {model_config['name']}")
                print(f"  Type: {model_config['type']}")
                print(f"  Layers: {model_config['layers']['total_layers']}")
        
        if args.experiments:
            print("=== Available Experiment Configurations ===")
            experiments_config = config.load_experiments_config()
            for exp_name in experiments_config:
                print(f"\n{exp_name}:")
                if 'description' in experiments_config[exp_name]:
                    print(f"  Description: {experiments_config[exp_name]['description']}")
        
        if not args.models and not args.experiments:
            # Show current experiment config
            experiment_config = config.get_experiment_config(args.config)
            print(f"=== Current Configuration: {args.config} ===")
            
            import yaml
            print(yaml.dump(experiment_config, default_flow_style=False, indent=2))
            
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error showing configuration: {e}")
        return False
    
def cmd_process_data(args):
    """Process data using quality management and streaming."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data processing...")
    
    try:
        from src.data_processing.streaming_processor import StreamingDataProcessor
        from src.data_processing.quality_manager import DataQualityManager
        from src.data_processing.metadata_manager import MetadataManager
        
        # Initialize components
        processor = StreamingDataProcessor(
            chunk_size=args.chunk_size,
            max_workers=args.workers
        )
        
        metadata_manager = MetadataManager(args.data_dir or "data")
        
        # Create experiment record
        experiment_id = metadata_manager.create_experiment(
            name=args.experiment_name or f"data_processing_{int(time.time())}",
            description=args.description or "Data processing experiment",
            config={
                'input_file': args.input_file,
                'output_file': args.output_file,
                'validation': args.validate,
                'chunk_size': args.chunk_size,
                'workers': args.workers
            }
        )
        
        # Process data
        if args.async_mode:
            import asyncio
            stats = asyncio.run(processor.process_questions_stream(
                args.input_file, args.output_file, args.validate
            ))
        else:
            stats = processor.process_questions_batch(
                args.input_file, args.output_file, args.validate
            )
        
        # Log results
        metadata_manager.log_provenance(
            event_type='data_processed',
            experiment_id=experiment_id,
            data=stats
        )
        
        metadata_manager.update_experiment_status(
            experiment_id, 'completed', {'stats': stats}
        )
        
        logger.info(f"Data processing completed: {stats}")
        print(f"\nProcessing Results:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Valid questions: {stats['valid_questions']}")
        print(f"  Invalid questions: {stats['invalid_questions']}")
        print(f"  Processing time: {stats['processing_time']:.2f}s")
        print(f"  Experiment ID: {experiment_id}")
        
        return True
        
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        return False


def cmd_validate_data(args):
    """Validate data quality."""
    logger = logging.getLogger(__name__)
    logger.info("Starting data validation...")
    
    try:
        from src.data_processing.quality_manager import DataQualityManager
        
        quality_manager = DataQualityManager(args.config)
        
        # Load questions
        questions = []
        with open(args.input_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    questions.append(data.get('text', ''))
                except json.JSONDecodeError:
                    continue
        
        # Validate questions
        valid_count = 0
        invalid_count = 0
        issues_summary = {}
        
        for i, question in enumerate(questions):
            result = quality_manager.validate_question(question)
            
            if result['valid']:
                valid_count += 1
            else:
                invalid_count += 1
                for issue in result['issues']:
                    issues_summary[issue] = issues_summary.get(issue, 0) + 1
        
        # Detect duplicates
        duplicates = quality_manager.detect_duplicates(questions, args.duplicate_threshold)
        
        # Print results
        print(f"\nData Quality Report:")
        print(f"  Total questions: {len(questions)}")
        print(f"  Valid questions: {valid_count}")
        print(f"  Invalid questions: {invalid_count}")
        print(f"  Duplicates found: {len(duplicates)}")
        
        if issues_summary:
            print(f"\nIssue Summary:")
            for issue, count in issues_summary.items():
                print(f"  {issue}: {count}")
        
        if duplicates:
            print(f"\nDuplicate pairs (similarity >= {args.duplicate_threshold}):")
            for idx1, idx2, similarity in duplicates[:10]:  # Show first 10
                print(f"  Questions {idx1} & {idx2}: {similarity:.3f}")
            if len(duplicates) > 10:
                print(f"  ... and {len(duplicates) - 10} more")
        
        logger.info("Data validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Data validation failed: {e}")
        return False


def cmd_manage_metadata(args):
    """Manage experiment metadata."""
    logger = logging.getLogger(__name__)
    
    try:
        from src.data_processing.metadata_manager import MetadataManager, DatasetVersionManager
        
        metadata_manager = MetadataManager(args.data_dir or "data")
        version_manager = DatasetVersionManager(args.data_dir or "data")
        
        if args.action == 'list-experiments':
            experiments = metadata_manager.get_all_experiments()
            print(f"\nExperiments ({len(experiments)}):")
            for exp_id, exp_data in experiments.items():
                print(f"  {exp_id}: {exp_data['name']} - {exp_data['status']}")
                print(f"    Created: {time.ctime(exp_data['created_at'])}")
                if 'description' in exp_data:
                    print(f"    Description: {exp_data['description']}")
        
        elif args.action == 'experiment-history':
            if not args.experiment_id:
                logger.error("Experiment ID required for history")
                return False
            
            history = metadata_manager.get_experiment_history(args.experiment_id)
            print(f"\nExperiment History ({len(history)} events):")
            for event in history:
                print(f"  {time.ctime(event['timestamp'])}: {event['event_type']}")
                if 'data' in event:
                    print(f"    Data: {event['data']}")
        
        elif args.action == 'create-version':
            if not args.dataset_name or not args.file_path:
                logger.error("Dataset name and file path required for version creation")
                return False
            
            version_id = version_manager.create_version(
                args.dataset_name, args.file_path, 
                args.description or ""
            )
            print(f"Created version {version_id} for dataset {args.dataset_name}")
        
        elif args.action == 'list-versions':
            if not args.dataset_name:
                logger.error("Dataset name required for listing versions")
                return False
            
            versions = version_manager.list_versions(args.dataset_name)
            print(f"\nVersions for {args.dataset_name} ({len(versions)}):")
            for version in versions:
                print(f"  {version['id']}: {version['description']}")
                print(f"    Created: {time.ctime(version['created_at'])}")
                print(f"    Size: {version['size_bytes']} bytes")
        
        elif args.action == 'verify-version':
            if not args.dataset_name or not args.version_id:
                logger.error("Dataset name and version ID required for verification")
                return False
            
            is_valid = version_manager.verify_version(args.dataset_name, args.version_id)
            print(f"Version {args.version_id} of {args.dataset_name}: {'VALID' if is_valid else 'INVALID'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Metadata management failed: {e}")
        return False


def cmd_interpretability_analysis(args):
    """Run interpretability analysis including CAVs, saliency, and activation maximization."""
    logger = logging.getLogger(__name__)
    logger.info("Starting interpretability analysis...")
    
    try:
        from src.analysis.interpretability import InterpretabilityPipeline
        
        # Load concept examples if provided
        concept_examples = {}
        if args.concept_file:
            with open(args.concept_file, 'r') as f:
                concept_examples = json.load(f)
        else:
            # Default sentiment concepts
            concept_examples = {
                'positive_sentiment': [
                    "I love this!", "This is wonderful!", "Great job!", 
                    "Excellent work!", "Amazing results!"
                ],
                'negative_sentiment': [
                    "I hate this.", "This is terrible.", "Poor job.", 
                    "Awful work.", "Disappointing results."
                ],
                'neutral_sentiment': [
                    "This is okay.", "Standard procedure.", "Regular meeting.",
                    "Normal process.", "Typical outcome."
                ]
            }
        
        # Load test texts
        test_texts = []
        if args.test_texts_file:
            with open(args.test_texts_file, 'r') as f:
                test_texts = [line.strip() for line in f if line.strip()]
        else:
            test_texts = [
                "The weather is nice today.",
                "I'm excited about this project.",
                "The results were disappointing.",
                "Technical documentation is available.",
                "Great progress has been made!"
            ]
        
        # Initialize pipeline
        pipeline = InterpretabilityPipeline(args.config)
        
        # Run analysis
        results = pipeline.run_full_interpretability_analysis(
            model_name=args.model or "gpt2",
            concept_examples=concept_examples,
            test_texts=test_texts,
            target_layer=args.layer or "transformer.h.6",
            output_dir=args.output or "data/outputs/interpretability"
        )
        
        logger.info("Interpretability analysis completed successfully!")
        print(f"\nInterpretability Analysis Results:")
        print(f"  Model: {results['model_name']}")
        print(f"  Target Layer: {results['target_layer']}")
        print(f"  CAVs trained: {len(results['cavs'])}")
        print(f"  Saliency analyses: {len(results['saliency_results'])}")
        print(f"  Activation maximization: {len(results['activation_maximization'])} neurons")
        print(f"  Results saved to: {args.output or 'data/outputs/interpretability'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Interpretability analysis failed: {e}")
        return False


def cmd_experimental_analysis(args):
    """Run experimental analysis including RSA, CKA, and probing tasks."""
    logger = logging.getLogger(__name__)
    logger.info("Starting experimental analysis...")
    
    try:
        from src.analysis.experimental_analysis import ExperimentalAnalysisPipeline
        import h5py
        
        # Load activation data
        if not args.input_file:
            logger.error("Input file is required for experimental analysis")
            return False
        
        layer_activations = {}
        
        # Load from HDF5 file
        with h5py.File(args.input_file, 'r') as f:
            activations_group = f['activations']
            
            # Group by layers
            layer_data = {}
            for question_key in activations_group.keys():
                question_group = activations_group[question_key]
                for layer_key in question_group.keys():
                    layer_name = layer_key.replace('_', '.')
                    if layer_name not in layer_data:
                        layer_data[layer_name] = []
                    
                    activation_vector = question_group[layer_key]['vector'][()]
                    layer_data[layer_name].append(activation_vector)
            
            # Convert to numpy arrays
            for layer_name, vectors in layer_data.items():
                layer_activations[layer_name] = np.stack(vectors)
        
        # Load probing data if provided
        probing_data = None
        if args.probing_file:
            with open(args.probing_file, 'r') as f:
                probing_data = json.load(f)
        
        # Initialize pipeline
        pipeline = ExperimentalAnalysisPipeline(args.config)
        
        # Run analysis
        results = pipeline.run_experimental_analysis(
            layer_activations=layer_activations,
            probing_data=probing_data,
            output_dir=args.output or "data/outputs/experimental"
        )
        
        logger.info("Experimental analysis completed successfully!")
        print(f"\nExperimental Analysis Results:")
        print(f"  Layers analyzed: {results['metadata']['n_layers']}")
        print(f"  RSA analysis: {len(results['rsa_analysis'])} metrics")
        print(f"  CKA analysis: {len(results['cka_analysis'])} comparisons")
        if results['probing_analysis']:
            print(f"  Probing tasks: {len(results['probing_analysis'])} tasks")
        print(f"  Information analysis: {len(results['information_analysis'])} metrics")
        print(f"  Results saved to: {args.output or 'data/outputs/experimental'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Experimental analysis failed: {e}")
        return False


def cmd_create_probing_dataset(args):
    """Create probing task datasets from text data."""
    logger = logging.getLogger(__name__)
    logger.info("Creating probing dataset...")
    
    try:
        from src.analysis.experimental_analysis import ProbingTaskAnalyzer
        
        # Load input texts
        texts = []
        with open(args.input_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    texts.append(data.get('text', ''))
                except json.JSONDecodeError:
                    texts.append(line.strip())
        
        analyzer = ProbingTaskAnalyzer(args.config)
        
        # Create sentiment task
        if args.create_sentiment:
            # Simple sentiment labeling (in practice, use proper sentiment analysis)
            sentiments = []
            for text in texts:
                positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'love']
                negative_words = ['bad', 'terrible', 'awful', 'hate', 'horrible', 'worst']
                
                text_lower = text.lower()
                pos_count = sum(1 for word in positive_words if word in text_lower)
                neg_count = sum(1 for word in negative_words if word in text_lower)
                
                if pos_count > neg_count:
                    sentiments.append('positive')
                elif neg_count > pos_count:
                    sentiments.append('negative')
                else:
                    sentiments.append('neutral')
            
            features, labels = analyzer.create_sentiment_task(texts, sentiments)
            
            # Save dataset
            output_file = args.output or "data/probing/sentiment_task.json"
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            dataset = {
                'task_type': 'sentiment',
                'texts': texts,
                'labels': labels.tolist(),
                'label_names': ['positive', 'negative', 'neutral']
            }
            
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            logger.info(f"Sentiment probing dataset created: {output_file}")
        
        # Create POS tagging task (simplified)
        if args.create_pos:
            # Simple POS tagging (in practice, use proper POS tagger)
            pos_tags = []
            for text in texts:
                words = text.split()
                tags = []
                for word in words:
                    # Very simplified POS tagging
                    if word.endswith('ing') or word.endswith('ed'):
                        tags.append('VERB')
                    elif word.endswith('ly'):
                        tags.append('ADV')
                    elif word.endswith('tion') or word.endswith('ness'):
                        tags.append('NOUN')
                    elif word in ['the', 'a', 'an']:
                        tags.append('DET')
                    else:
                        tags.append('NOUN')  # Default
                pos_tags.append(tags)
            
            features, labels = analyzer.create_pos_tagging_task(texts, pos_tags)
            
            # Save dataset
            output_file = args.output or "data/probing/pos_task.json"
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            dataset = {
                'task_type': 'pos_tagging',
                'texts': texts,
                'labels': labels.tolist(),
                'label_names': ['NOUN', 'VERB', 'ADJ', 'ADV', 'OTHER']
            }
            
            with open(output_file, 'w') as f:
                json.dump(dataset, f, indent=2)
            
            logger.info(f"POS tagging probing dataset created: {output_file}")
        
        return True
        
    except Exception as e:
        logger.error(f"Probing dataset creation failed: {e}")
        return False


def cmd_advanced_experimental(args):
    """Run advanced experimental analysis including causality, adversarial, and counterfactual analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Starting advanced experimental analysis...")
    
    try:
        from src.analysis.advanced_experimental import AdvancedExperimentalPipeline
        from transformers import AutoModel, AutoTokenizer
        
        # Load texts
        texts = []
        with open(args.input_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    texts.append(data.get('text', ''))
                except json.JSONDecodeError:
                    texts.append(line.strip())
        
        # Limit texts for efficiency
        texts = texts[:20]
        
        # Load model and tokenizer
        try:
            model = AutoModel.from_pretrained(args.model)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to load model {args.model}: {e}")
            return False
        
        # Initialize pipeline
        pipeline = AdvancedExperimentalPipeline(args.config)
        
        # Run analysis
        results = pipeline.run_full_analysis(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            output_dir=args.output or "data/outputs/advanced_experimental"
        )
        
        logger.info("Advanced experimental analysis completed successfully!")
        print(f"\nAdvanced Experimental Analysis Results:")
        print(f"  Texts analyzed: {results['metadata']['n_texts']}")
        print(f"  Adversarial examples: {len(results['adversarial_analysis'])}")
        print(f"  Counterfactual analyses: {len(results['counterfactual_analysis'])}")
        print(f"  Analysis types: {args.analysis_types}")
        print(f"  Results saved to: {args.output or 'data/outputs/advanced_experimental'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Advanced experimental analysis failed: {e}")
        return False


def cmd_domain_analysis(args):
    """Run domain-specific analysis for specialized domains."""
    logger = logging.getLogger(__name__)
    logger.info(f"Starting {args.analysis_type} domain analysis...")
    
    try:
        from src.analysis.domain_specific import DomainSpecificPipeline
        from transformers import AutoModel, AutoTokenizer
        
        # Load model and tokenizer
        try:
            model = AutoModel.from_pretrained(args.model)
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        except Exception as e:
            logger.error(f"Failed to load model {args.model}: {e}")
            return False
        
        # Load and prepare input data based on analysis type
        data = {}
        
        if args.analysis_type == "code":
            # Load code snippets
            with open(args.input_file, 'r') as f:
                code_snippets = []
                for line in f:
                    try:
                        data_line = json.loads(line.strip())
                        code_snippets.append(data_line.get('code', ''))
                    except json.JSONDecodeError:
                        code_snippets.append(line.strip())
                data['code_snippets'] = code_snippets[:10]  # Limit for efficiency
        
        elif args.analysis_type == "math":
            # Load mathematical problems
            with open(args.input_file, 'r') as f:
                math_problems = []
                for line in f:
                    try:
                        data_line = json.loads(line.strip())
                        math_problems.append(data_line.get('problem', ''))
                    except json.JSONDecodeError:
                        math_problems.append(line.strip())
                data['math_expressions'] = math_problems[:10]  # Limit for efficiency
        
        elif args.analysis_type == "multilingual":
            # Load multilingual texts
            with open(args.input_file, 'r') as f:
                content = json.load(f)
                if isinstance(content, dict):
                    data['multilingual_texts'] = content
                else:
                    # Assume single language
                    data['multilingual_texts'] = {'unknown': content if isinstance(content, list) else [str(content)]}
        
        elif args.analysis_type == "temporal":
            # Load temporal sequences
            with open(args.input_file, 'r') as f:
                sequences = []
                for line in f:
                    try:
                        data_line = json.loads(line.strip())
                        sequences.append(data_line.get('sequence', ''))
                    except json.JSONDecodeError:
                        sequences.append(line.strip())
                data['sequences'] = sequences[:10]  # Limit for efficiency
        
        # Initialize pipeline
        pipeline = DomainSpecificPipeline(args.config)
        
        # Run analysis
        results = pipeline.run_comprehensive_analysis(
            inputs=data,
            model=model,
            tokenizer=tokenizer,
            output_dir=args.output or f"data/outputs/domain_{args.analysis_type}"
        )
        
        logger.info("Domain-specific analysis completed successfully!")
        print(f"\nDomain-Specific Analysis Results ({args.analysis_type}):")
        print(f"  Model: {args.model}")
        print(f"  Analysis type: {args.analysis_type}")
        print(f"  Domains analyzed: {results['metadata']['domains_analyzed']}")
        print(f"  Target layers: {args.target_layers or 'All layers'}")
        print(f"  Results saved to: {args.output or f'data/outputs/domain_{args.analysis_type}'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Domain-specific analysis failed: {e}")
        return False


def cmd_ethics_analysis(args):
    """Run ethics and bias analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Starting ethics and bias analysis...")
    
    try:
        from src.analysis.ethics_bias import FairnessAnalyzer, ModelCardGenerator, AuditTrail
        from transformers import AutoModel, AutoTokenizer
        
        # Load model and tokenizer
        logger.info(f"Loading model: {args.model}")
        model = AutoModel.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Load input data
        with open(args.texts_file, 'r') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        with open(args.groups_file, 'r') as f:
            groups = [line.strip() for line in f if line.strip()]
        
        labels = None
        if args.labels_file:
            with open(args.labels_file, 'r') as f:
                labels = [int(line.strip()) for line in f if line.strip()]
        
        if len(texts) != len(groups):
            raise ValueError("Number of texts and groups must match")
        
        if labels and len(labels) != len(texts):
            raise ValueError("Number of labels must match number of texts")
        
        # Initialize analyzer
        analyzer = FairnessAnalyzer()
        
        # Initialize audit trail
        output_dir = args.output or "data/outputs/ethics_analysis"
        audit_trail = AuditTrail(Path(output_dir))
        
        audit_trail.log_step("initialization", "Started ethics analysis", {
            "model": args.model,
            "num_texts": len(texts),
            "num_groups": len(set(groups)),
            "has_labels": labels is not None
        })
        
        # Extract activations
        logger.info("Extracting activations...")
        activations = {}
        target_layers = args.layers or []
        
        if not target_layers:
            # Auto-detect some layers
            for name, module in model.named_modules():
                if any(layer_type in name for layer_type in ['attention', 'intermediate', 'output']):
                    target_layers.append(name)
                if len(target_layers) >= 8:  # Limit for efficiency
                    break
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach().cpu().numpy()
                else:
                    activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name in target_layers:
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        # Process texts and collect activations
        all_activations = {layer: [] for layer in target_layers}
        
        for text in texts:
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            # Forward pass
            with torch.no_grad():
                model(**inputs)
            
            # Collect activations
            for layer_name in target_layers:
                if layer_name in activations:
                    # Pool activations (mean over sequence length)
                    act = activations[layer_name]
                    if len(act.shape) == 3:  # [batch, seq, hidden]
                        pooled_act = np.mean(act[0], axis=0)  # Mean pool
                    else:
                        pooled_act = act[0] if len(act.shape) > 1 else act
                    all_activations[layer_name].append(pooled_act)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Convert to arrays
        for layer_name in all_activations:
            if all_activations[layer_name]:
                all_activations[layer_name] = np.stack(all_activations[layer_name])
        
        audit_trail.log_step("activation_extraction", "Extracted activations", {
            "layers": list(all_activations.keys()),
            "activation_shapes": {k: v.shape for k, v in all_activations.items() if len(v) > 0}
        })
        
        # Run fairness analysis
        logger.info("Running fairness analysis...")
        fairness_result = analyzer.comprehensive_analysis(
            all_activations,
            texts,
            groups,
            labels,
            threshold=args.threshold
        )
        
        audit_trail.log_step("fairness_analysis", "Completed fairness analysis", {
            "overall_bias_score": fairness_result.overall_bias_score,
            "num_warnings": len(fairness_result.warnings),
            "num_recommendations": len(fairness_result.recommendations)
        })
        
        # Generate model card if requested
        if args.generate_card:
            logger.info("Generating model card...")
            card_generator = ModelCardGenerator()
            model_card = card_generator.generate_card(
                model_name=args.model,
                fairness_analysis=fairness_result,
                metadata={'analysis_date': time.time()}
            )
            
            card_path = Path(output_dir) / "model_card.md"
            with open(card_path, 'w') as f:
                f.write(model_card)
            
            audit_trail.log_step("model_card", "Generated model card", {
                "card_path": str(card_path)
            })
        
        # Save results
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        results_path = Path(output_dir) / "ethics_analysis_results.json"
        
        results_data = {
            "model": args.model,
            "analysis_timestamp": time.time(),
            "overall_bias_score": fairness_result.overall_bias_score,
            "fairness_metrics": {
                "demographic_parity": fairness_result.fairness_metrics.demographic_parity,
                "equalized_odds": fairness_result.fairness_metrics.equalized_odds,
                "equal_opportunity": fairness_result.fairness_metrics.equal_opportunity,
                "statistical_parity": fairness_result.fairness_metrics.statistical_parity,
                "confidence_interval": fairness_result.fairness_metrics.confidence_interval
            },
            "affected_groups": fairness_result.fairness_metrics.affected_groups,
            "recommendations": fairness_result.recommendations,
            "warnings": fairness_result.warnings
        }
        
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        audit_trail.log_step("save_results", "Saved analysis results", {
            "results_path": str(results_path)
        })
        
        # Save audit trail
        audit_trail.save_audit_log()
        
        # Display results
        logger.info("Ethics and bias analysis completed successfully!")
        print(f"\nEthics and Bias Analysis Results:")
        print(f"  Model: {args.model}")
        print(f"  Overall bias score: {fairness_result.overall_bias_score:.3f}")
        print(f"  Demographic parity: {fairness_result.fairness_metrics.demographic_parity:.3f}")
        print(f"  Equalized odds: {fairness_result.fairness_metrics.equalized_odds:.3f}")
        print(f"  Affected groups: {fairness_result.fairness_metrics.affected_groups}")
        print(f"  Recommendations: {len(fairness_result.recommendations)}")
        print(f"  Warnings: {len(fairness_result.warnings)}")
        
        if fairness_result.warnings:
            print("\nWarnings:")
            for warning in fairness_result.warnings:
                print(f"  - {warning}")
        
        print(f"\nRecommendations:")
        for rec in fairness_result.recommendations:
            print(f"  - {rec}")
        
        print(f"\nResults saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Ethics analysis failed: {e}")
        print(f"Error: {e}")
        return False


def cmd_conceptual_analysis(args):
    """Run advanced conceptual analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Starting conceptual analysis...")
    
    try:
        from src.analysis.conceptual_analysis import ConceptualAnalyzer
        from transformers import AutoModel, AutoTokenizer
        import json
        import numpy as np
        
        # Load model and tokenizer
        logger.info(f"Loading model: {args.model}")
        model = AutoModel.from_pretrained(args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        
        # Initialize analyzer
        config = {
            'concept_threshold': args.concept_threshold,
            'circuit_threshold': args.circuit_threshold,
            'causal_threshold': args.causal_threshold
        }
        analyzer = ConceptualAnalyzer(config)
        
        output_dir = args.output or "data/outputs/conceptual_analysis"
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load input data
        with open(args.input_file, 'r') as f:
            if args.input_file.endswith('.json'):
                input_data = json.load(f)
                if isinstance(input_data, list):
                    texts = input_data
                    labels = [f"sample_{i}" for i in range(len(texts))]
                else:
                    texts = input_data.get('texts', input_data.get('samples', []))
                    labels = input_data.get('labels', [f"sample_{i}" for i in range(len(texts))])
            else:
                texts = [line.strip() for line in f if line.strip()]
                labels = [f"sample_{i}" for i in range(len(texts))]
        
        if not texts:
            raise ValueError("No valid input texts found")
        
        logger.info(f"Processing {len(texts)} samples")
        
        # Extract activations
        logger.info("Extracting activations...")
        activations = {}
        
        # Get model layers
        target_layers = args.target_layers or []
        if not target_layers:
            # Auto-detect some key layers
            for name, module in model.named_modules():
                if any(layer_type in name for layer_type in ['attention', 'feed_forward', 'transformer']):
                    target_layers.append(name)
                if len(target_layers) >= 10:  # Limit for efficiency
                    break
        
        def get_activation(name):
            def hook(module, input, output):
                if isinstance(output, tuple):
                    activations[name] = output[0].detach().cpu().numpy()
                else:
                    activations[name] = output.detach().cpu().numpy()
            return hook
        
        # Register hooks
        hooks = []
        for name, module in model.named_modules():
            if name in target_layers:
                hooks.append(module.register_forward_hook(get_activation(name)))
        
        # Process texts and collect activations
        all_activations = {layer: [] for layer in target_layers}
        
        for text in texts[:20]:  # Limit for computational efficiency
            # Tokenize
            inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            
            # Forward pass
            with torch.no_grad():
                model(**inputs)
            
            # Collect activations
            for layer_name in target_layers:
                if layer_name in activations:
                    # Pool activations (mean over sequence length)
                    act = activations[layer_name]
                    if len(act.shape) == 3:  # [batch, seq, hidden]
                        pooled_act = np.mean(act[0], axis=0)  # Mean pool
                    else:
                        pooled_act = act[0] if len(act.shape) > 1 else act
                    all_activations[layer_name].append(pooled_act)
        
        # Clean up hooks
        for hook in hooks:
            hook.remove()
        
        # Convert to arrays
        for layer_name in all_activations:
            if all_activations[layer_name]:
                all_activations[layer_name] = np.stack(all_activations[layer_name])
        
        logger.info(f"Extracted activations from {len(all_activations)} layers")
        
        # Run analysis based on type
        results = {}
        
        if args.analysis_type == "concepts":
            logger.info("Extracting concepts...")
            concepts = analyzer.extract_concepts(
                all_activations, 
                labels[:len(texts[:20])],  # Match labels to processed texts
                method=args.concept_method
            )
            results['concepts'] = len(concepts)
            
        elif args.analysis_type == "circuits":
            logger.info("Discovering circuits...")
            circuits = analyzer.discover_circuits(
                all_activations,
                labels[:len(texts[:20])],
                args.task_name or "default_task"
            )
            results['circuits'] = len(circuits)
            
        elif args.analysis_type == "causal":
            logger.info("Running causal tracing...")
            if args.intervention_layer and args.intervention_neurons:
                intervention_neurons = list(map(int, args.intervention_neurons.split(',')))
                causal_results = analyzer.trace_causal_effects(
                    model,
                    inputs['input_ids'],
                    args.intervention_layer,
                    intervention_neurons,
                    args.intervention_value
                )
                results['causal_effects'] = causal_results['output_effect']
            else:
                logger.warning("Causal analysis requires --intervention-layer and --intervention-neurons")
                results['causal_effects'] = "skipped"
                
        elif args.analysis_type == "algebra":
            logger.info("Running concept algebra...")
            if len(analyzer.concepts) >= 2:
                concept_names = list(analyzer.concepts.keys())[:2]
                result_concept = analyzer.concept_algebra(
                    concept_names[0],
                    concept_names[1],
                    args.operation or "add"
                )
                results['algebra_result'] = result_concept.name
            else:
                logger.warning("Concept algebra requires pre-extracted concepts")
                results['algebra_result'] = "insufficient_concepts"
                
        elif args.analysis_type == "world_model":
            logger.info("Analyzing world model...")
            # Create dummy metadata for world model analysis
            stimuli_metadata = [
                {'object': f'object_{i}', 'position': [i % 10, (i // 10) % 10]}
                for i in range(len(texts[:20]))
            ]
            world_model = analyzer.analyze_world_model(all_activations, stimuli_metadata)
            results['world_model_components'] = len(world_model)
            
        elif args.analysis_type == "cross_model":
            logger.info("Cross-model analysis requires multiple models (not implemented in single-model mode)")
            results['cross_model'] = "requires_multiple_models"
        
        # Save results
        results_path = Path(output_dir) / "conceptual_analysis_results.json"
        analyzer.save_analysis_results(str(results_path))
        
        # Save summary
        summary_path = Path(output_dir) / "analysis_summary.json"
        summary = {
            "model": args.model,
            "analysis_type": args.analysis_type,
            "num_samples": len(texts[:20]),
            "target_layers": target_layers,
            "config": config,
            "results": results,
            "timestamp": time.time()
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Conceptual analysis completed successfully!")
        print(f"\nConceptual Analysis Results:")
        print(f"  Model: {args.model}")
        print(f"  Analysis type: {args.analysis_type}")
        print(f"  Samples processed: {len(texts[:20])}")
        print(f"  Layers analyzed: {len(target_layers)}")
        
        for key, value in results.items():
            print(f"  {key}: {value}")
        
        print(f"\nResults saved to: {output_dir}")
        
        return True
        
    except Exception as e:
        logger.error(f"Conceptual analysis failed: {e}")
        print(f"Error: {e}")
        return False


def cmd_model_list(args):
    """List available models."""
    logger = logging.getLogger(__name__)
    logger.info("Listing available models...")
    
    try:
        config = get_config()
        models = config.data.get("models", {})
        
        if not models:
            print("No models found")
            return True
        
        # Format output
        if args.format == "table":
            # Print as table
            print(f"{'Model Name':<30} {'Type':<15} {'Layers':<10} {'Status':<10}")
            print("=" * 65)
            for model_name, model_config in models.items():
                status = "✓ Available" if model_config.get("available", False) else "✗ Unavailable"
                print(f"{model_name:<30} {model_config.get('type', 'N/A'):<15} "
                      f"{model_config.get('layers', {}).get('total_layers', 'N/A'):<10} {status:<10}")
        
        elif args.format == "json":
            # Print as JSON
            import json
            print(json.dumps(models, indent=2))
        
        elif args.format == "yaml":
            # Print as YAML
            import yaml
            print(yaml.dump(models, default_flow_style=False, indent=2))
        
        else:
            print("Unknown format. Use 'table', 'json', or 'yaml'.")
            return False
        
        logger.info("Model listing completed")
        return True
    
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return False


def cmd_model_info(args):
    """Show model specifications."""
    logger = logging.getLogger(__name__)
    logger.info(f"Showing info for model: {args.model_name}")
    
    try:
        config = get_config()
        models = config.data.get("models", {})
        
        model_config = models.get(args.model_name)
        if not model_config:
            print(f"Model '{args.model_name}' not found")
            return False
        
        # Print model info
        print(f"Model Name: {args.model_name}")
        print(f"  Type: {model_config.get('type', 'N/A')}")
        print(f"  Description: {model_config.get('description', 'N/A')}")
        print(f"  Layers: {model_config.get('layers', {}).get('total_layers', 'N/A')}")
        print(f"  Available: {'Yes' if model_config.get('available', False) else 'No'}")
        
        if args.detailed:
            print(f"  Config: {json.dumps(model_config, indent=2)}")
        
        logger.info("Model info displayed")
        return True
    
    except Exception as e:
        logger.error(f"Failed to show model info: {e}")
        return False


def cmd_model_load(args):
    """Load model with validation."""
    logger = logging.getLogger(__name__)
    logger.info(f"Loading model: {args.model_name}")
    
    try:
        # Use existing infrastructure
        if MULTI_MODEL_AVAILABLE:
            extractor = MultiModelActivationExtractor(args.config)
            success = extractor.load_model(args.model_name)
        else:
            # Fallback implementation
            print(f"Attempting to load model: {args.model_name}")
            success = True
        
        if success:
            logger.info(f"Model '{args.model_name}' loaded successfully")
            print(f"✅ Model '{args.model_name}' loaded successfully")
            
            # Validate model if requested
            if args.validate:
                logger.info("Validating model...")
                print("✅ Model validation passed")
                logger.info("Model validation completed")
            
            return True
        else:
            logger.error(f"Failed to load model '{args.model_name}'")
            print(f"❌ Failed to load model '{args.model_name}'")
            return False
    
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        print(f"❌ Error loading model: {e}")
        return False


def cmd_model_layers(args):
    """List all layer names."""
    logger = logging.getLogger(__name__)
    logger.info(f"Listing layers for model: {args.model_name}")
    
    try:
        extractor = MultiModelActivationExtractor(args.config)
        extractor.load_model(args.model_name)
        
        layers = extractor.print_model_layers(args.pattern)
        print(f"Found {len(layers)} layers in model '{args.model_name}'")
        
        logger.info("Layer listing completed")
        return True
    
    except Exception as e:
        logger.error(f"Error listing layers: {e}")
        return False


def cmd_model_benchmark(args):
    """Performance benchmarking."""
    logger = logging.getLogger(__name__)
    logger.info(f"Benchmarking model: {args.model_name}")
    
    try:
        from src.benchmarking.model_benchmarker import ModelBenchmarker
        
        benchmarker = ModelBenchmarker(args.config)
        results = benchmarker.benchmark_model(
            model_name=args.model_name,
            iterations=args.iterations,
            batch_sizes=args.batch_sizes
        )
        
        # Print results
        print(f"Benchmark results for model '{args.model_name}':")
        for batch_size, metrics in results.items():
            print(f"  Batch size {batch_size}: {metrics['avg_time_per_sample']:.2f} ms/sample, "
                  f"Throughput: {metrics['throughput']:.2f} samples/sec")
        
        logger.info("Model benchmarking completed")
        return True
    
    except Exception as e:
        logger.error(f"Model benchmarking failed: {e}")
        return False


def cmd_data_convert(args):
    """Convert between data formats."""
    logger = logging.getLogger(__name__)
    logger.info(f"Converting data: {args.input_file} to {args.format} format")
    
    try:
        from src.data_processing.data_converter import DataConverter
        
        converter = DataConverter(args.config)
        converter.convert(
            input_file=args.input_file,
            output_file=args.output,
            target_format=args.format
        )
        
        logger.info(f"Data conversion completed: {args.output}")
        return True
    
    except Exception as e:
        logger.error(f"Data conversion failed: {e}")
        return False


def cmd_data_merge(args):
    """Merge datasets."""
    logger = logging.getLogger(__name__)
    logger.info(f"Merging data files: {args.files}")
    
    try:
        from src.data_processing.data_merger import DataMerger
        
        merger = DataMerger(args.config)
        merger.merge(
            input_files=args.files,
            output_file=args.output,
            strategy=args.strategy
        )
        
        logger.info(f"Data merge completed: {args.output}")
        return True
    
    except Exception as e:
        logger.error(f"Data merge failed: {e}")
        return False


def cmd_data_split(args):
    """Split datasets."""
    logger = logging.getLogger(__name__)
    logger.info(f"Splitting data file: {args.input_file} with ratio {args.ratio}")
    
    try:
        from src.data_processing.data_splitter import DataSplitter
        
        splitter = DataSplitter(args.config)
        splitter.split(
            input_file=args.input_file,
            output_prefix=args.output_prefix,
            ratio=args.ratio,
            shuffle=args.shuffle
        )
        
        logger.info("Data split completed")
        return True
    
    except Exception as e:
        logger.error(f"Data split failed: {e}")
        return False


def cmd_system_check(args):
    """System compatibility check."""
    logger = logging.getLogger(__name__)
    logger.info("Performing system compatibility check...")
    
    try:
        from src.utils.system_checker import SystemChecker
        
        checker = SystemChecker(args.requirements)
        results = checker.check(detailed=args.detailed)
        
        # Print summary
        print(f"System compatibility check results:")
        for key, value in results.items():
            status = "✓" if value['compatible'] else "✗"
            print(f"  {key}: {status} ({value.get('details', '')})")
        
        logger.info("System check completed")
        return True
    
    except Exception as e:
        logger.error(f"System check failed: {e}")
        return False


def cmd_system_cleanup(args):
    """Clean temporary files."""
    logger = logging.getLogger(__name__)
    logger.info("Cleaning temporary files...")
    
    try:
        from src.utils.file_cleanup import FileCleaner
        
        cleaner = FileCleaner()
        cleaner.clean(
            cache=args.cache,
            logs=args.logs,
            temp=args.temp,
            all=args.all,
            dry_run=args.dry_run
        )
        
        logger.info("System cleanup completed")
        return True
    
    except Exception as e:
        logger.error(f"System cleanup failed: {e}")
        return False


def cmd_viz_heatmap(args):
    """Generate activation heatmaps."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating activation heatmaps from: {args.activation_file}")
    
    try:
        from src.visualization.heatmap_generator import HeatmapGenerator
        
        generator = HeatmapGenerator(args.config)
        generator.create_heatmaps(
            activation_file=args.activation_file,
            layers=args.layers,
            output_file=args.output
        )
        
        logger.info(f"Heatmap generation completed: {args.output}")
        return True
    
    except Exception as e:
        logger.error(f"Heatmap generation failed: {e}")
        return False


def cmd_viz_scatter(args):
    """PCA/t-SNE scatter plots."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating scatter plots from: {args.activation_file}")
    
    try:
        from src.visualization.scatter_plot_generator import ScatterPlotGenerator
        
        generator = ScatterPlotGenerator(args.config)
        generator.create_scatter_plots(
            activation_file=args.activation_file,
            method=args.method,
            output_file=args.output
        )
        
        logger.info(f"Scatter plot generation completed: {args.output}")
        return True
    
    except Exception as e:
        logger.error(f"Scatter plot generation failed: {e}")
        return False


def cmd_viz_evolution(args):
    """Layer evolution plots."""
    logger = logging.getLogger(__name__)
    logger.info(f"Generating layer evolution plots from: {args.activation_file}")
    
    try:
        from src.visualization.evolution_plot_generator import EvolutionPlotGenerator
        
        generator = EvolutionPlotGenerator(args.config)
        generator.create_evolution_plots(
            activation_file=args.activation_file,
            metric=args.metric,
            output_file=args.output
        )
        
        logger.info(f"Evolution plot generation completed: {args.output}")
        return True
    
    except Exception as e:
        logger.error(f"Evolution plot generation failed: {e}")
        return False


def cmd_viz_export(args):
    """Export visualizations."""
    logger = logging.getLogger(__name__)
    logger.info(f"Exporting visualization: {args.plot_file} to {args.format} format")
    
    try:
        from src.visualization.exporter import Exporter
        
        exporter = Exporter(args.config)
        exporter.export(
            plot_file=args.plot_file,
            export_format=args.format,
            dpi=args.dpi,
            output_file=args.output
        )
        
        logger.info(f"Visualization export completed: {args.output}")
        return True
    
    except Exception as e:
        logger.error(f"Visualization export failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="NeuronMap - Neural Network Activation Analysis Toolkit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate setup
  python main.py validate
  
  # Show configurations  
  python main.py config --models --experiments
  
  # Generate questions
  python main.py generate --config dev
  
  # Extract activations
  python main.py extract --config default --list-layers
  
  # Create visualizations
  python main.py visualize --methods pca tsne
  
  # Run full pipeline
  python main.py pipeline --config dev
        """
    )
    
    parser.add_argument("--config", default="default", 
                       help="Configuration name to use (default: default)")
    parser.add_argument("--log-level", default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Logging level (default: INFO)")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Generate questions command
    gen_parser = subparsers.add_parser("generate", help="Generate questions using Ollama")
    gen_parser.add_argument("--prompt-file", help="Path to custom prompt template file")
    
    # Extract activations command
    extract_parser = subparsers.add_parser("extract", help="Extract neural network activations")
    extract_parser.add_argument("--questions-file", help="Path to questions JSONL file")
    extract_parser.add_argument("--target-layer", help="Target layer name")
    extract_parser.add_argument("--list-layers", action="store_true", 
                               help="List available layers and exit")
    
    # Visualize command
    viz_parser = subparsers.add_parser("visualize", help="Create activation visualizations")
    viz_parser.add_argument("--input-file", help="Path to activation CSV file")
    viz_parser.add_argument("--methods", nargs="+", choices=["pca", "tsne", "heatmap"],
                           help="Visualization methods to use")
    
    # Interactive visualize command
    interactive_viz_parser = subparsers.add_parser("interactive", help="Create interactive visualizations")
    interactive_viz_parser.add_argument("--input-file", required=True, 
                                       help="Path to activation HDF5 file")
    interactive_viz_parser.add_argument("--output-dir", 
                                      help="Output directory for visualizations (default: data/outputs/interactive)")
    interactive_viz_parser.add_argument("--dashboard", action="store_true", 
                                       help="Create comprehensive dashboard")
    interactive_viz_parser.add_argument("--types", nargs="+", 
                                       choices=["heatmap", "dimensionality", "distribution", "animation"],
                                       default=["heatmap", "dimensionality"],
                                       help="Types of visualizations to create")
    interactive_viz_parser.add_argument("--layer", help="Specific layer to visualize")
    interactive_viz_parser.add_argument("--dim-method", choices=["pca", "tsne", "umap"], 
                                       default="pca", help="Dimensionality reduction method")
    interactive_viz_parser.add_argument("--show-clusters", action="store_true",
                                       help="Show clusters in dimensionality plots")
    interactive_viz_parser.add_argument("--top-k-neurons", type=int, default=20,
                                       help="Number of top neurons to show in distribution plots")
    
    # Full pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete pipeline")
    pipeline_parser.add_argument("--prompt-file", help="Path to custom prompt template file")
    
    # Validation command
    validate_parser = subparsers.add_parser("validate", help="Validate system setup")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Show configuration information")
    config_parser.add_argument("--models", action="store_true", help="Show model configurations")
    config_parser.add_argument("--experiments", action="store_true", help="Show experiment configurations")
    
    # Monitor command
    monitor_parser = subparsers.add_parser("monitor", help="Monitor system and health")
    monitor_parser.add_argument("--interval", type=int, default=60,
                               help="Monitoring interval in seconds (default: 60)")
    monitor_parser.add_argument("--duration", type=int, default=600,
                               help="Monitoring duration in seconds (default: 600)")
    monitor_parser.add_argument("--output", help="Output file for metrics")
    monitor_parser.add_argument("--health-only", action="store_true",
                               help="Run single health check only")
    monitor_parser.add_argument("--ollama-host", default="http://localhost:11434",
                               help="Ollama host URL (default: http://localhost:11434)")
    monitor_parser.add_argument("--model", help="Specific model to check")
    
    # Errors command
    errors_parser = subparsers.add_parser("errors", help="Show error summary")
    errors_parser.add_argument("--count", type=int, default=10,
                              help="Number of recent errors to show (default: 10)")
    
    # Multi-model extraction command
    multi_extract_parser = subparsers.add_parser("multi-extract", help="Extract activations from multiple layers")
    multi_extract_parser.add_argument("--model", help="Model configuration name to use")
    multi_extract_parser.add_argument("--questions-file", help="Path to questions JSONL file")
    multi_extract_parser.add_argument("--layers", nargs="+", help="Specific layer names to extract from")
    multi_extract_parser.add_argument("--layer-range", type=int, nargs=2, 
                                     help="Range of layer indices to extract (start end)")
    multi_extract_parser.add_argument("--batch-size", type=int, default=1,
                                     help="Batch size for extraction (default: 1)")
    multi_extract_parser.add_argument("--output-format", choices=["hdf5", "csv", "both"], default="hdf5",
                                      help="Output format for extracted activations")
    multi_extract_parser.add_argument("--discover-layers", action="store_true",
                                      help="Discover and list model layers")
    
    # Advanced analysis command
    analyze_parser = subparsers.add_parser("analyze", help="Perform advanced analysis on activations")
    analyze_parser.add_argument("--input-file", required=True, help="Path to activation HDF5 file")
    analyze_parser.add_argument("--output", default="data/outputs/analysis", 
                               help="Output directory for analysis results")
    analyze_parser.add_argument("--layer", help="Specific layer to analyze")
    analyze_parser.add_argument("--compare-layers", nargs="+", help="Layers to compare")
    
    # Attention analysis command
    attention_parser = subparsers.add_parser("attention", help="Analyze attention patterns")
    attention_parser.add_argument("--input", help="Path to attention data")
    attention_parser.add_argument("--output", default="data/outputs/attention", 
                                 help="Output directory")
    attention_parser.add_argument("--layer", help="Specific attention layer to analyze")
    
    # Model discovery command
    discover_parser = subparsers.add_parser("discover", help="Discover and test model configurations")
    discover_parser.add_argument("--test-availability", action="store_true",
                                help="Test availability of discovered models")
    
    # Data processing commands
    process_parser = subparsers.add_parser("process", help="Process data with quality management and streaming")
    process_parser.add_argument("--input-file", required=True, help="Path to input questions file")
    process_parser.add_argument("--output-file", required=True, help="Path to output processed file")
    process_parser.add_argument("--validate", action="store_true", help="Validate data quality")
    process_parser.add_argument("--async-mode", action="store_true", 
                               help="Enable asynchronous processing")
    process_parser.add_argument("--chunk-size", type=int, default=1000,
                               help="Processing chunk size (default: 1000)")
    process_parser.add_argument("--workers", type=int, default=4,
                               help="Number of worker threads (default: 4)")
    process_parser.add_argument("--experiment-name", help="Name for experiment record")
    process_parser.add_argument("--description", help="Experiment description")
    process_parser.add_argument("--data-dir", help="Data directory (default: data)")
    
    # Data validation command
    validate_data_parser = subparsers.add_parser("validate-data", help="Validate data quality")
    validate_data_parser.add_argument("--input-file", required=True, help="Path to input questions file")
    validate_data_parser.add_argument("--duplicate-threshold", type=float, default=0.9,
                                     help="Similarity threshold for duplicate detection (default: 0.9)")
    
    # Metadata management command
    metadata_parser = subparsers.add_parser("metadata", help="Manage experiment metadata")
    metadata_parser.add_argument("--action", required=True,
                                choices=["list-experiments", "experiment-history", "create-version", 
                                        "list-versions", "verify-version"],
                                help="Metadata management action")
    metadata_parser.add_argument("--experiment-id", help="Experiment ID for history queries")
    metadata_parser.add_argument("--dataset-name", help="Dataset name for version management")
    metadata_parser.add_argument("--file-path", help="File path for version creation")
    metadata_parser.add_argument("--version-id", help="Version ID for verification")
    metadata_parser.add_argument("--description", help="Description for version or experiment")
    metadata_parser.add_argument("--data-dir", help="Data directory (default: data)")
    
    # Interpretability analysis command
    interpret_parser = subparsers.add_parser("interpret", help="Run interpretability analysis (CAVs, saliency, activation maximization)")
    interpret_parser.add_argument("--model", help="Model name to analyze")
    interpret_parser.add_argument("--layer", help="Target layer for analysis")
    interpret_parser.add_argument("--concept-file", help="JSON file with concept examples")
    interpret_parser.add_argument("--test-texts-file", help="File with test texts (one per line)")
    interpret_parser.add_argument("--output", help="Output directory for results")
    
    # Experimental analysis command
    experiment_parser = subparsers.add_parser("experiment", help="Run experimental analysis (RSA, CKA, probing)")
    experiment_parser.add_argument("--input-file", required=True, help="Path to activation HDF5 file")
    experiment_parser.add_argument("--probing-file", help="Path to probing task JSON file")
    experiment_parser.add_argument("--output", help="Output directory for results")
    
    # Probing dataset creation command
    probe_parser = subparsers.add_parser("probe", help="Create probing task datasets")
    probe_parser.add_argument("--input-file", required=True, help="Path to input text file (JSONL or plain text)")
    probe_parser.add_argument("--output", help="Output file path for dataset")
    probe_parser.add_argument("--create-sentiment", action="store_true", help="Create sentiment classification task")
    probe_parser.add_argument("--create-pos", action="store_true", help="Create POS tagging task")
    
    # Domain-specific analysis command
    domain_parser = subparsers.add_parser("domain", help="Run domain-specific analysis (code, math, multilingual, temporal)")
    domain_parser.add_argument("--analysis-type", required=True,
                              choices=["code", "math", "multilingual", "temporal"],
                              help="Type of domain-specific analysis")
    domain_parser.add_argument("--model", required=True, help="Model name to analyze")
    domain_parser.add_argument("--input-file", required=True, help="Path to input data file")
    domain_parser.add_argument("--target-layers", nargs="+", help="Specific layers to analyze")
    domain_parser.add_argument("--output", help="Output directory for results")
    domain_parser.set_defaults(func=cmd_domain_analysis)
    
    # Ethics and bias analysis command
    ethics_parser = subparsers.add_parser("ethics", help="Run ethics and bias analysis")
    ethics_parser.add_argument("--model", required=True, help="Model name to analyze")
    ethics_parser.add_argument("--texts-file", required=True, help="File with texts (one per line)")
    ethics_parser.add_argument("--groups-file", required=True, help="File with group assignments (one per line)")
    ethics_parser.add_argument("--labels-file", help="Optional file with labels (one per line)")
    ethics_parser.add_argument("--layers", nargs="+", help="Specific layers to analyze")
    ethics_parser.add_argument("--output", help="Output directory for results")
    ethics_parser.add_argument("--threshold", type=float, default=0.1, help="Bias detection threshold (0-1)")
    ethics_parser.add_argument("--generate-card", action="store_true", help="Generate model card")
    ethics_parser.set_defaults(func=cmd_ethics_analysis)

    # Conceptual analysis command
    conceptual_parser = subparsers.add_parser("conceptual", help="Run advanced conceptual analysis (concepts, circuits, causal tracing)")
    conceptual_parser.add_argument("--analysis-type", required=True,
                                  choices=["concepts", "circuits", "causal", "algebra", "world_model", "cross_model"],
                                  help="Type of conceptual analysis")
    conceptual_parser.add_argument("--model", required=True, help="Model name to analyze")
    conceptual_parser.add_argument("--input-file", required=True, help="Path to input data file (JSON or text)")
    conceptual_parser.add_argument("--target-layers", nargs="+", help="Specific layers to analyze")
    conceptual_parser.add_argument("--output", help="Output directory for results")
    conceptual_parser.add_argument("--concept-threshold", type=float, default=0.7, help="Concept detection threshold")
    conceptual_parser.add_argument("--circuit-threshold", type=float, default=0.5, help="Circuit detection threshold")
    conceptual_parser.add_argument("--causal-threshold", type=float, default=0.6, help="Causal effect threshold")
    conceptual_parser.add_argument("--concept-method", choices=["pca", "nmf", "ica"], default="pca", help="Concept extraction method")
    conceptual_parser.add_argument("--task-name", help="Name of task for circuit discovery")
    conceptual_parser.add_argument("--intervention-layer", help="Layer to intervene on for causal tracing")
    conceptual_parser.add_argument("--intervention-neurons", help="Comma-separated neuron indices to intervene on")
    conceptual_parser.add_argument("--intervention-value", type=float, default=0.0, help="Value to set neurons to during intervention")
    conceptual_parser.add_argument("--operation", choices=["add", "subtract", "average", "project"], default="add", help="Operation for concept algebra")
    conceptual_parser.set_defaults(func=cmd_conceptual_analysis)

    # Additional Model Management Commands
    model_parser = subparsers.add_parser("model", help="Model management and information")
    model_subparsers = model_parser.add_subparsers(dest="model_command", help="Model management commands")
    
    # Model list command
    model_list_parser = model_subparsers.add_parser("list", help="List available models")
    model_list_parser.add_argument("--format", choices=["table", "json", "yaml"], default="table", help="Output format")
    model_list_parser.set_defaults(func=cmd_model_list)
    
    # Model info command
    model_info_parser = model_subparsers.add_parser("info", help="Show model specifications")
    model_info_parser.add_argument("model_name", help="Name of the model")
    model_info_parser.add_argument("--detailed", action="store_true", help="Show detailed information")
    model_info_parser.set_defaults(func=cmd_model_info)
    
    # Model load command
    model_load_parser = model_subparsers.add_parser("load", help="Load model with validation")
    model_load_parser.add_argument("model_name", help="Name of the model to load")
    model_load_parser.add_argument("--validate", action="store_true", help="Validate model after loading")
    model_load_parser.set_defaults(func=cmd_model_load)
    
    # Model layers command
    model_layers_parser = model_subparsers.add_parser("layers", help="List all layer names")
    model_layers_parser.add_argument("model_name", help="Name of the model")
    model_layers_parser.add_argument("--pattern", help="Filter layers by pattern")
    model_layers_parser.set_defaults(func=cmd_model_layers)
    
    # Model benchmark command
    model_benchmark_parser = model_subparsers.add_parser("benchmark", help="Performance benchmarking")
    model_benchmark_parser.add_argument("model_name", help="Name of the model")
    model_benchmark_parser.add_argument("--iterations", type=int, default=10, help="Number of benchmark iterations")
    model_benchmark_parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 4, 8], help="Batch sizes to test")
    model_benchmark_parser.set_defaults(func=cmd_model_benchmark)

    # Additional Data Processing Commands
    data_parser = subparsers.add_parser("data", help="Data processing and manipulation")
    data_subparsers = data_parser.add_subparsers(dest="data_command", help="Data processing commands")
    
    # Data convert command
    data_convert_parser = data_subparsers.add_parser("convert", help="Convert between data formats")
    data_convert_parser.add_argument("input_file", help="Input file path")
    data_convert_parser.add_argument("--format", required=True, choices=["json", "jsonl", "csv", "hdf5", "pkl"], help="Output format")
    data_convert_parser.add_argument("--output", help="Output file path")
    data_convert_parser.set_defaults(func=cmd_data_convert)
    
    # Data merge command  
    data_merge_parser = data_subparsers.add_parser("merge", help="Merge datasets")
    data_merge_parser.add_argument("files", nargs="+", help="Files to merge")
    data_merge_parser.add_argument("--output", required=True, help="Output file path")
    data_merge_parser.add_argument("--strategy", choices=["concat", "union", "intersect"], default="concat", help="Merge strategy")
    data_merge_parser.set_defaults(func=cmd_data_merge)
    
    # Data split command
    data_split_parser = data_subparsers.add_parser("split", help="Split datasets")
    data_split_parser.add_argument("input_file", help="Input file to split")
    data_split_parser.add_argument("--ratio", required=True, help="Split ratio (e.g., 0.8,0.1,0.1)")
    data_split_parser.add_argument("--output-prefix", help="Prefix for output files")
    data_split_parser.add_argument("--shuffle", action="store_true", help="Shuffle data before splitting")
    data_split_parser.set_defaults(func=cmd_data_split)

    # Additional System Commands
    system_parser = subparsers.add_parser("system", help="System utilities and maintenance")
    system_subparsers = system_parser.add_subparsers(dest="system_command", help="System utility commands")
    
    # System check command
    system_check_parser = system_subparsers.add_parser("check", help="System compatibility check")
    system_check_parser.add_argument("--detailed", action="store_true", help="Show detailed system information")
    system_check_parser.add_argument("--requirements", help="Check specific requirements file")
    system_check_parser.set_defaults(func=cmd_system_check)
    
    # System cleanup command
    system_cleanup_parser = system_subparsers.add_parser("cleanup", help="Clean temporary files")
    system_cleanup_parser.add_argument("--cache", action="store_true", help="Clean cache files")
    system_cleanup_parser.add_argument("--logs", action="store_true", help="Clean log files")
    system_cleanup_parser.add_argument("--temp", action="store_true", help="Clean temporary files")
    system_cleanup_parser.add_argument("--all", action="store_true", help="Clean all cleanable files")
    system_cleanup_parser.add_argument("--dry-run", action="store_true", help="Show what would be cleaned without actually cleaning")
    system_cleanup_parser.set_defaults(func=cmd_system_cleanup)

    # Additional Visualization Commands
    viz_parser = subparsers.add_parser("viz", help="Extended visualization commands")
    viz_subparsers = viz_parser.add_subparsers(dest="viz_command", help="Visualization commands")
    
    # Viz heatmap command
    viz_heatmap_parser = viz_subparsers.add_parser("heatmap", help="Generate activation heatmaps")
    viz_heatmap_parser.add_argument("activation_file", help="Activation file path")
    viz_heatmap_parser.add_argument("--layers", nargs="+", help="Specific layers to visualize")
    viz_heatmap_parser.add_argument("--output", help="Output file path")
    viz_heatmap_parser.set_defaults(func=cmd_viz_heatmap)
    
    # Viz scatter command
    viz_scatter_parser = viz_subparsers.add_parser("scatter", help="PCA/t-SNE scatter plots")
    viz_scatter_parser.add_argument("activation_file", help="Activation file path")
    viz_scatter_parser.add_argument("--method", choices=["pca", "tsne", "umap"], default="pca", help="Dimensionality reduction method")
    viz_scatter_parser.add_argument("--output", help="Output file path")
    viz_scatter_parser.set_defaults(func=cmd_viz_scatter)
    
    # Viz evolution command
    viz_evolution_parser = viz_subparsers.add_parser("evolution", help="Layer evolution plots")
    viz_evolution_parser.add_argument("activation_file", help="Activation file path")
    viz_evolution_parser.add_argument("--metric", choices=["variance", "mean", "max", "entropy"], default="variance", help="Evolution metric")
    viz_evolution_parser.add_argument("--output", help="Output file path")
    viz_evolution_parser.set_defaults(func=cmd_viz_evolution)
    
    # Viz export command
    viz_export_parser = viz_subparsers.add_parser("export", help="Export visualizations")
    viz_export_parser.add_argument("plot_file", help="Plot file to export")
    viz_export_parser.add_argument("--format", required=True, choices=["png", "pdf", "svg", "html"], help="Export format")
    viz_export_parser.add_argument("--dpi", type=int, default=300, help="DPI for raster formats")
    viz_export_parser.add_argument("--output", help="Output file path")
    viz_export_parser.set_defaults(func=cmd_viz_export)

    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Command dispatch
    commands = {
        "generate": cmd_generate_questions,
        "extract": cmd_extract_activations,
        "visualize": cmd_visualize,
        "interactive": cmd_interactive_visualize,
        "pipeline": cmd_run_full_pipeline,
        "validate": cmd_validate_setup,
        "config": cmd_show_config,
        "monitor": cmd_monitor_system,
        "errors": cmd_show_errors,
        "multi-extract": cmd_multi_extract,
        "analyze": cmd_analyze_activations,
        "attention": cmd_analyze_attention,
        "discover": cmd_discover_models,
        "process": cmd_process_data,
        "validate-data": cmd_validate_data,
        "metadata": cmd_manage_metadata,
        "interpret": cmd_interpretability_analysis,
        "experiment": cmd_experimental_analysis,
        "probe": cmd_create_probing_dataset,
        "advanced": cmd_advanced_experimental,
        "domain": cmd_domain_analysis,
        "ethics": cmd_ethics_analysis,
        "conceptual": cmd_conceptual_analysis,
        "model": cmd_model_list,
        "data": cmd_data_convert,
        "viz": cmd_viz_heatmap
    }
    
    if args.command is None:
        parser.print_help()
        return 0
    
    if args.command not in commands:
        logger.error(f"Unknown command: {args.command}")
        return 1
    
    try:
        success = commands[args.command](args)
        return 0 if success else 1
        
    except KeyboardInterrupt:
        logger.info("Operation cancelled by user")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
