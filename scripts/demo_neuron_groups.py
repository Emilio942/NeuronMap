#!/usr/bin/env python3
"""
Neuron Group Analysis Demo Script
================================

This script demonstrates how to use the NeuronGroupVisualizer to identify and 
visualize groups of neurons that activate together during learning tasks.

Usage:
    python scripts/demo_neuron_groups.py
"""

import sys
import logging
from pathlib import Path
import numpy as np
import pandas as pd

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from visualization.neuron_group_visualizer import (
    NeuronGroupVisualizer, 
    create_neuron_group_analysis
)
from utils.config import get_config

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_demo_data(n_samples: int = 100, n_neurons: int = 50) -> tuple:
    """Generate synthetic neural activation data for demonstration.
    
    Args:
        n_samples: Number of samples/questions
        n_neurons: Number of neurons
        
    Returns:
        Tuple of (activation_matrix, question_metadata)
    """
    logger.info(f"Generating demo data: {n_samples} samples, {n_neurons} neurons")
    
    # Create synthetic neuron groups
    np.random.seed(42)
    
    # Define 4 synthetic neuron groups with different activation patterns
    group_configs = [
        {'neurons': list(range(0, 12)), 'skill': 'mathematical', 'strength': 0.8},
        {'neurons': list(range(12, 22)), 'skill': 'linguistic', 'strength': 0.7},
        {'neurons': list(range(22, 32)), 'skill': 'logical', 'strength': 0.6},
        {'neurons': list(range(32, 40)), 'skill': 'memory', 'strength': 0.5}
    ]
    
    activation_matrix = np.random.normal(0.2, 0.1, (n_samples, n_neurons))
    
    # Create question metadata
    questions = []
    categories = []
    
    for i in range(n_samples):
        # Randomly choose which group(s) should be active for this sample
        active_group = np.random.choice(len(group_configs))
        group_config = group_configs[active_group]
        
        # Set higher activations for neurons in the active group
        for neuron_idx in group_config['neurons']:
            activation_matrix[i, neuron_idx] = np.random.normal(
                group_config['strength'], 0.15
            )
        
        # Sometimes activate multiple groups (complex tasks)
        if np.random.random() < 0.3:  # 30% chance of multi-group activation
            secondary_group = np.random.choice(len(group_configs))
            if secondary_group != active_group:
                secondary_config = group_configs[secondary_group]
                for neuron_idx in secondary_config['neurons']:
                    activation_matrix[i, neuron_idx] += np.random.normal(
                        secondary_config['strength'] * 0.6, 0.1
                    )
        
        # Generate question text and category
        skill = group_config['skill']
        categories.append(skill)
        
        if skill == 'mathematical':
            questions.append(f"What is {np.random.randint(1, 100)} + {np.random.randint(1, 100)}?")
        elif skill == 'linguistic':
            questions.append(f"What is the meaning of the word '{['cat', 'dog', 'tree', 'house'][np.random.randint(0, 4)]}'?")
        elif skill == 'logical':
            questions.append(f"If A implies B and B implies C, what can we conclude about A and C?")
        else:  # memory
            questions.append(f"What is the capital of {['France', 'Germany', 'Italy', 'Spain'][np.random.randint(0, 4)]}?")
    
    # Ensure all values are positive (typical for neural activations)
    activation_matrix = np.maximum(activation_matrix, 0.01)
    
    # Create metadata DataFrame
    question_metadata = pd.DataFrame({
        'question_id': range(n_samples),
        'question': questions,
        'category': categories,
        'sample_index': range(n_samples)
    })
    
    logger.info("Demo data generation completed")
    return activation_matrix, question_metadata


def demonstrate_basic_group_identification():
    """Demonstrate basic neuron group identification."""
    logger.info("=== Demonstrating Basic Group Identification ===")
    
    # Generate demo data
    activation_matrix, question_metadata = generate_demo_data(n_samples=80, n_neurons=45)
    
    # Initialize visualizer
    visualizer = NeuronGroupVisualizer(output_dir="demo_outputs/basic_groups")
    
    # Identify neuron groups using different methods
    methods = ['correlation_clustering', 'kmeans', 'hierarchical']
    
    for method in methods:
        logger.info(f"Testing {method} clustering...")
        
        neuron_groups = visualizer.identify_neuron_groups(
            activation_matrix,
            method=method,
            correlation_threshold=0.5 if method == 'correlation_clustering' else None,
            n_groups=4 if method != 'correlation_clustering' else None,
            min_group_size=3
        )
        
        logger.info(f"  Found {len(neuron_groups)} groups with {method}")
        for group in neuron_groups:
            logger.info(f"    Group {group.group_id}: {group.group_size} neurons, "
                       f"cohesion: {group.cohesion_score:.3f}")


def demonstrate_learning_pattern_analysis():
    """Demonstrate learning pattern analysis."""
    logger.info("=== Demonstrating Learning Pattern Analysis ===")
    
    # Generate demo data with temporal structure
    activation_matrix, question_metadata = generate_demo_data(n_samples=120, n_neurons=50)
    
    # Initialize visualizer
    visualizer = NeuronGroupVisualizer(output_dir="demo_outputs/learning_patterns")
    
    # Identify neuron groups
    neuron_groups = visualizer.identify_neuron_groups(
        activation_matrix,
        method='correlation_clustering',
        correlation_threshold=0.6,
        min_group_size=4
    )
    
    logger.info(f"Identified {len(neuron_groups)} neuron groups")
    
    # Analyze learning patterns
    learning_events = visualizer.analyze_learning_patterns(
        activation_matrix, neuron_groups, question_metadata
    )
    
    logger.info(f"Identified {len(learning_events)} learning events")
    
    # Analyze skill distribution
    skill_counts = {}
    for event in learning_events:
        skill_counts[event.skill_type] = skill_counts.get(event.skill_type, 0) + 1
    
    logger.info("Learning event distribution by skill type:")
    for skill, count in skill_counts.items():
        logger.info(f"  {skill}: {count} events")


def demonstrate_visualizations():
    """Demonstrate different visualization methods."""
    logger.info("=== Demonstrating Visualizations ===")
    
    # Generate demo data
    activation_matrix, question_metadata = generate_demo_data(n_samples=100, n_neurons=60)
    
    # Initialize visualizer
    visualizer = NeuronGroupVisualizer(output_dir="demo_outputs/visualizations")
    
    # Identify neuron groups
    neuron_groups = visualizer.identify_neuron_groups(
        activation_matrix,
        method='correlation_clustering',
        correlation_threshold=0.55,
        min_group_size=3
    )
    
    logger.info(f"Creating visualizations for {len(neuron_groups)} groups...")
    
    # Create different types of visualizations
    visualization_methods = ['heatmap', 'network', 'scatter']
    
    for method in visualization_methods:
        try:
            output_path = visualizer.visualize_neuron_groups(
                activation_matrix, neuron_groups, method=method
            )
            if output_path:
                logger.info(f"  {method.capitalize()} visualization saved to: {output_path}")
            else:
                logger.warning(f"  {method.capitalize()} visualization failed (missing dependencies?)")
        except Exception as e:
            logger.warning(f"  {method.capitalize()} visualization failed: {e}")


def demonstrate_interactive_dashboard():
    """Demonstrate interactive dashboard creation."""
    logger.info("=== Demonstrating Interactive Dashboard ===")
    
    # Generate demo data
    activation_matrix, question_metadata = generate_demo_data(n_samples=150, n_neurons=70)
    
    # Initialize visualizer
    visualizer = NeuronGroupVisualizer(output_dir="demo_outputs/dashboard")
    
    # Identify neuron groups
    neuron_groups = visualizer.identify_neuron_groups(
        activation_matrix,
        method='correlation_clustering',
        correlation_threshold=0.6,
        min_group_size=4
    )
    
    # Analyze learning patterns
    learning_events = visualizer.analyze_learning_patterns(
        activation_matrix, neuron_groups, question_metadata
    )
    
    try:
        # Create interactive dashboard
        dashboard_path = visualizer.create_interactive_group_dashboard(
            activation_matrix, neuron_groups, learning_events, question_metadata
        )
        
        if dashboard_path:
            logger.info(f"Interactive dashboard created: {dashboard_path}")
            logger.info("Open the HTML file in a web browser to explore the dashboard")
        else:
            logger.warning("Interactive dashboard creation failed (Plotly not available?)")
    except Exception as e:
        logger.warning(f"Interactive dashboard creation failed: {e}")


def demonstrate_complete_analysis():
    """Demonstrate complete analysis workflow."""
    logger.info("=== Demonstrating Complete Analysis Workflow ===")
    
    # Generate demo data
    activation_matrix, question_metadata = generate_demo_data(n_samples=200, n_neurons=80)
    
    try:
        # Run complete analysis using convenience function
        results = create_neuron_group_analysis(
            activation_matrix=activation_matrix,
            question_metadata=question_metadata,
            output_dir="demo_outputs/complete_analysis",
            config=None
        )
        
        logger.info("Complete analysis results:")
        logger.info(f"  Neuron groups found: {results['summary']['total_groups']}")
        logger.info(f"  Learning events identified: {results['summary']['total_learning_events']}")
        logger.info(f"  Analysis complete: {results['summary']['analysis_complete']}")
        
        # Log visualization paths
        logger.info("Generated visualizations:")
        for viz_type, path in results['visualizations'].items():
            if path:
                logger.info(f"  {viz_type}: {path}")
        
        logger.info(f"Analysis report: {results['report']}")
        
    except Exception as e:
        logger.error(f"Complete analysis failed: {e}")


def demonstrate_real_data_loading():
    """Demonstrate loading and analyzing real NeuronMap data."""
    logger.info("=== Demonstrating Real Data Analysis ===")
    
    # Try to find actual activation results from NeuronMap
    potential_data_files = [
        "data/processed/activation_results.csv",
        "data/outputs/activation_results.csv",
        "outputs/activation_results.csv"
    ]
    
    activation_file = None
    for file_path in potential_data_files:
        if Path(file_path).exists():
            activation_file = file_path
            break
    
    if activation_file:
        logger.info(f"Found real data file: {activation_file}")
        
        try:
            # Load real data
            df = pd.read_csv(activation_file)
            logger.info(f"Loaded {len(df)} samples from real data")
            
            # Extract activation vectors (assuming they're in 'activation_vector' column)
            if 'activation_vector' in df.columns:
                activation_matrix = []
                valid_indices = []
                
                for idx, row in df.iterrows():
                    try:
                        if isinstance(row['activation_vector'], str):
                            # Parse JSON string
                            import ast
                            activation_vector = ast.literal_eval(row['activation_vector'])
                        else:
                            activation_vector = row['activation_vector']
                        
                        if isinstance(activation_vector, list):
                            activation_matrix.append(activation_vector)
                            valid_indices.append(idx)
                    except Exception as e:
                        logger.warning(f"Error parsing activation vector at row {idx}: {e}")
                
                if activation_matrix:
                    activation_matrix = np.array(activation_matrix)
                    question_metadata = df.loc[valid_indices].reset_index(drop=True)
                    
                    logger.info(f"Extracted activation matrix: {activation_matrix.shape}")
                    
                    # Run analysis on real data
                    results = create_neuron_group_analysis(
                        activation_matrix=activation_matrix,
                        question_metadata=question_metadata,
                        output_dir="demo_outputs/real_data_analysis",
                        config=None
                    )
                    
                    logger.info("Real data analysis completed successfully!")
                    logger.info(f"  Groups found: {results['summary']['total_groups']}")
                    logger.info(f"  Learning events: {results['summary']['total_learning_events']}")
                else:
                    logger.warning("No valid activation vectors found in real data")
            else:
                logger.warning("No 'activation_vector' column found in real data")
                
        except Exception as e:
            logger.error(f"Failed to analyze real data: {e}")
    else:
        logger.info("No real data files found, skipping real data demonstration")
        logger.info("To analyze real data, run NeuronMap activation extraction first")


def main():
    """Main demonstration function."""
    logger.info("Starting Neuron Group Analysis Demonstration")
    logger.info("=" * 60)
    
    # Create demo output directory
    Path("demo_outputs").mkdir(exist_ok=True)
    
    try:
        # Run demonstrations
        demonstrate_basic_group_identification()
        print()
        
        demonstrate_learning_pattern_analysis()
        print()
        
        demonstrate_visualizations()
        print()
        
        demonstrate_interactive_dashboard()
        print()
        
        demonstrate_complete_analysis()
        print()
        
        demonstrate_real_data_loading()
        print()
        
        logger.info("=" * 60)
        logger.info("Demonstration completed successfully!")
        logger.info("Check the 'demo_outputs' directory for generated visualizations and reports")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()
