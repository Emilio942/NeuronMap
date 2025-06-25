#!/usr/bin/env python3
"""
Modern Visualizer Script - Enhanced Activation Visualization
===========================================================

Migrated from legacy visualizer.py with enhancements:
- Modern modular architecture
- Enhanced visualization options
- Configuration system integration
- Multiple dimensionality reduction techniques
- Interactive plotting capabilities
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.visualization.visualizer import ActivationVisualizer
from src.utils.config_manager import ConfigManager

def main():
    """Enhanced main function with CLI support."""
    parser = argparse.ArgumentParser(description="NeuronMap Activation Visualization")
    parser.add_argument("--input", default="activation_results.csv", help="Input CSV file")
    parser.add_argument("--output-dir", default="visualizations", help="Output directory")
    parser.add_argument("--method", choices=["pca", "tsne", "both"], default="both", help="Visualization method")
    parser.add_argument("--config", default="default", help="Configuration name")
    
    args = parser.parse_args()
    
    # Initialize visualizer
    visualizer = ActivationVisualizer(
        config_name=args.config,
        output_dir=args.output_dir
    )
    
    # Load and prepare data
    activation_matrix, df = visualizer.load_and_prepare_data(args.input)
    
    # Create visualizations
    if args.method in ["pca", "both"]:
        pca_result = visualizer.run_pca(activation_matrix)
        visualizer.plot_scatter(
            pca_result, 
            "PCA Visualization of Neural Activations",
            Path(args.output_dir) / "activation_pca_scatter.png",
            df
        )
    
    if args.method in ["tsne", "both"]:
        tsne_result = visualizer.run_tsne(activation_matrix)
        visualizer.plot_scatter(
            tsne_result,
            "t-SNE Visualization of Neural Activations", 
            Path(args.output_dir) / "activation_tsne_scatter.png",
            df
        )
    
    print(f"Visualizations saved to {args.output_dir}")

if __name__ == "__main__":
    main()
