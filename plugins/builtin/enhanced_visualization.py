"""
Enhanced Visualization Plugin
============================

Advanced visualization capabilities for neural network analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

import sys

# Add src directory to path
src_path = Path(__file__).parent.parent.parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from core.plugin_system import VisualizationPlugin, PluginMetadata

class EnhancedVisualizationPlugin(VisualizationPlugin):
    """Enhanced visualization plugin with advanced plotting capabilities."""
    
    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="Enhanced Visualization",
            version="1.0.0",
            author="NeuronMap Team",
            description="Advanced visualization tools for neural network analysis",
            plugin_type="visualization",
            dependencies=["matplotlib", "seaborn", "plotly", "numpy"],
            tags=["visualization", "plotting", "interactive"],
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat()
        )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the plugin."""
        self.config = config
        self.output_dir = Path(config.get('output_dir', 'data/outputs/plugins/visualization'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style preferences
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        return True
    
    def execute(self, *args, **kwargs) -> Any:
        """Execute visualization creation."""
        return self.create_visualization(*args, **kwargs)
    
    def create_visualization(self, data: Any, config: Dict[str, Any]) -> str:
        """Create comprehensive visualizations."""
        viz_type = config.get('type', 'comprehensive')
        
        if viz_type == 'activation_heatmap':
            return self._create_activation_heatmap(data, config)
        elif viz_type == 'layer_comparison':
            return self._create_layer_comparison(data, config)
        elif viz_type == 'activation_distribution':
            return self._create_activation_distribution(data, config)
        elif viz_type == 'interactive_3d':
            return self._create_interactive_3d(data, config)
        elif viz_type == 'comprehensive':
            return self._create_comprehensive_dashboard(data, config)
        else:
            return self._create_default_visualization(data, config)
    
    def _create_activation_heatmap(self, activations: Dict[str, np.ndarray], config: Dict[str, Any]) -> str:
        """Create detailed activation heatmaps."""
        try:
            num_layers = len(activations)
            if num_layers == 0:
                return ""
            
            # Create subplots
            cols = min(3, num_layers)
            rows = (num_layers + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
            if num_layers == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if hasattr(axes, '__iter__') else [axes]
            else:
                axes = axes.flatten()
            
            for idx, (layer_name, activation_data) in enumerate(activations.items()):
                if idx >= len(axes):
                    break
                
                ax = axes[idx]
                
                if activation_data is not None and len(activation_data) > 0:
                    # Handle different activation shapes
                    if activation_data.ndim > 2:
                        # For higher dimensional data, take mean across spatial dimensions
                        plot_data = np.mean(activation_data, axis=tuple(range(2, activation_data.ndim)))
                    else:
                        plot_data = activation_data
                    
                    # Limit size for visualization
                    if plot_data.shape[0] > 100:
                        plot_data = plot_data[:100]
                    if plot_data.ndim > 1 and plot_data.shape[1] > 100:
                        plot_data = plot_data[:, :100]
                    
                    im = ax.imshow(plot_data, aspect='auto', cmap='viridis', interpolation='nearest')
                    ax.set_title(f'Layer: {layer_name}')
                    ax.set_xlabel('Feature Dimension')
                    ax.set_ylabel('Sample')
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Layer: {layer_name}')
            
            # Hide unused subplots
            for idx in range(len(activations), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            output_path = self.output_dir / f'activation_heatmap_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error creating activation heatmap: {e}")
            return ""
    
    def _create_layer_comparison(self, activations: Dict[str, np.ndarray], config: Dict[str, Any]) -> str:
        """Create layer comparison visualizations."""
        try:
            # Compute statistics for each layer
            layer_stats = {}
            for layer_name, activation_data in activations.items():
                if activation_data is not None and len(activation_data) > 0:
                    flat_data = activation_data.flatten()
                    layer_stats[layer_name] = {
                        'mean': np.mean(flat_data),
                        'std': np.std(flat_data),
                        'min': np.min(flat_data),
                        'max': np.max(flat_data),
                        'median': np.median(flat_data)
                    }
            
            if not layer_stats:
                return ""
            
            # Create comparison plots
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # Mean values
            layers = list(layer_stats.keys())
            means = [layer_stats[layer]['mean'] for layer in layers]
            axes[0,0].bar(layers, means)
            axes[0,0].set_title('Mean Activation by Layer')
            axes[0,0].tick_params(axis='x', rotation=45)
            
            # Standard deviation
            stds = [layer_stats[layer]['std'] for layer in layers]
            axes[0,1].bar(layers, stds)
            axes[0,1].set_title('Standard Deviation by Layer')
            axes[0,1].tick_params(axis='x', rotation=45)
            
            # Range (max - min)
            ranges = [layer_stats[layer]['max'] - layer_stats[layer]['min'] for layer in layers]
            axes[1,0].bar(layers, ranges)
            axes[1,0].set_title('Activation Range by Layer')
            axes[1,0].tick_params(axis='x', rotation=45)
            
            # Box plot comparison (if we have multiple samples)
            sample_data = []
            sample_labels = []
            for layer_name, activation_data in activations.items():
                if activation_data is not None and len(activation_data) > 1:
                    flat_data = activation_data.flatten()
                    # Sample for visualization if too large
                    if len(flat_data) > 1000:
                        sample_indices = np.random.choice(len(flat_data), 1000, replace=False)
                        flat_data = flat_data[sample_indices]
                    sample_data.append(flat_data)
                    sample_labels.append(layer_name)
            
            if sample_data:
                axes[1,1].boxplot(sample_data, labels=sample_labels)
                axes[1,1].set_title('Activation Distribution by Layer')
                axes[1,1].tick_params(axis='x', rotation=45)
            else:
                axes[1,1].text(0.5, 0.5, 'Insufficient data for box plot', ha='center', va='center')
            
            plt.tight_layout()
            output_path = self.output_dir / f'layer_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error creating layer comparison: {e}")
            return ""
    
    def _create_activation_distribution(self, activations: Dict[str, np.ndarray], config: Dict[str, Any]) -> str:
        """Create activation distribution plots."""
        try:
            num_layers = len(activations)
            if num_layers == 0:
                return ""
            
            cols = min(2, num_layers)
            rows = (num_layers + cols - 1) // cols
            
            fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 4*rows))
            if num_layers == 1:
                axes = [axes]
            elif rows == 1:
                axes = axes if hasattr(axes, '__iter__') else [axes]
            else:
                axes = axes.flatten()
            
            for idx, (layer_name, activation_data) in enumerate(activations.items()):
                if idx >= len(axes):
                    break
                
                ax = axes[idx]
                
                if activation_data is not None and len(activation_data) > 0:
                    flat_data = activation_data.flatten()
                    
                    # Sample if too large
                    if len(flat_data) > 10000:
                        sample_indices = np.random.choice(len(flat_data), 10000, replace=False)
                        flat_data = flat_data[sample_indices]
                    
                    # Create histogram with KDE
                    ax.hist(flat_data, bins=50, alpha=0.7, density=True, label='Histogram')
                    
                    # Add KDE if we have enough data points
                    if len(flat_data) > 100:
                        from scipy.stats import gaussian_kde
                        kde = gaussian_kde(flat_data)
                        x_range = np.linspace(flat_data.min(), flat_data.max(), 100)
                        ax.plot(x_range, kde(x_range), 'r-', label='KDE', linewidth=2)
                    
                    ax.set_title(f'Distribution: {layer_name}')
                    ax.set_xlabel('Activation Value')
                    ax.set_ylabel('Density')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f'Distribution: {layer_name}')
            
            # Hide unused subplots
            for idx in range(len(activations), len(axes)):
                axes[idx].set_visible(False)
            
            plt.tight_layout()
            output_path = self.output_dir / f'activation_distribution_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(output_path)
            
        except Exception as e:
            print(f"Error creating activation distribution: {e}")
            return ""
    
    def _create_interactive_3d(self, activations: Dict[str, np.ndarray], config: Dict[str, Any]) -> str:
        """Create interactive 3D visualizations using Plotly."""
        try:
            # Create 3D scatter plot for the first layer with sufficient data
            for layer_name, activation_data in activations.items():
                if activation_data is not None and activation_data.size > 100:
                    # Flatten and sample data
                    flat_data = activation_data.flatten()
                    if len(flat_data) > 1000:
                        sample_indices = np.random.choice(len(flat_data), 1000, replace=False)
                        flat_data = flat_data[sample_indices]
                    
                    # Create 3D coordinates (this is a simple example)
                    n = len(flat_data)
                    x = np.arange(n) % int(np.sqrt(n))
                    y = np.arange(n) // int(np.sqrt(n))
                    z = flat_data
                    
                    fig = go.Figure(data=[go.Scatter3d(
                        x=x, y=y, z=z,
                        mode='markers',
                        marker=dict(
                            size=3,
                            color=z,
                            colorscale='Viridis',
                            showscale=True,
                            colorbar=dict(title="Activation Value")
                        ),
                        name=layer_name
                    )])
                    
                    fig.update_layout(
                        title=f'3D Activation Visualization: {layer_name}',
                        scene=dict(
                            xaxis_title='X Coordinate',
                            yaxis_title='Y Coordinate',
                            zaxis_title='Activation Value'
                        ),
                        width=800,
                        height=600
                    )
                    
                    output_path = self.output_dir / f'interactive_3d_{layer_name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
                    fig.write_html(str(output_path))
                    return str(output_path)
            
            return ""
            
        except Exception as e:
            print(f"Error creating interactive 3D visualization: {e}")
            return ""
    
    def _create_comprehensive_dashboard(self, activations: Dict[str, np.ndarray], config: Dict[str, Any]) -> str:
        """Create a comprehensive visualization dashboard."""
        try:
            # Create multiple visualizations and combine them
            plots_created = []
            
            # 1. Activation heatmap
            heatmap_path = self._create_activation_heatmap(activations, config)
            if heatmap_path:
                plots_created.append(('Activation Heatmap', heatmap_path))
            
            # 2. Layer comparison
            comparison_path = self._create_layer_comparison(activations, config)
            if comparison_path:
                plots_created.append(('Layer Comparison', comparison_path))
            
            # 3. Distribution analysis
            distribution_path = self._create_activation_distribution(activations, config)
            if distribution_path:
                plots_created.append(('Activation Distributions', distribution_path))
            
            # 4. Interactive 3D (if requested)
            if config.get('include_3d', False):
                interactive_path = self._create_interactive_3d(activations, config)
                if interactive_path:
                    plots_created.append(('Interactive 3D', interactive_path))
            
            # Create HTML dashboard
            if plots_created:
                dashboard_path = self._create_html_dashboard(plots_created, activations, config)
                return dashboard_path
            
            return ""
            
        except Exception as e:
            print(f"Error creating comprehensive dashboard: {e}")
            return ""
    
    def _create_html_dashboard(self, plots: List[tuple], activations: Dict[str, np.ndarray], config: Dict[str, Any]) -> str:
        """Create an HTML dashboard combining all visualizations."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>NeuronMap Visualization Dashboard</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
                    .header {{ text-align: center; margin-bottom: 30px; }}
                    .plot-section {{ background: white; margin: 20px 0; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
                    .plot-title {{ font-size: 18px; font-weight: bold; margin-bottom: 10px; color: #333; }}
                    .plot-image {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 4px; }}
                    .stats-table {{ width: 100%; border-collapse: collapse; margin-top: 20px; }}
                    .stats-table th, .stats-table td {{ padding: 8px; text-align: left; border-bottom: 1px solid #ddd; }}
                    .stats-table th {{ background-color: #f2f2f2; }}
                    .timestamp {{ color: #666; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>NeuronMap Visualization Dashboard</h1>
                    <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            """
            
            # Add summary statistics
            html_content += """
                <div class="plot-section">
                    <div class="plot-title">Analysis Summary</div>
                    <table class="stats-table">
                        <tr><th>Layer</th><th>Shape</th><th>Mean</th><th>Std</th><th>Min</th><th>Max</th></tr>
            """
            
            for layer_name, activation_data in activations.items():
                if activation_data is not None:
                    flat_data = activation_data.flatten()
                    html_content += f"""
                        <tr>
                            <td>{layer_name}</td>
                            <td>{activation_data.shape}</td>
                            <td>{np.mean(flat_data):.4f}</td>
                            <td>{np.std(flat_data):.4f}</td>
                            <td>{np.min(flat_data):.4f}</td>
                            <td>{np.max(flat_data):.4f}</td>
                        </tr>
                    """
            
            html_content += """
                    </table>
                </div>
            """
            
            # Add each plot
            for plot_title, plot_path in plots:
                plot_filename = Path(plot_path).name
                if plot_path.endswith('.html'):
                    # For interactive plots, embed iframe
                    html_content += f"""
                        <div class="plot-section">
                            <div class="plot-title">{plot_title}</div>
                            <iframe src="{plot_filename}" width="100%" height="600" frameborder="0"></iframe>
                        </div>
                    """
                else:
                    # For static images
                    html_content += f"""
                        <div class="plot-section">
                            <div class="plot-title">{plot_title}</div>
                            <img src="{plot_filename}" alt="{plot_title}" class="plot-image">
                        </div>
                    """
            
            html_content += """
            </body>
            </html>
            """
            
            # Save dashboard
            dashboard_path = self.output_dir / f'dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
            with open(dashboard_path, 'w') as f:
                f.write(html_content)
            
            return str(dashboard_path)
            
        except Exception as e:
            print(f"Error creating HTML dashboard: {e}")
            return ""
    
    def _create_default_visualization(self, data: Any, config: Dict[str, Any]) -> str:
        """Create a default visualization."""
        return self._create_comprehensive_dashboard(data, config)
