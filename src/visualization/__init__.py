"""
Visualization module for NeuronMap
=================================

Provides comprehensive visualization capabilities for neural network analysis
including interactive plots, static visualizations, and dashboard generation.
"""

# Core visualization components
from .core_visualizer import CoreVisualizer
from .interactive_plots import InteractivePlots, PlotConfig, create_interactive_analysis

# Import existing visualization modules if available
try:
    from .activation_visualizer import ActivationVisualizer
except ImportError:
    pass

try:
    from .visualizer import ActivationVisualizer as LegacyVisualizer
except ImportError:
    pass

__all__ = [
    # Core components
    'CoreVisualizer',
    'InteractivePlots',
    'PlotConfig',
    'create_interactive_analysis',

    # Legacy components (if available)
    'ActivationVisualizer',
    'LegacyVisualizer'
]