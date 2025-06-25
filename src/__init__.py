"""
NeuronMap - Neural Network Activation Analysis Framework
=======================================================

A comprehensive framework for analyzing neural network activations
with support for multiple model architectures and advanced visualization.
"""

__version__ = "1.0.0"

# Core components
from .core.plugin_system import PluginManager, PluginBase
from .utils.config_manager import ConfigManager, get_config

# Data generation components
from .data_generation.question_generator import QuestionGenerator

# Analysis components
from .analysis.activation_extractor import ActivationExtractor
from .analysis.layer_inspector import LayerInspector

# Visualization components
from .visualization.core_visualizer import CoreVisualizer
from .visualization.interactive_plots import InteractivePlots

# Data processing
from .data_processing.quality_manager import DataQualityManager

__all__ = [
    # Core
    "PluginManager",
    "PluginBase",
    "ConfigManager",
    "get_config",

    # Data generation
    "QuestionGenerator",

    # Analysis
    "ActivationExtractor",
    "LayerInspector",

    # Visualization
    "CoreVisualizer",
    "InteractivePlots",

    # Data
    "DataQualityManager"
]
