"""
Visualization module for NeuronMap
=================================

Provides comprehensive visualization capabilities for neural network analysis
including interactive plots, static visualizations, dashboard generation,
and advanced neuron group analysis.
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

# Import advanced neuron group analysis components
try:
    from .neuron_group_visualizer import (
        NeuronGroupVisualizer, 
        NeuronGroup, 
        LearningEvent,
        create_neuron_group_analysis
    )
    NEURON_GROUP_AVAILABLE = True
except ImportError:
    NEURON_GROUP_AVAILABLE = False

# Import PyTorch neuron group analysis components
try:
    from .pytorch_neuron_group_visualizer import (
        PyTorchNeuronGroupVisualizer,
        PyTorchNeuronGroup,
        PyTorchLearningEvent,
        create_pytorch_neuron_group_analysis
    )
    PYTORCH_NEURON_GROUP_AVAILABLE = True
except ImportError:
    PYTORCH_NEURON_GROUP_AVAILABLE = False

# Import enhanced analysis workflow
try:
    from .enhanced_analysis import (
        EnhancedAnalysisWorkflow,
        integrate_neuron_group_analysis
    )
    ENHANCED_ANALYSIS_AVAILABLE = True
except ImportError:
    ENHANCED_ANALYSIS_AVAILABLE = False

__all__ = [
    # Core components
    'CoreVisualizer',
    'InteractivePlots',
    'PlotConfig',
    'create_interactive_analysis',

    # Legacy components (if available)
    'ActivationVisualizer',
    'LegacyVisualizer',
    
    # Advanced neuron group analysis (if available)
    'NeuronGroupVisualizer',
    'NeuronGroup',
    'LearningEvent', 
    'create_neuron_group_analysis',
    
    # PyTorch neuron group analysis (if available)
    'PyTorchNeuronGroupVisualizer',
    'PyTorchNeuronGroup',
    'PyTorchLearningEvent',
    'create_pytorch_neuron_group_analysis',
    
    # Enhanced analysis workflow (if available)
    'EnhancedAnalysisWorkflow',
    'integrate_neuron_group_analysis',
    
    # Availability flags
    'NEURON_GROUP_AVAILABLE',
    'PYTORCH_NEURON_GROUP_AVAILABLE',
    'ENHANCED_ANALYSIS_AVAILABLE'
]