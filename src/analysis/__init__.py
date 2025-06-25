"""
Analysis module for NeuronMap
============================

Provides comprehensive neural network analysis capabilities including
layer inspection, activation extraction, and advanced experimental methods.
"""

# Core analysis components
from .base_model_handler import BaseModelHandler, ModelConfig, ActivationResult, ModelFactory
from .activation_extractor import ActivationExtractor
from .layer_inspector import LayerInspector, LayerInfo, LayerActivationStats, analyze_model_layers

# Model handlers
try:
    from .t5_model_handler import T5ModelHandler, T5ActivationResult
except ImportError:
    pass

try:
    from .llama_model_handler import LlamaModelHandler, LlamaActivationResult
except ImportError:
    pass

# Import existing modules if available
try:
    from .activation_analyzer import ActivationAnalyzer
except ImportError:
    pass

try:
    from .advanced_experimental import AdvancedExperimentalAnalyzer
except ImportError:
    pass

try:
    from .scientific_rigor import StatisticalTester, ExperimentLogger
except ImportError:
    pass

try:
    from .ethics_bias import FairnessAnalyzer, ModelCardGenerator, AuditTrail
except ImportError:
    pass

try:
    from .conceptual_analysis import ConceptualAnalyzer
except ImportError:
    pass

# Universal model support
try:
    from .universal_model_support import (
        UniversalModelSupport, UniversalLayerMapper, ArchitectureRegistry,
        DomainAdapterRegistry, ArchitectureType, LayerType,
        AdapterConfig, create_universal_model_support, analyze_model, get_layer_info
    )
except ImportError:
    pass

try:
    from .universal_advanced_analyzer import (
        UniversalAdvancedAnalyzer, PerformanceAnalyzer, DomainSpecificAnalyzer,
        CrossArchitectureAnalyzer, PerformanceMetrics, OptimizationRecommendation,
        CrossArchitectureComparison
    )
except ImportError:
    pass

__all__ = [
    # Core components
    'ActivationExtractor',
    'LayerInspector',
    'LayerInfo',
    'LayerActivationStats',
    'analyze_model_layers',

    # Legacy components (if available)
    'ActivationAnalyzer',
    'AdvancedExperimentalAnalyzer',
    'StatisticalTester',
    'ExperimentLogger',
    'FairnessAnalyzer',
    'ModelCardGenerator',
    'AuditTrail',
    'ConceptualAnalyzer'
]
